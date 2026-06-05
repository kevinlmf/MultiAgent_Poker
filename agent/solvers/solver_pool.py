"""
Parallel solver pool — MIP / Greedy / Native race for fast incumbents.

Pattern: multiple solver agents on the same sub-problem; first good feasible
solution can be used immediately; best cost wins at timeout (Learning-to-Search
style incumbent sharing without replacing exact MIP when it finishes first).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agent.or_optimization.capacity_planning import CapacityPlanningSolver
from agent.or_optimization.production_planning import ProductionPlanningSolver
from agent.solvers.heuristic import greedy_capacity_plan, greedy_production_plan

logger = logging.getLogger(__name__)


@dataclass
class SolverOutcome:
    payload: Dict
    solver: str
    elapsed_ms: float
    cost: float
    is_exact: bool = False


class ParallelSolverPool:
    """
    Multi-threaded solver race on a single sub-problem.

    - MIP agent: exact (when scipy succeeds)
    - Greedy agent: fast feasible incumbent
    - Native agent: OpenMP multi-start greedy (optional C++ extension)
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_sec: float = 3.0,
        use_native: bool = True,
    ):
        self.max_workers = max(1, max_workers)
        self.timeout_sec = timeout_sec
        self.use_native = use_native
        self._capacity_mip = CapacityPlanningSolver()
        self._production_lp = ProductionPlanningSolver()
        self._stats: Dict[str, int] = {"mip": 0, "greedy": 0, "native": 0, "lp": 0}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def race_capacity(
        self,
        site_build_costs: np.ndarray,
        site_base_capacities: np.ndarray,
        quarterly_demand: float,
        budget: float,
    ) -> SolverOutcome:
        tasks: List[Tuple[str, Callable[[], Dict]]] = [
            ("mip", lambda: self._capacity_mip.optimize(
                site_build_costs, site_base_capacities, quarterly_demand, budget
            )),
            ("greedy", lambda: greedy_capacity_plan(
                site_build_costs, site_base_capacities, quarterly_demand, budget,
                line_capacity=self._capacity_mip.line_capacity,
                line_fixed_cost=self._capacity_mip.line_fixed_cost,
                worker_productivity=self._capacity_mip.worker_productivity,
                worker_quarterly_cost=self._capacity_mip.worker_annual_cost,
                max_lines_per_site=self._capacity_mip.max_lines_per_site,
                max_workers=self._capacity_mip.max_workers,
            )),
        ]
        if self.use_native:
            tasks.append(("native", lambda: self._native_capacity(
                site_build_costs, site_base_capacities, quarterly_demand, budget
            )))
        outcome = self._race(tasks, cost_key="total_cost", exact_solvers={"mip"})
        self._stats[outcome.solver] = self._stats.get(outcome.solver, 0) + 1
        return outcome

    def race_production(
        self,
        product_prices: np.ndarray,
        unit_production_costs: np.ndarray,
        raw_material_costs: np.ndarray,
        bom: np.ndarray,
        monthly_demand: np.ndarray,
        capacity_limit: float,
        raw_supply_limit: np.ndarray,
    ) -> SolverOutcome:
        lp_args = (
            product_prices, unit_production_costs, raw_material_costs,
            bom, monthly_demand, capacity_limit, raw_supply_limit,
        )
        tasks: List[Tuple[str, Callable[[], Dict]]] = [
            ("lp", lambda: self._production_lp.optimize(*lp_args)),
            ("greedy", lambda: greedy_production_plan(*lp_args)),
        ]
        outcome = self._race(tasks, cost_key="total_cost", exact_solvers={"lp"}, minimize=False)
        # for production we maximize margin — flip comparison via net_margin
        if outcome.payload.get("net_margin") is not None:
            outcome.cost = -float(outcome.payload["net_margin"])
        self._stats[outcome.solver] = self._stats.get(outcome.solver, 0) + 1
        return outcome

    def _race(
        self,
        tasks: List[Tuple[str, Callable[[], Dict]]],
        cost_key: str,
        exact_solvers: Optional[set] = None,
        minimize: bool = True,
    ) -> SolverOutcome:
        exact_solvers = exact_solvers or set()
        deadline = time.perf_counter() + self.timeout_sec
        best: Optional[SolverOutcome] = None

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            future_map: Dict[Future, Tuple[str, float]] = {}
            for name, fn in tasks:
                t0 = time.perf_counter()
                future_map[pool.submit(fn)] = (name, t0)

            pending = set(future_map.keys())
            while pending and time.perf_counter() < deadline:
                done, pending = wait(
                    pending,
                    timeout=min(0.05, max(0.001, deadline - time.perf_counter())),
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    name, t0 = future_map[fut]
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    try:
                        payload = fut.result()
                    except Exception as exc:
                        logger.debug("Solver %s failed: %s", name, exc)
                        continue
                    if not payload.get("success"):
                        continue
                    cost = float(payload.get(cost_key, payload.get("net_margin", 0)))
                    if not minimize and "net_margin" in payload:
                        cost = -float(payload["net_margin"])
                    cand = SolverOutcome(
                        payload=payload,
                        solver=name,
                        elapsed_ms=elapsed,
                        cost=cost,
                        is_exact=name in exact_solvers,
                    )
                    if best is None or cand.cost < best.cost:
                        best = cand
                    if cand.is_exact:
                        return cand

            for fut in pending:
                fut.cancel()

        if best is not None:
            return best
        return SolverOutcome(
            payload={"success": False, "status": "all_solvers_failed"},
            solver="none",
            elapsed_ms=0.0,
            cost=float("inf"),
        )

    def _native_capacity(
        self,
        site_build_costs: np.ndarray,
        site_base_capacities: np.ndarray,
        quarterly_demand: float,
        budget: float,
    ) -> Dict:
        try:
            from native.bindings import native_capacity_greedy
            return native_capacity_greedy(
                site_build_costs, site_base_capacities, quarterly_demand, budget,
                line_capacity=self._capacity_mip.line_capacity,
                line_fixed_cost=self._capacity_mip.line_fixed_cost,
                worker_productivity=self._capacity_mip.worker_productivity,
                worker_quarterly_cost=self._capacity_mip.worker_annual_cost,
                num_trials=32,
            )
        except Exception:
            return self._python_parallel_greedy_starts(
                site_build_costs, site_base_capacities, quarterly_demand, budget
            )

    def _python_parallel_greedy_starts(
        self,
        site_build_costs: np.ndarray,
        site_base_capacities: np.ndarray,
        quarterly_demand: float,
        budget: float,
    ) -> Dict:
        """Multi-start greedy without C++ — parallel permutations."""
        m = len(site_build_costs)
        orders = [np.argsort(site_build_costs)]
        rng = np.random.default_rng(42)
        for _ in range(min(8, max(1, self.max_workers))):
            orders.append(rng.permutation(m))

        best: Optional[Dict] = None
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(
                    greedy_capacity_plan,
                    site_build_costs,
                    site_base_capacities,
                    quarterly_demand,
                    budget,
                    site_order=order,
                    line_capacity=self._capacity_mip.line_capacity,
                    line_fixed_cost=self._capacity_mip.line_fixed_cost,
                    worker_productivity=self._capacity_mip.worker_productivity,
                    worker_quarterly_cost=self._capacity_mip.worker_annual_cost,
                )
                for order in orders
            ]
            for fut in futures:
                r = fut.result()
                if r.get("success") and (best is None or r["total_cost"] < best["total_cost"]):
                    best = r
        return best or {"success": False, "status": "parallel_greedy_failed"}
