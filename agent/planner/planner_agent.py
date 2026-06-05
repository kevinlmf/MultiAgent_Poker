"""
Master Planner Agent — problem decomposition + parallel sub-agents.

Decomposition pattern (Dantzig-Wolfe / Benders spirit):
  Master: budget + demand coupling
  Sub-agents: one per factory site (parallel greedy / local MIP)

Solver pool pattern (parallel B&B incumbent sharing):
  MIP + Greedy + Native run concurrently; best feasible wins.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.decisions import StrategicDecision, TacticalDecision
from agent.risk.control_agent import RiskAdjustment
from agent.solvers.solver_pool import ParallelSolverPool

logger = logging.getLogger(__name__)


class SiteSubAgent:
    """Sub-agent: local capacity plan for one candidate site."""

    def __init__(self, site_id: int, build_cost: float, base_capacity: float):
        self.site_id = site_id
        self.build_cost = build_cost
        self.base_capacity = base_capacity

    def solve_local(
        self,
        budget_share: float,
        demand_share: float,
        *,
        line_capacity: float = 120.0,
        line_fixed_cost: float = 80_000.0,
    ) -> Dict:
        if budget_share < self.build_cost:
            return {"site_id": self.site_id, "feasible": False, "cost": float("inf"), "capacity": 0.0}
        lines = min(12, max(1, int(demand_share / max(line_capacity, 1))))
        line_cost = lines * line_fixed_cost
        if self.build_cost + line_cost > budget_share:
            lines = max(0, int((budget_share - self.build_cost) / line_fixed_cost))
        cap = self.base_capacity + lines * line_capacity
        cost = self.build_cost + lines * line_fixed_cost
        return {
            "site_id": self.site_id,
            "feasible": cap >= demand_share * 0.3 or lines > 0,
            "lines": lines,
            "cost": cost,
            "capacity": cap,
        }


class PlannerAgent:
    """
    Master planner coordinating decomposed sub-agents and solver races.

    Modes:
      - decompose=True: parallel site sub-agents, master merges (fast query path)
      - solver_pool: MIP || Greedy || Native race (fast incumbent path)
    """

    def __init__(
        self,
        site_build_costs: Optional[np.ndarray] = None,
        site_base_capacities: Optional[np.ndarray] = None,
        solver_pool: Optional[ParallelSolverPool] = None,
        decompose: bool = True,
    ):
        self.site_build_costs = site_build_costs or np.array([500_000.0, 420_000.0, 380_000.0])
        self.site_base_capacities = site_base_capacities or np.array([200.0, 150.0, 120.0])
        self.pool = solver_pool or ParallelSolverPool()
        self.decompose = decompose
        self.last_solver: Optional[str] = None
        self.last_elapsed_ms: float = 0.0

    def solve_strategic(
        self,
        quarter: int,
        daily_demand: float,
        budget: float,
        risk: Optional[RiskAdjustment] = None,
    ) -> StrategicDecision:
        risk = risk or RiskAdjustment()
        # Same units as CapacityPlanningSolver: forecast DAILY throughput for the quarter
        q_demand = daily_demand * risk.demand_forecast_multiplier

        if self.decompose:
            decomp = self._decomposed_capacity(q_demand, budget * risk.budget_scale)
            if decomp is not None:
                return self._to_strategic(quarter, decomp, risk)

        outcome = self.pool.race_capacity(
            self.site_build_costs,
            self.site_base_capacities,
            q_demand,
            budget * risk.budget_scale,
        )
        self.last_solver = outcome.solver
        self.last_elapsed_ms = outcome.elapsed_ms
        return self._to_strategic(quarter, outcome.payload, risk)

    def solve_tactical(
        self,
        month: int,
        product_demand: np.ndarray,
        capacity_limit: float,
        raw_cost_multiplier: float,
        price_multiplier: float,
        product_prices: np.ndarray,
        unit_production_costs: np.ndarray,
        bom: np.ndarray,
        raw_base_costs: np.ndarray,
        risk: Optional[RiskAdjustment] = None,
    ) -> TacticalDecision:
        risk = risk or RiskAdjustment()
        demand = product_demand * risk.demand_forecast_multiplier
        cap = capacity_limit * risk.capacity_multiplier * risk.production_scale
        raw_limits = demand @ bom.T + 500.0
        prices = product_prices * price_multiplier
        raw_costs = raw_base_costs * raw_cost_multiplier

        outcome = self.pool.race_production(
            prices, unit_production_costs, raw_costs, bom,
            demand, cap, raw_limits,
        )
        self.last_solver = outcome.solver
        self.last_elapsed_ms = outcome.elapsed_ms
        r = outcome.payload
        if not r.get("success"):
            n = len(product_prices)
            return TacticalDecision(
                month=month,
                production_volume=np.zeros(n),
                raw_material_procurement=np.zeros(bom.shape[0]),
                revenue=0.0,
                total_cost=0.0,
                is_feasible=False,
            )
        return TacticalDecision(
            month=month,
            production_volume=r["production"],
            raw_material_procurement=r["raw_procurement"],
            revenue=float(r["revenue"]),
            total_cost=float(r["total_cost"]),
            is_feasible=True,
        )

    def _decomposed_capacity(self, quarterly_demand: float, budget: float) -> Optional[Dict]:
        """Parallel site sub-agents → master knapsack-style merge."""
        m = len(self.site_build_costs)
        sub_agents = [
            SiteSubAgent(i, float(self.site_build_costs[i]), float(self.site_base_capacities[i]))
            for i in range(m)
        ]
        demand_share = quarterly_demand / max(m, 1)
        budget_share = budget / max(m, 1)

        locals_: List[Dict] = []
        with ThreadPoolExecutor(max_workers=m) as pool:
            futs = [
                pool.submit(a.solve_local, budget_share * 1.5, demand_share)
                for a in sub_agents
            ]
            for fut in as_completed(futs):
                locals_.append(fut.result())

        # Master: pick cheapest sites until demand met (parallel B&B leaf evaluation)
        locals_.sort(key=lambda x: x["cost"] / max(x["capacity"], 1.0))
        open_sites: List[int] = []
        lines: Dict[int, int] = {}
        total_cap = 0.0
        total_cost = 0.0
        for loc in locals_:
            if total_cap >= quarterly_demand:
                break
            if not loc["feasible"] or total_cost + loc["cost"] > budget:
                continue
            sid = loc["site_id"]
            open_sites.append(sid)
            lines[sid] = int(loc["lines"])
            total_cap += loc["capacity"]
            total_cost += loc["cost"]

        if total_cap < quarterly_demand * 0.5:
            return None

        workers = 0
        w_cost = 11_250.0
        w_prod = 8.0
        while total_cap < quarterly_demand and total_cost + 20 * w_cost <= budget:
            workers += 20
            total_cost += 20 * w_cost
            total_cap += 20 * w_prod

        if total_cap < quarterly_demand * 0.75:
            return None

        self.last_solver = "decomposed_sites"
        return {
            "success": True,
            "status": "decomposed_master",
            "open_sites": open_sites,
            "lines_per_site": lines,
            "workforce": workers,
            "total_capacity": total_cap,
            "total_cost": total_cost,
        }

    @staticmethod
    def _to_strategic(quarter: int, result: Dict, risk: RiskAdjustment) -> StrategicDecision:
        if not result.get("success"):
            return StrategicDecision(
                quarter=quarter,
                open_factories=[],
                lines_per_factory={},
                workforce=0,
                daily_capacity=0.0,
                investment_cost=0.0,
                is_feasible=False,
            )
        q_cap = float(result["total_capacity"])
        return StrategicDecision(
            quarter=quarter,
            open_factories=result.get("open_sites", []),
            lines_per_factory=result.get("lines_per_site", {}),
            workforce=int(result.get("workforce", 0)),
            daily_capacity=q_cap * risk.capacity_multiplier,
            investment_cost=float(result["total_cost"]),
            is_feasible=True,
        )
