"""战略层 Agent — 季度 MIP：建厂 / 产线 / 员工（支持增量扩产）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from agent.decisions import StrategicDecision
from agent.or_optimization.capacity_planning import CapacityPlanningSolver
from agent.risk.control_agent import RiskAdjustment

if TYPE_CHECKING:
    from agent.coordinator import EnterpriseState


class StrategicAgent:
    layer = "strategic"
    model = "MIP"
    cadence = "quarterly"

    def __init__(
        self,
        site_build_costs: Optional[np.ndarray] = None,
        site_base_capacities: Optional[np.ndarray] = None,
    ):
        self.solver = CapacityPlanningSolver()
        self.site_build_costs = site_build_costs or np.array([500_000.0, 420_000.0, 380_000.0])
        self.site_base_capacities = site_base_capacities or np.array([200.0, 150.0, 120.0])

    def decide(
        self,
        quarter: int,
        daily_demand_target: float,
        budget: float,
        risk: Optional[RiskAdjustment] = None,
        state: Optional["EnterpriseState"] = None,
        macro_phase: str = "neutral",
    ) -> StrategicDecision:
        risk = risk or RiskAdjustment()
        target = daily_demand_target * risk.demand_forecast_multiplier * 1.05

        if state is not None and state.daily_capacity > 0:
            return self._decide_incremental(quarter, target, budget, risk, state, macro_phase)

        result = self.solver.optimize(
            site_build_costs=self.site_build_costs,
            site_base_capacities=self.site_base_capacities,
            quarterly_demand=target,
            budget=budget * risk.budget_scale,
        )
        return self._from_solver_result(quarter, result, risk)

    def _decide_incremental(
        self,
        quarter: int,
        target: float,
        budget: float,
        risk: RiskAdjustment,
        state: "EnterpriseState",
        macro_phase: str,
    ) -> StrategicDecision:
        """Only invest when capacity gap exists; avoid repeated factory CapEx."""
        current = state.daily_capacity * risk.capacity_multiplier
        if current >= target:
            return StrategicDecision(
                quarter=quarter,
                open_factories=list(state.open_factories),
                lines_per_factory=dict(state.lines_per_factory),
                workforce=state.workforce,
                daily_capacity=current,
                investment_cost=0.0,
                is_feasible=True,
            )

        gap = target - current
        recession = macro_phase in ("recession", "trough", "steady")
        hot = macro_phase in ("expansion", "peak", "recovery")
        eff_budget = budget * risk.budget_scale * (0.45 if recession else (1.15 if hot else 1.0))

        s = self.solver
        workers_needed = int(np.ceil(gap / max(s.worker_productivity, 1.0)))
        workers_needed = min(workers_needed, s.max_workers - state.workforce)
        worker_cost = workers_needed * s.worker_annual_cost

        if workers_needed > 0 and worker_cost <= eff_budget:
            new_workers = state.workforce + workers_needed
            new_cap = current + workers_needed * s.worker_productivity
            return StrategicDecision(
                quarter=quarter,
                open_factories=list(state.open_factories),
                lines_per_factory=dict(state.lines_per_factory),
                workforce=new_workers,
                daily_capacity=new_cap,
                investment_cost=float(worker_cost),
                is_feasible=True,
            )

        if recession:
            return StrategicDecision(
                quarter=quarter,
                open_factories=list(state.open_factories),
                lines_per_factory=dict(state.lines_per_factory),
                workforce=state.workforce,
                daily_capacity=current,
                investment_cost=0.0,
                is_feasible=True,
            )

        remaining = eff_budget - worker_cost if workers_needed > 0 else eff_budget
        if remaining <= 0:
            return StrategicDecision(
                quarter=quarter,
                open_factories=list(state.open_factories),
                lines_per_factory=dict(state.lines_per_factory),
                workforce=state.workforce,
                daily_capacity=current,
                investment_cost=0.0,
                is_feasible=True,
            )

        open_sites = list(state.open_factories)
        lines = dict(state.lines_per_factory)
        spent = max(0.0, worker_cost if workers_needed > 0 else 0.0)
        new_cap = current + max(0, workers_needed) * s.worker_productivity
        new_workers = state.workforce + max(0, workers_needed)

        if not recession:
            for i in range(len(self.site_build_costs)):
                if new_cap >= target:
                    break
                if i in open_sites:
                    add_lines = min(2, s.max_lines_per_site - lines.get(i, 0))
                    line_cost = add_lines * s.line_fixed_cost
                    if add_lines > 0 and spent + line_cost <= eff_budget:
                        lines[i] = lines.get(i, 0) + add_lines
                        spent += line_cost
                        new_cap += add_lines * s.line_capacity
                    continue
                build = float(self.site_build_costs[i])
                if spent + build > eff_budget:
                    continue
                open_sites.append(i)
                lines[i] = 2
                spent += build + 2 * s.line_fixed_cost
                new_cap += float(self.site_base_capacities[i]) + 2 * s.line_capacity

        return StrategicDecision(
            quarter=quarter,
            open_factories=open_sites,
            lines_per_factory=lines,
            workforce=new_workers,
            daily_capacity=new_cap,
            investment_cost=float(spent),
            is_feasible=True,
        )

    @staticmethod
    def _from_solver_result(quarter: int, result: dict, risk: RiskAdjustment) -> StrategicDecision:
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
        return StrategicDecision(
            quarter=quarter,
            open_factories=result.get("open_sites", []),
            lines_per_factory=result.get("lines_per_site", {}),
            workforce=int(result.get("workforce", 0)),
            daily_capacity=float(result["total_capacity"]) * risk.capacity_multiplier,
            investment_cost=float(result["total_cost"]),
            is_feasible=True,
        )
