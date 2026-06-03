"""战略层 Agent — 季度 MIP：建厂 / 产线 / 员工"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.decisions import StrategicDecision
from agent.or_optimization.capacity_planning import CapacityPlanningSolver
from agent.risk.control_agent import RiskAdjustment


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
    ) -> StrategicDecision:
        risk = risk or RiskAdjustment()
        result = self.solver.optimize(
            site_build_costs=self.site_build_costs,
            site_base_capacities=self.site_base_capacities,
            quarterly_demand=daily_demand_target * risk.demand_forecast_multiplier,
            budget=budget * risk.budget_scale,
        )
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
