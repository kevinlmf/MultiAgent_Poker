"""战术层 Agent — 月度 LP：产量 / 原料采购"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.decisions import TacticalDecision
from agent.or_optimization.production_planning import ProductionPlanningSolver
from agent.risk.control_agent import RiskAdjustment


class TacticalAgent:
    layer = "tactical"
    model = "LP"
    cadence = "monthly"

    def __init__(
        self,
        product_prices: Optional[np.ndarray] = None,
        unit_production_costs: Optional[np.ndarray] = None,
        bom: Optional[np.ndarray] = None,
        raw_base_costs: Optional[np.ndarray] = None,
    ):
        self.solver = ProductionPlanningSolver()
        self.product_prices = product_prices or np.array([45.0, 38.0, 52.0])
        self.unit_production_costs = unit_production_costs or np.array([22.0, 18.0, 26.0])
        self.bom = bom or np.array([[2.0, 1.0, 3.0], [1.0, 2.0, 1.5], [0.5, 0.5, 1.0]])
        self.raw_base_costs = raw_base_costs or np.array([8.0, 6.5, 5.0])

    def decide(
        self,
        month: int,
        product_demand: np.ndarray,
        capacity_limit: float,
        raw_cost_multiplier: float = 1.0,
        price_multiplier: float = 1.0,
        risk: Optional[RiskAdjustment] = None,
    ) -> TacticalDecision:
        risk = risk or RiskAdjustment()
        demand = product_demand * risk.demand_forecast_multiplier
        cap = capacity_limit * risk.capacity_multiplier * risk.production_scale
        raw_limits = demand @ self.bom.T + 500.0

        result = self.solver.optimize(
            product_prices=self.product_prices * price_multiplier,
            unit_production_costs=self.unit_production_costs,
            raw_material_costs=self.raw_base_costs * raw_cost_multiplier,
            bom=self.bom,
            monthly_demand=demand,
            capacity_limit=cap,
            raw_supply_limit=raw_limits,
        )
        if not result.get("success"):
            n = len(self.product_prices)
            return TacticalDecision(
                month=month,
                production_volume=np.zeros(n),
                raw_material_procurement=np.zeros(self.bom.shape[0]),
                revenue=0.0,
                total_cost=0.0,
                is_feasible=False,
            )
        return TacticalDecision(
            month=month,
            production_volume=np.array(result["production"]),
            raw_material_procurement=np.array(result["raw_procurement"]),
            revenue=float(result["revenue"]),
            total_cost=float(result["total_cost"]),
            is_feasible=True,
        )

    def decide_robust(
        self,
        month: int,
        demand_point: np.ndarray,
        demand_upper: np.ndarray,
        capacity_limit: float,
        raw_cost_multiplier: float = 1.0,
        price_multiplier: float = 1.0,
        risk: Optional[RiskAdjustment] = None,
    ) -> TacticalDecision:
        risk = risk or RiskAdjustment()
        point = demand_point * risk.demand_forecast_multiplier
        upper = demand_upper * risk.demand_forecast_multiplier
        cap = capacity_limit * risk.capacity_multiplier * risk.production_scale
        raw_limits = upper @ self.bom.T + 500.0
        result = self.solver.optimize_robust(
            product_prices=self.product_prices * price_multiplier,
            unit_production_costs=self.unit_production_costs,
            raw_material_costs=self.raw_base_costs * raw_cost_multiplier,
            bom=self.bom,
            demand_point=point,
            demand_upper=upper,
            capacity_limit=cap,
            raw_supply_limit=raw_limits,
        )
        if not result.get("success"):
            n = len(self.product_prices)
            return TacticalDecision(
                month=month,
                production_volume=np.zeros(n),
                raw_material_procurement=np.zeros(self.bom.shape[0]),
                revenue=0.0,
                total_cost=0.0,
                is_feasible=False,
            )
        return TacticalDecision(
            month=month,
            production_volume=np.array(result["production"]),
            raw_material_procurement=np.array(result["raw_procurement"]),
            revenue=float(result["revenue"]),
            total_cost=float(result["total_cost"]),
            is_feasible=True,
        )
