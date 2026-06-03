"""运营层 Agent — 每日 DP：库存 / 维护 / 订单执行"""

from __future__ import annotations

from typing import List, Optional, Tuple

from agent.decisions import OperationalDecision
from agent.or_optimization.daily_operations import DailyOperationsSolver
from agent.risk.control_agent import RiskAdjustment


class OperationalAgent:
    layer = "operational"
    model = "DP"
    cadence = "daily"

    def __init__(self):
        self.solver = DailyOperationsSolver()

    def decide(
        self,
        day: int,
        inventory: float,
        demand: float,
        backlog: List[Tuple[float, float]],
        equipment_health: float,
        breakdown_risk: float,
        inbound_production: float = 0.0,
        risk: Optional[RiskAdjustment] = None,
    ) -> OperationalDecision:
        risk = risk or RiskAdjustment()
        inventory += risk.extra_inventory_injection

        raw = self.solver.optimize_day(
            inventory=inventory,
            demand=demand * risk.demand_forecast_multiplier,
            backlog_orders=backlog,
            equipment_health=equipment_health,
            breakdown_risk=breakdown_risk,
            inbound_from_production=inbound_production,
        )
        reorder = raw.get("reorder_qty", 0)
        if risk.reorder_boost > 0:
            reorder = min(self.solver.max_reorder, reorder + risk.reorder_boost)

        cost = raw["operating_cost"] + risk.contingency_cost
        maintain = bool(raw["maintain_equipment"])
        if getattr(risk, "force_maintain", False):
            maintain = True
            cost += self.solver.maintenance_cost

        return OperationalDecision(
            day=day,
            reorder_qty=int(reorder),
            maintain_equipment=maintain,
            inventory_end=float(raw["ending_inventory"]),
            units_sold=float(raw["sales"]),
            backlog_fulfilled=float(raw.get("fulfilled_backlog", 0)),
            stockout=float(raw["stockout"]),
            operating_cost=cost,
        )
