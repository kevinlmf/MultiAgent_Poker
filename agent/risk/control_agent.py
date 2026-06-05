"""风险管控 Agent — 异常检测 + 应急预案，协调三层决策"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from agent.risk.anomaly_detector import DemandAnomalyDetector, Anomaly, RiskLevel
from agent.risk.contingency_planner import ContingencyPlanner, ContingencyPlan


@dataclass
class RiskAdjustment:
    risk_level: str = "low"
    demand_forecast_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    reorder_boost: int = 0
    extra_inventory_injection: float = 0.0
    budget_scale: float = 1.0
    production_scale: float = 1.0
    force_maintain: bool = False
    contingency_plan_id: Optional[str] = None
    contingency_cost: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class RiskAgentDecision:
    day: int
    adjustment: RiskAdjustment
    top_plan: Optional[ContingencyPlan]
    raw_anomalies: List[Anomaly]


class RiskControlAgent:
    def __init__(
        self,
        holding_cost: float = 2.0,
        stockout_cost: float = 50.0,
        unit_price: float = 42.0,
        unit_cost: float = 18.0,
        sensitivity: str = "medium",
    ):
        self.detector = DemandAnomalyDetector(sensitivity=sensitivity, min_history=30)
        self.planner = ContingencyPlanner(
            holding_cost=holding_cost, stockout_cost=stockout_cost,
            unit_price=unit_price, unit_cost=unit_cost,
        )
        self._fitted = False

    def fit(self, historical_demand: np.ndarray) -> None:
        if len(historical_demand) >= self.detector.min_history:
            self.detector.fit(historical_demand)
        else:
            self.detector.baseline_mean = float(np.mean(historical_demand))
            self.detector.baseline_std = max(float(np.std(historical_demand)), 1.0)
            self.detector.is_fitted = True
        self._fitted = True

    def act(
        self, day: int, demand_history: np.ndarray, current_demand: float,
        inventory: float, pending_orders: float = 0.0,
    ) -> RiskAgentDecision:
        if not self._fitted:
            self.fit(demand_history)
        ts = datetime(2024, 1, 1) + timedelta(days=day)
        anomalies = self.detector.detect(demand_history[-30:], current_demand, ts)
        adjustment = RiskAdjustment()
        top_plan = None
        if anomalies:
            primary = max(anomalies, key=lambda a: a.anomaly_score)
            adjustment.anomalies = [a.anomaly_type.value for a in anomalies]
            adjustment.risk_level = primary.risk_level.value
            adjustment.recommended_actions = primary.recommended_actions[:3]
            forecast = np.full(14, float(np.mean(demand_history[-7:])))
            plans = self.planner.generate_plans(primary, inventory, pending_orders, forecast)
            if plans:
                top_plan = plans[0]
                adjustment = self._apply_contingency(adjustment, top_plan, primary)
        return RiskAgentDecision(day, adjustment, top_plan, anomalies)

    def _apply_contingency(self, adj: RiskAdjustment, plan: ContingencyPlan, anomaly: Anomaly) -> RiskAdjustment:
        adj.contingency_plan_id = plan.plan_id
        adj.contingency_cost = plan.total_cost
        for action in plan.actions:
            at = action.action_type.value
            if at == "emergency_order":
                adj.extra_inventory_injection += action.parameters.get("quantity", 0) * 0.15
                adj.reorder_boost += int(action.parameters.get("quantity", 0) * 0.05)
            elif at == "adjust_safety_stock":
                adj.demand_forecast_multiplier *= 1.1
                adj.reorder_boost += 50
            elif at == "reduce_order_quantity":
                adj.production_scale *= 0.85
        if anomaly.risk_level == RiskLevel.CRITICAL:
            adj.budget_scale = 1.15
            adj.reorder_boost += 100
        elif anomaly.risk_level == RiskLevel.HIGH:
            adj.reorder_boost += 50
        return adj
