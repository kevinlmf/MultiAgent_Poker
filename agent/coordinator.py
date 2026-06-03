"""Multi-Agent 协调器：OR (MIP/LP/DP) 与 ML/DL 可切换"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from agent.decisions import OperationalDecision, StrategicDecision, TacticalDecision
from agent.ml.operational_dl import OperationalDLAgent
from agent.ml.strategic_ml import MLStrategicAgent
from agent.ml.tactical_ml import MLTacticalAgent
from agent.operational_agent import OperationalAgent
from agent.risk.control_agent import RiskAdjustment, RiskControlAgent, RiskAgentDecision
from agent.strategic_agent import StrategicAgent
from agent.tactical_agent import TacticalAgent


class EnterpriseState:
    def __init__(
        self,
        day: int = 0,
        inventory: float = 400.0,
        equipment_health: float = 0.92,
        daily_capacity: float = 800.0,
        demand_history: Optional[List[float]] = None,
        backlog: Optional[List[tuple]] = None,
    ):
        self.day = day
        self.inventory = inventory
        self.equipment_health = equipment_health
        self.daily_capacity = daily_capacity
        self.demand_history: List[float] = demand_history or []
        self.backlog: List[tuple] = backlog or []


class MultiAgentCoordinator:
    """
    method_family:
      - or    : 传统 MIP / LP / DP
      - ml    : ML / DL (RF + GBR + MLP)
    """

    def __init__(
        self,
        enable_risk: bool = True,
        unit_price: float = 42.0,
        unit_cost: float = 18.0,
        method_family: str = "or",
        product_prices: Optional[np.ndarray] = None,
        unit_production_costs: Optional[np.ndarray] = None,
        bom: Optional[np.ndarray] = None,
        raw_base_costs: Optional[np.ndarray] = None,
    ):
        self.method_family = method_family
        self.enable_risk = enable_risk
        self.unit_price = unit_price

        if method_family == "ml":
            self.strategic = MLStrategicAgent()
            self.tactical = MLTacticalAgent(product_prices, unit_production_costs, bom, raw_base_costs)
            self.operational = OperationalDLAgent()
            self.method_label = "ML/DL"
        else:
            self.strategic = StrategicAgent()
            self.tactical = TacticalAgent(product_prices, unit_production_costs, bom, raw_base_costs)
            self.operational = OperationalAgent()
            self.method_label = "OR"

        self.risk = RiskControlAgent(unit_price=unit_price, unit_cost=unit_cost)
        self._ml_fitted = False

    def fit_ml(self, demand: np.ndarray) -> None:
        if self.method_family != "ml":
            return
        self.strategic.fit(demand)
        self.tactical.fit(demand)
        self.operational.fit(demand)
        self._ml_fitted = True

    def fit_risk(self, history: np.ndarray) -> None:
        self.risk.fit(history)

    def run_strategic(
        self, state: EnterpriseState, quarter: int, daily_demand: float, budget: float,
        risk: Optional[RiskAdjustment] = None,
    ) -> StrategicDecision:
        d = self.strategic.decide(quarter, daily_demand, budget, risk)
        if d.is_feasible:
            state.daily_capacity = d.daily_capacity
        return d

    def run_tactical(
        self, state: EnterpriseState, month: int, product_demand: np.ndarray,
        days_in_month: int, raw_mult: float = 1.0, price_mult: float = 1.0,
        risk: Optional[RiskAdjustment] = None, scenario_cap_mult: float = 1.0,
    ) -> TacticalDecision:
        cap = state.daily_capacity * scenario_cap_mult * days_in_month / 30.0
        return self.tactical.decide(month, product_demand, cap, raw_mult, price_mult, risk)

    def run_operational(
        self, state: EnterpriseState, day: int, demand: float, inbound: float,
        breakdown_risk: float, risk: Optional[RiskAdjustment] = None,
    ) -> OperationalDecision:
        return self.operational.decide(
            day, state.inventory, demand, state.backlog,
            state.equipment_health, breakdown_risk, inbound, risk,
        )

    def run_risk(self, state: EnterpriseState, current_demand: float) -> RiskAgentDecision:
        if not self.enable_risk:
            return RiskAgentDecision(state.day, RiskAdjustment(), None, [])
        hist = np.array(state.demand_history[-60:] or [current_demand])
        return self.risk.act(state.day, hist, current_demand, state.inventory)
