"""Multi-Agent 协调器：战略 MIP / 战术 LP / 运营 DP"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from agent.decisions import OperationalDecision, StrategicDecision, TacticalDecision
from agent.operational_agent import OperationalAgent
from agent.risk.control_agent import RiskAdjustment, RiskControlAgent, RiskAgentDecision
from agent.planner.planner_agent import PlannerAgent
from agent.solvers.solver_pool import ParallelSolverPool
from agent.strategic_agent import StrategicAgent
from agent.tactical_agent import TacticalAgent


class EnterpriseState:
    def __init__(
        self,
        day: int = 0,
        inventory: float = 400.0,
        equipment_health: float = 0.92,
        daily_capacity: float = 0.0,
        demand_history: Optional[List[float]] = None,
        backlog: Optional[List[tuple]] = None,
        open_factories: Optional[List[int]] = None,
        lines_per_factory: Optional[dict] = None,
        workforce: int = 0,
    ):
        self.day = day
        self.inventory = inventory
        self.equipment_health = equipment_health
        self.daily_capacity = daily_capacity
        self.demand_history: List[float] = demand_history or []
        self.backlog: List[tuple] = backlog or []
        self.open_factories: List[int] = open_factories or []
        self.lines_per_factory: dict = lines_per_factory or {}
        self.workforce = workforce


class MultiAgentCoordinator:
    def __init__(
        self,
        enable_risk: bool = True,
        unit_price: float = 42.0,
        unit_cost: float = 18.0,
        product_prices: Optional[np.ndarray] = None,
        unit_production_costs: Optional[np.ndarray] = None,
        bom: Optional[np.ndarray] = None,
        raw_base_costs: Optional[np.ndarray] = None,
        parallel_solvers: bool = False,
        solver_workers: int = 4,
        use_native_solver: bool = True,
        decompose_strategic: bool = True,
    ):
        self.enable_risk = enable_risk
        self.unit_price = unit_price
        self.parallel_solvers = parallel_solvers
        self.planner: Optional[PlannerAgent] = None
        self.method_label = "OR"

        self.strategic = StrategicAgent()
        self.tactical = TacticalAgent(product_prices, unit_production_costs, bom, raw_base_costs)
        self.operational = OperationalAgent()

        if parallel_solvers:
            pool = ParallelSolverPool(
                max_workers=solver_workers,
                use_native=use_native_solver,
            )
            self.planner = PlannerAgent(
                solver_pool=pool,
                decompose=decompose_strategic,
            )
            self.method_label = "OR+Parallel"

        self.risk = RiskControlAgent(unit_price=unit_price, unit_cost=unit_cost)

    def fit_risk(self, history: np.ndarray) -> None:
        self.risk.fit(history)

    def run_strategic(
        self, state: EnterpriseState, quarter: int, daily_demand: float, budget: float,
        risk: Optional[RiskAdjustment] = None,
        macro_phase: str = "neutral",
    ) -> StrategicDecision:
        if self.planner is not None:
            d = self.planner.solve_strategic(quarter, daily_demand, budget, risk)
        else:
            d = self.strategic.decide(
                quarter, daily_demand, budget, risk,
                state=state, macro_phase=macro_phase,
            )
        if d.is_feasible:
            state.daily_capacity = d.daily_capacity
            state.open_factories = list(d.open_factories)
            state.lines_per_factory = dict(d.lines_per_factory)
            state.workforce = d.workforce
        return d

    def run_tactical(
        self, state: EnterpriseState, month: int, product_demand: np.ndarray,
        days_in_month: int, raw_mult: float = 1.0, price_mult: float = 1.0,
        risk: Optional[RiskAdjustment] = None, scenario_cap_mult: float = 1.0,
        product_demand_upper: Optional[np.ndarray] = None,
        use_robust_lp: bool = False,
    ) -> TacticalDecision:
        cap = state.daily_capacity * scenario_cap_mult * days_in_month / 30.0
        if use_robust_lp and product_demand_upper is not None:
            return self.tactical.decide_robust(
                month, product_demand, product_demand_upper, cap,
                raw_mult, price_mult, risk,
            )
        if self.planner is not None:
            return self.planner.solve_tactical(
                month,
                product_demand,
                cap,
                raw_mult,
                price_mult,
                self.tactical.product_prices,
                self.tactical.unit_production_costs,
                self.tactical.bom,
                self.tactical.raw_base_costs,
                risk,
            )
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
