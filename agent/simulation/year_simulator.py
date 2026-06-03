"""
一年企业 Multi-Agent 模拟
战略(季度/MIP) → 战术(月度/LP) → 运营(每日/DP) + 风险管控 + SQLite
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.coordinator import EnterpriseState, MultiAgentCoordinator
from agent.decisions import OperationalDecision, StrategicDecision, TacticalDecision
from agent.recommendation.or_advisor import ORAdvisor, ORAdviceReport
from agent.recommendation.ml_advisor import MLAdvisor, MLAdviceReport
from agent.rl.residual_policy import ResidualRLPolicy
from agent.scenarios.profiles import ScenarioProfile, get_scenario
from agent.simulation.data_loader import RealDataBundle, load_enterprise_data
from agent.simulation.dynamic_events import EventImpactCalculator
from db.repository import SimulationRepository

logger = logging.getLogger(__name__)

QUARTERS, MONTHS = 4, 12


@dataclass
class SimulationConfig:
    seed: int = 42
    base_daily_demand: float = 280.0
    unit_price: float = 42.0
    initial_inventory: float = 400.0
    initial_equipment_health: float = 0.92
    quarterly_budget: float = 1_200_000.0
    num_products: int = 3
    product_mix: tuple = (0.5, 0.3, 0.2)
    output_dir: str = "results"
    data_path: Optional[str] = None
    use_synthetic_demand: bool = False
    enable_risk_agent: bool = True
    db_path: Optional[str] = None
    persist_db: bool = True
    scenario_id: str = "baseline"
    policy_mode: str = "or"          # or | ml | or_rl
    simulation_days: int = 365
    train_rl: bool = False
    rl_episodes: int = 30


@dataclass
class YearSimulationResult:
    run_id: Optional[int]
    annual_profit: float
    annual_revenue: float
    annual_cost: float
    service_level: float
    strategic: List[StrategicDecision]
    tactical: List[TacticalDecision]
    daily_summary: pd.DataFrame
    quarterly_summary: pd.DataFrame
    data_bundle: Optional[RealDataBundle] = None
    scenario_id: str = "baseline"
    policy_mode: str = "or"
    or_advice: Optional[ORAdviceReport] = None
    ml_advice: Optional[MLAdviceReport] = None
    method_family: str = "or"


class YearEnterpriseSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.scenario: ScenarioProfile = get_scenario(self.config.scenario_id)
        self.events = EventImpactCalculator(self.scenario.events)
        family = "or" if self.config.policy_mode in ("or", "or_rl") else "ml"
        self.coordinator = MultiAgentCoordinator(
            enable_risk=self.config.enable_risk_agent,
            unit_price=self.config.unit_price,
            method_family=family,
        )
        self.rl_policy: Optional[ResidualRLPolicy] = None
        if self.config.policy_mode == "or_rl":
            self.rl_policy = ResidualRLPolicy()
        self.or_advisor = ORAdvisor()
        self.ml_advisor = MLAdvisor()
        self.repo = SimulationRepository(self.config.db_path) if self.config.persist_db else None
        self.data_bundle: Optional[RealDataBundle] = None

    def run(self) -> YearSimulationResult:
        if self.config.train_rl and self.rl_policy:
            self._train_rl()
        return self._simulate()

    def _train_rl(self) -> None:
        cfg = self.config
        orig_days = cfg.simulation_days
        cfg.simulation_days = min(90, orig_days)
        cfg.persist_db = False
        for ep in range(cfg.rl_episodes):
            cfg.seed = self.config.seed + ep
            self.rng = np.random.default_rng(cfg.seed)
            result = self._simulate(train_rl=True)
        cfg.simulation_days = orig_days
        cfg.persist_db = self.config.persist_db
        if self.rl_policy:
            self.rl_policy.epsilon = 0.05

    def _simulate(self, train_rl: bool = False) -> YearSimulationResult:
        cfg = self.config
        sc = self.scenario
        demand, prices = self._load_demand()
        demand = demand * sc.demand_scale
        n_days = min(cfg.simulation_days, len(demand))
        demand = demand[:n_days]
        if prices is not None:
            prices = prices[:n_days]
        mods = [self.events.day_modifiers(d) for d in range(n_days)]
        for m in mods:
            m["raw_cost_multiplier"] *= sc.raw_cost_scale
            m["capacity_multiplier"] *= sc.capacity_scale

        state = EnterpriseState(
            inventory=sc.initial_inventory,
            equipment_health=cfg.initial_equipment_health,
        )

        run_id = None
        if self.repo:
            run_id = self.repo.create_run(
                cfg.seed,
                self.data_bundle.source_path if self.data_bundle else None,
                cfg.use_synthetic_demand,
                cfg.enable_risk_agent,
                scenario_id=cfg.scenario_id,
                policy_mode=cfg.policy_mode,
                simulation_days=n_days,
            )
            dates = None
            if self.data_bundle and self.data_bundle.dates is not None:
                dates = [str(d.date()) for d in self.data_bundle.dates[:n_days]]
            self.repo.save_demand_series(run_id, demand, prices, dates)

        if len(demand) >= 30:
            self.coordinator.fit_risk(demand[:60])
        if self.coordinator.method_family == "ml":
            self.coordinator.fit_ml(demand)

        strategic, tactical = [], []
        sample_ops: List[OperationalDecision] = []
        op_rows, risk_rows, scen_rows = [], [], []
        total_demand = total_fulfilled = 0.0
        prev_rl_state = None
        prev_action = 0
        month_ranges = self._month_ranges()
        budget = sc.quarterly_budget

        for quarter in range(QUARTERS):
            q_start, q_end = quarter * 91, min((quarter + 1) * 91, n_days)
            q_mods = mods[q_start:q_end]
            q_days = demand[q_start:q_end]
            daily_need = (float(np.sum(q_days * [m["demand_multiplier"] for m in q_mods])) / max(1, q_end - q_start)) * 1.25

            q_risk = self.coordinator.run_risk(state, float(np.mean(q_days))).adjustment if state.demand_history else None
            strat = self.coordinator.run_strategic(state, quarter + 1, daily_need, budget, q_risk)
            strat.investment_cost += sum(m["extra_fixed_cost"] for m in q_mods) / max(1, len(q_mods))
            strategic.append(strat)
            if run_id:
                self.repo.save_strategic(run_id, strat)

            for month in range(quarter * 3, min(quarter * 3 + 3, MONTHS)):
                m_start, m_end = month_ranges[month]
                if m_start >= n_days:
                    continue
                m_end = min(m_end, n_days)
                m_mods = mods[m_start:m_end]
                cap_mult = float(np.mean([m["capacity_multiplier"] for m in m_mods]))
                prod_demand = self._split_demand(float(np.sum(demand[m_start:m_end] * [x["demand_multiplier"] for x in m_mods])))
                m_risk = self.coordinator.run_risk(state, float(np.mean(demand[m_start:m_end]))).adjustment if state.demand_history else None
                tac = self.coordinator.run_tactical(
                    state, month + 1, prod_demand, m_end - m_start,
                    float(np.mean([x["raw_cost_multiplier"] for x in m_mods])),
                    float(np.mean([x["price_multiplier"] for x in m_mods])),
                    m_risk,
                    scenario_cap_mult=cap_mult,
                )
                tactical.append(tac)
                if run_id:
                    self.repo.save_tactical(run_id, tac)
                daily_inbound = float(np.sum(tac.production_volume)) / max(1, m_end - m_start) if tac.is_feasible else 0.0

                for day in range(m_start, m_end):
                    mod = mods[day]
                    base_d = demand[day]
                    real_d = base_d * mod["demand_multiplier"]
                    state.day = day
                    state.demand_history.append(base_d)
                    total_demand += real_d

                    if mod["active_event_names"]:
                        scen_rows.append({"day": day + 1, "events": "|".join(mod["active_event_names"])})

                    price = float(prices[day]) * mod["price_multiplier"] if prices is not None else cfg.unit_price * mod["price_multiplier"]
                    period_cost = (tac.total_cost if day == m_start else 0) + (strat.investment_cost if day == q_start else 0)
                    if mod["extra_fixed_cost"] > 0 and not mod.get("_charged"):
                        period_cost += mod["extra_fixed_cost"]
                        mod["_charged"] = True

                    risk_dec = self.coordinator.run_risk(state, base_d)
                    final_adj = risk_dec.adjustment
                    rl_action_id = 0
                    if self.rl_policy and cfg.policy_mode == "or_rl":
                        stress = min(1.0, len(mod["active_event_names"]) * 0.35)
                        backlog_vol = sum(b[0] for b in state.backlog)
                        rl_state = ResidualRLPolicy.build_state(
                            state.inventory, state.demand_history,
                            state.equipment_health, backlog_vol, stress, base_d,
                        )
                        rl_adj = self.rl_policy.select(rl_state, explore=train_rl)
                        rl_action_id = rl_adj.action_id
                        final_adj = ResidualRLPolicy.merge(risk_dec.adjustment, rl_adj)

                    op = self.coordinator.run_operational(
                        state, day + 1, real_d, daily_inbound,
                        0.1 if mod["capacity_multiplier"] < 0.9 else 0.03,
                        final_adj,
                    )
                    if len(sample_ops) < 8:
                        sample_ops.append(op)
                    fulfilled = op.units_sold + op.backlog_fulfilled
                    total_fulfilled += min(fulfilled, real_d)
                    revenue = fulfilled * price
                    profit = revenue - op.operating_cost - period_cost
                    day_reward = profit / 2000.0 + (1.0 if op.stockout == 0 else -0.5)

                    if self.rl_policy and cfg.policy_mode == "or_rl" and train_rl and prev_rl_state is not None:
                        self.rl_policy.update(prev_rl_state, prev_action, day_reward, rl_state)
                    if self.rl_policy and cfg.policy_mode == "or_rl":
                        prev_rl_state = rl_state
                        prev_action = rl_action_id

                    state.inventory = op.inventory_end
                    if op.maintain_equipment:
                        state.equipment_health = min(1.0, state.equipment_health + 0.02)
                    else:
                        state.equipment_health = max(0.5, state.equipment_health - 0.005)
                    if op.stockout > 0:
                        state.backlog.append((op.stockout, 1.2))
                    state.backlog = [(q * 0.95, w) for q, w in state.backlog if q > 1.0][:20]

                    op_rows.append({
                        "day": day + 1, "quarter": quarter + 1, "month": month + 1,
                        "demand": real_d, "fulfilled": fulfilled, "inventory": state.inventory,
                        "reorder": op.reorder_qty, "maintain": op.maintain_equipment,
                        "revenue": revenue, "profit": profit, "cost": op.operating_cost + period_cost,
                        "risk_level": risk_dec.adjustment.risk_level,
                        "policy_mode": cfg.policy_mode,
                        "rl_action": rl_action_id if cfg.policy_mode == "or_rl" else -1,
                        "units_sold": op.units_sold, "stockout": op.stockout,
                        "inventory_end": op.inventory_end, "operating_cost": op.operating_cost,
                    })
                    risk_rows.append({
                        "day": day + 1, "risk_level": risk_dec.adjustment.risk_level,
                        "anomalies": "|".join(risk_dec.adjustment.anomalies) or "-",
                        "plan_id": risk_dec.adjustment.contingency_plan_id or "-",
                        "cost": risk_dec.adjustment.contingency_cost,
                    })

        daily_df = pd.DataFrame(op_rows)
        quarterly_df = daily_df.groupby("quarter").agg(
            revenue=("revenue", "sum"), cost=("cost", "sum"), profit=("profit", "sum")
        ).reset_index()
        sl = total_fulfilled / total_demand if total_demand else 0.0
        ann_rev, ann_cost = float(daily_df["revenue"].sum()), float(daily_df["cost"].sum())
        ann_profit = float(daily_df["profit"].sum())
        if self.coordinator.method_family == "ml":
            ml_advice = self.ml_advisor.build_report(cfg.scenario_id, strategic, tactical)
            advice = None
        else:
            ml_advice = None
            advice = self.or_advisor.build_report(cfg.scenario_id, cfg.policy_mode, strategic, tactical, sample_ops)

        if run_id:
            self.repo.save_operational_batch(run_id, op_rows)
            self.repo.save_risk_batch(run_id, risk_rows)
            if scen_rows:
                self.repo.save_scenario_events(run_id, scen_rows)
            if advice:
                self.repo.save_or_recommendations(run_id, advice)
            elif ml_advice:
                from agent.recommendation.or_advisor import ORAdviceReport
                self.repo.save_or_recommendations(
                    run_id,
                    ORAdviceReport(cfg.scenario_id, "ml", ml_advice.recommendations),
                )
            self.repo.finalize_run(run_id, ann_rev, ann_cost, ann_profit, sl)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"{cfg.scenario_id}_{cfg.policy_mode}"
        daily_df.to_csv(out / f"sim_{tag}_daily.csv", index=False)
        quarterly_df.to_csv(out / f"sim_{tag}_quarterly.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / f"sim_{tag}_risk.csv", index=False)
        txt = advice.to_text() if advice else (ml_advice.to_text() if ml_advice else "")
        (out / f"sim_{tag}_advice.txt").write_text(txt, encoding="utf-8")

        return YearSimulationResult(
            run_id, ann_profit, ann_rev, ann_cost, sl,
            strategic, tactical, daily_df, quarterly_df, self.data_bundle,
            cfg.scenario_id, cfg.policy_mode, advice,
            ml_advice, self.coordinator.method_family,
        )

    def _load_demand(self):
        cfg = self.config
        sc = self.scenario
        if sc.data_mode == "synthetic" or cfg.use_synthetic_demand:
            pass
        elif sc.data_mode in ("auto", "real") and not cfg.use_synthetic_demand:
            try:
                self.data_bundle = load_enterprise_data(cfg.data_path, min_days=365, max_days=365)
                p = self.data_bundle.unit_prices
                return self.data_bundle.demand.copy(), p
            except Exception as e:
                logger.warning("Real data failed (%s), using synthetic", e)
        t = np.arange(365, dtype=float)
        base = self.config.base_daily_demand
        d = base * (1 + 0.25 * np.sin(2 * np.pi * t / 365.25)) * self.rng.lognormal(0, 0.1, 365)
        return d, None

    def _split_demand(self, total: float) -> np.ndarray:
        mix = np.array(self.config.product_mix[: self.config.num_products])
        return total * (mix / mix.sum())

    @staticmethod
    def _month_ranges():
        lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        ranges, s = [], 0
        for L in lengths:
            ranges.append((s, s + L))
            s += L
        return ranges


def print_summary(r: YearSimulationResult, db_path: str = "data/operations.db") -> None:
    print("\n" + "=" * 56)
    method = "MIP/LP/DP" if r.method_family == "or" else "ML/DL"
    print(f"Multi-Agent 模拟 | 场景={r.scenario_id} | 方法={method} | 策略={r.policy_mode.upper()}")
    print("=" * 56)
    print("\n战略层(季度·MIP) → 建厂 / 产线 / 员工")
    for s in r.strategic:
        print(f"  Q{s.quarter}: 厂{s.open_factories} 产线{s.lines_per_factory} 员工{s.workforce} 产能{s.daily_capacity:.0f}/日 ${s.investment_cost:,.0f}")
    print("\n战术层(月度·LP) → 产量 / 原料采购")
    for t in r.tactical[:4]:
        print(f"  M{t.month}: 产量{np.round(t.production_volume,1).tolist()} 原料{np.round(t.raw_material_procurement,1).tolist()} 利润${t.profit:,.0f}")
    print(f"  ... 共 {len(r.tactical)} 个月")
    print("\n运营层(每日·DP) → 库存 / 维护 / 订单")
    print(f"  {len(r.daily_summary)} 天 | 满足率 {r.service_level:.1%}")
    print(f"\n年度: 营收 ${r.annual_revenue:,.0f} | 成本 ${r.annual_cost:,.0f} | 净利润 ${r.annual_profit:,.0f}")
    if r.run_id:
        print(f"数据库: {db_path} (run_id={r.run_id})")
