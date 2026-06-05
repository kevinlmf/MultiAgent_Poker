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
from agent.rl.residual_policy import ResidualRLPolicy
from agent.scenarios.profiles import ScenarioProfile, get_scenario
from agent.simulation.data_loader import RealDataBundle, load_enterprise_data
from agent.simulation.dynamic_events import EventImpactCalculator
from db.repository import SimulationRepository

try:
    from agent.forecast.forecast_agent import ForecastAgent
except ImportError:
    ForecastAgent = None  # type: ignore

logger = logging.getLogger(__name__)

QUARTERS, MONTHS = 4, 12


def _num_quarters(n_days: int) -> int:
    return max(1, (n_days + 90) // 91)


def _month_ranges_for_days(n_days: int) -> List[tuple]:
    lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ranges, s, mi = [], 0, 0
    while s < n_days:
        L = lengths[mi % 12]
        e = min(s + L, n_days)
        ranges.append((s, e))
        s = e
        mi += 1
    return ranges


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
    policy_mode: str = "or"          # or | or_rl
    simulation_days: int = 365
    train_rl: bool = False
    rl_episodes: int = 30
    parallel_solvers: bool = False
    solver_workers: int = 4
    use_native_solver: bool = True
    decompose_strategic: bool = True
    use_forecast: bool = False
    forecast_model: str = "gbr"
    forecast_fit_end: Optional[int] = None
    forecast_interval_z: float = 1.28
    use_robust_lp: bool = False
    safety_stock_days: float = 2.0
    capex_amortize_days: int = 365
    use_memory: bool = False
    apply_memory_hints: bool = True
    apply_memory_policy: bool = True
    memory_rl_min_similarity: float = 0.55
    auto_rl_from_memory: bool = True
    rl_warmstart_episodes: int = 5
    rl_cold_episodes: int = 20
    save_rl_checkpoint: bool = True
    save_to_memory: bool = True
    cycle_years: int = 0


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
    memory_recall: Optional[Any] = None
    rl_warmstarted: bool = False
    rl_train_episodes: int = 0
    cycle_phase_summary: Optional[pd.DataFrame] = None


class YearEnterpriseSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.scenario: ScenarioProfile = get_scenario(self.config.scenario_id)
        wrap_events = self.config.simulation_days > 365 or self.config.scenario_id == "economic_cycle"
        self.events = EventImpactCalculator(self.scenario.events, wrap_annual=wrap_events)
        self.coordinator = MultiAgentCoordinator(
            enable_risk=self.config.enable_risk_agent,
            unit_price=self.config.unit_price,
            parallel_solvers=self.config.parallel_solvers,
            solver_workers=self.config.solver_workers,
            use_native_solver=self.config.use_native_solver,
            decompose_strategic=self.config.decompose_strategic,
        )
        self.rl_policy: Optional[ResidualRLPolicy] = None
        if self.config.policy_mode == "or_rl":
            self.rl_policy = ResidualRLPolicy()
        self.or_advisor = ORAdvisor()
        self.repo = SimulationRepository(self.config.db_path) if self.config.persist_db else None
        self.data_bundle: Optional[RealDataBundle] = None
        self._fixed_demand: Optional[np.ndarray] = None
        self._forecast_fit_series: Optional[np.ndarray] = None
        self.forecast_agent: Optional[ForecastAgent] = None
        self.memory_agent = None
        self.memory_recall = None
        self.rl_warmstarted = False
        self.memory_triggered_rl = False
        self.rl_train_episodes = 0
        if (self.config.use_forecast or self.config.use_robust_lp) and ForecastAgent is not None:
            self.forecast_agent = ForecastAgent(model=self.config.forecast_model)
        if self.config.use_memory or self.config.save_to_memory:
            from agent.memory.memory_agent import StrategyMemoryAgent
            self.memory_agent = StrategyMemoryAgent(self.repo)

    def run(self) -> YearSimulationResult:
        if self.config.use_memory and self.memory_agent is not None:
            preview = self._preview_demand()
            self._apply_memory_recall(preview)

        if self._should_train_rl():
            episodes = self._resolve_rl_episodes()
            self.rl_train_episodes = episodes
            orig_episodes = self.config.rl_episodes
            self.config.rl_episodes = episodes
            self._train_rl()
            self.config.rl_episodes = orig_episodes
        return self._simulate()

    def _preview_demand(self) -> np.ndarray:
        demand, _ = self._load_demand()
        demand = demand * self.scenario.demand_scale
        n = min(self.config.simulation_days, len(demand))
        return demand[:n]

    def _ensure_or_rl_stack(self) -> None:
        if self.config.policy_mode != "or_rl":
            return
        if self.rl_policy is None:
            self.rl_policy = ResidualRLPolicy()
        self._refresh_coordinator()

    def _refresh_coordinator(self) -> None:
        cfg = self.config
        self.coordinator = MultiAgentCoordinator(
            enable_risk=cfg.enable_risk_agent,
            unit_price=cfg.unit_price,
            parallel_solvers=cfg.parallel_solvers,
            solver_workers=cfg.solver_workers,
            use_native_solver=cfg.use_native_solver,
            decompose_strategic=cfg.decompose_strategic,
        )

    def _apply_memory_recall(self, demand: np.ndarray) -> None:
        cfg = self.config
        if self.memory_agent is None:
            return
        window = demand[: min(90, len(demand))]
        self.memory_recall = self.memory_agent.recall(window, cfg.scenario_id)
        if not self.memory_recall:
            return

        recall = self.memory_recall
        if cfg.apply_memory_hints:
            if recall.forecast_model not in ("none", "oracle"):
                cfg.forecast_model = recall.forecast_model
                cfg.use_forecast = True
                if self.forecast_agent is None and ForecastAgent is not None:
                    self.forecast_agent = ForecastAgent(model=cfg.forecast_model)
                    self.forecast_agent.fit(demand[: max(30, int(len(demand) * 0.6))])
            cfg.use_robust_lp = recall.use_robust_lp
            if recall.parallel_solvers and not cfg.parallel_solvers:
                cfg.parallel_solvers = True
                self._refresh_coordinator()

        use_rl_hint = (
            cfg.apply_memory_policy
            and cfg.auto_rl_from_memory
            and recall.policy_mode == "or_rl"
            and recall.similarity >= cfg.memory_rl_min_similarity
        )
        if use_rl_hint and cfg.policy_mode != "or_rl":
            cfg.policy_mode = "or_rl"
            self.memory_triggered_rl = True
            logger.info(
                "Memory policy hint: switch to or_rl (sim=%.2f, profit=$%s)",
                recall.similarity,
                f"{recall.annual_profit:,.0f}",
            )

        if cfg.policy_mode == "or_rl":
            self._ensure_or_rl_stack()
            if recall.rl_checkpoint and self.rl_policy is not None:
                self.rl_policy.load_checkpoint(recall.rl_checkpoint)
                self.rl_warmstarted = True
                self.rl_policy.epsilon = 0.12
                logger.info(
                    "RL warm-start from memory id=%s (%d states, sim=%.2f)",
                    recall.memory_id,
                    self.rl_policy.num_states,
                    recall.similarity,
                )
            elif use_rl_hint:
                self.memory_triggered_rl = True

        if cfg.apply_memory_hints or use_rl_hint:
            logger.info("Memory hint (sim=%.2f): %s", recall.similarity, recall.rationale)

    def _should_train_rl(self) -> bool:
        if self.config.policy_mode != "or_rl" or self.rl_policy is None:
            return False
        if self.config.train_rl:
            return True
        return self.memory_triggered_rl or self.rl_warmstarted

    def _resolve_rl_episodes(self) -> int:
        cfg = self.config
        if cfg.train_rl and not self.memory_triggered_rl:
            return cfg.rl_episodes
        if self.rl_warmstarted:
            return cfg.rl_warmstart_episodes
        return cfg.rl_cold_episodes

    def _train_rl(self) -> None:
        cfg = self.config
        orig_days = cfg.simulation_days
        orig_persist = cfg.persist_db
        orig_save_memory = cfg.save_to_memory
        cfg.simulation_days = min(90, orig_days)
        cfg.persist_db = False
        cfg.save_to_memory = False
        for ep in range(cfg.rl_episodes):
            cfg.seed = self.config.seed + ep
            self.rng = np.random.default_rng(cfg.seed)
            self._simulate(train_rl=True)
        cfg.simulation_days = orig_days
        cfg.persist_db = orig_persist
        cfg.save_to_memory = orig_save_memory
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

        bootstrap_cap = 280.0 if cfg.scenario_id == "economic_cycle" else 0.0
        bootstrap_workers = 35 if cfg.scenario_id == "economic_cycle" else 0
        state = EnterpriseState(
            inventory=sc.initial_inventory,
            equipment_health=cfg.initial_equipment_health,
            daily_capacity=bootstrap_cap,
            workforce=bootstrap_workers,
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
        if self.forecast_agent is not None:
            fit_end = cfg.forecast_fit_end or max(30, int(len(demand) * 0.6))
            fit_series = self._forecast_fit_series if self._forecast_fit_series is not None else demand[:fit_end]
            self.forecast_agent.fit(fit_series)

        use_robust = cfg.use_robust_lp or cfg.use_forecast

        strategic, tactical = [], []
        sample_ops: List[OperationalDecision] = []
        op_rows, risk_rows, scen_rows = [], [], []
        total_demand = total_fulfilled = 0.0
        prev_rl_state = None
        prev_action = 0
        month_ranges = _month_ranges_for_days(n_days)
        n_months = len(month_ranges)
        n_quarters = _num_quarters(n_days)
        budget = sc.quarterly_budget
        active_amort: List[List[float]] = []
        cycle_phases: List[str] = []

        if cfg.scenario_id == "economic_cycle":
            from agent.scenarios.economic_cycle import phase_at_day
        else:
            phase_at_day = None  # type: ignore

        for quarter in range(n_quarters):
            q_start = quarter * 91
            if q_start >= n_days:
                break
            q_end = min((quarter + 1) * 91, n_days)
            q_mods = mods[q_start:q_end]
            q_days = demand[q_start:q_end]
            macro_phase = "neutral"
            q_budget = budget
            if phase_at_day is not None:
                macro_phase, phase_budget_scale = phase_at_day(q_start)
                q_budget = budget * phase_budget_scale
            daily_need = self._planning_daily_need(
                state, q_days, q_mods, q_end - q_start, use_forecast=cfg.use_forecast,
                macro_phase=macro_phase,
            )

            q_risk = self.coordinator.run_risk(state, float(np.mean(q_days))).adjustment if state.demand_history else None
            strat = self.coordinator.run_strategic(
                state, quarter + 1, daily_need, q_budget, q_risk, macro_phase=macro_phase,
            )
            strat.investment_cost += sum(m["extra_fixed_cost"] for m in q_mods) / max(1, len(q_mods))
            if cfg.capex_amortize_days > 0 and strat.investment_cost > 0:
                active_amort.append([
                    strat.investment_cost / cfg.capex_amortize_days,
                    float(cfg.capex_amortize_days),
                ])
            strategic.append(strat)
            if run_id:
                self.repo.save_strategic(run_id, strat)

            for month in range(quarter * 3, quarter * 3 + 3):
                if month >= n_months:
                    break
                m_start, m_end = month_ranges[month]
                m_end = min(m_end, n_days)
                m_mods = mods[m_start:m_end]
                cap_mult = float(np.mean([m["capacity_multiplier"] for m in m_mods]))
                prod_point, prod_upper = self._planning_monthly_demand(
                    state, demand, m_start, m_end, m_mods, use_forecast=cfg.use_forecast,
                    macro_phase=macro_phase,
                )
                m_risk = self.coordinator.run_risk(state, float(np.mean(demand[m_start:m_end]))).adjustment if state.demand_history else None
                tac = self.coordinator.run_tactical(
                    state, month + 1, prod_point, m_end - m_start,
                    float(np.mean([x["raw_cost_multiplier"] for x in m_mods])),
                    float(np.mean([x["price_multiplier"] for x in m_mods])),
                    m_risk,
                    scenario_cap_mult=cap_mult,
                    product_demand_upper=prod_upper if use_robust else None,
                    use_robust_lp=use_robust,
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
                    period_cost = tac.total_cost if day == m_start else 0.0
                    if cfg.capex_amortize_days <= 0:
                        if day == q_start:
                            period_cost += strat.investment_cost
                    else:
                        period_cost += sum(a[0] for a in active_amort if a[1] > 0)
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

                    if use_robust and self.forecast_agent and len(state.demand_history) >= 7:
                        fc1 = self.forecast_agent.predict(
                            np.array(state.demand_history, dtype=float),
                            horizon=1, z=cfg.forecast_interval_z,
                        )
                        gap = max(0.0, float(fc1.upper[0] - fc1.point[0]))
                        boost = int(min(100, gap * cfg.safety_stock_days * 2))
                        if cfg.scenario_id == "economic_cycle" and day_phase in ("expansion", "peak"):
                            boost += 35
                        elif cfg.scenario_id == "economic_cycle":
                            boost += 15
                        final_adj.reorder_boost = getattr(final_adj, "reorder_boost", 0) + boost

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

                    day_phase = macro_phase
                    if phase_at_day is not None:
                        day_phase, _ = phase_at_day(day)
                    cycle_phases.append(day_phase)

                    op_rows.append({
                        "day": day + 1, "quarter": quarter + 1, "month": month + 1,
                        "cycle_phase": day_phase,
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
                    for a in active_amort:
                        if a[1] > 0:
                            a[1] -= 1.0

        daily_df = pd.DataFrame(op_rows)
        quarterly_df = daily_df.groupby("quarter").agg(
            revenue=("revenue", "sum"), cost=("cost", "sum"), profit=("profit", "sum")
        ).reset_index()
        phase_df = None
        if "cycle_phase" in daily_df.columns and daily_df["cycle_phase"].nunique() > 1:
            phase_df = daily_df.groupby("cycle_phase").agg(
                revenue=("revenue", "sum"),
                cost=("cost", "sum"),
                profit=("profit", "sum"),
                demand=("demand", "sum"),
                fulfilled=("fulfilled", "sum"),
                days=("day", "count"),
            ).reset_index()
            phase_df["service_level"] = phase_df["fulfilled"] / phase_df["demand"].clip(lower=1.0)
        sl = total_fulfilled / total_demand if total_demand else 0.0
        ann_rev, ann_cost = float(daily_df["revenue"].sum()), float(daily_df["cost"].sum())
        ann_profit = float(daily_df["profit"].sum())
        advice = self.or_advisor.build_report(cfg.scenario_id, cfg.policy_mode, strategic, tactical, sample_ops)

        if run_id:
            self.repo.save_operational_batch(run_id, op_rows)
            self.repo.save_risk_batch(run_id, risk_rows)
            if scen_rows:
                self.repo.save_scenario_events(run_id, scen_rows)
            if advice:
                self.repo.save_or_recommendations(run_id, advice)
            self.repo.finalize_run(run_id, ann_rev, ann_cost, ann_profit, sl)

        if self.memory_agent is not None and cfg.save_to_memory:
            solver_hint = None
            if self.coordinator.planner:
                solver_hint = self.coordinator.planner.last_solver
            rl_q = None
            if (
                cfg.save_rl_checkpoint
                and cfg.policy_mode == "or_rl"
                and self.rl_policy
                and self.rl_policy.num_states > 0
            ):
                rl_q = self.rl_policy.to_dict()
            self.memory_agent.record(
                scenario_id=cfg.scenario_id,
                demand_series=demand,
                policy_mode=cfg.policy_mode,
                forecast_model=cfg.forecast_model if cfg.use_forecast else None,
                parallel_solvers=cfg.parallel_solvers,
                use_robust_lp=use_robust,
                use_forecast=cfg.use_forecast,
                annual_profit=ann_profit,
                service_level=sl,
                solver_hint=solver_hint,
                simulation_run_id=run_id,
                rl_q_table=rl_q,
            )

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"{cfg.scenario_id}_{cfg.policy_mode}"
        daily_df.to_csv(out / f"sim_{tag}_daily.csv", index=False)
        quarterly_df.to_csv(out / f"sim_{tag}_quarterly.csv", index=False)
        if phase_df is not None:
            phase_df.to_csv(out / f"sim_{tag}_cycle_phases.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / f"sim_{tag}_risk.csv", index=False)
        if advice:
            (out / f"sim_{tag}_advice.txt").write_text(advice.to_text(), encoding="utf-8")

        return YearSimulationResult(
            run_id, ann_profit, ann_rev, ann_cost, sl,
            strategic, tactical, daily_df, quarterly_df, self.data_bundle,
            cfg.scenario_id, cfg.policy_mode, advice,
            self.memory_recall,
            self.rl_warmstarted, self.rl_train_episodes, phase_df,
        )

    def _load_demand(self):
        cfg = self.config
        if self._fixed_demand is not None:
            return self._fixed_demand.copy(), None
        sc = self.scenario
        if sc.data_mode == "synthetic" or cfg.use_synthetic_demand:
            pass
        elif sc.data_mode in ("auto", "real") and not cfg.use_synthetic_demand:
            try:
                from agent.scenarios.data_factory import resolve_scenario_data_path
                from agent.scenarios.economic_cycle import ensure_economic_cycle_dataset
                path = cfg.data_path
                if cfg.scenario_id == "economic_cycle":
                    years = cfg.cycle_years or 5
                    path = ensure_economic_cycle_dataset(years=years)
                    max_days = cfg.simulation_days or years * 365
                elif path is None and sc.use_dedicated_data:
                    path = resolve_scenario_data_path(sc.id, auto_build=True)
                    max_days = cfg.simulation_days or 365
                else:
                    max_days = cfg.simulation_days or 365
                self.data_bundle = load_enterprise_data(
                    path or cfg.data_path, min_days=90, max_days=max_days,
                )
                p = self.data_bundle.unit_prices
                return self.data_bundle.demand.copy(), p
            except Exception as e:
                logger.warning("Real data failed (%s), using synthetic", e)
        n = max(cfg.simulation_days, 365)
        t = np.arange(n, dtype=float)
        base = self.config.base_daily_demand
        d = base * (1 + 0.25 * np.sin(2 * np.pi * t / 365.25)) * self.rng.lognormal(0, 0.1, n)
        return d, None

    def _split_demand(self, total: float) -> np.ndarray:
        mix = np.array(self.config.product_mix[: self.config.num_products])
        return total * (mix / mix.sum())

    def _planning_daily_need(
        self,
        state: EnterpriseState,
        q_days: np.ndarray,
        q_mods: list,
        n_qdays: int,
        use_forecast: bool,
        macro_phase: str = "neutral",
    ) -> float:
        mult = float(np.mean([m["demand_multiplier"] for m in q_mods])) if q_mods else 1.0
        if use_forecast and self.forecast_agent is not None and len(state.demand_history) >= 7:
            hist = np.array(state.demand_history, dtype=float)
            fc = self.forecast_agent.predict(hist, horizon=max(7, n_qdays), z=self.config.forecast_interval_z)
            base = float(np.mean(fc.point))
            need = base * mult * 1.10
        else:
            need = (float(np.sum(q_days * [m["demand_multiplier"] for m in q_mods])) / max(1, n_qdays)) * 1.25
        if macro_phase in ("expansion", "peak"):
            need *= 1.12
        return need

    def _planning_monthly_demand(
        self,
        state: EnterpriseState,
        demand: np.ndarray,
        m_start: int,
        m_end: int,
        m_mods: list,
        use_forecast: bool,
        macro_phase: str = "neutral",
    ) -> tuple:
        cfg = self.config
        if use_forecast and self.forecast_agent is not None and len(state.demand_history) >= 7:
            days = m_end - m_start
            hist = np.array(state.demand_history, dtype=float)
            fc = self.forecast_agent.predict(hist, horizon=max(1, days), z=cfg.forecast_interval_z)
            mults = [x["demand_multiplier"] for x in m_mods[:days]]
            while len(mults) < days:
                mults.append(1.0)
            point_total = float(np.sum(fc.point[:days] * mults))
            upper_total = float(np.sum(fc.upper[:days] * mults))
            if macro_phase in ("expansion", "peak"):
                oracle = float(np.sum(demand[m_start:m_end] * mults))
                point_total = max(point_total, oracle * 0.92)
                upper_total = max(upper_total, oracle * 1.05)
            return self._split_demand(point_total), self._split_demand(upper_total)
        total = float(np.sum(demand[m_start:m_end] * [x["demand_multiplier"] for x in m_mods]))
        point = self._split_demand(total)
        upper = self._split_demand(total * 1.12)
        return point, upper

    @staticmethod
    def _month_ranges():
        return _month_ranges_for_days(365)


def print_cycle_summary(phase_df: pd.DataFrame) -> None:
    print("\n经济周期分阶段汇总:")
    for _, row in phase_df.iterrows():
        sl = row.get("service_level", 0)
        print(
            f"  {row['cycle_phase']:12s}  {int(row['days']):4d}天  "
            f"利润 ${row['profit']:>12,.0f}  满足率 {sl:.1%}"
        )


def print_summary(r: YearSimulationResult, db_path: str = "data/operations.db") -> None:
    print("\n" + "=" * 56)
    horizon = f"{len(r.daily_summary)}天"
    if r.scenario_id == "economic_cycle":
        horizon = f"{len(r.daily_summary)}天 · 经济周期"
    print(f"Multi-Agent 模拟 | 场景={r.scenario_id} | 方法=MIP/LP/DP | 策略={r.policy_mode.upper()}")
    print("=" * 56)
    print("\n战略层(季度·MIP) → 建厂 / 产线 / 员工")
    show_q = r.strategic if len(r.strategic) <= 8 else r.strategic[:4] + r.strategic[-2:]
    for s in show_q:
        print(f"  Q{s.quarter}: 厂{s.open_factories} 产线{s.lines_per_factory} 员工{s.workforce} 产能{s.daily_capacity:.0f}/日 ${s.investment_cost:,.0f}")
    if len(r.strategic) > 8:
        print(f"  ... 共 {len(r.strategic)} 个季度")
    print("\n战术层(月度·LP) → 产量 / 原料采购")
    for t in r.tactical[:4]:
        print(f"  M{t.month}: 产量{np.round(t.production_volume,1).tolist()} 原料{np.round(t.raw_material_procurement,1).tolist()} 利润${t.profit:,.0f}")
    print(f"  ... 共 {len(r.tactical)} 个月")
    print("\n运营层(每日·DP) → 库存 / 维护 / 订单")
    print(f"  {horizon} | 满足率 {r.service_level:.1%}")
    label = "周期合计" if r.scenario_id == "economic_cycle" else "年度"
    print(f"\n{label}: 营收 ${r.annual_revenue:,.0f} | 成本 ${r.annual_cost:,.0f} | 净利润 ${r.annual_profit:,.0f}")
    if getattr(r, "cycle_phase_summary", None) is not None and not r.cycle_phase_summary.empty:
        print_cycle_summary(r.cycle_phase_summary)
    if getattr(r, "memory_recall", None):
        mr = r.memory_recall
        print(f"Memory: sim={mr.similarity:.2f} | {mr.rationale[:90]}...")
    if getattr(r, "rl_warmstarted", False):
        print(f"RL: warm-started ({r.rl_train_episodes} fine-tune episodes)")
    elif getattr(r, "rl_train_episodes", 0) > 0:
        print(f"RL: trained {r.rl_train_episodes} episodes")
    if r.run_id:
        print(f"数据库: {db_path} (run_id={r.run_id})")
