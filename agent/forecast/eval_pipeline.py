"""Forecast walk-forward evaluation + optional downstream simulation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.forecast.forecast_agent import ForecastAgent, SUPPORTED_MODELS
from agent.forecast.metrics import ForecastMetrics
from agent.forecast.split import TimeSeriesSplit, chronological_split
from agent.simulation.data_loader import load_enterprise_data
from agent.simulation.year_simulator import SimulationConfig, YearEnterpriseSimulator
from db.repository import SimulationRepository


def run_forecast_benchmark(
    demand: np.ndarray,
    data_source: Optional[str],
    models: List[str],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    refit_every: int = 30,
    repo: Optional[SimulationRepository] = None,
) -> pd.DataFrame:
    split = chronological_split(len(demand), train_ratio, val_ratio, test_ratio)
    rows = []

    for model in models:
        agent = ForecastAgent(model=model)
        t0 = time.perf_counter()
        metrics_map = agent.evaluate_splits(demand, split, refit_every=refit_every)
        total_ms = (time.perf_counter() - t0) * 1000.0

        exp_id = None
        if repo:
            exp_id = repo.create_forecast_experiment(
                model_name=model,
                data_source=data_source,
                train_days=split.train_days,
                val_days=split.val_days,
                test_days=split.test_days,
                fit_latency_ms=agent.last_fit_ms,
                notes=f"refit_every={refit_every}",
            )
            for split_name, m in metrics_map.items():
                repo.save_forecast_metrics(
                    exp_id, split_name, m.mae, m.rmse, m.mape, m.smape, m.bias, m.n_samples,
                    predict_latency_ms=agent.last_predict_ms,
                )

        for split_name, m in metrics_map.items():
            rows.append({
                "model": model,
                "split": split_name,
                "mape": m.mape,
                "smape": m.smape,
                "rmse": m.rmse,
                "bias": m.bias,
                "mae": m.mae,
                "n_samples": m.n_samples,
                "experiment_id": exp_id,
                "eval_total_ms": total_ms,
            })

    return pd.DataFrame(rows)


def run_downstream_comparison(
    demand: np.ndarray,
    split: TimeSeriesSplit,
    model: str,
    scenario_id: str = "baseline",
    policy_mode: str = "or",
    parallel_solvers: bool = False,
    repo: Optional[SimulationRepository] = None,
    forecast_experiment_id: Optional[int] = None,
    test_metrics: Optional[ForecastMetrics] = None,
) -> Dict[str, object]:
    """
    Compare oracle planning (actual demand visible to planner) vs forecast-driven
    on the test window only.
    """
    test_days = split.test_days
    results = {}

    for mode, use_fc in (("oracle", False), ("forecast", True)):
        cfg = SimulationConfig(
            scenario_id=scenario_id,
            policy_mode=policy_mode,
            simulation_days=test_days,
            persist_db=repo is not None,
            use_forecast=use_fc,
            forecast_model=model,
            forecast_fit_end=split.test.start,
            parallel_solvers=parallel_solvers,
            data_path=None,
            use_synthetic_demand=False,
        )
        sim = YearEnterpriseSimulator(cfg)
        sim._fixed_demand = demand[split.test]
        sim._forecast_fit_series = demand[: split.test.start]
        t0 = time.perf_counter()
        r = sim.run()
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        solver_ms = 0.0
        if sim.coordinator.planner:
            solver_ms = sim.coordinator.planner.last_elapsed_ms

        unified_id = None
        if repo:
            unified_id = repo.save_unified_metrics(
                forecast_experiment_id=forecast_experiment_id if mode == "forecast" else None,
                simulation_run_id=r.run_id,
                split_name="test",
                model_name=model if mode == "forecast" else "oracle",
                policy_mode=policy_mode,
                scenario_id=scenario_id,
                forecast_mape=test_metrics.mape if test_metrics and mode == "forecast" else None,
                forecast_smape=test_metrics.smape if test_metrics and mode == "forecast" else None,
                annual_profit=r.annual_profit,
                service_level=r.service_level,
                solver_latency_ms=solver_ms,
                end_to_end_latency_ms=e2e_ms,
                notes=f"planning={mode}, test_days={test_days}",
            )

        results[mode] = {
            "profit": r.annual_profit,
            "service_level": r.service_level,
            "run_id": r.run_id,
            "e2e_ms": e2e_ms,
            "unified_id": unified_id,
        }

    return results


def load_demand_series(data_path: Optional[str]) -> Tuple[np.ndarray, Optional[str]]:
    bundle = load_enterprise_data(data_path, min_days=90, max_days=365)
    return bundle.demand.copy(), bundle.source_path


def write_report(
    forecast_df: pd.DataFrame,
    downstream: Optional[Dict],
    output_path: Path,
    split: TimeSeriesSplit,
) -> None:
    lines = [
        "# Forecast Agent Evaluation Report",
        "",
        f"- Train / Val / Test days: {split.train_days} / {split.val_days} / {split.test_days}",
        "",
        "## Forecast metrics (walk-forward one-step)",
        "",
    ]
    cols = ["model", "split", "mape", "smape", "rmse", "bias", "mae", "n_samples"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in forecast_df.iterrows():
        lines.append("| " + " | ".join(
            f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in cols
        ) + " |")
    lines.append("")
    if downstream:
        lines.extend([
            "## Downstream simulation on TEST window",
            "",
            "| Planning | Profit | Service Level | E2E ms |",
            "|----------|--------|---------------|--------|",
        ])
        for mode, d in downstream.items():
            lines.append(
                f"| {mode} | ${d['profit']:,.0f} | {d['service_level']:.1%} | {d['e2e_ms']:.0f} |"
            )
        lines.append("")
        if "oracle" in downstream and "forecast" in downstream:
            dp = downstream["forecast"]["profit"] - downstream["oracle"]["profit"]
            ds = downstream["forecast"]["service_level"] - downstream["oracle"]["service_level"]
            lines.append(f"- Forecast vs Oracle Δprofit: ${dp:,.0f}")
            lines.append(f"- Forecast vs Oracle Δservice: {ds:+.1%}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
