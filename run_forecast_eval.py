#!/usr/bin/env python3
"""
Forecast Agent evaluation — walk-forward train/val/test + downstream simulation.

Phase 1 pipeline:
  1. Chronological split (no shuffle)
  2. Forecast metrics: MAPE, sMAPE, RMSE, bias per split
  3. Optional: oracle vs forecast-driven planning on TEST window
  4. Persist to SQLite (forecast_experiments, forecast_metrics, unified_experiment_metrics)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.forecast.eval_pipeline import (
    load_demand_series,
    run_downstream_comparison,
    run_forecast_benchmark,
    write_report,
)
from agent.forecast.forecast_agent import SUPPORTED_MODELS
from agent.forecast.split import chronological_split
from db.repository import SimulationRepository

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Forecast Agent walk-forward evaluation")
    p.add_argument("--data", type=str, default=None, help="Demand CSV path")
    p.add_argument("--models", type=str, default="naive,exp_smooth,gbr,mlp",
                   help=f"Comma-separated models: {','.join(SUPPORTED_MODELS)}")
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--refit-every", type=int, default=30, help="Walk-forward refit interval (days)")
    p.add_argument("--downstream", action="store_true",
                   help="Run oracle vs forecast simulation on TEST window")
    p.add_argument("--best-model", type=str, default=None,
                   help="Model for downstream sim (default: best test sMAPE)")
    p.add_argument("--scenario", type=str, default="baseline")
    p.add_argument("--policy", type=str, default="or", choices=["or", "or_rl"])
    p.add_argument("--parallel-solvers", action="store_true")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--output", type=str, default="results/forecast_eval_report.md")
    args = p.parse_args()

    demand, source = load_demand_series(args.data)
    split = chronological_split(
        len(demand), args.train_ratio, args.val_ratio, args.test_ratio,
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    repo = None if args.no_db else SimulationRepository(args.db)

    print(f"Data: {source or 'synthetic'} | days={len(demand)}")
    print(f"Split: train={split.train_days} val={split.val_days} test={split.test_days}")
    print(f"Models: {models}\n")

    forecast_df = run_forecast_benchmark(
        demand, source, models,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        refit_every=args.refit_every,
        repo=repo,
    )

    test_df = forecast_df[forecast_df["split"] == "test"].sort_values("smape")
    print("=== Test split (lower sMAPE is better) ===")
    print(test_df[["model", "mape", "smape", "rmse", "bias"]].to_string(index=False))

    downstream = None
    if args.downstream:
        best = args.best_model or str(test_df.iloc[0]["model"])
        from agent.forecast.forecast_agent import ForecastAgent
        from agent.forecast.metrics import compute_forecast_metrics

        agent = ForecastAgent(model=best)
        agent.fit(demand[split.train])
        preds, actuals, test_m = agent.rolling_evaluate(demand, split.test, refit_every=args.refit_every)
        exp_row = test_df[test_df["model"] == best]
        exp_id = int(exp_row["experiment_id"].iloc[0]) if repo and len(exp_row) and pd.notna(exp_row["experiment_id"].iloc[0]) else None

        print(f"\n=== Downstream sim on TEST ({split.test_days}d) model={best} ===")
        downstream = run_downstream_comparison(
            demand, split, best,
            scenario_id=args.scenario,
            policy_mode=args.policy,
            parallel_solvers=args.parallel_solvers,
            repo=repo,
            forecast_experiment_id=exp_id if repo else None,
            test_metrics=test_m,
        )
        for mode, d in downstream.items():
            print(f"  {mode:10s} profit=${d['profit']:,.0f}  SL={d['service_level']:.1%}  e2e={d['e2e_ms']:.0f}ms")

    out = Path(args.output)
    write_report(forecast_df, downstream, out, split)
    forecast_df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nReport: {out}")
    print(f"CSV:    {out.with_suffix('.csv')}")
    if repo:
        print(f"DB:     {args.db}")


if __name__ == "__main__":
    main()
