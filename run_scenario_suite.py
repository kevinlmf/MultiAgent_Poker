#!/usr/bin/env python3
"""
Multi-scenario real-case evaluation suite.

Each scenario uses its own CSV under data/scenarios/ (derived from base real demand).
Runs AI+OR stack: Forecast + Robust LP + CapEx amortization + Memory.

Usage:
  python run_scenario_suite.py
  python run_scenario_suite.py --quick
  python run_scenario_suite.py --rebuild-data
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.scenarios.data_factory import ensure_scenario_datasets
from agent.scenarios.profiles import SCENARIOS
from agent.simulation.year_simulator import SimulationConfig, YearEnterpriseSimulator
from db.repository import SimulationRepository


def main():
    p = argparse.ArgumentParser(description="5 real-case scenarios × AI+OR evaluation")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    p.add_argument("--policies", nargs="*", default=["or", "or_rl"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--use-forecast", action="store_true", default=True)
    p.add_argument("--no-forecast", action="store_true")
    p.add_argument("--forecast-model", type=str, default="exp_smooth")
    p.add_argument("--parallel-solvers", action="store_true")
    p.add_argument("--use-memory", action="store_true")
    p.add_argument("--apply-memory", action="store_true")
    p.add_argument("--capex-amortize-days", type=int, default=365)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--output", type=str, default="results")
    args = p.parse_args()

    use_forecast = args.use_forecast and not args.no_forecast
    days = 90 if args.quick else 365
    tag = "90d" if args.quick else "365d"

    paths = ensure_scenario_datasets(force=args.rebuild_data)
    print("=" * 72)
    print("Real-Case Scenario Suite (dedicated CSV per scenario)")
    print("=" * 72)
    print(f"Days: {days} | Forecast: {use_forecast} ({args.forecast_model}) | CapEx amort: {args.capex_amortize_days}d")
    for sid in args.scenarios:
        if sid in paths:
            print(f"  {sid:18s} → {paths[sid]}")
    print()

    rows = []
    for sid in args.scenarios:
        if sid not in SCENARIOS:
            continue
        sc = SCENARIOS[sid]
        print(f"--- {sc.name} ({sid}) ---")
        for policy in args.policies:
            cfg = SimulationConfig(
                seed=args.seed,
                scenario_id=sid,
                policy_mode=policy,
                simulation_days=days,
                output_dir=args.output,
                db_path=args.db,
                persist_db=not args.no_db,
                use_forecast=use_forecast,
                forecast_model=args.forecast_model,
                use_robust_lp=use_forecast,
                capex_amortize_days=args.capex_amortize_days,
                parallel_solvers=args.parallel_solvers,
                use_memory=args.use_memory,
                apply_memory_hints=args.apply_memory,
                save_to_memory=not args.no_db,
                train_rl=(policy == "or_rl"),
                rl_episodes=20,
            )
            r = YearEnterpriseSimulator(cfg).run()
            src = r.data_bundle.source_path if r.data_bundle else "-"
            rows.append({
                "scenario": sid,
                "scenario_name": sc.name,
                "policy": policy,
                "data_source": src,
                "mean_demand": r.data_bundle.mean_demand if r.data_bundle else None,
                "profit": r.annual_profit,
                "revenue": r.annual_revenue,
                "cost": r.annual_cost,
                "service_level": r.service_level,
                "run_id": r.run_id,
                "days": len(r.daily_summary),
            })
            print(
                f"  [{policy:6}] profit ${r.annual_profit:>12,.0f}  SL {r.service_level:>6.1%}  "
                f"data={Path(src).name if src != '-' else src}"
            )
        print()

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"scenario_suite_{tag}.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        f"# Real-Case Scenario Suite ({tag})",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Forecast: {use_forecast} | Robust LP: {use_forecast} | CapEx amort: {args.capex_amortize_days}d",
        "",
        "## Per-scenario data files",
        "",
    ]
    for sid in args.scenarios:
        if sid in paths:
            sub = df[df["scenario"] == sid]
            md = sub["mean_demand"].iloc[0] if not sub.empty else 0
            md_lines.append(f"- `{sid}`: `{paths[sid]}` (sim mean demand ≈ {md:.0f})")
    md_lines.extend(["", "## Results", ""])
    md_lines.append(df.to_string(index=False))
    md_path = out / f"scenario_suite_{tag}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 72)
    print(df.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"MD:  {md_path}")
    if not args.no_db:
        print(f"DB:  {args.db} (strategy_memory + runs)")
    print("=" * 72)


if __name__ == "__main__":
    main()
