#!/usr/bin/env python3
"""
Full macro economic cycle simulation (default 5 years).

Phases: recovery → expansion → peak → recession → rebound
  - Dedicated multi-year demand CSV with cycle_phase column
  - Phase-aware strategic budgeting (recession caps CapEx)
  - Incremental capacity (no repeated factory builds)
  - Annual calendar events repeat each year

Usage:
  python run_economic_cycle.py --years 5 --policy or_rl --use-forecast --train-rl
  python run_economic_cycle.py --quick          # 2-year mini cycle
  python run_economic_cycle.py --years 5 --use-memory

Note: on 5y cycles, OR+RL needs --train-rl (default ON for or_rl).
      Pure OR without RL often loses money on expansion/peak — that is expected stress.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.scenarios.economic_cycle import ensure_economic_cycle_dataset
from agent.simulation.year_simulator import (
    SimulationConfig,
    YearEnterpriseSimulator,
    print_summary,
)


def main():
    p = argparse.ArgumentParser(description="Multi-year economic cycle simulation")
    p.add_argument("--years", type=int, default=5, help="Cycle length in years (default 5)")
    p.add_argument("--quick", action="store_true", help="2-year mini cycle (recovery+expansion only)")
    p.add_argument("--policy", type=str, default="or_rl", choices=["or", "or_rl"])
    p.add_argument("--train-rl", action="store_true", help="Train RL before main run (default ON for or_rl)")
    p.add_argument("--no-train-rl", action="store_true", help="Skip RL training (or_rl will behave like OR)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-forecast", action="store_true", default=True)
    p.add_argument("--no-forecast", action="store_true")
    p.add_argument("--forecast-model", type=str, default="gbr",
                   choices=["naive", "exp_smooth", "gbr", "mlp", "lstm"])
    p.add_argument("--parallel-solvers", action="store_true")
    p.add_argument("--capex-amortize-days", type=int, default=365)
    p.add_argument("--use-memory", action="store_true")
    p.add_argument("--apply-memory", action="store_true")
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--output", type=str, default="results")
    args = p.parse_args()

    years = 2 if args.quick else args.years
    days = years * 365
    use_forecast = args.use_forecast and not args.no_forecast
    train_rl = (args.policy == "or_rl" and not args.no_train_rl) or args.train_rl

    if args.rebuild_data:
        ensure_economic_cycle_dataset(years=5, force=True)
        if years != 5:
            ensure_economic_cycle_dataset(years=years, force=True)
    data_path = ensure_economic_cycle_dataset(years=years, force=False)

    print("=" * 72)
    print("Economic Cycle Simulation")
    print("=" * 72)
    print(f"Years: {years} ({days} days) | Policy: {args.policy.upper()}")
    print(f"Forecast: {use_forecast} ({args.forecast_model}) | RL train: {train_rl}")
    print(f"CapEx amort: {args.capex_amortize_days}d")
    print(f"Data: {data_path}")
    if args.quick:
        print("Note: --quick = recovery + expansion only (hard phases, often negative without RL)")
    else:
        print("Phases: recovery → expansion → peak → recession → rebound")
    if args.policy == "or_rl" and not train_rl:
        print("WARNING: or_rl without RL training ≈ pure OR → expect low SL in expansion/peak")
    print()

    cfg = SimulationConfig(
        seed=args.seed,
        scenario_id="economic_cycle",
        policy_mode=args.policy,
        simulation_days=days,
        cycle_years=years,
        data_path=data_path,
        output_dir=args.output,
        db_path=args.db,
        persist_db=not args.no_db,
        use_forecast=use_forecast,
        forecast_model=args.forecast_model,
        use_robust_lp=use_forecast,
        safety_stock_days=3.0 if days > 365 else 2.0,
        capex_amortize_days=args.capex_amortize_days,
        parallel_solvers=args.parallel_solvers,
        train_rl=train_rl,
        rl_episodes=15 if days > 365 else 20,
        use_memory=args.use_memory,
        apply_memory_hints=args.apply_memory or args.use_memory,
        save_to_memory=not args.no_db,
    )

    t0 = datetime.now()
    result = YearEnterpriseSimulator(cfg).run()
    elapsed = (datetime.now() - t0).total_seconds()

    print_summary(result, args.db)
    out = Path(args.output)
    tag = f"economic_cycle_{years}y_{args.policy}"
    md = out / f"{tag}_report.md"
    lines = [
        f"# Economic Cycle Report ({years}y · {args.policy})",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Elapsed: {elapsed:.0f}s | RL trained: {train_rl}",
        f"- Days: {days} | Service level: {result.service_level:.1%}",
        f"- Total profit: ${result.annual_profit:,.0f}",
        f"- Revenue: ${result.annual_revenue:,.0f} | Cost: ${result.annual_cost:,.0f}",
        "",
    ]
    if result.cycle_phase_summary is not None and not result.cycle_phase_summary.empty:
        lines.append("## Phase breakdown")
        lines.append("")
        lines.append(result.cycle_phase_summary.to_string(index=False))
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")
    print(f"Phase CSV: {out / f'sim_economic_cycle_{args.policy}_cycle_phases.csv'}")
    if result.annual_profit < 0 and args.policy == "or":
        print("\nTip: 5y pure OR often loses in expansion/peak. Try:")
        print("  python run_economic_cycle.py --years 5 --policy or_rl --train-rl")
    print("=" * 72)


if __name__ == "__main__":
    main()
