#!/usr/bin/env python3
"""一年 Multi-Agent 企业运营模拟 — 主入口"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.scenarios.profiles import list_scenarios
from agent.simulation.year_simulator import YearEnterpriseSimulator, SimulationConfig, print_summary
from db.repository import SimulationRepository


def main():
    p = argparse.ArgumentParser(description="Multi-Agent 企业模拟")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", type=str, default="baseline", help="场景 ID")
    p.add_argument("--policy", type=str, default="or", choices=["or", "or_rl"], help="or=纯OR, or_rl=OR+RL")
    p.add_argument("--train-rl", action="store_true")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-risk", action="store_true")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--parallel-solvers", action="store_true", help="MIP+Greedy+Native parallel race")
    p.add_argument("--solver-workers", type=int, default=4)
    p.add_argument("--no-native", action="store_true", help="Skip C++ OpenMP, use Python threads only")
    p.add_argument("--no-decompose", action="store_true", help="Disable site-level decomposition")
    p.add_argument("--use-forecast", action="store_true", help="Forecast Agent drives planner (not oracle demand)")
    p.add_argument("--forecast-model", type=str, default="gbr",
                   choices=["naive", "exp_smooth", "gbr", "mlp", "lstm"])
    p.add_argument("--no-robust-lp", action="store_true", help="Disable robust LP even with forecast")
    p.add_argument("--capex-amortize-days", type=int, default=365,
                   help="Amortize strategic CapEx over N days (0=lump sum)")
    p.add_argument("--use-memory", action="store_true", help="Enable strategy memory retrieval")
    p.add_argument("--apply-memory", action="store_true", help="Apply memory hints to forecast/solver settings")
    p.add_argument("--no-apply-memory-policy", action="store_true",
                   help="Do not switch to or_rl from memory recall")
    p.add_argument("--memory-rl-sim", type=float, default=0.55,
                   help="Min similarity to apply memory or_rl + RL warm-start")
    p.add_argument("--no-auto-rl-train", action="store_true",
                   help="Disable auto RL train when memory triggers or_rl")
    p.add_argument("--no-save-memory", action="store_true", help="Do not write run to strategy memory")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--list-runs", action="store_true")
    args = p.parse_args()

    if args.list_scenarios:
        for s in list_scenarios():
            print(f"  {s.id}: {s.name} — {s.description}")
        return

    if args.list_runs:
        print(SimulationRepository(args.db).list_runs().to_string(index=False))
        return

    cfg = SimulationConfig(
        seed=args.seed,
        scenario_id=args.scenario,
        policy_mode=args.policy,
        train_rl=args.train_rl,
        simulation_days=args.days,
        data_path=args.data,
        use_synthetic_demand=args.synthetic,
        enable_risk_agent=not args.no_risk,
        db_path=args.db,
        persist_db=not args.no_db,
        parallel_solvers=args.parallel_solvers,
        solver_workers=args.solver_workers,
        use_native_solver=not args.no_native,
        decompose_strategic=not args.no_decompose,
        use_forecast=args.use_forecast,
        forecast_model=args.forecast_model,
        use_robust_lp=args.use_forecast and not args.no_robust_lp,
        capex_amortize_days=args.capex_amortize_days,
        use_memory=args.use_memory,
        apply_memory_hints=args.apply_memory or args.use_memory,
        apply_memory_policy=not args.no_apply_memory_policy,
        memory_rl_min_similarity=args.memory_rl_sim,
        auto_rl_from_memory=not args.no_auto_rl_train,
        save_to_memory=not args.no_save_memory,
    )
    mode = args.policy.upper()
    if args.parallel_solvers:
        mode += " + PARALLEL"
    if args.use_forecast:
        mode += f" + FCST({args.forecast_model})"
    if args.use_forecast and not args.no_robust_lp:
        mode += " + ROBUST-LP"
    if args.capex_amortize_days > 0:
        mode += f" + CapEx/{args.capex_amortize_days}d"
    if args.use_memory:
        mode += " + MEMORY(RL)"
    print(f"场景: {args.scenario} | 策略: {mode} | 天数: {args.days}")
    result = YearEnterpriseSimulator(cfg).run()
    print_summary(result, args.db)
    if result.or_advice:
        print("\n" + result.or_advice.to_text()[:1200] + "\n  ...")


if __name__ == "__main__":
    main()
