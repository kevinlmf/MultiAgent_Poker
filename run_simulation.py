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
    )
    print(f"场景: {args.scenario} | 策略: {args.policy.upper()} | 天数: {args.days}")
    result = YearEnterpriseSimulator(cfg).run()
    print_summary(result, args.db)
    if result.or_advice:
        print("\n" + result.or_advice.to_text()[:1200] + "\n  ...")


if __name__ == "__main__":
    main()
