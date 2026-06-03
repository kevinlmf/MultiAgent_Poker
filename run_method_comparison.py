#!/usr/bin/env python3
"""
三层决策方法对比（5 场景 + 动态事件）:
  - OR  : 传统 MIP / LP / DP
  - ML  : RandomForest + GBR + MLP(DL)
  - OR+RL: OR 基线 + Residual RL 增强

用法:
  python run_method_comparison.py
  python run_method_comparison.py --quick
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.scenarios.profiles import SCENARIOS
from agent.simulation.year_simulator import YearEnterpriseSimulator, SimulationConfig


METHODS = [
    ("or", "OR (MIP/LP/DP)"),
    ("ml", "ML/DL (RF/GBR/MLP)"),
    ("or_rl", "OR + Residual RL"),
]


def rank_methods(rows: list) -> str:
    best = max(rows, key=lambda r: (r["profit"], r["service_level"]))
    return best["policy"]


def main():
    p = argparse.ArgumentParser(description="OR vs ML/DL vs OR+RL 全场景对比")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-train-rl", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    days = 90 if args.quick else 365
    all_rows = []
    layer_win = {"strategic": {}, "tactical": {}, "operational": {}}

    print("=" * 70)
    print("三层决策 · 方法对比 (OR vs ML/DL vs OR+RL)")
    print("=" * 70)
    print(f"天数: {days} | 场景: {len(args.scenarios)} | 动态事件: 已启用\n")

    for sid in args.scenarios:
        if sid not in SCENARIOS:
            continue
        sc = SCENARIOS[sid]
        print(f"### 场景: {sc.name} ({sid})")
        scenario_rows = []

        for policy, label in METHODS:
            cfg = SimulationConfig(
                seed=args.seed,
                scenario_id=sid,
                policy_mode=policy,
                simulation_days=days,
                use_synthetic_demand=args.synthetic,
                output_dir=args.output,
                persist_db=False,
                train_rl=(policy == "or_rl" and not args.no_train_rl),
                rl_episodes=20,
            )
            r = YearEnterpriseSimulator(cfg).run()
            backends = {
                "or": ("MIP", "LP", "DP"),
                "ml": ("ML-RF", "ML-GBR", "DL-MLP"),
                "or_rl": ("MIP", "LP", "DP+RL"),
            }
            b = backends.get(policy, ("?", "?", "?"))
            row = {
                "scenario": sid,
                "scenario_name": sc.name,
                "policy": policy,
                "method_label": label,
                "strategic": b[0],
                "tactical": b[1],
                "operational": b[2],
                "profit": r.annual_profit,
                "revenue": r.annual_revenue,
                "cost": r.annual_cost,
                "service_level": r.service_level,
                "days": len(r.daily_summary),
            }
            scenario_rows.append(row)
            all_rows.append(row)
            print(f"  {label:22} 利润 ${r.annual_profit:>12,.0f}  满足率 {r.service_level:>6.1%}")

        winner = max(scenario_rows, key=lambda x: (x["profit"], x["service_level"]))
        print(f"  → 本场景最优: {winner['method_label']}\n")

    df = pd.DataFrame(all_rows)
    tag = "90d" if args.quick else "365d"
    csv_path = Path(args.output) / f"method_comparison_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    summary_lines = [
        f"# 方法对比报告 ({tag})",
        f"- 时间: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 按场景最优方法",
        "",
    ]
    for sid in df["scenario"].unique():
        sub = df[df["scenario"] == sid]
        w = sub.loc[sub["profit"].idxmax()]
        summary_lines.append(
            f"- **{w['scenario_name']}**: {w['method_label']} "
            f"(利润 ${w['profit']:,.0f}, 满足率 {w['service_level']:.1%})"
        )

    summary_lines.extend([
        "",
        "## 三层方法说明",
        "",
        "| 层级 | OR (传统) | ML/DL (新) |",
        "|------|-----------|------------|",
        "| 战略/季度 | MIP 建厂/产线/员工 | RandomForest 产能预测 |",
        "| 战术/月度 | LP 产量/原料 | GBR 需求预测 + 学习分配 |",
        "| 运营/每日 | DP 库存/维护/订单 | MLP 神经网络补货+维护 |",
        "",
        "## 全表",
        "",
        df.to_string(index=False),
    ])
    md_path = Path(args.output) / f"method_comparison_{tag}.md"
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 70)
    print(df.groupby("policy")[["profit", "service_level"]].mean().round(2))
    print(f"\nCSV: {csv_path}")
    print(f"MD:  {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
