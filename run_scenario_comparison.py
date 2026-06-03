#!/usr/bin/env python3
"""
多场景 × 双策略对比：OR 基线 vs OR+RL
生成 CSV 汇总、Markdown 年度报告、SQLite（含 or_recommendations）
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.scenarios.profiles import SCENARIOS
from agent.simulation.year_simulator import YearEnterpriseSimulator, SimulationConfig
from db.repository import SimulationRepository


def _df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_markdown_report(df: pd.DataFrame, runs_df: pd.DataFrame, out: Path, days: int) -> None:
    lines = [
        "# 企业运营全场景年度报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 模拟天数: {days}",
        f"- 场景数: {df['scenario'].nunique()}",
        "",
        "## 1. OR vs OR+RL 汇总",
        "",
        _df_to_md(df),
        "",
        "## 2. 分场景胜负",
        "",
    ]
    for sid in df["scenario"].unique():
        sub = df[df["scenario"] == sid]
        or_r = sub[sub["policy"] == "or"].iloc[0]
        rl_r = sub[sub["policy"] == "or_rl"].iloc[0]
        dp = rl_r["profit"] - or_r["profit"]
        ds = rl_r["service_level"] - or_r["service_level"]
        win = "OR+RL" if dp > 0 or (dp == 0 and ds > 0) else "OR"
        lines.append(
            f"- **{or_r['scenario_name']}** (`{sid}`): 更优={win}，"
            f"利润差 ${dp:,.0f}，满足率差 {ds:+.1%}"
        )
    lines.extend(["", "## 3. 数据库 Run 记录", ""])
    if not runs_df.empty:
        lines.append(runs_df.to_string(index=False))
    lines.extend([
        "",
        "## 4. OR 建议查询",
        "",
        "```sql",
        "SELECT layer, period, priority, action FROM or_recommendations",
        "WHERE run_id = <id> ORDER BY layer, id;",
        "```",
        "",
        "文本备份: `results/sim_<场景>_<策略>_or_advice.txt`",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="全场景 OR vs OR+RL 对比报告")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-train-rl", action="store_true", help="跳过 RL 预训练（默认会训练）")
    p.add_argument("--quick", action="store_true", help="90 天")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results")
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--no-db", action="store_true")
    args = p.parse_args()

    train_rl = not args.no_train_rl
    days = 90 if args.quick else 365
    rows = []
    run_ids = []

    print("=" * 60)
    print("企业运营全场景报告")
    print("=" * 60)
    print(f"天数: {days} | 场景: {args.scenarios} | DB: {not args.no_db} | RL训练: {train_rl}\n")

    for sid in args.scenarios:
        if sid not in SCENARIOS:
            continue
        sc = SCENARIOS[sid]
        print(f"--- {sc.name} ({sid}) ---")

        for mode in ("or", "or_rl"):
            cfg = SimulationConfig(
                seed=args.seed,
                scenario_id=sid,
                policy_mode=mode,
                simulation_days=days,
                data_path=args.data,
                use_synthetic_demand=args.synthetic,
                output_dir=args.output,
                db_path=args.db,
                persist_db=not args.no_db,
                train_rl=(mode == "or_rl" and train_rl),
                rl_episodes=25,
            )
            result = YearEnterpriseSimulator(cfg).run()
            n_rec = len(result.or_advice.recommendations) if result.or_advice else 0
            rows.append({
                "scenario": sid,
                "scenario_name": sc.name,
                "policy": mode,
                "run_id": result.run_id,
                "profit": result.annual_profit,
                "revenue": result.annual_revenue,
                "cost": result.annual_cost,
                "service_level": result.service_level,
                "or_recommendations": n_rec,
                "days": len(result.daily_summary),
            })
            if result.run_id:
                run_ids.append(result.run_id)
            print(
                f"  [{mode:5}] 利润 ${result.annual_profit:,.0f} | "
                f"满足率 {result.service_level:.1%} | OR建议 {n_rec} | run_id={result.run_id}"
            )

        or_r = next(r for r in rows if r["scenario"] == sid and r["policy"] == "or")
        rl_r = next(r for r in rows if r["scenario"] == sid and r["policy"] == "or_rl")
        print(
            f"  → 更优: {'OR+RL' if rl_r['profit'] > or_r['profit'] else 'OR'} "
            f"(Δ利润 ${rl_r['profit']-or_r['profit']:,.0f})\n"
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / ("scenario_comparison_90d.csv" if args.quick else "scenario_comparison_365d.csv")
    df.to_csv(csv_path, index=False)

    runs_df = pd.DataFrame()
    if not args.no_db:
        repo = SimulationRepository(args.db)
        runs_df = repo.list_runs_detail(limit=len(run_ids) + 5)

    md_path = out_dir / ("full_year_report_90d.md" if args.quick else "full_year_report_365d.md")
    write_markdown_report(df, runs_df, md_path, days)

    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nCSV:  {csv_path}")
    print(f"报告: {md_path}")
    if not args.no_db:
        print(f"数据库: {args.db}  (表 or_recommendations 已写入)")
    print("=" * 60)


if __name__ == "__main__":
    main()
