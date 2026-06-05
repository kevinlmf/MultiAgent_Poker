#!/usr/bin/env python3
"""Query strategy memory (RAG) for similar past runs and recommendations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.memory.memory_agent import StrategyMemoryAgent
from agent.simulation.data_loader import load_enterprise_data
from db.repository import SimulationRepository


def main():
    p = argparse.ArgumentParser(description="Strategy memory / RAG retrieval")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--scenario", type=str, default="baseline")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--db", type=str, default="data/operations.db")
    p.add_argument("--list", action="store_true", help="List recent memory entries")
    args = p.parse_args()

    repo = SimulationRepository(args.db)
    mem = StrategyMemoryAgent(repo)

    if args.list:
        df = repo.list_strategy_memory(scenario_id=args.scenario, limit=20)
        print(df.to_string(index=False) if not df.empty else "No memory entries yet.")
        return

    bundle = load_enterprise_data(args.data, min_days=90, max_days=365)
    recall = mem.recall(bundle.demand[:90], scenario_id=args.scenario)
    print(f"Scenario: {args.scenario} | demand days: {len(bundle.demand)}")
    if recall is None:
        print("No similar memory found. Run simulations with --save-memory first.")
        return
    print(f"\nBest match (similarity={recall.similarity:.2f}):")
    print(f"  Policy:    {recall.policy_mode}")
    print(f"  Forecast:  {recall.forecast_model}")
    print(f"  Parallel:  {recall.parallel_solvers}")
    print(f"  Robust LP: {recall.use_robust_lp}")
    print(f"  Profit:    ${recall.annual_profit:,.0f}")
    print(f"  Service:   {recall.service_level:.1%}")
    if recall.rl_checkpoint:
        n_states = len(recall.rl_checkpoint.get("q", {}))
        print(f"  RL checkpoint: yes ({n_states} Q-table states)")
    else:
        print("  RL checkpoint: no")
    print(f"  Rationale: {recall.rationale}")

    top = mem.recall_top_k(bundle.demand[:90], scenario_id=args.scenario, k=args.k)
    if len(top) > 1:
        print(f"\nTop-{args.k} similar episodes:")
        for i, r in enumerate(top, 1):
            print(f"  {i}. sim={r.similarity:.2f} {r.policy_mode}+{r.forecast_model} "
                  f"profit=${r.annual_profit:,.0f} SL={r.service_level:.1%}")


if __name__ == "__main__":
    main()
