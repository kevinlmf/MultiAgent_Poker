# Multi-Agent Operations System

> Hierarchical multi-agent simulation of manufacturing operations—compare OR (MIP/LP/DP), ML/DL, and OR+RL across scenarios, with P&L, service level, and strategy recommendations.

A research platform that simulates **one year of manufacturing enterprise operations** under multiple stress scenarios, compares **traditional operations research (OR)** with **machine learning / deep learning (ML/DL)**, and optionally improves OR decisions with **residual reinforcement learning (RL)**.

[中文版 README](README_zh.md) · [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org) · [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Problem Statement

Manufacturing firms must coordinate decisions at **different time horizons** while facing uncertain demand, supply shocks, and equipment risk:

- **Quarterly**: whether to build plants, add production lines, and scale workforce  
- **Monthly**: how much to produce per SKU and how much raw material to procure  
- **Daily**: inventory replenishment, preventive maintenance, and order fulfillment  

Single-level optimizers (e.g., inventory-only EOQ) ignore capacity and investment constraints. Pure ML forecasts often violate feasibility and lack explainability. This project asks:

> *Under realistic yearly scenarios, can we simulate enterprise P&L, compare OR vs ML/DL layer-by-layer, and produce actionable strategy recommendations?*

---

## Data

| Source | Description |
|--------|-------------|
| **Real CSV** | Default: `data/sample_manufacturing_demand_2024.csv` (365 days, auto-created if missing) |
| **Custom CSV** | `date`, `demand` (or `sales`), optional `unit_price`, `unit_cost` |
| **Synthetic** | Seasonal + weekly + promotional patterns via `--synthetic` |

**Dynamic events** (`dynamic_events.py`): holidays, supply shocks, 618/Double-11, equipment failure, recalls, etc.

---

## Methods

Three **policy modes** run on the same multi-agent pipeline:

| Mode | Strategic (quarterly) | Tactical (monthly) | Operational (daily) |
|------|----------------------|-------------------|---------------------|
| **`or`** | MIP — plants, lines, headcount | LP — production & procurement | DP — inventory, maintenance, orders |
| **`ml`** | RandomForest capacity model | GBR demand forecast + learned mix | MLP (64,32) reorder & maintenance |
| **`or_rl`** | OR + residual RL adjustments | OR + RL production scale | OR + RL reorder / maintain boost |

**Risk agent** (anomaly detection + contingency) and **OR/ML advisors** → SQLite `or_recommendations`.

```
                    ┌─────────────────┐
  Scenarios + Data  │  Risk Control   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   Strategic MIP/ML    Tactical LP/ML     Operational DP/MLP
         └───────────────────┬───────────────────┘
                             ▼
            Profit · Service level · CSV · SQLite
```

---

## Why Multi-Agent?

Enterprise planning is naturally **hierarchical**. A monolithic model cannot enforce:

1. **Top-down constraints** — quarterly capacity caps monthly production; monthly inflow shapes daily inventory.  
2. **Role separation** — capital budgeting (strategic) vs. supply planning (tactical) vs. execution (operational) use different math and different data frequencies.  
3. **Explainability** — each agent emits typed decisions (`agent/decisions.py`) stored per layer in the database.  
4. **Method swapping** — the coordinator (`agent/coordinator.py`) switches OR vs ML backends without rewriting the simulation loop.  
5. **Risk as a fourth agent** — detects demand anomalies and adjusts all layers via shared `RiskAdjustment` signals.

Mirrors **S&OP / IBP**: capacity → production planning → daily control tower.

---

## Preset Scenarios

Defined in `agent/scenarios/profiles.py`:

| ID | Intent |
|----|--------|
| `baseline` | Normal year with standard event calendar |
| `growth` | Higher demand scale, larger budget |
| `recession` | Lower demand, tighter budget |
| `supply_crisis` | Raw-cost inflation + capacity stress |
| `promotion_heavy` | High promo peaks, larger starting inventory |

---

## Results (Summary)

### 365-day · OR vs OR+RL · 5 scenarios · real sample demand

From `results/scenario_comparison_365d.csv` (trained RL, dynamic events on):

| Scenario | OR profit | OR+RL profit | OR fill rate | OR+RL fill rate | Winner |
|----------|-----------|--------------|--------------|-----------------|--------|
| baseline | -$2.56M | **-$1.68M** | 39% | **67%** | OR+RL |
| growth | -$3.02M | **-$2.81M** | 34% | **40%** | OR+RL |
| recession | -$2.03M | **-$1.15M** | 47% | **78%** | OR+RL |
| supply_crisis | -$2.58M | **-$1.68M** | 38% | **67%** | OR+RL |
| promotion_heavy | -$2.88M | **-$1.99M** | 35% | **62%** | OR+RL |

**OR+RL wins all five scenarios** on profit and service level: residual RL adds reorder boosts and maintenance when OR baselines under-react to shocks.

### 90-day · OR vs ML/DL vs OR+RL · quick benchmark

From `results/method_comparison_90d.csv` (ML trained on demand history at run start):

| Scenario | OR profit | ML/DL profit | OR fill | ML fill |
|----------|-----------|--------------|---------|---------|
| baseline | **-$49K** | -$985K | **66%** | 28% |
| recession | **+$50K** | -$725K | **76%** | 30% |

**Takeaway:** With light training and no hard constraints, **OR (MIP/LP/DP) is more stable**; **ML/DL needs more labels and constraint-aware learning**. **OR+RL** combines interpretability with adaptive execution (best long-horizon results in our 365-day runs).

Full tables: `results/full_year_report_365d.md`, `results/method_comparison_365d.md`.

---
```bash
## Quick Start

# Clone the repository
git clone https://github.com/kevinlmf/Operations_Agent_System
cd Operations_Agent_System

```bash
pip install -r requirements.txt

# Compare OR vs ML/DL vs OR+RL across all scenarios (365 days)
python run_method_comparison.py

# Quick 90-day benchmark
python run_method_comparison.py --quick

# OR vs OR+RL only, full annual report + SQLite
python run_scenario_comparison.py

# Single run
python run_simulation.py --scenario recession --policy or
python run_simulation.py --scenario baseline --policy ml
python run_simulation.py --scenario growth --policy or_rl --train-rl

python run_simulation.py --data /path/to/demand.csv
```

---

## Outputs

| Artifact | Content |
|----------|---------|
| `results/method_comparison_365d.csv` | 5 scenarios × 3 methods |
| `results/scenario_comparison_365d.csv` | 5 scenarios × OR vs OR+RL |
| `results/full_year_report_365d.md` | Annual markdown summary |
| `results/sim_<scenario>_<policy>_*.csv` | Daily / quarterly metrics |
| `results/sim_*_advice.txt` | OR or ML strategy narrative |
| `data/operations.db` | SQLite: runs, layer decisions, `or_recommendations` |

```sql
SELECT layer, period, priority, action
FROM or_recommendations WHERE run_id = 83;
```

---

## Project Structure

```
├── run_method_comparison.py      # OR vs ML vs OR+RL
├── run_scenario_comparison.py    # OR vs OR+RL annual
├── run_simulation.py
├── agent/
│   ├── strategic_agent.py        # MIP
│   ├── tactical_agent.py         # LP
│   ├── operational_agent.py      # DP
│   ├── ml/                       # RF, GBR, MLP
│   ├── rl/residual_policy.py
│   ├── recommendation/           # OR & ML advisors
│   ├── scenarios/profiles.py
│   └── simulation/year_simulator.py
├── db/schema.sql · repository.py
└── evaluation/risk_management/
```

---

## Future Extensions

1. **Constraint-aware ML** — Lagrangian or differentiable LP layers so ML respects BOM and capacity like MIP/LP.  
2. **Deep forecasting** — LSTM/Transformer demand modules feeding tactical agent; pre-train on multi-SKU history.  
3. **LLM strategy layer** — natural-language S&OP briefs from `or_recommendations` + scenario context.  
4. **Multi-facility network** — extend MIP to multi-echelon location–allocation; RL for routing.  
5. **Online learning** — update residual RL and GBR models each month with rolling demand.  
6. **Public datasets** — adapters for M5, Rossmann, or internal ERP exports.  
7. **Interactive dashboard** — Streamlit on `operations.db` for quarter-by-quarter drill-down.  
8. **Stochastic programming** — two-stage MIP for demand uncertainty sets per scenario.

---

## License

MIT — for education and research only. Validate all decisions with domain experts before production use.

## Disclaimer

 **Important**: This project is provided for educational, academic research, and learning purposes only. The inventory decisions generated by this system should be validated by domain experts before implementation in production environments.

---

May our lives keep optimizing, like finding balance in every step😊.

