# Multi-Agent Operations System

> AI forecasts demand → OR plans under constraints → RL fine-tunes daily execution → Memory reuses what worked.

[中文版 README](README_zh.md) · [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org) · [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Motivation

Small and mid-size manufacturers (mixed SKUs, tight capacity and cash, small planning teams) often face:

| Pain point | What it looks like |
|------------|-------------------|
| **Opaque demand** | Seasonality, promos, macro cycles; forecasts miss, production and stock drift |
| **Rigid capacity & budget** | Slow scale-up in peaks, idle assets in troughs; big CapEx hard to repeat |
| **Siloed plans** | Quarterly investment, monthly output, daily replenishment decided separately |
| **Inventory trap** | Too much stock ties cash; too little loses orders — worst on promo spikes |
| **Supply & equipment** | Raw limits, aging assets, budget cuts — no executable fallback |
| **Lost lessons** | Similar macro shapes restart from scratch; no record of what worked |

This repo uses **simulation + multi-agent orchestration** to test whether **forecast → constrained optimization → daily execution → experience reuse** can address these pains in a reproducible way.

---

## 2. Data & Scenarios

| Source | Path |
|--------|------|
| Base series | `data/sample_manufacturing_demand_2024.csv` |
| Five cases | `data/scenarios/{baseline,growth,recession,supply_crisis,promotion_heavy}_demand.csv` |
| **Economic cycle** | `data/scenarios/economic_cycle_5y_demand.csv` (`cycle_phase`) |

| Scenario ID | Profile |
|-------------|---------|
| `baseline` / `growth` / `recession` / `supply_crisis` / `promotion_heavy` | Single-year real-case demand each |
| **`economic_cycle`** | **recovery → expansion → peak → recession → rebound** (1825 days) |

Simulations also inject calendar events (raw price spikes, supply disruption, equipment failure, promos, etc.) via `agent/simulation/dynamic_events.py`. Demand CSV is input; P&L and decisions land in SQLite (see §6).

---

## 3. Role of AI / ML

AI **does not replace** OR — it feeds signals and adaptation into tactical/operational layers:

```
CSV demand → Forecast (walk-forward) → Memory? → strategic/tactical/ops agents → P&L
```

| Module | Models / methods | Role |
|--------|------------------|------|
| **ForecastAgent** | naive, exp_smooth, **GBR**, MLP, optional LSTM | Point + interval forecasts; walk-forward to avoid leakage |
| **Robust tactical** | Interval upper bound + `--use-robust-lp` | Tactical LP stocks to forecast upper bound |
| **RiskControlAgent** | Demand anomaly detection + contingency plans | Scales demand/capacity/budget; maintenance / stock boosts |
| **Residual RL** | Q-learning (see §5) | Adjusts operational DP suggestions only |

**By macro phase, AI mainly improves “reading the trend”:** recovery/expansion tracks upturns; recession softens forecasts while OR tightens budget; rebound stabilizes forecasts so built capacity pays off (phase P&L in §7).

---

## 4. Optimization Methods

Core solvers live in `agent/or_optimization/`: **SciPy MIP / LP + daily DP**, with greedy fallback. `RiskControlAgent` adjusts constraints before each solve.

| Layer | Model | Variables | Objective & constraints (plain) | Solver |
|-------|-------|-----------|--------------------------------|--------|
| **Strategic** (quarter) | **MIP** | Open site? #lines? #workers? | Min CapEx + labor; capacity ≥ demand; budget cap | `scipy.optimize.milp` |
| **Tactical** (month) | **LP** | Production per SKU, raw buy | Max margin; capacity, BOM, demand limits | `scipy.optimize.linprog` |
| **Operational** (day) | **DP** | Reorder qty, maintain? | Min holding + stockout + maint; fulfill backlog | Enumerate / short-horizon DP |

**Flow:** MIP sets daily capacity → LP allocates monthly output & materials → DP runs inventory/maintain each day.

**Parallel race** (`--parallel-solvers`, `agent/solvers/solver_pool.py`): on strategic sub-problems, **MIP ∥ Python greedy ∥ native C++** run in threads; fastest **feasible** incumbent wins within a timeout — exact MIP when it finishes first, else a good greedy plan.

**Optional C++** (`native/`, not required): OpenMP multi-start greedy (`capacity_greedy.cpp`) returns a feasible incumbent in milliseconds when MIP stalls. `make -C native` + `--parallel-solvers`; skip with `--no-native`. Bindings: `native/bindings.py`. Next: VRP, large MIP decomposition.

`ORAdvisor` turns MIP/LP/DP output into readable management advice (`or_recommendations` trail).

---

## 5. Why RL

OR (MIP/LP/DP) already plans from strategy to daily ops, but **forecast + robust LP can still under-stock on promo spike days**. RL is a **residual on the operational layer only** (`agent/rl/residual_policy.py`) — it does **not** replace MIP/LP/DP:

```
Daily decision = OR (DP + risk)  +  RL_adjustment
                 ↑ strategic/tactical already fixed by OR
```

| Point | Detail |
|-------|--------|
| **Where** | After tactical LP and DP propose reorder/maintain, RL picks one of **5 actions**: trust OR, light reorder (+25), aggressive (+60), maintain-first, crisis (+100) |
| **State** | Inventory ratio, 7/30-day demand trend, equipment health, backlog, event stress (discrete Q-learning) |
| **Training** | Pre-train on 90-day windows; reward = daily profit proxy − stockout penalty |
| **vs `or`** | `or` = DP only; `or_rl` adds RL on top. Without `--train-rl`, Q-table is neutral (≈ `or`); with `--train-rl`, RL boosts reorder on spike days |

**When to enable:** Use `or_rl` on long horizons, promos, or economic cycles; add **`--train-rl`** when RL should steer execution (default in `run_economic_cycle.py`). Trained Q-tables can be saved to **Memory** (§6).

---

## 6. Memory & Database

**SQLite** (`data/operations.db`) — one run produces thousands of layered decisions. CSV exports (`results/sim_*.csv`) are for charts; the DB is the **system of record**:

| Need | Stored |
|------|--------|
| Audit & replay | `simulation_runs`, daily/strategic/tactical/operational rows |
| Policy comparison | Profit, SL, `policy_mode` per scenario |
| Forecast science | `forecast_experiments`, `forecast_metrics` |
| OR advice trail | `or_recommendations` |
| Cross-run learning | `strategy_memory` (config + optional **RL Q-table**) |

**Strategy Memory** (`agent/memory/`, RAG-like) — avoid cold-start config search on every new demand window:

```
Encode demand shape (mean, std, trend, peak_ratio)
  → kNN over strategy_memory
  → match: policy, forecast, robust LP, profit, SL, rl_q_table_json
  → inject as hints (--use-memory) before simulation
```

| RAG idea | This system |
|----------|-------------|
| Retrieval | kNN on demand signature + scenario filter |
| Context | Best past config + `recommendation_text` |
| Generation | Agents re-run OR/RL with retrieved settings (not an LLM) |

**Value:** Config reuse · RL warm-start (5 fine-tune episodes vs 15 cold) · Run 2 of a long cycle inherits run 1. Query: `python run_memory_query.py --scenario economic_cycle`.

```sql
SELECT scenario_id, policy_mode, forecast_model, annual_profit, service_level
FROM strategy_memory ORDER BY id DESC LIMIT 5;
```

---

## 7. Evaluation Results

> Reproduce via Quick Start below. Artifacts: `results/scenario_suite_90d.csv` · `results/economic_cycle_5y_or_rl_report.md` · `results/forecast_eval_report.csv` (seed=42, 5y default forecast=**gbr**)

**90-day suite** (`run_scenario_suite.py --quick`, five scenarios × `or`): profit **+$176K–+$254K** per scenario, SL **56–80%**.

| Scenario | OR profit | SL | or_rl profit | SL |
|----------|-----------|-----|--------------|-----|
| baseline | +$236K | 66% | +$342K | 89% |
| growth | +$176K | 56% | +$273K | 78% |
| recession | +$254K | 80% | +$280K | 85% |
| supply_crisis | +$247K | 68% | +$350K | 92% |
| promotion_heavy | +$244K | 66% | +$321K | 76% |

**5-year economic cycle** (`run_economic_cycle.py --years 5 --policy or_rl --use-forecast --train-rl`):

| Metric | Value |
|--------|-------|
| Total profit | **+$1,438,524** |
| Service level | **67.2%** |
| RL pre-train | 15 × 90-day episodes |

| Phase | Profit | SL | AI + OR note |
|-------|--------|-----|--------------|
| rebound | +$564K | **85%** | Stable forecast, reuse capacity |
| recovery | +$381K | 72% | Uptrend forecast, incremental hiring |
| recession | +$231K | 79% | Softer forecast, tighter budget |
| expansion | +$152K | 60% | Intervals → robust stocking |
| peak | +$111K | 60% | High mean demand within budget |

**Run modes** (same OR stack; differences are AI/RL and horizon):

| Mode | Meaning | When to use |
|------|---------|-------------|
| `--policy or` | Full MIP/LP/DP, no RL on ops | Short-horizon pure-OR baseline |
| `--policy or_rl` (no `--train-rl`) | RL slot present, Q untrained — moves ≈ OR | A/B the RL slot |
| `--policy or_rl --train-rl` | RL pre-trained, adjusts daily reorder | Long horizon, economic cycle |
| `--use-forecast` | Forecast + robust LP | Tactical upper-bound stocking |
| `--use-memory` | Retrieve past config ± Q-table | Warm-start on similar shapes |
| `--quick` (economic cycle) | 2-year expansion slice | Shorter stress test |

**Single scenario:** 90-day `recession` + or_rl + train-rl → **+$341K**, **99.3%** SL. Forecast test sMAPE (exp_smooth): **11.5%**.

---

## 8. Directions to Extend (smart-factory roadmap)

**Position:** **early Stage-2 prototype** — CSV simulation of forecast → OR → layered agents; not live MES/PLC Copilot or lights-out factory.

On this architecture, **directions you can extend toward** include:

| Stage | Industry vision | This repo today | Can extend toward |
|-------|-----------------|-----------------|-------------------|
| **Stage 1 Copilot** | Telemetry → AI → **advice → human OK** | `ORAdvisor` text only; sim auto-executes | MES/ERP/SCADA read-only; approval gate; **predictive maintenance** |
| **Stage 2 autonomous** | Forecast + optimize + **execute** agents | **Forecast + MIP/LP/DP + residual RL** | **CP-SAT** scheduling, **VRP**, OPC-UA mock |
| **Stage 3 lights-out** | Robots + AGV, 24/7 | Not covered | PLC / robot closed-loop sandbox |

| Algorithm layer | Built | Can extend toward |
|-----------------|-------|-------------------|
| ① Prediction | Demand GBR / LSTM / intervals | Fault, energy, tariff; Transformer / GNN |
| ② Optimization | **MIP / LP / DP** | Energy in objective; `native/` VRP / decomposition |
| ③ RL | 5 discrete ops Q-learning | Domain-split continuous control |
| ④ Multi-agent | Vertical strategic/tactical/ops + risk + memory | Horizontal procurement/logistics/energy agents |

**Engineering directions:** Kafka/ERP streams · time-series store · LLM S&OP briefs · Memory with equipment signatures

---
```bash
## Quick Start

# Clone the repository
git clone https://github.com/kevinlmf/Operations_Agent_System
cd Operations_Agent_System

```bash
git clone https://github.com/kevinlmf/Operations_Agent_System
cd Operations_Agent_System
pip install -r requirements.txt
make -C native   # optional C++ greedy solver

# ★ 5-year economic cycle (recommended)
python run_economic_cycle.py --years 5 --policy or_rl --use-forecast --train-rl

# 5 scenarios × 90d
python run_scenario_suite.py --quick

# Single scenario
python run_simulation.py --scenario recession --policy or_rl \
  --use-forecast --days 90 --train-rl

python run_forecast_eval.py --downstream
python run_memory_query.py --scenario economic_cycle
```

**Entry scripts:** `run_economic_cycle.py` · `run_scenario_suite.py` · `run_simulation.py` · `run_forecast_eval.py` · `run_memory_query.py`  
**Code:** `agent/{forecast,memory,planner,solvers,strategic,tactical,operational,risk,rl,scenarios}/` · `native/` · `db/`

---

## License

MIT — research and education only. Validate with domain experts before production use.

**Disclaimer:** For educational and research use only. Inventory and capacity decisions should be validated by domain experts before any production deployment.
