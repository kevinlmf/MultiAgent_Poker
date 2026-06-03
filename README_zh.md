# 多智能体企业运营系统

> 分层多智能体仿真制造企业运营：多场景下对比 OR（MIP/LP/DP）、ML/DL 与 OR+RL，输出损益、满足率与战略建议。

面向制造企业的研究平台：在多种压力场景下模拟 **全年运营**，对比 **传统运筹学（OR）** 与 **机器学习/深度学习（ML/DL）**，并可用 **残差强化学习（RL）** 在 OR 基线之上做自适应修正。

[English README](README.md) · [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org) · [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 解决的问题

制造企业在不确定需求、供应冲击与设备风险下，需要在 **不同时间尺度** 上协同决策：

- **季度（战略）**：是否建厂、扩产线、调整人力  
- **月度（战术）**：各 SKU 产量、原料采购量  
- **每日（运营）**：补货、预防性维护、订单履约  

单层优化（如仅做 EOQ 库存）无法体现产能与投资约束；纯 ML 预测又常违反可行性、难以解释。本项目要回答：

> *在贴近真实的年度场景下，能否仿真企业损益、逐层对比 OR 与 ML/DL，并输出可执行的战略建议？*

---

## 数据

| 来源 | 说明 |
|------|------|
| **真实 CSV** | 默认 `data/sample_manufacturing_demand_2024.csv`（365 天，缺失时自动生成） |
| **自定义 CSV** | 列：`date`、`demand`（或 `sales`），可选 `unit_price`、`unit_cost` |
| **合成数据** | `--synthetic` 生成季节 + 周内 + 促销模式 |

**动态事件**（`agent/simulation/dynamic_events.py`）：春节淡季、原料涨价、断供、设备故障、618/双 11、用工短缺、召回、价格战、环保检查等。

---

## 方法

三种 **策略模式**（`--policy`）共用同一多智能体流水线：

| 模式 | 战略（季度） | 战术（月度） | 运营（每日） |
|------|-------------|-------------|-------------|
| **`or`** | **MIP** 建厂/产线/员工 | **LP** 产量与采购 | **DP** 库存/维护/订单 |
| **`ml`** | **RandomForest** 产能投资 | **GBR** 需求预测 + 学习分配 | **MLP (64,32)** 补货与维护 |
| **`or_rl`** | OR + 残差 RL | OR + RL 产量缩放 | OR + RL 补货/维护增强 |

**风控智能体**（异常检测 + 应急预案）与 **OR/ML 顾问** 将建议写入 SQLite 表 `or_recommendations`。

```
                    ┌─────────────────┐
  场景 + 需求数据    │   风控智能体     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   战略 MIP / ML        战术 LP / ML        运营 DP / MLP
         └───────────────────┬───────────────────┘
                             ▼
           利润 · 满足率 · CSV · SQLite
```

---

## 为什么用多智能体？

企业计划天然 **分层**，单体模型难以同时保证：

1. **自上而下约束** — 季度产能限制月度产量，月度到货影响每日库存。  
2. **职责分离** — 资本预算（战略）、供应计划（战术）、执行（运营）使用不同数学模型与数据频率。  
3. **可解释性** — 各层输出结构化决策（`agent/decisions.py`），按层写入数据库。  
4. **方法可切换** — 协调器（`agent/coordinator.py`）在 OR 与 ML 后端间切换，无需重写仿真主循环。  
5. **风控作为第四智能体** — 检测需求异常，通过 `RiskAdjustment` 信号联动三层。

对应现实中的 **S&OP / 集成业务计划（IBP）**：产能决策 → 月度排产 → 日度控制塔。

---

## 预置场景

定义于 `agent/scenarios/profiles.py`：

| ID | 含义 |
|----|------|
| `baseline` | 基准年度，标准事件日历 |
| `growth` | 需求放大、预算更宽松 |
| `recession` | 需求萎缩、预算收紧 |
| `supply_crisis` | 原料成本上涨 + 产能压力 |
| `promotion_heavy` | 大促峰值高、初始库存更大 |

---

## 实验结果（摘要）

### 365 天 · OR vs OR+RL · 5 场景 · 样本需求 CSV

数据来源：`results/scenario_comparison_365d.csv`（开启 RL 训练与动态事件）

| 场景 | OR 利润 | OR+RL 利润 | OR 满足率 | OR+RL 满足率 | 更优 |
|------|---------|------------|-----------|--------------|------|
| baseline 基准 | -256 万 | **-168 万** | 39% | **67%** | OR+RL |
| growth 增长 | -302 万 | **-281 万** | 34% | **40%** | OR+RL |
| recession 衰退 | -203 万 | **-115 万** | 47% | **78%** | OR+RL |
| supply_crisis 供应链 | -258 万 | **-168 万** | 38% | **67%** | OR+RL |
| promotion_heavy 大促 | -288 万 | **-199 万** | 35% | **62%** | OR+RL |

**五个场景 OR+RL 均在利润与满足率上优于纯 OR**：残差 RL 在 OR 对冲击反应不足时加强补货与维护。

> 注：全年利润常为负，主要因战略层大额建厂/扩产固定成本相对样本需求偏高，属当前参数设定下的仿真现象，非代码错误。

### 90 天快速对比 · OR vs ML/DL vs OR+RL

数据来源：`results/method_comparison_90d.csv`（运行开始时用历史需求训练 ML）

| 场景 | OR 利润 | ML/DL 利润 | OR 满足率 | ML 满足率 |
|------|---------|------------|-----------|-----------|
| baseline | **-4.9 万** | -98.5 万 | **66%** | 28% |
| recession | **+5.0 万** | -72.5 万 | **76%** | 30% |

**结论**：在轻量训练、无硬约束条件下，**OR（MIP/LP/DP）更稳健**；**ML/DL 需更多样本与约束感知学习**。长期（365 天）**OR+RL** 兼顾可解释性与执行层自适应，表现最佳。

完整报告见 `results/full_year_report_365d.md`、`results/method_comparison_365d.md`。

---

## 快速开始

```bash
pip install -r requirements.txt

# 五场景 × 三种方法（OR / ML / OR+RL），默认 365 天
python run_method_comparison.py

# 90 天快速 benchmark
python run_method_comparison.py --quick

# 仅 OR vs OR+RL，年度报告 + 数据库
python run_scenario_comparison.py

# 单次仿真
python run_simulation.py --scenario recession --policy or
python run_simulation.py --scenario baseline --policy ml
python run_simulation.py --scenario growth --policy or_rl --train-rl

python run_simulation.py --data /path/to/demand.csv
python run_simulation.py --list-scenarios
```

---

## 输出文件

| 路径 | 内容 |
|------|------|
| `results/method_comparison_365d.csv` | 5 场景 × 3 方法指标 |
| `results/scenario_comparison_365d.csv` | 5 场景 × OR vs OR+RL |
| `results/full_year_report_365d.md` | 年度 Markdown 汇总 |
| `results/sim_<场景>_<策略>_*.csv` | 日度 / 季度指标 |
| `results/sim_*_advice.txt` | OR 或 ML 文字建议 |
| `data/operations.db` | 运行记录、分层决策、`or_recommendations` |

```sql
SELECT layer, period, priority, action
FROM or_recommendations WHERE run_id = 83;
```

---

## 项目结构

```
├── run_method_comparison.py      # OR vs ML vs OR+RL
├── run_scenario_comparison.py    # OR vs OR+RL 年度对比
├── run_simulation.py             # 单场景单次运行
├── agent/
│   ├── strategic_agent.py        # MIP
│   ├── tactical_agent.py         # LP
│   ├── operational_agent.py      # DP
│   ├── ml/                       # RF、GBR、MLP
│   ├── rl/residual_policy.py
│   ├── recommendation/           # OR 与 ML 顾问
│   ├── scenarios/profiles.py
│   └── simulation/year_simulator.py
├── db/schema.sql · repository.py
└── evaluation/risk_management/
```

---

## 未来扩展

1. **约束感知 ML** — 拉格朗日或可微 LP 层，使 ML 输出满足 BOM 与产能（对齐 MIP/LP）。  
2. **深度预测** — LSTM/Transformer 需求模块接入战术层；多 SKU 历史预训练。  
3. **LLM 战略层** — 基于 `or_recommendations` 与场景上下文生成自然语言 S&OP 简报。  
4. **多工厂网络** — MIP 扩展为多级选址–分配；RL 做运输/路由。  
5. **在线学习** — 按月滚动更新残差 RL 与 GBR。  
6. **公开数据集** — 适配 M5、Rossmann 或企业 ERP 导出格式。  
7. **交互看板** — 基于 `operations.db` 的 Streamlit 季度下钻。  
8. **随机规划** — 按场景构造需求不确定集的两阶段 MIP。

---

## 许可证

MIT — 仅供学习与研究。生产环境使用前须经领域专家校验决策结果。
