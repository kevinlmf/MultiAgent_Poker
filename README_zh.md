# 多智能体企业运营系统

> AI 预测需求 → OR 在约束下排产 → RL 微调日度执行 → Memory 沉淀可复用经验。

[English README](README.md) · [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org) · [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. 动机

中小型制造企业（多品种、产能与现金流有限、计划团队小）常同时遇到：

| 痛点 | 典型表现 |
|------|----------|
| **需求看不清** | 季节、促销、宏观周期叠加；点预测偏差大，产量与备货对不上 |
| **产能与预算僵** | 旺季扩产慢、淡季闲置；大额建厂难反复投，人工/产线调配滞后 |
| **计划各说各话** | 季度投资、月度排产、日度补货分头拍板，战略与执行脱节 |
| **库存两难** | 备多占现金、备少缺货丢单；促销尖峰时尤其明显 |
| **供应与设备** | 原料上限、设备老化、预算收紧时缺少可执行的降级方案 |
| **经验难复用** | 相似行情重来仍从零试策略，不知道上次哪种配置更赚 |

本仓库用 **仿真 + 多 Agent** 把「预测 → 约束优化 → 日度执行 → 经验沉淀」串成可复现实验，验证上述痛点能否被系统性缓解。

---

## 2. 数据与场景

| 来源 | 路径 |
|------|------|
| 基准序列 | `data/sample_manufacturing_demand_2024.csv` |
| 五场景 | `data/scenarios/{baseline,growth,recession,supply_crisis,promotion_heavy}_demand.csv` |
| **经济周期** | `data/scenarios/economic_cycle_5y_demand.csv`（含 `cycle_phase`） |

| 场景 ID | 特征 |
|---------|------|
| `baseline` / `growth` / `recession` / `supply_crisis` / `promotion_heavy` | 单年 real-case 需求曲线 |
| **`economic_cycle`** | **复苏 → 扩张 → 过热 → 衰退 → 反弹**（1825 天） |

仿真还会注入日历事件（原料涨价、断供、设备故障、大促等），见 `agent/simulation/dynamic_events.py`。需求 CSV 为输入；损益与决策写入 SQLite（见 §6）。

---

## 3. AI / ML 的作用

AI 侧**不替代** OR，主要为战术/运营层提供信号与适应：

```
CSV 需求 → Forecast（walk-forward）→ Memory? → 战略/战术/运营 Agent → 损益
```

| 模块 | 模型 / 方法 | 作用 |
|------|-------------|------|
| **ForecastAgent** | naive、exp_smooth、**GBR**、MLP、可选 LSTM | 点预测 + 预测区间；walk-forward 避免泄漏 |
| **鲁棒战术** | 区间上界 + `--use-robust-lp` | 战术 LP 按预测上界备料，抗波动 |
| **RiskControlAgent** | 需求异常检测 + 应急预案 | 缩放需求/产能/预算，触发维护或加库存 |
| **Residual RL** | Q-learning（见 §5） | 仅在运营层修正 DP 建议 |

**按宏观阶段，AI 主要补「看清趋势」：** 复苏/扩张跟住回升；衰退走弱预测配合 OR 缩预算；反弹阶段预测趋稳，便于复用已建产能（阶段损益见 §7）。

---

## 4. 优化方法

核心在 `agent/or_optimization/`：**SciPy MIP / LP + 日度 DP**；求解失败时贪心兜底。`RiskControlAgent` 在每次求解前调整约束。

| 层级 | 模型 | 决策变量 | 目标与约束（白话） | 求解器 |
|------|------|----------|-------------------|--------|
| **战略**（季度） | **MIP** | 开厂？产线数？工人数？ | 最小 CapEx+人工；产能≥预测需求；预算上限 | `scipy.optimize.milp` |
| **战术**（月） | **LP** | 各 SKU 产量、原料采购 | 最大毛利；产能、BOM、需求约束 | `scipy.optimize.linprog` |
| **运营**（日） | **DP** | 补货量、是否维护 | 最小持有+缺货+维护；履约 backlog | 枚举 / 短视距 DP |

**链路：** MIP 定日产能 → LP 分配月产量与原料 → DP 逐日库存/维护。

**并行竞赛**（`--parallel-solvers`，`agent/solvers/solver_pool.py`）：战略子问题上 **MIP ∥ Python 贪心 ∥ C++** 多线程竞速；超时内最先得到**可行解**的胜出 — MIP 先完则用精确解，否则用优质贪心解。

**可选 C++**（`native/`，非必需）：OpenMP 多起点贪心（`capacity_greedy.cpp`）在 MIP 慢/超时时毫秒级返回可行 incumbent。`make -C native` + `--parallel-solvers`；`--no-native` 可关。绑定：`native/bindings.py`。后续：VRP、大规模 MIP 分解。

`ORAdvisor` 将 MIP/LP/DP 输出转为管理层可读建议（`or_recommendations` 表留痕）。

---

## 5. 为什么用 RL

OR（MIP/LP/DP）已覆盖战略到日度计划，但**预测 + 鲁棒 LP 在促销尖峰日仍可能备货偏少**。RL 作为**运营层残差**（`agent/rl/residual_policy.py`），不替代 MIP/LP/DP：

```
日决策 = OR（DP + 风控） + RL_调整量
         ↑ 战略/战术已由 OR 定好
```

| 要点 | 说明 |
|------|------|
| **位置** | 战术 LP 定月产量、DP 给出补货/维护后，RL 在 **5 档动作** 中选：跟 OR、轻补 (+25)、积极补 (+60)、优先维护、危机 (+100) |
| **状态** | 库存比、7/30 日需求趋势、设备健康、backlog、事件压力（离散 Q-learning） |
| **训练** | 90 天窗口上预训练；奖励 = 日利润代理 − 缺货惩罚 |
| **与 `or` 区别** | `or` 完全由 DP 定日决策；`or_rl` 多一档 RL。未 `--train-rl` 时 Q 表中性，行为≈`or`；`--train-rl` 后峰值日倾向加强补货 |

**何时启用：** 长周期、大促、经济周期用 `or_rl`；要让 RL 参与执行需 **`--train-rl`**（`run_economic_cycle.py` 默认开启）。训练好的 Q 表可写入 Memory（§6）。

---

## 6. Memory 与 Database

**SQLite**（`data/operations.db`）— 多 Agent 一次运行产生海量分层决策，CSV 导出（`results/sim_*.csv`）便于画图，DB 为**权威数据源**：

| 需求 | 存储 |
|------|------|
| 审计与回放 | `simulation_runs`、战略/战术/运营逐日记录 |
| 策略对比 | 同场景 or / or_rl 利润、满足率 |
| 预测评估 | `forecast_experiments`、`forecast_metrics` |
| OR 建议留痕 | `or_recommendations` |
| 跨次学习 | `strategy_memory`（配置 + 可选 **RL Q 表**） |

**Strategy Memory**（`agent/memory/`，类 RAG）— 新需求窗口不必从零选配置：

```
编码需求形态（均值、波动、趋势、峰值比）
  → kNN 检索 strategy_memory
  → 命中：policy、forecast、robust LP、利润、SL、rl_q_table_json
  → --use-memory 仿真前注入 hint
```

| RAG 概念 | 本系统 |
|----------|--------|
| 检索 | 需求特征 kNN + 场景过滤 |
| 上下文 | 历史最优配置 + `recommendation_text` |
| 生成 | Agent 用检索结果重跑 OR/RL（非 LLM） |

**价值：** 配置继承 · RL warm-start（5 轮微调 vs 15 轮冷启动）· 长周期第二次跑继承第一次经验。查询：`python run_memory_query.py --scenario economic_cycle`。

```sql
SELECT scenario_id, policy_mode, forecast_model, annual_profit, service_level
FROM strategy_memory ORDER BY id DESC LIMIT 5;
```

---

## 7. 评估结果

> 复现命令见 §快速开始。明细：`results/scenario_suite_90d.csv` · `results/economic_cycle_5y_or_rl_report.md` · `results/forecast_eval_report.csv`（seed=42，5 年默认 forecast=**gbr**）

**90 天套件**（`run_scenario_suite.py --quick`，五场景 × `or`）：利润 **+17.6～25.4 万/场景**，满足率 **56～80%**。

| 场景 | OR 利润 | SL | or_rl 利润 | SL |
|------|---------|-----|------------|-----|
| baseline | +23.6 万 | 66% | +34.2 万 | 89% |
| growth | +17.6 万 | 56% | +27.3 万 | 78% |
| recession | +25.4 万 | 80% | +28.0 万 | 85% |
| supply_crisis | +24.7 万 | 68% | +35.0 万 | 92% |
| promotion_heavy | +24.4 万 | 66% | +32.1 万 | 76% |

**5 年经济周期**（`run_economic_cycle.py --years 5 --policy or_rl --use-forecast --train-rl`）：

| 指标 | 数值 |
|------|------|
| 周期总利润 | **+143.9 万 USD** |
| 满足率 | **67.2%** |
| RL 预训练 | 15 × 90 天 |

| 阶段 | 利润 | SL | AI + OR 要点 |
|------|------|-----|----------------|
| rebound | +56.4 万 | **85%** | 预测趋稳，复用产能 |
| recovery | +38.1 万 | 72% | 预测回升，增量增员 |
| recession | +23.1 万 | 79% | 走弱预测，预算收紧 |
| expansion | +15.2 万 | 60% | 区间→鲁棒备料 |
| peak | +11.1 万 | 60% | 高均值 + 预算内加人/线 |

**不同跑法对比**（同一 OR 栈，差异在 AI/RL 与周期）：

| 跑法 | 含义 | 适用 |
|------|------|------|
| `--policy or` | 完整 MIP/LP/DP，运营层无 RL | 短周期纯 OR 基准 |
| `--policy or_rl`（无 `--train-rl`） | RL 槽位在，Q 未训练，日决策≈OR | 对照 RL 槽位 |
| `--policy or_rl --train-rl` | RL 预训练后参与日补货 | 长周期、经济周期 |
| `--use-forecast` | 预测 + 鲁棒 LP | 战术层区间上界 |
| `--use-memory` | 检索历史配置 ± Q 表 | 相似形态 warm-start |
| `--quick`（经济周期） | 2 年扩张切片 | 缩短压力测试 |

**单场景：** 90 天 `recession` + or_rl + train-rl → **+34.1 万**，**99.3%** SL。预测 test sMAPE（exp_smooth）：**11.5%**。

---

## 8. 可延伸方向（智能工厂路线图）

**定位：** **二阶段早期原型** — CSV 仿真验证「预测 → OR → 分层 Agent」；非 MES/PLC 在线 Copilot，非关灯无人厂。

在本架构上，**可 extend 的方向**包括：

| 演进阶段 | 行业愿景 | 本项目现状 | 可延伸方向 |
|----------|----------|------------|------------|
| **一阶段 Copilot** | 设备遥测 → AI → **建议 → 人工确认** | `ORAdvisor` 文字建议；仿真自动执行 | MES/ERP/SCADA 只读；审批闸；**预测性维护** |
| **二阶段自主优化** | 预测 + 优化 + **执行** 三 Agent | **Forecast + MIP/LP/DP + RL 残差** | **CP-SAT** 排程、**VRP**、OPC-UA mock |
| **三阶段无人化** | 机器人 + AGV，24h 关灯运行 | 未覆盖 | PLC / 机械臂闭环 sandbox |

| 算法四层 | 已有 | 可延伸方向 |
|----------|------|------------|
| ① 预测 | 需求 GBR / LSTM / 区间 | 故障、能耗、电价；Transformer / GNN |
| ② 优化 | **MIP / LP / DP** | 能源进目标；`native/` VRP / 分解 |
| ③ RL | 运营 5 档 Q-learning | 分域连续控制（储能、调参） |
| ④ 多 Agent | 纵向战略/战术/运营 + 风控 + Memory | 横向采购/物流/能源 Agent、协商协议 |

**工程方向：** Kafka/ERP 事件流 · 时序库 · LLM S&OP 简报 · Memory 扩设备签名

---

## 快速开始

```bash
pip install -r requirements.txt
make -C native   # 可选 C++

python run_economic_cycle.py --years 5 --policy or_rl --use-forecast --train-rl
python run_scenario_suite.py --quick
python run_simulation.py --scenario recession --policy or_rl --use-forecast --days 90 --train-rl
python run_forecast_eval.py --downstream
python run_memory_query.py --scenario economic_cycle
```

**入口：** `run_economic_cycle.py` · `run_scenario_suite.py` · `run_simulation.py` · `run_forecast_eval.py` · `run_memory_query.py`  
**代码：** `agent/{forecast,memory,planner,solvers,strategic,tactical,operational,risk,rl,scenarios}/` · `native/` · `db/`

---

## 许可证

MIT — 仅供学习与研究，生产使用前需经业务专家校验。
