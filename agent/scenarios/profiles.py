"""预定义运营场景 — 用于模拟不同环境下的企业策略输出"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.simulation.dynamic_events import EnterpriseEvent, default_year_events


@dataclass
class ScenarioProfile:
    id: str
    name: str
    description: str
    events: List[EnterpriseEvent] = field(default_factory=list)
    demand_scale: float = 1.0
    raw_cost_scale: float = 1.0
    capacity_scale: float = 1.0
    quarterly_budget: float = 1_200_000.0
    initial_inventory: float = 400.0
    data_mode: str = "auto"  # auto | synthetic | real
    data_path: Optional[str] = None  # per-scenario real CSV under data/scenarios/
    use_dedicated_data: bool = True  # load scenario CSV; demand_scale then applies lightly


SCENARIOS: Dict[str, ScenarioProfile] = {
    "baseline": ScenarioProfile(
        id="baseline",
        name="基准运营",
        description="正常年度：季节性波动 + 常规供应链事件",
        events=default_year_events(),
        data_mode="auto",
        data_path="data/scenarios/baseline_demand.csv",
        demand_scale=1.0,
    ),
    "growth": ScenarioProfile(
        id="growth",
        name="增长扩张",
        description="需求持续上升，大促更猛，需积极扩产",
        events=default_year_events(),
        demand_scale=1.05,
        quarterly_budget=1_500_000.0,
        initial_inventory=550.0,
        data_path="data/scenarios/growth_demand.csv",
    ),
    "recession": ScenarioProfile(
        id="recession",
        name="需求衰退",
        description="宏观下行：需求萎缩、价格战、库存积压风险",
        events=default_year_events(),
        demand_scale=1.0,
        raw_cost_scale=1.05,
        quarterly_budget=900_000.0,
        initial_inventory=600.0,
        data_path="data/scenarios/recession_demand.csv",
    ),
    "supply_crisis": ScenarioProfile(
        id="supply_crisis",
        name="供应链危机",
        description="原料涨价 + 断供 + 设备故障叠加",
        events=default_year_events(),
        raw_cost_scale=1.15,
        capacity_scale=0.88,
        quarterly_budget=1_100_000.0,
        demand_scale=1.0,
        data_path="data/scenarios/supply_crisis_demand.csv",
    ),
    "promotion_heavy": ScenarioProfile(
        id="promotion_heavy",
        name="大促驱动",
        description="618/双十一峰值极高，履约压力最大",
        events=default_year_events(),
        demand_scale=1.0,
        initial_inventory=700.0,
        data_path="data/scenarios/promotion_heavy_demand.csv",
    ),
    "economic_cycle": ScenarioProfile(
        id="economic_cycle",
        name="经济周期 (5年)",
        description="recovery → expansion → peak → recession → rebound 完整宏观周期",
        events=default_year_events(),
        demand_scale=1.0,
        quarterly_budget=1_200_000.0,
        initial_inventory=450.0,
        data_mode="auto",
        data_path="data/scenarios/economic_cycle_5y_demand.csv",
        use_dedicated_data=True,
    ),
}


def get_scenario(scenario_id: str) -> ScenarioProfile:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Choose from: {list(SCENARIOS)}")
    return SCENARIOS[scenario_id]


def list_scenarios() -> List[ScenarioProfile]:
    return list(SCENARIOS.values())
