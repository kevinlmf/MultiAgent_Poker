"""
Realistic enterprise events over a one-year horizon (365 days).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class EventType(Enum):
    DEMAND_SURGE = "demand_surge"
    DEMAND_DROP = "demand_drop"
    RAW_PRICE_SPIKE = "raw_price_spike"
    SUPPLY_DISRUPTION = "supply_disruption"
    EQUIPMENT_FAILURE = "equipment_failure"
    LABOR_SHORTAGE = "labor_shortage"
    QUALITY_RECALL = "quality_recall"
    COMPETITOR_PRICE_WAR = "competitor_price_war"
    REGULATORY_AUDIT = "regulatory_audit"
    HOLIDAY_SEASON = "holiday_season"


@dataclass
class EnterpriseEvent:
    name: str
    event_type: EventType
    start_day: int
    duration_days: int
    demand_multiplier: float = 1.0
    raw_cost_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    price_multiplier: float = 1.0
    extra_fixed_cost: float = 0.0
    description: str = ""


def default_year_events() -> List[EnterpriseEvent]:
    """Calendar of problems a mid-size manufacturer might face in one year."""
    return [
        EnterpriseEvent(
            "春节淡季",
            EventType.DEMAND_DROP,
            start_day=30,
            duration_days=14,
            demand_multiplier=0.65,
            description="春节前后需求下滑",
        ),
        EnterpriseEvent(
            "原料涨价",
            EventType.RAW_PRICE_SPIKE,
            start_day=75,
            duration_days=45,
            raw_cost_multiplier=1.18,
            description="上游原材料合同重谈，采购价上升",
        ),
        EnterpriseEvent(
            "供应商断供",
            EventType.SUPPLY_DISRUPTION,
            start_day=95,
            duration_days=10,
            raw_cost_multiplier=1.25,
            capacity_multiplier=0.85,
            description="关键供应商物流中断",
        ),
        EnterpriseEvent(
            "设备故障",
            EventType.EQUIPMENT_FAILURE,
            start_day=120,
            duration_days=7,
            capacity_multiplier=0.55,
            extra_fixed_cost=35_000,
            description="主产线检修停机",
        ),
        EnterpriseEvent(
            "618大促",
            EventType.DEMAND_SURGE,
            start_day=165,
            duration_days=18,
            demand_multiplier=1.45,
            description="电商大促订单激增",
        ),
        EnterpriseEvent(
            "用工紧张",
            EventType.LABOR_SHORTAGE,
            start_day=200,
            duration_days=21,
            capacity_multiplier=0.88,
            extra_fixed_cost=20_000,
            description="旺季临时工招聘困难",
        ),
        EnterpriseEvent(
            "质量召回",
            EventType.QUALITY_RECALL,
            start_day=230,
            duration_days=5,
            demand_multiplier=0.75,
            extra_fixed_cost=50_000,
            description="批次质量问题召回处理",
        ),
        EnterpriseEvent(
            "竞品价格战",
            EventType.COMPETITOR_PRICE_WAR,
            start_day=260,
            duration_days=30,
            price_multiplier=0.92,
            demand_multiplier=1.08,
            description="竞争对手降价，销量升但毛利受压",
        ),
        EnterpriseEvent(
            "双十一",
            EventType.HOLIDAY_SEASON,
            start_day=305,
            duration_days=20,
            demand_multiplier=1.55,
            description="年末促销高峰",
        ),
        EnterpriseEvent(
            "环保督查",
            EventType.REGULATORY_AUDIT,
            start_day=330,
            duration_days=10,
            capacity_multiplier=0.75,
            extra_fixed_cost=15_000,
            description="环保合规检查，部分产线限产",
        ),
    ]


class EventImpactCalculator:
    """Combine active events for a given day into multipliers."""

    def __init__(self, events: Optional[List[EnterpriseEvent]] = None):
        self.events = events or default_year_events()

    def active_events(self, day: int) -> List[EnterpriseEvent]:
        return [
            e
            for e in self.events
            if e.start_day <= day < e.start_day + e.duration_days
        ]

    def day_modifiers(self, day: int) -> Dict[str, float]:
        demand = 1.0
        raw_cost = 1.0
        capacity = 1.0
        price = 1.0
        fixed_extra = 0.0
        labels: List[str] = []

        for e in self.active_events(day):
            demand *= e.demand_multiplier
            raw_cost *= e.raw_cost_multiplier
            capacity *= e.capacity_multiplier
            price *= e.price_multiplier
            fixed_extra += e.extra_fixed_cost
            labels.append(e.name)

        return {
            "demand_multiplier": demand,
            "raw_cost_multiplier": raw_cost,
            "capacity_multiplier": capacity,
            "price_multiplier": price,
            "extra_fixed_cost": fixed_extra,
            "active_event_names": labels,
        }
