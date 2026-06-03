"""Typed decisions for the three-layer multi-agent system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StrategicDecision:
    """季度 · MIP — 建厂、产线数量、员工规模"""
    quarter: int
    open_factories: List[int]
    lines_per_factory: Dict[int, int]
    workforce: int
    daily_capacity: float
    investment_cost: float
    is_feasible: bool = True


@dataclass
class TacticalDecision:
    """月度 · LP — 产品产量、原料采购量"""
    month: int
    production_volume: np.ndarray
    raw_material_procurement: np.ndarray
    revenue: float
    total_cost: float
    is_feasible: bool = True

    @property
    def profit(self) -> float:
        return self.revenue - self.total_cost


@dataclass
class OperationalDecision:
    """每日 · DP — 库存控制、设备维护、订单执行"""
    day: int
    reorder_qty: int
    maintain_equipment: bool
    inventory_end: float
    units_sold: float
    backlog_fulfilled: float
    stockout: float
    operating_cost: float
