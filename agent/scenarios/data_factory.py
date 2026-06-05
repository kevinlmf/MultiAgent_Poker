"""
Build per-scenario real-case CSV files from a base demand series.

Each scenario gets its own file under data/scenarios/ so evaluation uses
distinct time series (not only runtime multipliers on one CSV).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from agent.scenarios.profiles import SCENARIOS, ScenarioProfile
from agent.simulation.data_loader import load_enterprise_data

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "scenarios"
DEFAULT_BASE = Path(__file__).parent.parent.parent / "data" / "sample_manufacturing_demand_2024.csv"


def _transform_series(
    demand: np.ndarray,
    prices: Optional[np.ndarray],
    costs: Optional[np.ndarray],
    profile: ScenarioProfile,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    n = len(demand)
    t = np.arange(n, dtype=float)
    d = demand.copy().astype(float)
    p = prices.copy().astype(float) if prices is not None else None
    c = costs.copy().astype(float) if costs is not None else None

    sid = profile.id
    if sid == "baseline":
        pass
    elif sid == "growth":
        trend = 1.0 + 0.0008 * t
        d = d * 1.12 * trend
    elif sid == "recession":
        trend = 1.0 - 0.0005 * t
        d = d * 0.78 * np.maximum(trend, 0.85)
        if p is not None:
            p = p * 0.92
    elif sid == "supply_crisis":
        if c is not None:
            shock = np.ones(n)
            shock[75:120] = 1.22
            shock[95:105] = 1.35
            c = c * shock
        d = d * 0.95
    elif sid == "promotion_heavy":
        promo = np.ones(n)
        promo[164:182] = 1.55
        promo[304:324] = 1.72
        promo[130:145] = 1.25
        d = d * promo

    d = np.maximum(d, 0.1)
    return d, p, c


def scenario_csv_path(scenario_id: str, data_dir: Optional[Path] = None) -> Path:
    root = data_dir or DATA_DIR
    return root / f"{scenario_id}_demand.csv"


def ensure_scenario_datasets(
    base_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Create missing scenario CSVs from base real data. Returns scenario_id → path."""
    base_path = Path(base_path or DEFAULT_BASE)
    out_dir = Path(data_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_enterprise_data(base_path, min_days=90, max_days=365)
    dates = bundle.dates
    paths: Dict[str, str] = {}

    for sid, profile in SCENARIOS.items():
        out = scenario_csv_path(sid, out_dir)
        if out.exists() and not force:
            paths[sid] = str(out.resolve())
            continue
        d, p, c = _transform_series(
            bundle.demand, bundle.unit_prices, bundle.unit_costs, profile,
        )
        rows = {"demand": np.round(d, 1)}
        if dates is not None and len(dates) >= len(d):
            rows["date"] = [x.strftime("%Y-%m-%d") for x in dates[: len(d)]]
        else:
            rows["date"] = pd.date_range("2024-01-01", periods=len(d), freq="D").strftime("%Y-%m-%d")
        if p is not None:
            rows["unit_price"] = np.round(p, 2)
        if c is not None:
            rows["unit_cost"] = np.round(c, 2)
        pd.DataFrame(rows).to_csv(out, index=False)
        paths[sid] = str(out.resolve())
    return paths


def resolve_scenario_data_path(
    scenario_id: str,
    override: Optional[str] = None,
    auto_build: bool = True,
) -> Optional[str]:
    if override:
        return override
    profile = SCENARIOS.get(scenario_id)
    if profile is None:
        return None
    if profile.data_path:
        p = Path(profile.data_path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        if p.exists():
            return str(p.resolve())
    if auto_build:
        ensure_scenario_datasets()
        p = scenario_csv_path(scenario_id)
        if p.exists():
            return str(p.resolve())
    return None
