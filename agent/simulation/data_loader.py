"""
Load real enterprise time-series data for the year simulation.
Supports CSV with flexible column names; generates a sample dataset if none provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


DEMAND_ALIASES = ("demand", "sales", "units_sold", "quantity", "qty", "order_volume")
DATE_ALIASES = ("date", "datetime", "timestamp", "day", "period")
PRICE_ALIASES = ("unit_price", "price", "avg_price", "selling_price")
COST_ALIASES = ("unit_cost", "cost", "raw_material_cost")


@dataclass
class RealDataBundle:
    """Normalized daily series for simulation."""
    demand: np.ndarray
    dates: Optional[pd.DatetimeIndex]
    unit_prices: Optional[np.ndarray]
    unit_costs: Optional[np.ndarray]
    source_path: Optional[str]
    n_days: int
    mean_demand: float
    std_demand: float

    @property
    def is_real(self) -> bool:
        return self.source_path is not None


def _find_column(df: pd.DataFrame, aliases: tuple) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_enterprise_data(
    path: Optional[Union[str, Path]] = None,
    min_days: int = 365,
    max_days: Optional[int] = 365,
    resample: str = "D",
) -> RealDataBundle:
    """
    Load demand (and optional price/cost) from CSV.

    Expected formats:
      - Single column: demand / sales / units_sold
      - Full: date + demand [+ unit_price] [+ unit_cost]

    If path is None or missing, uses bundled sample data under data/.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "sample_manufacturing_demand_2024.csv"
    path = Path(path)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_sample_csv(path, n_days=max(min_days, 365))

    df = pd.read_csv(path)
    date_col = _find_column(df, DATE_ALIASES)
    demand_col = _find_column(df, DEMAND_ALIASES)

    if demand_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            raise ValueError(f"No numeric demand column in {path}")
        demand_col = numeric[0]

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        if resample:
            agg = {demand_col: "sum"}
            price_col = _find_column(df.reset_index(), PRICE_ALIASES)
            cost_col = _find_column(df.reset_index(), COST_ALIASES)
            if price_col and price_col in df.columns:
                agg[price_col] = "mean"
            if cost_col and cost_col in df.columns:
                agg[cost_col] = "mean"
            df = df.resample(resample).agg(agg)
        dates = df.index
    else:
        dates = None

    demand = df[demand_col].astype(float).values
    demand = np.maximum(demand, 0.1)

    if len(demand) < min_days:
        raise ValueError(f"Need at least {min_days} days in {path}, got {len(demand)}")

    if max_days and len(demand) > max_days:
        demand = demand[:max_days]
        if dates is not None:
            dates = dates[:max_days]

    price_col = _find_column(df.reset_index() if dates is not None else df, PRICE_ALIASES)
    cost_col = _find_column(df.reset_index() if dates is not None else df, COST_ALIASES)
    prices = None
    costs = None
    if price_col and price_col in df.columns:
        prices = df[price_col].astype(float).values[: len(demand)]
    if cost_col and cost_col in df.columns:
        costs = df[cost_col].astype(float).values[: len(demand)]

    return RealDataBundle(
        demand=demand,
        dates=dates,
        unit_prices=prices,
        unit_costs=costs,
        source_path=str(path.resolve()),
        n_days=len(demand),
        mean_demand=float(np.mean(demand)),
        std_demand=float(np.std(demand)),
    )


def _write_sample_csv(path: Path, n_days: int = 365, seed: int = 2024) -> None:
    """Create illustrative 'real' manufacturing demand series (Retail / FMCG style)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    base = 265.0
    annual = 1.0 + 0.22 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    weekly = 1.0 + 0.1 * np.sin(2 * np.pi * t / 7)
    promo = np.ones(n_days)
    promo[164:182] = 1.35
    promo[304:324] = 1.5
    promo[29:43] = 0.7
    noise = rng.lognormal(0, 0.1, n_days)
    demand = base * annual * weekly * promo * noise
    unit_price = 40.0 + 2.0 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.5, n_days)
    unit_cost = 18.0 + 0.5 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.3, n_days)

    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "demand": np.round(demand, 1),
            "unit_price": np.round(unit_price, 2),
            "unit_cost": np.round(unit_cost, 2),
        }
    ).to_csv(path, index=False)
