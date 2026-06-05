"""
Multi-year macro economic cycle demand generator.

Typical 5-phase business cycle (≈5 years):
  recovery → expansion → peak → recession → rebound

Tiles the base real demand pattern and applies smooth macro multipliers,
prices, and costs per phase. Output CSV includes `cycle_phase` for
phase-aware strategic budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.simulation.data_loader import load_enterprise_data

DEFAULT_BASE = Path(__file__).parent.parent.parent / "data" / "sample_manufacturing_demand_2024.csv"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "scenarios"

# (phase_id, days, demand_mult_start, demand_mult_end, budget_mult, raw_cost_mult)
DEFAULT_5Y_PHASES: Tuple[Tuple[str, int, float, float, float, float], ...] = (
    ("recovery", 365, 0.88, 1.02, 1.00, 1.02),
    ("expansion", 365, 1.02, 1.26, 1.25, 1.00),
    ("peak", 365, 1.26, 1.12, 1.05, 1.03),
    ("recession", 365, 1.12, 0.70, 0.50, 1.08),
    ("rebound", 365, 0.70, 0.96, 0.85, 1.04),
)

PHASE_BUDGET_SCALE: Dict[str, float] = {
    "recovery": 1.0,
    "expansion": 1.25,
    "peak": 1.05,
    "recession": 0.50,
    "rebound": 0.85,
}


@dataclass
class CyclePhaseSpan:
    phase: str
    start_day: int
    end_day: int
    budget_scale: float


def phase_spans(phases: Tuple = DEFAULT_5Y_PHASES) -> List[CyclePhaseSpan]:
    spans: List[CyclePhaseSpan] = []
    s = 0
    for phase, days, _, _, budget_mult, _ in phases:
        spans.append(CyclePhaseSpan(phase, s, s + days, budget_mult))
        s += days
    return spans


def phase_at_day(day: int, phases: Tuple = DEFAULT_5Y_PHASES) -> Tuple[str, float]:
    """Return (phase_id, budget_scale) for simulation day index."""
    s = 0
    for phase, days, _, _, budget_mult, _ in phases:
        if day < s + days:
            return phase, budget_mult
        s += days
    last = phases[-1]
    return last[0], last[4]


def build_economic_cycle_series(
    years: int = 5,
    base_path: Optional[Path] = None,
    seed: int = 2024,
    phases: Tuple = DEFAULT_5Y_PHASES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Build daily demand, prices, costs, macro multiplier, dates, phase labels.
    """
    base_path = Path(base_path or DEFAULT_BASE)
    bundle = load_enterprise_data(base_path, min_days=90, max_days=365)
    base_d = bundle.demand.astype(float)
    base_p = bundle.unit_prices.astype(float) if bundle.unit_prices is not None else np.full(len(base_d), 42.0)
    base_c = bundle.unit_costs.astype(float) if bundle.unit_costs is not None else np.full(len(base_d), 18.0)
    rng = np.random.default_rng(seed)

    use_phases = phases[:years] if years <= len(phases) else phases + tuple(
        [("steady", 365, 1.0, 1.0, 1.0, 1.0)] * (years - len(phases))
    )
    n_total = sum(p[1] for p in use_phases)

    demand = np.zeros(n_total)
    prices = np.zeros(n_total)
    costs = np.zeros(n_total)
    macro = np.zeros(n_total)
    phase_labels: List[str] = []
    dates = pd.date_range("2024-01-01", periods=n_total, freq="D")

    idx = 0
    for phase, days, d0, d1, _, raw_mult in use_phases:
        for i in range(days):
            intra = i / max(days - 1, 1)
            m = d0 + (d1 - d0) * intra
            bday = i % len(base_d)
            noise = float(rng.lognormal(0, 0.04))
            demand[idx] = max(0.1, base_d[bday] * m * noise)
            macro[idx] = m
            phase_labels.append(phase)
            p = base_p[bday]
            c = base_c[bday]
            if phase == "recession":
                p *= 0.94 - 0.04 * intra
                c *= raw_mult
            elif phase == "expansion":
                p *= 1.0 + 0.03 * intra
            elif phase == "peak":
                c *= raw_mult
            else:
                c *= raw_mult
            prices[idx] = p
            costs[idx] = c
            idx += 1

    return demand, prices, costs, macro, dates, phase_labels


def economic_cycle_csv_path(years: int = 5, data_dir: Optional[Path] = None) -> Path:
    root = data_dir or DATA_DIR
    return root / f"economic_cycle_{years}y_demand.csv"


def ensure_economic_cycle_dataset(
    years: int = 5,
    base_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> str:
    out = economic_cycle_csv_path(years, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return str(out.resolve())

    d, p, c, macro, dates, phases = build_economic_cycle_series(years=years, base_path=base_path)
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "demand": np.round(d, 1),
        "unit_price": np.round(p, 2),
        "unit_cost": np.round(c, 2),
        "macro_multiplier": np.round(macro, 4),
        "cycle_phase": phases,
    }).to_csv(out, index=False)
    return str(out.resolve())
