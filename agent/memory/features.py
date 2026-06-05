"""Demand feature vector for strategy memory retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass
class DemandSignature:
    mean: float
    std: float
    trend: float
    cv: float
    peak_ratio: float

    def to_vector(self) -> np.ndarray:
        return np.array([self.mean, self.std, self.trend, self.cv, self.peak_ratio], dtype=float)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def encode_demand(series: np.ndarray) -> DemandSignature:
    s = np.asarray(series, dtype=float)
    if len(s) < 2:
        m = float(np.mean(s)) if len(s) else 0.0
        return DemandSignature(mean=m, std=0.0, trend=0.0, cv=0.0, peak_ratio=1.0)
    mean = float(np.mean(s))
    std = float(np.std(s))
    trend = float((np.mean(s[max(0, len(s) // 2) :]) - np.mean(s[: max(1, len(s) // 2)])) / max(mean, 1e-6))
    cv = std / max(mean, 1e-6)
    peak_ratio = float(np.max(s) / max(mean, 1e-6))
    return DemandSignature(mean=mean, std=std, trend=trend, cv=cv, peak_ratio=peak_ratio)
