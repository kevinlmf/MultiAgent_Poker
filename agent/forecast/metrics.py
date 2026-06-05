"""Forecast accuracy metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    bias: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        return {k: v for k, v in d.items() if k != "n_samples"}


def compute_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    if n == 0:
        return ForecastMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    a = actual[:n]
    p = predicted[:n]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    denom = np.maximum(np.abs(a), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape_denom = np.maximum(np.abs(a) + np.abs(p), 1e-6)
    smape = float(np.mean(2.0 * np.abs(err) / smape_denom) * 100.0)
    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape, smape=smape, bias=bias, n_samples=n)


def pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, q: float = 0.9) -> float:
    """Pinball loss for quantile forecasts (optional probabilistic layer)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(quantile_pred, dtype=float)
    diff = a - p
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1.0) * diff)))
