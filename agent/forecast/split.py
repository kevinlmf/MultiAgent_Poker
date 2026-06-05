"""Chronological train / val / test splits for time series (no shuffle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: slice
    val: slice
    test: slice
    train_days: int
    val_days: int
    test_days: int

    def ranges(self) -> Tuple[slice, slice, slice]:
        return self.train, self.val, self.test


def chronological_split(
    n: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> TimeSeriesSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if n < 30:
        raise ValueError(f"Need at least 30 days, got {n}")

    n_train = max(14, int(n * train_ratio))
    n_val = max(7, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 7:
        n_test = 7
        n_train = n - n_val - n_test

    i1 = n_train
    i2 = n_train + n_val
    return TimeSeriesSplit(
        train=slice(0, i1),
        val=slice(i1, i2),
        test=slice(i2, n),
        train_days=i1,
        val_days=i2 - i1,
        test_days=n - i2,
    )


def split_series(
    demand: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> Tuple[TimeSeriesSplit, np.ndarray, Optional[np.ndarray]]:
    split = chronological_split(len(demand), train_ratio, val_ratio, test_ratio)
    d_dates = dates[split.ranges()[0]] if dates is not None else None
    return split, demand, dates
