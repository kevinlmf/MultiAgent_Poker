"""ML 训练公用工具"""

from __future__ import annotations

import numpy as np


def build_lag_matrix(series: np.ndarray, lags: int = 7) -> tuple:
    """构造监督学习样本: X[t] = series[t-lags:t], y[t] = series[t]."""
    series = np.asarray(series, dtype=float)
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags : i])
        y.append(series[i])
    if not X:
        return np.zeros((1, lags)), np.array([float(np.mean(series))])
    return np.array(X), np.array(y)


def exp_smooth_forecast(history: np.ndarray, horizon: int, alpha: float = 0.35) -> np.ndarray:
    """指数平滑预测。"""
    if len(history) == 0:
        return np.full(horizon, 100.0)
    level = float(history[0])
    for x in history:
        level = alpha * x + (1 - alpha) * level
    return np.full(horizon, level)


def try_import_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
        return True, {
            "GBR": GradientBoostingRegressor,
            "RFR": RandomForestRegressor,
            "MLPR": MLPRegressor,
            "MLPC": MLPClassifier,
            "Scaler": StandardScaler,
        }
    except ImportError:
        return False, {}
