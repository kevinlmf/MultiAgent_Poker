"""
Forecast Agent — independent demand forecasting for Planner / tactical layers.

Models (no extra deps beyond sklearn):
  - naive      : seasonal naive (same weekday last week)
  - exp_smooth : exponential smoothing
  - gbr        : GradientBoosting on lag features
  - mlp        : MLPRegressor on lag features (lightweight sequence model)

Optional (if torch installed):
  - lstm       : small LSTM on lag windows
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.forecast.metrics import ForecastMetrics, compute_forecast_metrics
from agent.forecast.split import TimeSeriesSplit
from agent.forecast.base import build_lag_matrix, exp_smooth_forecast, try_import_sklearn

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("naive", "exp_smooth", "gbr", "mlp", "lstm")


@dataclass
class ForecastResult:
    point: np.ndarray
    model: str
    horizon: int
    latency_ms: float = 0.0
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    residual_std: float = 0.0


class ForecastAgent:
    layer = "forecast"
    cadence = "daily"

    def __init__(
        self,
        model: str = "gbr",
        lags: int = 14,
        horizon: int = 7,
    ):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {SUPPORTED_MODELS}")
        self.model_name = model
        self.lags = lags
        self.horizon = horizon
        self._sklearn_ok, self._sk = try_import_sklearn()
        self._regressor = None
        self._scaler = None
        self._lstm = None
        self._train_series: np.ndarray = np.array([])
        self._exp_alpha = 0.35
        self._residual_std: float = 0.0
        self.last_fit_ms: float = 0.0
        self.last_predict_ms: float = 0.0

    def fit(self, series: np.ndarray) -> None:
        t0 = time.perf_counter()
        self._train_series = np.asarray(series, dtype=float)
        if self.model_name == "naive" or self.model_name == "exp_smooth":
            pass
        elif self.model_name in ("gbr", "mlp"):
            self._fit_sklearn(self._train_series)
        elif self.model_name == "lstm":
            self._fit_lstm(self._train_series)
        self._residual_std = self._estimate_residual_std(self._train_series)
        self.last_fit_ms = (time.perf_counter() - t0) * 1000.0

    def predict(self, history: np.ndarray, horizon: Optional[int] = None, z: float = 1.28) -> ForecastResult:
        h = horizon or self.horizon
        t0 = time.perf_counter()
        hist = np.asarray(history, dtype=float)
        if self.model_name == "naive":
            point = self._predict_naive(hist, h)
        elif self.model_name == "exp_smooth":
            point = exp_smooth_forecast(hist, h, self._exp_alpha)
        elif self.model_name in ("gbr", "mlp"):
            point = self._predict_sklearn_recursive(hist, h)
        elif self.model_name == "lstm":
            point = self._predict_lstm(hist, h)
        else:
            point = exp_smooth_forecast(hist, h)
        std = self._residual_std if self._residual_std > 0 else max(5.0, float(np.std(hist[-30:]) * 0.5))
        lower = np.maximum(0.0, point - z * std)
        upper = point + z * std
        latency = (time.perf_counter() - t0) * 1000.0
        self.last_predict_ms = latency
        return ForecastResult(
            point=point, lower=lower, upper=upper, model=self.model_name,
            horizon=h, latency_ms=latency, residual_std=std,
        )

    def predict_one(self, history: np.ndarray) -> float:
        return float(self.predict(history, horizon=1).point[0])

    def rolling_evaluate(
        self,
        series: np.ndarray,
        index_slice: slice,
        refit_every: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, ForecastMetrics]:
        """
        Walk-forward one-step evaluation on index_slice.
        Uses actual history up to t-1 to predict t (no future leakage in features).
        """
        series = np.asarray(series, dtype=float)
        start, stop = index_slice.start, index_slice.stop
        preds, actuals = [], []
        for t in range(start, stop):
            history = series[:t]
            if len(history) < 2:
                continue
            if refit_every and (t - start) % refit_every == 0:
                self.fit(history)
            preds.append(self.predict_one(history))
            actuals.append(series[t])
        p = np.array(preds)
        a = np.array(actuals)
        return p, a, compute_forecast_metrics(a, p)

    def _estimate_residual_std(self, series: np.ndarray, tail: int = 40) -> float:
        s = np.asarray(series, dtype=float)
        if len(s) < 5:
            return 10.0
        start = max(self.lags, len(s) - tail)
        if start >= len(s) - 1:
            return max(1.0, float(np.std(s)))
        errs = []
        for t in range(start, len(s)):
            pred = self.predict_one(s[:t])
            errs.append(pred - s[t])
        return max(1.0, float(np.sqrt(np.mean(np.array(errs) ** 2))))

    def evaluate_splits(
        self,
        series: np.ndarray,
        split: TimeSeriesSplit,
        refit_every: Optional[int] = 30,
    ) -> Dict[str, ForecastMetrics]:
        self.fit(series[split.train])
        out: Dict[str, ForecastMetrics] = {}
        _, _, out["train"] = self.rolling_evaluate(series, split.train, refit_every=refit_every)
        self.fit(series[split.train])
        _, _, out["val"] = self.rolling_evaluate(series, split.val, refit_every=refit_every)
        self.fit(series[: split.val.stop])
        _, _, out["test"] = self.rolling_evaluate(series, split.test, refit_every=refit_every)
        return out

    def _fit_sklearn(self, series: np.ndarray) -> None:
        if not self._sklearn_ok:
            logger.warning("sklearn unavailable; falling back to exp_smooth")
            self.model_name = "exp_smooth"
            return
        X, y = build_lag_matrix(series, self.lags)
        self._scaler = self._sk["Scaler"]()
        Xs = self._scaler.fit_transform(X)
        if self.model_name == "gbr":
            self._regressor = self._sk["GBR"](
                n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42
            )
        else:
            self._regressor = self._sk["MLPR"](
                hidden_layer_sizes=(64, 32), max_iter=400, random_state=42
            )
        self._regressor.fit(Xs, y)

    def _predict_sklearn_recursive(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._regressor is None or self._scaler is None:
            return exp_smooth_forecast(history, horizon, self._exp_alpha)
        seq = list(history[-max(self.lags, len(history)) :])
        preds = []
        for _ in range(horizon):
            window = np.array(seq[-self.lags :], dtype=float)
            if len(window) < self.lags:
                window = np.pad(window, (self.lags - len(window), 0), mode="edge")
            x = self._scaler.transform(window.reshape(1, -1))
            p = float(self._regressor.predict(x)[0])
            preds.append(max(0.0, p))
            seq.append(p)
        return np.array(preds)

    def _predict_naive(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) < 8:
            return exp_smooth_forecast(history, horizon, self._exp_alpha)
        out = []
        for h in range(horizon):
            idx = len(history) + h - 7
            ref = history[idx] if idx >= 0 else history[-1]
            out.append(float(ref))
        return np.array(out)

    def _fit_lstm(self, series: np.ndarray) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("torch not installed; using mlp instead of lstm")
            self.model_name = "mlp"
            self._fit_sklearn(series)
            return

        X, y = build_lag_matrix(series, self.lags)
        if len(X) < 10:
            self.model_name = "exp_smooth"
            return

        class _TinyLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.fc(o[:, -1, :])

        self._torch = torch
        self._lstm = _TinyLSTM()
        opt = torch.optim.Adam(self._lstm.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self._lstm.train()
        for _ in range(60):
            opt.zero_grad()
            loss = loss_fn(self._lstm(Xt), yt)
            loss.backward()
            opt.step()
        self._lstm.eval()

    def _predict_lstm(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._lstm is None:
            return self._predict_sklearn_recursive(history, horizon)
        torch = self._torch
        seq = list(history.astype(float))
        preds = []
        for _ in range(horizon):
            window = np.array(seq[-self.lags :], dtype=float)
            if len(window) < self.lags:
                window = np.pad(window, (self.lags - len(window), 0), mode="edge")
            with torch.no_grad():
                x = torch.tensor(window, dtype=torch.float32).reshape(1, self.lags, 1)
                p = float(self._lstm(x).item())
            p = max(0.0, p)
            preds.append(p)
            seq.append(p)
        return np.array(preds)
