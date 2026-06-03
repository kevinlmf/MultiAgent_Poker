"""战术层 ML — 对比 LP：需求预测(GBR) + 学习化产量/原料分配"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.decisions import TacticalDecision
from agent.ml.base import build_lag_matrix, exp_smooth_forecast, try_import_sklearn
from agent.risk.control_agent import RiskAdjustment


class MLTacticalAgent:
    layer = "tactical"
    model = "ML-GBR"
    cadence = "monthly"

    def __init__(
        self,
        product_prices: Optional[np.ndarray] = None,
        unit_production_costs: Optional[np.ndarray] = None,
        bom: Optional[np.ndarray] = None,
        raw_base_costs: Optional[np.ndarray] = None,
    ):
        self.product_prices = product_prices or np.array([45.0, 38.0, 52.0])
        self.unit_production_costs = unit_production_costs or np.array([22.0, 18.0, 26.0])
        self.bom = bom or np.array([[2.0, 1.0, 3.0], [1.0, 2.0, 1.5], [0.5, 0.5, 1.0]])
        self.raw_base_costs = raw_base_costs or np.array([8.0, 6.5, 5.0])
        self._use_sklearn, self._sk = try_import_sklearn()
        self._forecasters = []
        self._margin_weights = None
        self._history: np.ndarray = np.array([])

    def fit(self, demand: np.ndarray) -> None:
        self._history = np.asarray(demand, dtype=float)
        n_p = len(self.product_prices)
        mix = np.array([0.5, 0.3, 0.2][:n_p])
        mix /= mix.sum()

        self._forecasters = []
        for p in range(n_p):
            series = self._history * mix[p]
            X, y = build_lag_matrix(series, lags=14)
            if self._use_sklearn and len(X) > 20:
                model = self._sk["GBR"](n_estimators=50, max_depth=4, random_state=42 + p)
                model.fit(X, y)
                self._forecasters.append(model)
            else:
                self._forecasters.append(None)

        margin = self.product_prices - self.unit_production_costs
        self._margin_weights = np.maximum(margin, 0.1) / margin.sum()

    def _forecast_product_demands(self, month_total: float) -> np.ndarray:
        n_p = len(self.product_prices)
        out = np.zeros(n_p)
        mix = self._margin_weights if self._margin_weights is not None else np.ones(n_p) / n_p
        for p in range(n_p):
            series = self._history * mix[p] if len(self._history) else np.array([month_total / n_p])
            if self._forecasters[p] is not None and len(series) >= 15:
                X, _ = build_lag_matrix(series, 14)
                pred = float(self._forecasters[p].predict(X[-1:])[0])
            else:
                pred = float(exp_smooth_forecast(series, 1)[0])
            out[p] = max(0.0, pred * 30)
        total = out.sum()
        if total <= 0:
            return month_total * mix
        scale = month_total / total
        return out * scale

    def decide(
        self,
        month: int,
        product_demand: np.ndarray,
        capacity_limit: float,
        raw_cost_multiplier: float = 1.0,
        price_multiplier: float = 1.0,
        risk: Optional[RiskAdjustment] = None,
    ) -> TacticalDecision:
        risk = risk or RiskAdjustment()
        month_total = float(np.sum(product_demand)) * risk.demand_forecast_multiplier
        if len(self._history) >= 30:
            ml_forecast = self._forecast_product_demands(month_total)
            production = 0.6 * product_demand + 0.4 * ml_forecast
        else:
            production = product_demand.copy()

        cap = capacity_limit * risk.production_scale * risk.capacity_multiplier
        if production.sum() > cap:
            production = production * (cap / production.sum())

        raw = production @ self.bom.T
        raw = np.maximum(raw, production.sum() * 0.1)
        prices = self.product_prices * price_multiplier
        revenue = float(np.dot(prices, production))
        prod_cost = float(np.dot(self.unit_production_costs, production))
        raw_cost = float(np.dot(self.raw_base_costs * raw_cost_multiplier, raw))

        return TacticalDecision(
            month=month,
            production_volume=production,
            raw_material_procurement=raw,
            revenue=revenue,
            total_cost=prod_cost + raw_cost,
            is_feasible=True,
        )
