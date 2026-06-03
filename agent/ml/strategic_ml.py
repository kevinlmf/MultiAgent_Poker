"""战略层 ML — 对比 MIP：用集成学习预测产能/人力投资"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.decisions import StrategicDecision
from agent.ml.base import try_import_sklearn
from agent.risk.control_agent import RiskAdjustment


class MLStrategicAgent:
    layer = "strategic"
    model = "ML-RF"
    cadence = "quarterly"

    def __init__(self):
        self._fitted = False
        self._use_sklearn, self._sk = try_import_sklearn()
        self._reg = None
        self._cost_per_cap = 1200.0

    def fit(self, demand: np.ndarray) -> None:
        d = np.asarray(demand, dtype=float)
        n = len(d)
        X, y = [], []
        for q in range(4):
            start = int(q * n / 4)
            end = int((q + 1) * n / 4)
            seg = d[start:end]
            if len(seg) < 5:
                continue
            feat = [
                float(np.mean(seg)),
                float(np.std(seg)),
                float(np.max(seg)),
                float(np.percentile(seg, 90)),
                q / 4.0,
            ]
            target_cap = float(np.mean(seg)) * 1.25
            X.append(feat)
            y.append(target_cap)
        X = np.array(X) if X else np.zeros((1, 5))
        y = np.array(y) if len(y) else np.array([np.mean(d) * 1.2])

        if self._use_sklearn:
            self._reg = self._sk["RFR"](n_estimators=40, max_depth=5, random_state=42)
            self._reg.fit(X, y)
        else:
            self._w = np.linalg.lstsq(
                np.hstack([X, np.ones((len(X), 1))]), y, rcond=None
            )[0]
        self._fitted = True

    def decide(
        self,
        quarter: int,
        daily_demand_target: float,
        budget: float,
        risk: Optional[RiskAdjustment] = None,
    ) -> StrategicDecision:
        risk = risk or RiskAdjustment()
        if not self._fitted:
            self.fit(np.array([daily_demand_target] * 90))

        feat = np.array([[
            daily_demand_target,
            daily_demand_target * 0.15,
            daily_demand_target * 1.1,
            daily_demand_target * 1.25,
            (quarter - 1) / 4.0,
        ]])
        if self._use_sklearn and self._reg is not None:
            cap = float(self._reg.predict(feat)[0])
        else:
            cap = float(np.dot(feat[0], self._w[:-1]) + self._w[-1])

        cap = max(daily_demand_target, cap) * risk.capacity_multiplier * risk.demand_forecast_multiplier
        budget_use = min(budget * risk.budget_scale, cap * self._cost_per_cap * 90)
        workforce = int(min(500, cap / 8))
        lines = max(1, int(cap / 120))
        open_sites = [2] if budget_use > 400_000 else []
        lines_map = {2: lines} if open_sites else {}

        return StrategicDecision(
            quarter=quarter,
            open_factories=open_sites,
            lines_per_factory=lines_map,
            workforce=workforce,
            daily_capacity=cap,
            investment_cost=budget_use * 0.85,
            is_feasible=True,
        )
