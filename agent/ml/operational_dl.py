"""运营层 DL — 对比 DP：MLP 神经网络预测补货与维护"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from agent.decisions import OperationalDecision
from agent.ml.base import build_lag_matrix, try_import_sklearn
from agent.risk.control_agent import RiskAdjustment


class OperationalDLAgent:
    layer = "operational"
    model = "DL-MLP"
    cadence = "daily"

    def __init__(self):
        self._use_sklearn, self._sk = try_import_sklearn()
        self._reorder_model = None
        self._maintain_model = None
        self._scaler = None
        self._fitted = False
        self.max_reorder = 200

    def fit(self, demand: np.ndarray) -> None:
        d = np.asarray(demand, dtype=float)
        X, y_d = build_lag_matrix(d, lags=7)
        inv_proxy = np.maximum(0, np.cumsum(y_d[::-1])[::-1] * 0.01)[: len(y_d)]
        y_reorder = np.clip((y_d - inv_proxy) * 0.5, 0, self.max_reorder)
        vol = _rolling_std(y_d, 7) if len(y_d) > 7 else np.zeros(len(y_d))
        y_maintain = (vol > np.percentile(vol, 70)).astype(int) if len(vol) else np.zeros(len(y_d), dtype=int)

        feats = self._features_from_matrix(X, inv_proxy[: len(X)])
        if self._use_sklearn and len(feats) > 30:
            self._scaler = self._sk["Scaler"]()
            Xs = self._scaler.fit_transform(feats)
            self._reorder_model = self._sk["MLPR"](
                hidden_layer_sizes=(64, 32), max_iter=200, random_state=42
            )
            self._reorder_model.fit(Xs, y_reorder)
            self._maintain_model = self._sk["MLPC"](
                hidden_layer_sizes=(32, 16), max_iter=200, random_state=42
            )
            self._maintain_model.fit(Xs, y_maintain)
        self._fitted = True
        self._fallback_mean_reorder = float(np.median(y_reorder))

    def _features_from_matrix(self, X: np.ndarray, inv: np.ndarray) -> np.ndarray:
        return np.column_stack([
            X,
            inv,
            X.mean(axis=1),
            X.std(axis=1),
        ])

    def _predict(self, demand_history: List[float], inventory: float, equipment_health: float) -> Tuple[int, bool]:
        hist = np.array(demand_history[-7:] if demand_history else [100.0])
        if len(hist) < 7:
            hist = np.pad(hist, (7 - len(hist), 0), constant_values=hist.mean() if len(hist) else 100)
        inv = inventory / max(hist.mean() * 7, 1)
        feat = np.array([np.concatenate([hist, [inv, hist.mean(), hist.std()]])])

        if self._use_sklearn and self._reorder_model is not None and self._scaler is not None:
            feat_s = self._scaler.transform(feat)
            reorder = int(np.clip(self._reorder_model.predict(feat_s)[0], 0, self.max_reorder))
            maintain = bool(self._maintain_model.predict(feat_s)[0])
            return reorder, maintain

        # Numpy 回退
        gap = max(0.0, hist.mean() * 3 - inventory)
        return int(min(self.max_reorder, gap * 0.4)), equipment_health < 0.88

    def decide(
        self,
        day: int,
        inventory: float,
        demand: float,
        backlog: List[Tuple[float, float]],
        equipment_health: float,
        breakdown_risk: float,
        inbound_production: float = 0.0,
        risk: Optional[RiskAdjustment] = None,
    ) -> OperationalDecision:
        risk = risk or RiskAdjustment()
        if not self._fitted:
            self.fit(np.array([demand] * 60))

        reorder, maintain = self._predict(
            [demand] * 7, inventory, equipment_health
        )
        reorder = min(self.max_reorder, reorder + risk.reorder_boost)
        if getattr(risk, "force_maintain", False):
            maintain = True

        available = (inventory + inbound_production + reorder + risk.extra_inventory_injection)
        cap_mult = equipment_health * (1.0 - 0.3 * breakdown_risk)
        available *= cap_mult

        backlog_vol = sum(q for q, _ in backlog)
        sales = min(available, demand)
        fulfill_backlog = min(max(0.0, available - sales), backlog_vol)
        stockout = max(0.0, demand - sales)
        end_inv = max(0.0, available - sales - fulfill_backlog)

        op_cost = reorder * 0.3 + (2000.0 if maintain else 0) + end_inv * 0.5 + stockout * 8.0
        op_cost += risk.contingency_cost

        return OperationalDecision(
            day=day,
            reorder_qty=reorder,
            maintain_equipment=maintain,
            inventory_end=end_inv,
            units_sold=sales,
            backlog_fulfilled=fulfill_backlog,
            stockout=stockout,
            operating_cost=op_cost,
        )


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        seg = arr[max(0, i - w + 1) : i + 1]
        out[i] = np.std(seg)
    return out
