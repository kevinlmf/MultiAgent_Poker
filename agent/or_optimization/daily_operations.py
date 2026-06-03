"""
Operational daily planning (DP) — inventory, maintenance, order fulfillment.
"""

from typing import Dict, List, Tuple

import numpy as np


class DailyOperationsSolver:
    """
    Short-horizon DP for one operating day.
    Stages: maintenance decision -> replenishment -> order fulfillment (knapsack).
    """

    def __init__(
        self,
        holding_cost: float = 0.5,
        stockout_penalty: float = 8.0,
        maintenance_cost: float = 2_000.0,
        maintenance_reliability_gain: float = 0.15,
        max_reorder: int = 200,
        reorder_step: int = 50,
    ):
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.maintenance_cost = maintenance_cost
        self.maintenance_reliability_gain = maintenance_reliability_gain
        self.max_reorder = max_reorder
        self.reorder_step = reorder_step

    def optimize_day(
        self,
        inventory: float,
        demand: float,
        backlog_orders: List[Tuple[float, float]],
        equipment_health: float,
        breakdown_risk: float,
        inbound_from_production: float = 0.0,
    ) -> Dict:
        """
        Args:
            inventory: starting inventory units
            demand: realized demand today
            backlog_orders: list of (qty, priority_weight)
            equipment_health: 0-1
            breakdown_risk: 0-1 probability-like stress factor
            inbound_from_production: finished goods arriving today
        """
        reorder_options = list(range(0, self.max_reorder + 1, self.reorder_step))
        maintain_options = [0, 1]

        best = None
        for maintain in maintain_options:
            eff_health = min(1.0, equipment_health + maintain * self.maintenance_reliability_gain)
            # Deterministic effective capacity (breakdown risk reduces throughput)
            cap_mult = eff_health * (1.0 - 0.4 * breakdown_risk * (1.0 - eff_health))
            for reorder in reorder_options:
                available = (inventory + inbound_from_production + reorder) * cap_mult
                # Fulfill backlog + demand by priority knapsack then demand
                fulfilled_backlog, backlog_value = self._fulfill_orders(available, backlog_orders)
                remaining = available - fulfilled_backlog
                sales = min(remaining, demand)
                stockout = max(0.0, demand - sales)
                end_inv = max(0.0, remaining - sales)

                cost = (
                    maintain * self.maintenance_cost
                    + reorder * 0.3
                    + end_inv * self.holding_cost
                    + stockout * self.stockout_penalty
                )
                revenue = (sales + fulfilled_backlog) * 10.0  # placeholder; overridden by caller
                score = revenue - cost

                if best is None or score > best["score"]:
                    best = {
                        "maintain_equipment": bool(maintain),
                        "reorder_qty": reorder,
                        "fulfilled_backlog": fulfilled_backlog,
                        "sales": sales,
                        "stockout": stockout,
                        "ending_inventory": end_inv,
                        "operating_cost": cost,
                        "capacity_multiplier": cap_mult,
                        "score": score,
                    }

        return {"success": True, **best}

    def _fulfill_orders(
        self, available: float, orders: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if not orders or available <= 0:
            return 0.0, 0.0
        n = len(orders)
        cap = int(min(available, 500))
        if cap <= 0:
            return 0.0, 0.0

        qtys = [int(min(o[0], cap)) for o in orders]
        weights = [max(1, int(o[0])) for o in orders]
        values = [o[1] * o[0] for o in orders]

        dp = np.zeros((n + 1, cap + 1))
        for i in range(1, n + 1):
            for w in range(cap + 1):
                dp[i][w] = dp[i - 1][w]
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

        fulfilled = 0.0
        w = cap
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                fulfilled += qtys[i - 1]
                w -= weights[i - 1]
        return min(fulfilled, available), dp[n][cap]
