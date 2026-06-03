"""
Tactical production planning (LP) — monthly decisions:
- Production volume per product
- Raw material procurement
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


class ProductionPlanningSolver:
    """Monthly LP: maximize margin subject to capacity and BOM constraints."""

    def optimize(
        self,
        product_prices: np.ndarray,
        unit_production_costs: np.ndarray,
        raw_material_costs: np.ndarray,
        bom: np.ndarray,
        monthly_demand: np.ndarray,
        capacity_limit: float,
        raw_supply_limit: np.ndarray,
    ) -> Dict:
        """
        Args:
            product_prices: unit selling price per product (P,)
            unit_production_costs: variable production cost (P,)
            raw_material_costs: procurement cost per unit material (M,)
            bom: (M, P) material units needed per product unit
            monthly_demand: forecast demand per product (P,)
            capacity_limit: max total production from strategic layer
            raw_supply_limit: max purchasable raw material (M,)
        """
        p = len(product_prices)
        m = len(raw_material_costs)
        if bom.shape != (m, p):
            raise ValueError(f"BOM shape {bom.shape} must be ({m}, {p})")

        # x[0:p] production, r[p:p+m] procurement
        n = p + m
        margin_prod = product_prices - unit_production_costs
        c = np.concatenate([-margin_prod, raw_material_costs])  # minimize negative profit + material cost

        # Production <= demand (don't overproduce beyond forecast)
        A_ub_demand = np.hstack([np.eye(p), np.zeros((p, m))])
        b_ub_demand = monthly_demand

        # Total production <= capacity
        A_ub_cap = np.zeros((1, n))
        A_ub_cap[0, :p] = 1
        b_ub_cap = np.array([capacity_limit])

        # BOM: sum_p bom_mp * x_p <= r_m
        A_ub_bom = np.hstack([bom.T, -np.eye(m)])
        b_ub_bom = np.zeros(m)

        # Raw procurement limits
        A_ub_raw = np.hstack([np.zeros((m, p)), np.eye(m)])
        b_ub_raw = raw_supply_limit

        A_ub = np.vstack([A_ub_demand, A_ub_cap, A_ub_bom, A_ub_raw])
        b_ub = np.concatenate([b_ub_demand, b_ub_cap, b_ub_bom, b_ub_raw])

        bounds = [(0, None)] * n

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return {"success": False, "status": res.message}

        x = res.x[:p]
        r = res.x[p:]
        revenue = float(np.dot(product_prices, x))
        prod_cost = float(np.dot(unit_production_costs, x))
        raw_cost = float(np.dot(raw_material_costs, r))
        return {
            "success": True,
            "status": res.message,
            "production": x,
            "raw_procurement": r,
            "revenue": revenue,
            "production_cost": prod_cost,
            "raw_material_cost": raw_cost,
            "total_cost": prod_cost + raw_cost,
            "net_margin": revenue - prod_cost - raw_cost,
        }
