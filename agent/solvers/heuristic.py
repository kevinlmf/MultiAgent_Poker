"""Fast heuristic solvers — feasible incumbents for parallel solver races."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def greedy_capacity_plan(
    site_build_costs: np.ndarray,
    site_base_capacities: np.ndarray,
    quarterly_demand: float,
    budget: float,
    *,
    line_capacity: float = 120.0,
    line_fixed_cost: float = 80_000.0,
    worker_productivity: float = 8.0,
    worker_quarterly_cost: float = 11_250.0,
    max_lines_per_site: int = 12,
    max_workers: int = 500,
    site_order: np.ndarray | None = None,
) -> Dict:
    """Greedy capacity plan — O(sites) incumbent for MIP warm-start / racing."""
    m = len(site_build_costs)
    order = site_order if site_order is not None else np.argsort(site_build_costs)
    open_sites: List[int] = []
    lines: Dict[int, int] = {}
    workers = 0
    capacity = 0.0
    spent = 0.0

    for i in order:
        if capacity >= quarterly_demand:
            break
        cost = float(site_build_costs[i])
        if spent + cost > budget:
            continue
        open_sites.append(int(i))
        lines[int(i)] = 2
        spent += cost + 2 * line_fixed_cost
        capacity += float(site_base_capacities[i]) + 2 * line_capacity

    while capacity < quarterly_demand and spent < budget:
        w_batch = 20
        w_cost = w_batch * worker_quarterly_cost
        if spent + w_cost > budget:
            break
        workers += w_batch
        spent += w_cost
        capacity += w_batch * worker_productivity

    if capacity < quarterly_demand * 0.75:
        extra_w = min(max_workers, int((budget - spent) / max(worker_quarterly_cost, 1)))
        workers += extra_w
        spent += extra_w * worker_quarterly_cost
        capacity += extra_w * worker_productivity

    if capacity < quarterly_demand * 0.5:
        return {"success": False, "status": "heuristic_infeasible"}

    return {
        "success": True,
        "status": "greedy_heuristic",
        "open_sites": open_sites,
        "lines_per_site": lines,
        "workforce": workers,
        "total_capacity": capacity,
        "total_cost": spent,
        "within_budget": spent <= budget + 1e-6,
    }


def greedy_production_plan(
    product_prices: np.ndarray,
    unit_production_costs: np.ndarray,
    raw_material_costs: np.ndarray,
    bom: np.ndarray,
    monthly_demand: np.ndarray,
    capacity_limit: float,
    raw_supply_limit: np.ndarray,
) -> Dict:
    """Margin-ranked fill — fast incumbent for monthly LP."""
    p = len(product_prices)
    margin = product_prices - unit_production_costs
    order = np.argsort(-margin)
    production = np.zeros(p)
    raw = np.zeros(len(raw_material_costs))
    cap_left = capacity_limit

    for idx in order:
        if cap_left <= 0:
            break
        qty = min(monthly_demand[idx], cap_left)
        need = bom[:, idx] * qty
        if np.any(need > raw_supply_limit - raw + 1e-9):
            scale = np.min((raw_supply_limit - raw) / np.maximum(need, 1e-9))
            if scale <= 0:
                continue
            qty *= scale
            need = bom[:, idx] * qty
        production[idx] = qty
        raw += need
        cap_left -= qty

    revenue = float(np.dot(product_prices, production))
    prod_cost = float(np.dot(unit_production_costs, production))
    raw_cost = float(np.dot(raw_material_costs, raw))
    total = prod_cost + raw_cost
    return {
        "success": True,
        "status": "greedy_heuristic",
        "production": production,
        "raw_procurement": raw,
        "revenue": revenue,
        "production_cost": prod_cost,
        "raw_material_cost": raw_cost,
        "total_cost": total,
        "net_margin": revenue - total,
    }
