"""
Strategic capacity planning (MIP) — quarterly decisions:
- Open / expand factory sites
- Production line count per site
- Workforce scale
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

logger = logging.getLogger(__name__)


class CapacityPlanningSolver:
    """
    Mixed-integer model for quarterly strategic capacity.
    Minimizes investment + labor cost while meeting forecast demand.
    """

    def __init__(
        self,
        line_capacity: float = 120.0,
        worker_productivity: float = 8.0,
        line_fixed_cost: float = 80_000.0,
        worker_annual_cost: float = 45_000.0,
        max_lines_per_site: int = 12,
        max_workers: int = 500,
    ):
        self.line_capacity = line_capacity
        self.worker_productivity = worker_productivity
        self.line_fixed_cost = line_fixed_cost
        self.worker_annual_cost = worker_annual_cost / 4  # quarterly labor
        self.max_lines_per_site = max_lines_per_site
        self.max_workers = max_workers

    def optimize(
        self,
        site_build_costs: np.ndarray,
        site_base_capacities: np.ndarray,
        quarterly_demand: float,
        budget: float,
    ) -> Dict:
        """
        Args:
            site_build_costs: fixed cost to open each candidate site
            site_base_capacities: baseline capacity if site opened with 0 extra lines
            quarterly_demand: forecast units for the quarter
            budget: capital budget for the quarter
        """
        m = len(site_build_costs)
        max_l = self.max_lines_per_site
        max_w = self.max_workers

        # Variables: y[m] binary, lines[m] int, workers int
        # Approximate lines as continuous then round (scipy milp integrality)
        num_vars = m + m + 1  # y, lines per site, workers
        y_idx = slice(0, m)
        line_idx = slice(m, 2 * m)
        w_idx = 2 * m

        c = np.zeros(num_vars)
        c[y_idx] = site_build_costs
        c[line_idx] = self.line_fixed_cost
        c[w_idx] = self.worker_annual_cost

        # Capacity >= demand: base*y + line_cap*lines + prod*workers >= demand
        A_lb = np.zeros((1, num_vars))
        A_lb[0, y_idx] = site_base_capacities
        A_lb[0, line_idx] = self.line_capacity
        A_lb[0, w_idx] = self.worker_productivity
        demand_constraint = LinearConstraint(A_lb, lb=np.array([quarterly_demand]), ub=np.inf)

        # Budget: build + lines + workers <= budget
        A_budget = np.zeros((1, num_vars))
        A_budget[0, y_idx] = site_build_costs
        A_budget[0, line_idx] = self.line_fixed_cost
        A_budget[0, w_idx] = self.worker_annual_cost
        budget_constraint = LinearConstraint(A_budget, lb=-np.inf, ub=np.array([budget]))

        # lines_i <= max_l * y_i
        A_line = np.zeros((m, num_vars))
        for i in range(m):
            A_line[i, m + i] = 1
            A_line[i, i] = -max_l
        line_cap_constraint = LinearConstraint(A_line, lb=-np.inf, ub=np.zeros(m))

        constraints = [demand_constraint, budget_constraint, line_cap_constraint]

        integrality = np.zeros(num_vars)
        integrality[y_idx] = 1
        integrality[line_idx] = 1
        integrality[w_idx] = 1

        lb = np.zeros(num_vars)
        ub = np.concatenate([
            np.ones(m),
            np.full(m, max_l),
            np.array([max_w]),
        ])
        bounds = Bounds(lb, ub)

        try:
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
            if not res.success:
                return self._greedy_fallback(
                    site_build_costs, site_base_capacities, quarterly_demand, budget
                )

            y = np.round(res.x[y_idx]).astype(int)
            lines = np.round(res.x[line_idx]).astype(int)
            workers = int(np.round(res.x[w_idx]))
            open_sites = [i for i in range(m) if y[i] >= 0.5]
            total_capacity = float(
                np.sum(site_base_capacities * y)
                + self.line_capacity * np.sum(lines)
                + self.worker_productivity * workers
            )
            return {
                "success": True,
                "status": res.message,
                "open_sites": open_sites,
                "lines_per_site": {i: int(lines[i]) for i in open_sites},
                "workforce": workers,
                "total_capacity": total_capacity,
                "total_cost": float(res.fun),
                "within_budget": float(res.fun) <= budget + 1e-6,
            }
        except Exception as e:
            logger.exception("Capacity MIP failed")
            return self._greedy_fallback(
                site_build_costs, site_base_capacities, quarterly_demand, budget
            )

    def _greedy_fallback(
        self,
        site_build_costs: np.ndarray,
        site_base_capacities: np.ndarray,
        quarterly_demand: float,
        budget: float,
    ) -> Dict:
        """Feasible plan when MIP is infeasible: prioritize lowest-cost capacity."""
        order = np.argsort(site_build_costs)
        open_sites: List[int] = []
        lines: Dict[int, int] = {}
        workers = 0
        capacity = 0.0
        spent = 0.0

        for i in order:
            if capacity >= quarterly_demand:
                break
            cost = site_build_costs[i]
            if spent + cost > budget:
                continue
            open_sites.append(i)
            lines[i] = 2
            spent += cost + 2 * self.line_fixed_cost
            capacity += site_base_capacities[i] + 2 * self.line_capacity

        while capacity < quarterly_demand and spent < budget:
            w_batch = 20
            w_cost = w_batch * self.worker_annual_cost
            if spent + w_cost > budget:
                break
            workers += w_batch
            spent += w_cost
            capacity += w_batch * self.worker_productivity

        if capacity < quarterly_demand * 0.9:
            # Last resort: maximize capacity within budget
            workers = min(self.max_workers, int((budget - spent) / max(self.worker_annual_cost, 1)))
            capacity = capacity + workers * self.worker_productivity
            spent += workers * self.worker_annual_cost
            if capacity < quarterly_demand * 0.75:
                return {"success": False, "status": "infeasible_under_budget"}

        return {
            "success": True,
            "status": "greedy_fallback",
            "open_sites": open_sites,
            "lines_per_site": lines,
            "workforce": workers,
            "total_capacity": capacity,
            "total_cost": spent,
            "within_budget": spent <= budget,
        }
