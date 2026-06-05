/**
 * OpenMP parallel multi-start greedy capacity planner.
 * Each thread tries a different site ordering; best feasible plan wins.
 *
 * Build: make -C native
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct CapacityResult {
    int success;
    int open_count;
    int open_sites[16];
    int lines[16];
    int workforce;
    double total_capacity;
    double total_cost;
};

static void greedy_trial(
    const double* site_costs,
    const double* site_caps,
    int m,
    double demand,
    double budget,
    double line_cap,
    double line_cost,
    double w_prod,
    double w_cost,
    int site_order[],
    CapacityResult* out
) {
    std::vector<int> open_sites;
    std::vector<int> lines(m, 0);
    double capacity = 0.0;
    double spent = 0.0;

    for (int k = 0; k < m; ++k) {
        if (capacity >= demand) break;
        int i = site_order[k];
        double cost = site_costs[i];
        if (spent + cost > budget) continue;
        open_sites.push_back(i);
        lines[i] = 2;
        spent += cost + 2.0 * line_cost;
        capacity += site_caps[i] + 2.0 * line_cap;
    }

    int workers = 0;
    while (capacity < demand && spent + 20.0 * w_cost <= budget) {
        workers += 20;
        spent += 20.0 * w_cost;
        capacity += 20.0 * w_prod;
    }

    out->success = (capacity >= demand * 0.75) ? 1 : 0;
    out->open_count = static_cast<int>(open_sites.size());
    for (int i = 0; i < 16; ++i) {
        out->open_sites[i] = (i < out->open_count) ? open_sites[i] : -1;
        out->lines[i] = lines[i];
    }
    out->workforce = workers;
    out->total_capacity = capacity;
    out->total_cost = spent;
}

extern "C" {

void capacity_greedy_parallel(
    const double* site_costs,
    const double* site_caps,
    int m,
    double demand,
    double budget,
    double line_cap,
    double line_cost,
    double w_prod,
    double w_cost,
    int num_trials,
    CapacityResult* out
) {
    CapacityResult best{};
    best.success = 0;
    best.total_cost = 1e18;

    if (m <= 0 || m > 16 || out == nullptr) {
        if (out) *out = best;
        return;
    }

    std::vector<int> base_order(m);
    for (int i = 0; i < m; ++i) base_order[i] = i;

#ifdef _OPENMP
    #pragma omp parallel
    {
        CapacityResult local{};
        local.total_cost = 1e18;
        #pragma omp for schedule(dynamic)
        for (int t = 0; t < num_trials; ++t) {
            std::vector<int> order = base_order;
            // Deterministic shuffle per trial (LCG seed)
            uint32_t seed = static_cast<uint32_t>(t * 2654435761u + 42u);
            for (int i = m - 1; i > 0; --i) {
                seed = seed * 1664525u + 1013904223u;
                int j = static_cast<int>(seed % static_cast<uint32_t>(i + 1));
                std::swap(order[i], order[j]);
            }
            CapacityResult cur{};
            greedy_trial(site_costs, site_caps, m, demand, budget,
                         line_cap, line_cost, w_prod, w_cost,
                         order.data(), &cur);
            if (cur.success && cur.total_cost < local.total_cost) {
                local = cur;
            }
        }
        #pragma omp critical
        {
            if (local.success && local.total_cost < best.total_cost) {
                best = local;
            }
        }
    }
#else
    for (int t = 0; t < num_trials; ++t) {
        std::vector<int> order = base_order;
        CapacityResult cur{};
        greedy_trial(site_costs, site_caps, m, demand, budget,
                     line_cap, line_cost, w_prod, w_cost,
                     order.data(), &cur);
        if (cur.success && cur.total_cost < best.total_cost) {
            best = cur;
        }
    }
#endif

    *out = best;
}

}  // extern "C"
