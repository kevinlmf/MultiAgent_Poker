"""ctypes bindings for native OpenMP capacity greedy solver."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict

import numpy as np

_LIB = None
_LIB_PATHS = [
    Path(__file__).parent / "libcapacity_greedy.dylib",
    Path(__file__).parent / "libcapacity_greedy.so",
]


class _CapacityResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_int),
        ("open_count", ctypes.c_int),
        ("open_sites", ctypes.c_int * 16),
        ("lines", ctypes.c_int * 16),
        ("workforce", ctypes.c_int),
        ("total_capacity", ctypes.c_double),
        ("total_cost", ctypes.c_double),
    ]


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    for p in _LIB_PATHS:
        if p.exists():
            _LIB = ctypes.CDLL(str(p))
            _LIB.capacity_greedy_parallel.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.POINTER(_CapacityResult),
            ]
            _LIB.capacity_greedy_parallel.restype = None
            return _LIB
    raise OSError(
        "Native library not built. Run: make -C native"
    )


def is_native_available() -> bool:
    return any(p.exists() for p in _LIB_PATHS)


def native_capacity_greedy(
    site_build_costs: np.ndarray,
    site_base_capacities: np.ndarray,
    quarterly_demand: float,
    budget: float,
    *,
    line_capacity: float = 120.0,
    line_fixed_cost: float = 80_000.0,
    worker_productivity: float = 8.0,
    worker_quarterly_cost: float = 11_250.0,
    num_trials: int = 32,
) -> Dict:
    lib = _load_lib()
    costs = np.ascontiguousarray(site_build_costs, dtype=np.float64)
    caps = np.ascontiguousarray(site_base_capacities, dtype=np.float64)
    m = len(costs)
    res = _CapacityResult()
    lib.capacity_greedy_parallel(
        costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        caps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        m,
        float(quarterly_demand),
        float(budget),
        float(line_capacity),
        float(line_fixed_cost),
        float(worker_productivity),
        float(worker_quarterly_cost),
        int(num_trials),
        ctypes.byref(res),
    )
    if not res.success:
        return {"success": False, "status": "native_infeasible"}
    open_sites = [int(res.open_sites[i]) for i in range(res.open_count)]
    lines = {sid: int(res.lines[sid]) for sid in open_sites if res.lines[sid] > 0}
    return {
        "success": True,
        "status": "native_openmp",
        "open_sites": open_sites,
        "lines_per_site": lines,
        "workforce": int(res.workforce),
        "total_capacity": float(res.total_capacity),
        "total_cost": float(res.total_cost),
        "within_budget": float(res.total_cost) <= budget + 1e-6,
    }
