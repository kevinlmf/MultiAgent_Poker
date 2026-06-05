"""Strategy memory — store and retrieve past scenario/policy outcomes (simple RAG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from agent.memory.features import DemandSignature, encode_demand
from db.repository import SimulationRepository


@dataclass
class MemoryRecall:
    policy_mode: str
    forecast_model: str
    parallel_solvers: bool
    use_robust_lp: bool
    annual_profit: float
    service_level: float
    forecast_mape: Optional[float]
    similarity: float
    rationale: str
    source_run_id: Optional[int] = None
    memory_id: Optional[int] = None
    rl_checkpoint: Optional[dict] = None


class StrategyMemoryAgent:
    """
    Lightweight memory layer (no external vector DB):
      - Encode demand shape → feature vector
      - kNN retrieve similar past runs
      - Recommend policy / forecast / solver settings
    """

    layer = "memory"

    def __init__(self, repo: Optional[SimulationRepository] = None):
        self.repo = repo or SimulationRepository()

    def record(
        self,
        *,
        scenario_id: str,
        demand_series: np.ndarray,
        policy_mode: str,
        forecast_model: Optional[str],
        parallel_solvers: bool,
        use_robust_lp: bool,
        use_forecast: bool,
        annual_profit: float,
        service_level: float,
        forecast_mape: Optional[float] = None,
        solver_hint: Optional[str] = None,
        simulation_run_id: Optional[int] = None,
        notes: Optional[str] = None,
        rl_q_table: Optional[dict] = None,
    ) -> int:
        sig = encode_demand(demand_series)
        text = self._build_recommendation_text(
            scenario_id, policy_mode, forecast_model, annual_profit, service_level, forecast_mape,
        )
        rl_json = None
        if rl_q_table:
            import json
            rl_json = json.dumps(rl_q_table)
        return self.repo.save_strategy_memory(
            scenario_id=scenario_id,
            signature=sig,
            policy_mode=policy_mode,
            forecast_model=forecast_model or "none",
            parallel_solvers=int(parallel_solvers),
            use_robust_lp=int(use_robust_lp),
            use_forecast=int(use_forecast),
            annual_profit=annual_profit,
            service_level=service_level,
            forecast_mape=forecast_mape,
            solver_hint=solver_hint,
            simulation_run_id=simulation_run_id,
            recommendation_text=text,
            notes=notes,
            rl_q_table_json=rl_json,
        )

    def recall(
        self,
        demand_series: np.ndarray,
        scenario_id: Optional[str] = None,
        k: int = 5,
        min_similarity: float = 0.35,
    ) -> Optional[MemoryRecall]:
        entries = self.repo.list_strategy_memory(scenario_id=scenario_id, limit=200)
        if entries.empty:
            return None
        query = encode_demand(demand_series).to_vector()
        scored: List[tuple] = []
        for _, row in entries.iterrows():
            vec = np.array([
                row["demand_mean"], row["demand_std"], row["demand_trend"],
                row["demand_cv"], row["demand_peak_ratio"],
            ], dtype=float)
            sim = self._similarity(query, vec)
            if sim >= min_similarity:
                scored.append((sim, row))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], -float(x[1]["annual_profit"])))
        sim, best = scored[0]
        return self._row_to_recall(sim, best)

    def recall_top_k(
        self,
        demand_series: np.ndarray,
        scenario_id: Optional[str] = None,
        k: int = 3,
    ) -> List[MemoryRecall]:
        entries = self.repo.list_strategy_memory(scenario_id=scenario_id, limit=200)
        if entries.empty:
            return []
        query = encode_demand(demand_series).to_vector()
        scored = []
        for _, row in entries.iterrows():
            vec = np.array([
                row["demand_mean"], row["demand_std"], row["demand_trend"],
                row["demand_cv"], row["demand_peak_ratio"],
            ], dtype=float)
            scored.append((self._similarity(query, vec), row))
        scored.sort(key=lambda x: (-x[0], -float(x[1]["annual_profit"])))
        out = []
        for sim, row in scored[:k]:
            out.append(self._row_to_recall(sim, row))
        return out

    @staticmethod
    def _parse_rl_checkpoint(row) -> Optional[dict]:
        raw = row.get("rl_q_table_json") if hasattr(row, "get") else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
            return None
        import json
        try:
            return json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return None

    def _row_to_recall(self, sim: float, row) -> MemoryRecall:
        return MemoryRecall(
            policy_mode=str(row["policy_mode"]),
            forecast_model=str(row["forecast_model"]),
            parallel_solvers=bool(row["parallel_solvers"]),
            use_robust_lp=bool(row["use_robust_lp"]),
            annual_profit=float(row["annual_profit"]),
            service_level=float(row["service_level"]),
            forecast_mape=(
                float(row["forecast_mape"])
                if row.get("forecast_mape") is not None and str(row.get("forecast_mape")) != "nan"
                else None
            ),
            similarity=float(sim),
            rationale=str(row["recommendation_text"]),
            source_run_id=int(row["simulation_run_id"]) if row.get("simulation_run_id") else None,
            memory_id=int(row["id"]) if row.get("id") is not None else None,
            rl_checkpoint=self._parse_rl_checkpoint(row),
        )

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        cos = float(np.dot(a, b) / (na * nb))
        scale = 1.0 / (1.0 + np.mean(np.abs(a - b)) / max(np.mean(np.abs(a)), 1.0))
        return max(0.0, min(1.0, 0.5 * cos + 0.5 * scale))

    @staticmethod
    def _build_recommendation_text(
        scenario_id: str,
        policy_mode: str,
        forecast_model: Optional[str],
        profit: float,
        sl: float,
        mape: Optional[float],
    ) -> str:
        fc = forecast_model or "oracle"
        mape_s = f", forecast MAPE {mape:.1f}%" if mape is not None else ""
        return (
            f"Scenario '{scenario_id}': policy={policy_mode}, forecast={fc}{mape_s} "
            f"achieved profit ${profit:,.0f} and service level {sl:.1%}."
        )
