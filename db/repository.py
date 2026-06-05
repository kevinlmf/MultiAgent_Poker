"""SQLite persistence for simulation runs and agent decisions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.decisions import OperationalDecision, StrategicDecision, TacticalDecision
from agent.recommendation.or_advisor import ORAdviceReport, ORRecommendation

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB = Path(__file__).parent.parent / "data" / "operations.db"


class SimulationRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(sql)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns/tables for older databases."""
        cols = {r[1] for r in conn.execute("PRAGMA table_info(simulation_runs)").fetchall()}
        for spec in (
            ("scenario_id", "TEXT"),
            ("policy_mode", "TEXT"),
            ("simulation_days", "INTEGER DEFAULT 365"),
        ):
            if spec[0] not in cols:
                conn.execute(f"ALTER TABLE simulation_runs ADD COLUMN {spec[0]} {spec[1]}")
        # Ensure forecast / unified tables exist (schema.sql also creates them)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                data_source TEXT, model_name TEXT NOT NULL, lags INTEGER DEFAULT 14,
                train_days INTEGER, val_days INTEGER, test_days INTEGER,
                fit_latency_ms REAL, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS forecast_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES forecast_experiments(id) ON DELETE CASCADE,
                split_name TEXT NOT NULL, mae REAL, rmse REAL, mape REAL, smape REAL,
                bias REAL, n_samples INTEGER, predict_latency_ms REAL,
                UNIQUE (experiment_id, split_name)
            );
            CREATE TABLE IF NOT EXISTS unified_experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                forecast_experiment_id INTEGER REFERENCES forecast_experiments(id) ON DELETE SET NULL,
                simulation_run_id INTEGER REFERENCES simulation_runs(id) ON DELETE SET NULL,
                split_name TEXT, model_name TEXT, policy_mode TEXT, scenario_id TEXT,
                forecast_mape REAL, forecast_smape REAL, annual_profit REAL,
                service_level REAL, solver_latency_ms REAL, end_to_end_latency_ms REAL,
                constraint_violations INTEGER DEFAULT 0, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS strategy_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                scenario_id TEXT, demand_mean REAL, demand_std REAL, demand_trend REAL,
                demand_cv REAL, demand_peak_ratio REAL, policy_mode TEXT, forecast_model TEXT,
                parallel_solvers INTEGER DEFAULT 0, use_robust_lp INTEGER DEFAULT 0,
                use_forecast INTEGER DEFAULT 0, annual_profit REAL, service_level REAL,
                forecast_mape REAL, solver_hint TEXT,
                simulation_run_id INTEGER REFERENCES simulation_runs(id) ON DELETE SET NULL,
                recommendation_text TEXT, notes TEXT, rl_q_table_json TEXT
            );
            """
        )
        mem_cols = {r[1] for r in conn.execute("PRAGMA table_info(strategy_memory)").fetchall()}
        if "rl_q_table_json" not in mem_cols:
            conn.execute("ALTER TABLE strategy_memory ADD COLUMN rl_q_table_json TEXT")

    def create_run(
        self,
        seed: int,
        data_source: Optional[str],
        use_synthetic: bool,
        enable_risk: bool,
        scenario_id: Optional[str] = None,
        policy_mode: Optional[str] = None,
        simulation_days: int = 365,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO simulation_runs
                   (seed, data_source, use_synthetic, enable_risk, scenario_id, policy_mode, simulation_days)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    seed, data_source, int(use_synthetic), int(enable_risk),
                    scenario_id, policy_mode, simulation_days,
                ),
            )
            return int(cur.lastrowid)

    def save_or_recommendations(self, run_id: int, report: ORAdviceReport) -> None:
        rows = [
            (run_id, r.layer, r.period, r.priority, r.action, r.rationale, r.expected_impact)
            for r in report.recommendations
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO or_recommendations
                   (run_id, layer, period, priority, action, rationale, expected_impact)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )

    def query_or_recommendations(self, run_id: int, layer: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT * FROM or_recommendations WHERE run_id=?"
        params: tuple = (run_id,)
        if layer:
            sql += " AND layer=?"
            params = (run_id, layer)
        sql += " ORDER BY id"
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def list_runs_detail(self, limit: int = 30) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """SELECT id, created_at, scenario_id, policy_mode, simulation_days,
                          annual_profit, service_level, data_source
                   FROM simulation_runs ORDER BY id DESC LIMIT ?""",
                conn,
                params=(limit,),
            )

    def save_demand_series(self, run_id: int, demand: np.ndarray, prices: Optional[np.ndarray], dates: Optional[List[str]]) -> None:
        rows = []
        for i, d in enumerate(demand):
            rows.append((
                run_id, i, dates[i] if dates else None, float(d),
                float(prices[i]) if prices is not None else None,
            ))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO demand_records (run_id, day_index, record_date, demand, unit_price) VALUES (?,?,?,?,?)",
                rows,
            )

    def save_strategic(self, run_id: int, d: StrategicDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategic_decisions
                   (run_id, quarter, open_factories, lines_per_factory, workforce, daily_capacity, investment_cost)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    run_id, d.quarter, json.dumps(d.open_factories), json.dumps(d.lines_per_factory),
                    d.workforce, d.daily_capacity, d.investment_cost,
                ),
            )

    def save_tactical(self, run_id: int, d: TacticalDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tactical_decisions
                   (run_id, month, production_json, raw_material_json, revenue, total_cost, profit)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    run_id, d.month,
                    json.dumps(d.production_volume.tolist()),
                    json.dumps(d.raw_material_procurement.tolist()),
                    d.revenue, d.total_cost, d.profit,
                ),
            )

    def save_operational_batch(self, run_id: int, rows: List[Dict[str, Any]]) -> None:
        data = [
            (
                run_id, r["day"], r["reorder"], int(r["maintain"]), r["inventory_end"],
                r["units_sold"], r["stockout"], r["operating_cost"], r.get("revenue"), r.get("profit"),
            )
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO operational_decisions
                   (run_id, day_index, reorder_qty, maintain_equipment, inventory_end,
                    units_sold, stockout, operating_cost, revenue, profit)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                data,
            )

    def save_risk_batch(self, run_id: int, rows: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO risk_events (run_id, day_index, risk_level, anomalies, plan_id, contingency_cost)
                   VALUES (?,?,?,?,?,?)""",
                [(run_id, r["day"], r["risk_level"], r["anomalies"], r["plan_id"], r.get("cost", 0)) for r in rows],
            )

    def save_scenario_events(self, run_id: int, events: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO scenario_events (run_id, day_index, event_names) VALUES (?,?,?)",
                [(run_id, e["day"], e["events"]) for e in events],
            )

    def finalize_run(self, run_id: int, revenue: float, cost: float, profit: float, service_level: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE simulation_runs
                   SET annual_revenue=?, annual_cost=?, annual_profit=?, service_level=?
                   WHERE id=?""",
                (revenue, cost, profit, service_level, run_id),
            )

    def get_run_summary(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM simulation_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def query_daily_pnl(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT day_index, revenue, profit, inventory_end FROM operational_decisions WHERE run_id=? ORDER BY day_index",
                conn, params=(run_id,),
            )

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM simulation_runs WHERE id=?", (run_id,))

    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT id, created_at, data_source, annual_profit, service_level FROM simulation_runs ORDER BY id DESC LIMIT ?",
                conn, params=(limit,),
            )

    def create_forecast_experiment(
        self,
        model_name: str,
        data_source: Optional[str],
        train_days: int,
        val_days: int,
        test_days: int,
        lags: int = 14,
        fit_latency_ms: float = 0.0,
        notes: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO forecast_experiments
                   (data_source, model_name, lags, train_days, val_days, test_days, fit_latency_ms, notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (data_source, model_name, lags, train_days, val_days, test_days, fit_latency_ms, notes),
            )
            return int(cur.lastrowid)

    def save_forecast_metrics(
        self,
        experiment_id: int,
        split_name: str,
        mae: float,
        rmse: float,
        mape: float,
        smape: float,
        bias: float,
        n_samples: int,
        predict_latency_ms: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO forecast_metrics
                   (experiment_id, split_name, mae, rmse, mape, smape, bias, n_samples, predict_latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (experiment_id, split_name, mae, rmse, mape, smape, bias, n_samples, predict_latency_ms),
            )

    def save_unified_metrics(
        self,
        *,
        forecast_experiment_id: Optional[int] = None,
        simulation_run_id: Optional[int] = None,
        split_name: Optional[str] = None,
        model_name: Optional[str] = None,
        policy_mode: Optional[str] = None,
        scenario_id: Optional[str] = None,
        forecast_mape: Optional[float] = None,
        forecast_smape: Optional[float] = None,
        annual_profit: Optional[float] = None,
        service_level: Optional[float] = None,
        solver_latency_ms: Optional[float] = None,
        end_to_end_latency_ms: Optional[float] = None,
        constraint_violations: int = 0,
        notes: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO unified_experiment_metrics
                   (forecast_experiment_id, simulation_run_id, split_name, model_name, policy_mode,
                    scenario_id, forecast_mape, forecast_smape, annual_profit, service_level,
                    solver_latency_ms, end_to_end_latency_ms, constraint_violations, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    forecast_experiment_id, simulation_run_id, split_name, model_name, policy_mode,
                    scenario_id, forecast_mape, forecast_smape, annual_profit, service_level,
                    solver_latency_ms, end_to_end_latency_ms, constraint_violations, notes,
                ),
            )
            return int(cur.lastrowid)

    def list_forecast_experiments(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """SELECT e.id, e.created_at, e.model_name, e.train_days, e.val_days, e.test_days,
                          m.split_name, m.mape, m.smape, m.rmse, m.bias
                   FROM forecast_experiments e
                   LEFT JOIN forecast_metrics m ON m.experiment_id = e.id
                   ORDER BY e.id DESC, m.split_name LIMIT ?""",
                conn,
                params=(limit,),
            )

    def list_unified_metrics(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM unified_experiment_metrics ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    def save_strategy_memory(
        self,
        *,
        scenario_id: str,
        signature: Any,
        policy_mode: str,
        forecast_model: str,
        parallel_solvers: int,
        use_robust_lp: int,
        use_forecast: int,
        annual_profit: float,
        service_level: float,
        forecast_mape: Optional[float] = None,
        solver_hint: Optional[str] = None,
        simulation_run_id: Optional[int] = None,
        recommendation_text: str = "",
        notes: Optional[str] = None,
        rl_q_table_json: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO strategy_memory
                   (scenario_id, demand_mean, demand_std, demand_trend, demand_cv, demand_peak_ratio,
                    policy_mode, forecast_model, parallel_solvers, use_robust_lp, use_forecast,
                    annual_profit, service_level, forecast_mape, solver_hint,
                    simulation_run_id, recommendation_text, notes, rl_q_table_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    scenario_id, signature.mean, signature.std, signature.trend, signature.cv,
                    signature.peak_ratio, policy_mode, forecast_model, parallel_solvers,
                    use_robust_lp, use_forecast, annual_profit, service_level, forecast_mape,
                    solver_hint, simulation_run_id, recommendation_text, notes, rl_q_table_json,
                ),
            )
            return int(cur.lastrowid)

    def list_strategy_memory(
        self,
        scenario_id: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        sql = "SELECT * FROM strategy_memory"
        params: tuple = ()
        if scenario_id:
            sql += " WHERE scenario_id=?"
            params = (scenario_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        params = params + (limit,)
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)
