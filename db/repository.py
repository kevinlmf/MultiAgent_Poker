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
