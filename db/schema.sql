-- Multi-Agent Operations System — SQLite schema

CREATE TABLE IF NOT EXISTS simulation_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    seed            INTEGER,
    scenario_id     TEXT,
    policy_mode     TEXT,
    simulation_days INTEGER DEFAULT 365,
    data_source     TEXT,
    use_synthetic   INTEGER NOT NULL DEFAULT 0,
    enable_risk     INTEGER NOT NULL DEFAULT 1,
    annual_revenue  REAL,
    annual_cost     REAL,
    annual_profit   REAL,
    service_level   REAL
);

CREATE TABLE IF NOT EXISTS demand_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    day_index       INTEGER NOT NULL,
    record_date     TEXT,
    demand          REAL NOT NULL,
    unit_price      REAL,
    UNIQUE (run_id, day_index)
);

CREATE TABLE IF NOT EXISTS strategic_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    quarter         INTEGER NOT NULL,
    open_factories  TEXT NOT NULL,
    lines_per_factory TEXT NOT NULL,
    workforce       INTEGER NOT NULL,
    daily_capacity  REAL NOT NULL,
    investment_cost REAL NOT NULL,
    UNIQUE (run_id, quarter)
);

CREATE TABLE IF NOT EXISTS tactical_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    month           INTEGER NOT NULL,
    production_json TEXT NOT NULL,
    raw_material_json TEXT NOT NULL,
    revenue         REAL NOT NULL,
    total_cost      REAL NOT NULL,
    profit          REAL NOT NULL,
    UNIQUE (run_id, month)
);

CREATE TABLE IF NOT EXISTS operational_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    day_index       INTEGER NOT NULL,
    reorder_qty     INTEGER NOT NULL,
    maintain_equipment INTEGER NOT NULL,
    inventory_end   REAL NOT NULL,
    units_sold      REAL NOT NULL,
    stockout        REAL NOT NULL,
    operating_cost  REAL NOT NULL,
    revenue         REAL,
    profit          REAL,
    UNIQUE (run_id, day_index)
);

CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    day_index       INTEGER NOT NULL,
    risk_level      TEXT NOT NULL,
    anomalies       TEXT,
    plan_id         TEXT,
    contingency_cost REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scenario_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    day_index       INTEGER NOT NULL,
    event_names     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS or_recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    layer           TEXT NOT NULL,
    period          TEXT NOT NULL,
    priority        TEXT NOT NULL,
    action          TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    expected_impact TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ops_run_day ON operational_decisions(run_id, day_index);
CREATE INDEX IF NOT EXISTS idx_risk_run_day ON risk_events(run_id, day_index);
CREATE INDEX IF NOT EXISTS idx_or_rec_run ON or_recommendations(run_id, layer);

-- Forecast experiments (Phase 1: walk-forward eval)
CREATE TABLE IF NOT EXISTS forecast_experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    data_source     TEXT,
    model_name      TEXT NOT NULL,
    lags            INTEGER DEFAULT 14,
    train_days      INTEGER,
    val_days        INTEGER,
    test_days       INTEGER,
    fit_latency_ms  REAL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS forecast_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES forecast_experiments(id) ON DELETE CASCADE,
    split_name      TEXT NOT NULL,
    mae             REAL,
    rmse            REAL,
    mape            REAL,
    smape           REAL,
    bias            REAL,
    n_samples       INTEGER,
    predict_latency_ms REAL,
    UNIQUE (experiment_id, split_name)
);

-- Unified metrics: forecast quality + downstream simulation
CREATE TABLE IF NOT EXISTS unified_experiment_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    forecast_experiment_id  INTEGER REFERENCES forecast_experiments(id) ON DELETE SET NULL,
    simulation_run_id       INTEGER REFERENCES simulation_runs(id) ON DELETE SET NULL,
    split_name              TEXT,
    model_name              TEXT,
    policy_mode             TEXT,
    scenario_id             TEXT,
    forecast_mape           REAL,
    forecast_smape          REAL,
    annual_profit           REAL,
    service_level           REAL,
    solver_latency_ms       REAL,
    end_to_end_latency_ms   REAL,
    constraint_violations   INTEGER DEFAULT 0,
    notes                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecast_exp_model ON forecast_experiments(model_name);
CREATE INDEX IF NOT EXISTS idx_unified_exp ON unified_experiment_metrics(forecast_experiment_id);

-- Strategy memory / RAG layer
CREATE TABLE IF NOT EXISTS strategy_memory (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    scenario_id         TEXT,
    demand_mean         REAL,
    demand_std          REAL,
    demand_trend        REAL,
    demand_cv           REAL,
    demand_peak_ratio   REAL,
    policy_mode         TEXT,
    forecast_model      TEXT,
    parallel_solvers    INTEGER DEFAULT 0,
    use_robust_lp       INTEGER DEFAULT 0,
    use_forecast        INTEGER DEFAULT 0,
    annual_profit       REAL,
    service_level       REAL,
    forecast_mape       REAL,
    solver_hint         TEXT,
    simulation_run_id   INTEGER REFERENCES simulation_runs(id) ON DELETE SET NULL,
    recommendation_text TEXT,
    notes               TEXT,
    rl_q_table_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_strategy_memory_scenario ON strategy_memory(scenario_id);
