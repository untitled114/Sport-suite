-- Axiom schema (was cephalon_axiom, port 5541)
SET search_path TO axiom, public;

-- ==========================================================================
-- PIPELINE RUNS — structured telemetry per run (Operational Excellence)
-- ==========================================================================
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    run_date DATE NOT NULL,
    run_number INTEGER NOT NULL,
    run_type TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    duration_ms INTEGER,
    tasks JSONB,                              -- per-task breakdown [{name, status, duration_ms, metrics}]
    anomalies JSONB,                          -- [{type, severity, message}]
    summary JSONB,                            -- {picks_generated, feature_count, props_fetched, ...}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pr_date ON pipeline_runs(run_date, run_number);
CREATE INDEX idx_pr_status ON pipeline_runs(status) WHERE status != 'success';
CREATE INDEX idx_pr_anomalies ON pipeline_runs(run_date) WHERE anomalies IS NOT NULL;

-- ==========================================================================
-- MODEL REGISTRY — tracks which model is production/shadow/rolled_back
-- ==========================================================================
CREATE TABLE model_registry (
    id              SERIAL PRIMARY KEY,
    version         TEXT NOT NULL UNIQUE,     -- 'v4_POINTS_20260322'
    market          TEXT NOT NULL,            -- 'POINTS', 'REBOUNDS'
    status          TEXT NOT NULL DEFAULT 'training',  -- training/shadow/production/rolled_back
    auc             NUMERIC(6,4),
    r2              NUMERIC(6,4),
    feature_count   INTEGER,
    win_rate        NUMERIC(5,4),             -- populated during shadow/production
    pkl_path        TEXT NOT NULL DEFAULT '',
    training_samples INTEGER,
    promoted_at     TIMESTAMPTZ,
    rolled_back_at  TIMESTAMPTZ,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mr_market ON model_registry(market, status);
CREATE INDEX idx_mr_production ON model_registry(market) WHERE status = 'production';

-- ==========================================================================
-- PIPELINE AUDIT — one row per pipeline run (legacy, kept for backward compat)
-- ==========================================================================
CREATE TABLE axiom_pipeline_audit (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    run_number INTEGER NOT NULL,
    run_type TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL,
    props_fetched INTEGER,
    books_available INTEGER,
    injuries_updated BOOLEAN,
    games_found INTEGER,
    duration_seconds INTEGER,
    picks_generated INTEGER,
    xl_picks INTEGER,
    v3_picks INTEGER,
    error_message TEXT,
    error_traceback TEXT,
    anomalies JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_date, run_number)
);

CREATE INDEX idx_audit_date ON axiom_pipeline_audit(run_date, run_number);
CREATE INDEX idx_audit_status ON axiom_pipeline_audit(run_date, status);

-- ==========================================================================
-- PREDICTION HISTORY — one row per (run, player, stat_type, model_version)
-- ==========================================================================
CREATE TABLE nba_prediction_history (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    run_number INTEGER NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    player_name TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    tier TEXT,
    line NUMERIC(5,1),
    p_over NUMERIC(5,4),
    edge NUMERIC(6,3),
    spread NUMERIC(5,1),
    book TEXT,
    game_time TIMESTAMPTZ,
    opponent_team TEXT,
    is_home BOOLEAN,
    context_snapshot JSONB,
    actual_result NUMERIC(5,1),
    is_hit BOOLEAN,
    result_source TEXT,
    result_recorded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ph_date ON nba_prediction_history(run_date, run_number);
CREATE INDEX idx_ph_player ON nba_prediction_history(run_date, player_name, stat_type);
CREATE INDEX idx_ph_outcome ON nba_prediction_history(run_date, is_hit) WHERE is_hit IS NOT NULL;

-- ==========================================================================
-- CONVICTION — recomputed after every pipeline run
-- ==========================================================================
CREATE TABLE axiom_conviction (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    player_name TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    conviction NUMERIC(5,3) NOT NULL,
    conviction_label TEXT NOT NULL,
    appearances INTEGER NOT NULL,
    total_runs INTEGER NOT NULL,
    entry_run INTEGER NOT NULL,
    last_seen_run INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL,
    run_pattern TEXT NOT NULL,
    line_at_entry NUMERIC(5,1),
    line_latest NUMERIC(5,1),
    line_movement NUMERIC(5,1),
    line_direction TEXT,
    book_latest TEXT,
    p_over_at_entry NUMERIC(5,4),
    p_over_latest NUMERIC(5,4),
    p_over_trend NUMERIC(5,4),
    p_over_std NUMERIC(5,4),
    status TEXT NOT NULL,
    context_snapshot JSONB,
    narrative TEXT,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_date, player_name, stat_type)
);

CREATE INDEX idx_conv_date ON axiom_conviction(run_date);
CREATE INDEX idx_conv_status ON axiom_conviction(run_date, status);
CREATE INDEX idx_conv_label ON axiom_conviction(run_date, conviction_label);

-- ==========================================================================
-- POSTS — every message Axiom has sent (prevents duplicates)
-- ==========================================================================
CREATE TABLE axiom_posts (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    post_type TEXT NOT NULL,
    trigger TEXT,
    picks_sent JSONB,
    channel_id TEXT,
    message_id TEXT,
    sent_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_posts_date ON axiom_posts(run_date, post_type);

-- ==========================================================================
-- MEMORY — persistent key-value store
-- ==========================================================================
CREATE TABLE axiom_memory (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================================================================
-- ATLAS REGISTRY (data sources, ingestion, coverage, heartbeats, alerts)
-- ==========================================================================
CREATE TABLE data_sources (
    source_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    source_type TEXT NOT NULL,
    description TEXT,
    cost_monthly_usd NUMERIC(8,2),
    expected_frequency TEXT,
    sla_max_age_hours INTEGER,
    markets TEXT[],
    books TEXT[],
    enabled BOOLEAN DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE ingestion_log (
    id SERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds NUMERIC(10,2),
    status TEXT NOT NULL DEFAULT 'running',
    records_fetched INTEGER DEFAULT 0,
    records_new INTEGER DEFAULT 0,
    records_duplicate INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,
    bytes_transferred BIGINT DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ing_source ON ingestion_log(source_name, started_at DESC);
CREATE INDEX idx_ing_status ON ingestion_log(status) WHERE status != 'success';
CREATE INDEX idx_ing_recent ON ingestion_log(started_at DESC);

CREATE TABLE data_coverage (
    id SERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    game_date DATE NOT NULL,
    market TEXT,
    book_name TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    player_count INTEGER DEFAULT 0,
    has_actuals BOOLEAN DEFAULT FALSE,
    has_enrichment BOOLEAN DEFAULT FALSE,
    quality_score NUMERIC(3,2),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_cov_unique
    ON data_coverage (source_name, game_date, COALESCE(market, ''), COALESCE(book_name, ''));
CREATE INDEX idx_cov_gaps ON data_coverage(game_date, source_name) WHERE record_count = 0;

CREATE TABLE system_heartbeats (
    service_name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'healthy',
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    uptime_seconds INTEGER,
    version TEXT,
    metadata JSONB
);

CREATE TABLE atlas_alerts (
    id SERIAL PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    source TEXT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_unresolved ON atlas_alerts(created_at DESC) WHERE resolved = FALSE;
CREATE INDEX idx_alerts_severity ON atlas_alerts(severity, created_at DESC);

-- ==========================================================================
-- VALIDATION RUNS — walk-forward results per model version
-- ==========================================================================
CREATE TABLE validation_runs (
    id SERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    market TEXT NOT NULL,
    run_date DATE NOT NULL DEFAULT CURRENT_DATE,
    auc_mean NUMERIC(6,4),
    auc_std NUMERIC(6,4),
    wr_mean NUMERIC(6,4),
    roi_mean NUMERIC(6,4),
    fold_count INTEGER NOT NULL,
    beats_baseline BOOLEAN,
    promoted BOOLEAN DEFAULT FALSE,
    promoted_at TIMESTAMPTZ,
    rolled_back BOOLEAN DEFAULT FALSE,
    rolled_back_at TIMESTAMPTZ,
    rollback_reason TEXT,
    raw_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_vr_version ON validation_runs(model_version, market);
CREATE INDEX idx_vr_promoted ON validation_runs(market, promoted) WHERE promoted = TRUE;
CREATE INDEX idx_vr_date ON validation_runs(run_date DESC);

-- ==========================================================================
-- MODEL REGISTRY — which version is production for each market
-- ==========================================================================
CREATE TABLE model_registry (
    market TEXT PRIMARY KEY,
    production_version TEXT NOT NULL,
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validation_run_id INTEGER REFERENCES validation_runs(id),
    previous_version TEXT,
    rollback_count INTEGER DEFAULT 0
);

INSERT INTO model_registry (market, production_version) VALUES
    ('POINTS', 'v3'),
    ('REBOUNDS', 'v3')
ON CONFLICT (market) DO NOTHING;

-- Seed data sources
INSERT INTO data_sources (name, provider, source_type, description, cost_monthly_usd, expected_frequency, sla_max_age_hours, markets, books) VALUES
    ('bettingpros_props', 'bettingpros', 'api', 'Multi-book prop lines', 9.99, '6x_daily', 4,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{draftkings,fanduel,betmgm,caesars,betrivers,espnbet,underdog}'),
    ('bettingpros_cheatsheet', 'bettingpros', 'api', 'Projections, hit rates, streaks', 0, '6x_daily', 4,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{prizepicks,underdog}'),
    ('prizepicks', 'prizepicks', 'proxy', 'PrizePicks projections via proxy', 7.00, '6x_daily', 6,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{prizepicks}'),
    ('espn_schedule', 'espn', 'api', 'Game schedule and scores', 0, 'daily', 12, NULL, NULL),
    ('injuries', 'multiple', 'api', 'Player injury reports', 0, 'daily', 24, NULL, NULL),
    ('player_game_logs', 'nba_api', 'api', 'Official NBA box scores', 0, 'daily', 24,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', NULL)
ON CONFLICT (name) DO NOTHING;
