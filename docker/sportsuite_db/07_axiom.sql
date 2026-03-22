-- Axiom schema (was cephalon_axiom, port 5541)
SET search_path TO axiom, public;

-- ==========================================================================
-- PIPELINE AUDIT — one row per pipeline run
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
