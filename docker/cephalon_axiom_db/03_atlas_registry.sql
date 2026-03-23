-- Atlas Data Registry Schema
-- Tracks every piece of data flowing through the quant lab.
-- Atlas reads these tables for morning briefs, escalation context, and health monitoring.
-- Claude Code reads them on escalation to understand what happened without SSHing.

SET timezone = 'UTC';

-- ─────────────────────────────────────────────────────────────────
-- data_sources
-- Catalog of every data source in the system. What we have, what it costs,
-- and how fresh it should be. Atlas checks SLA compliance against this.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_sources (
    source_id           SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,        -- 'bettingpros_props', 'prizepicks', etc.
    provider            TEXT NOT NULL,               -- 'bettingpros', 'prizepicks', 'espn'
    source_type         TEXT NOT NULL,               -- 'api', 'scraper', 'proxy', 'database'
    description         TEXT,
    cost_monthly_usd    NUMERIC(8,2),               -- What we're paying for this source
    expected_frequency  TEXT,                        -- '6x_daily', 'daily', 'hourly', 'on_demand'
    sla_max_age_hours   INTEGER,                    -- Alert if data older than this
    markets             TEXT[],                      -- {'POINTS','REBOUNDS'} or NULL for all
    books               TEXT[],                      -- {'draftkings','fanduel'} or NULL
    enabled             BOOLEAN DEFAULT TRUE,
    config              JSONB,                       -- Source-specific config
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────
-- ingestion_log
-- Every fetch, load, backfill, and refresh operation. The heartbeat of the system.
-- First thing Atlas checks: "did data arrive? how much? any errors?"
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_log (
    id                  SERIAL PRIMARY KEY,
    source_name         TEXT NOT NULL,               -- FK-like ref to data_sources.name
    operation           TEXT NOT NULL,               -- 'fetch', 'backfill', 'refresh', 'load', 'enrich'
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    duration_seconds    NUMERIC(10,2),
    status              TEXT NOT NULL DEFAULT 'running',  -- 'running', 'success', 'partial', 'failed'

    -- Volume metrics
    records_fetched     INTEGER DEFAULT 0,
    records_new         INTEGER DEFAULT 0,
    records_duplicate   INTEGER DEFAULT 0,
    records_failed      INTEGER DEFAULT 0,

    -- Cost metrics
    api_calls_made      INTEGER DEFAULT 0,
    bytes_transferred   BIGINT DEFAULT 0,

    -- Error tracking
    error_count         INTEGER DEFAULT 0,
    error_message       TEXT,

    -- Flexible metadata: {market, book, date_range, game_date, etc.}
    metadata            JSONB,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────
-- data_coverage
-- Date-level inventory: what do we have, and is it complete?
-- Atlas uses this to spot gaps: "we have DraftKings for every day in January
-- except the 15th — investigate."
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_coverage (
    id                  SERIAL PRIMARY KEY,
    source_name         TEXT NOT NULL,
    game_date           DATE NOT NULL,
    market              TEXT,                        -- 'POINTS', 'REBOUNDS', NULL = all
    book_name           TEXT,                        -- 'draftkings', NULL = all
    record_count        INTEGER NOT NULL DEFAULT 0,
    player_count        INTEGER DEFAULT 0,           -- unique players
    has_actuals         BOOLEAN DEFAULT FALSE,
    has_enrichment      BOOLEAN DEFAULT FALSE,       -- projections, hit rates, EV
    quality_score       NUMERIC(3,2),                -- 0.00-1.00
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_coverage_unique
    ON data_coverage (source_name, game_date, COALESCE(market, ''), COALESCE(book_name, ''));

-- ─────────────────────────────────────────────────────────────────
-- system_heartbeats
-- Services check in here. Atlas monitors for missing heartbeats.
-- "Axiom hasn't reported in 2 hours — something's wrong."
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_heartbeats (
    service_name        TEXT PRIMARY KEY,            -- 'axiom', 'solace', 'lumen', 'airflow', 'api'
    status              TEXT NOT NULL DEFAULT 'healthy', -- 'healthy', 'degraded', 'down'
    heartbeat_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    uptime_seconds      INTEGER,
    version             TEXT,                        -- git hash or version string
    metadata            JSONB                        -- service-specific metrics
);

-- ─────────────────────────────────────────────────────────────────
-- atlas_alerts
-- Every alert Atlas has ever raised. Escalation trail.
-- "At 3:47 AM, Atlas flagged BettingPros returning 504s. Resolved at 4:12 AM."
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS atlas_alerts (
    id                  SERIAL PRIMARY KEY,
    alert_type          TEXT NOT NULL,               -- 'data_gap', 'source_down', 'sla_breach', 'drift', 'cost'
    severity            TEXT NOT NULL,               -- 'info', 'warning', 'critical'
    source              TEXT,                        -- which system/source triggered it
    title               TEXT NOT NULL,
    message             TEXT NOT NULL,
    metadata            JSONB,
    acknowledged        BOOLEAN DEFAULT FALSE,
    acknowledged_by     TEXT,
    acknowledged_at     TIMESTAMPTZ,
    resolved            BOOLEAN DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────
-- Indexes — fast lookups for Atlas dashboards and escalation queries
-- ─────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ingestion_source_date     ON ingestion_log (source_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_status          ON ingestion_log (status) WHERE status != 'success';
CREATE INDEX IF NOT EXISTS idx_ingestion_recent          ON ingestion_log (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_coverage_source_date      ON data_coverage (source_name, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_coverage_gaps             ON data_coverage (game_date, source_name) WHERE record_count = 0;
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved         ON atlas_alerts (created_at DESC) WHERE resolved = FALSE;
CREATE INDEX IF NOT EXISTS idx_alerts_severity           ON atlas_alerts (severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_heartbeats_stale          ON system_heartbeats (heartbeat_at);

-- ─────────────────────────────────────────────────────────────────
-- Seed data sources — our current inventory
-- ─────────────────────────────────────────────────────────────────
INSERT INTO data_sources (name, provider, source_type, description, cost_monthly_usd, expected_frequency, sla_max_age_hours, markets, books) VALUES
    ('bettingpros_props', 'bettingpros', 'api', 'Multi-book prop lines with consensus, odds, and projections', 9.99, '6x_daily', 4,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{draftkings,fanduel,betmgm,caesars,betrivers,espnbet,underdog}'),
    ('bettingpros_cheatsheet', 'bettingpros', 'api', 'Projections, hit rates, streaks, opposition rank per player/stat', 0, '6x_daily', 4,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{prizepicks,underdog}'),
    ('bettingpros_pick_recs', 'bettingpros', 'api', 'Premium pick recommendations with star ratings and EV', 0, '6x_daily', 6,
     NULL, NULL),
    ('bettingpros_hit_rates', 'bettingpros', 'api', 'Player hit rate history from BettingPros cheatsheet', 0, '6x_daily', 6,
     '{POINTS,REBOUNDS}', NULL),
    ('bettingpros_historical', 'bettingpros', 'api', 'Historical props with actuals and enrichment (backfill)', 0, 'on_demand', NULL,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{draftkings,fanduel,betmgm,caesars,bet365,betrivers,fanatics}'),
    ('prizepicks', 'prizepicks', 'proxy', 'PrizePicks projections via IPRoyal residential proxy (Colorado)', 7.00, '6x_daily', 6,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', '{prizepicks}'),
    ('espn_schedule', 'espn', 'api', 'Game schedule and scores (free, no key)', 0, 'daily', 12,
     NULL, NULL),
    ('vegas_lines', 'bettingpros', 'api', 'Game-level Vegas totals and spreads', 0, 'daily', 12,
     NULL, NULL),
    ('injuries', 'multiple', 'api', 'Player injury reports from CBS/Rotowire/BettingPros', 0, 'daily', 24,
     NULL, NULL),
    ('player_game_logs', 'nba_api', 'api', 'Official NBA box scores for outcome grading', 0, 'daily', 24,
     '{POINTS,REBOUNDS,ASSISTS,THREES}', NULL),
    ('player_rolling_stats', 'computed', 'database', 'EMA rolling stats L3/L5/L10/L20 computed from game logs', 0, 'daily', 24,
     NULL, NULL)
ON CONFLICT (name) DO NOTHING;
