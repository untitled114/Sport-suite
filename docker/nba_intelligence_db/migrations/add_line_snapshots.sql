-- Migration: Add line snapshots table + fetch_source tracking
-- Date: 2026-03-16
-- Purpose: Support direct sportsbook fetching with line movement history
--
-- Run on nba_intelligence (port 5539):
--   PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence -f add_line_snapshots.sql

-- ==========================================================================
-- LINE SNAPSHOTS (append-only line movement history)
-- ==========================================================================
-- Every fetch appends a row — no upserts. This tracks line movement
-- across the day for each book, enabling V4 features like
-- line_movement_velocity, opening_vs_current, and bp_line_latency.

CREATE TABLE IF NOT EXISTS nba_line_snapshots (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    book_name VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    over_line DECIMAL(5,2),
    over_odds INTEGER,
    under_line DECIMAL(5,2),
    under_odds INTEGER,
    fetch_source VARCHAR(50) NOT NULL DEFAULT 'bettingpros',  -- 'bettingpros' | 'direct'
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_snapshots_player_date
    ON nba_line_snapshots (player_name, game_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_book_date
    ON nba_line_snapshots (book_name, game_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_date
    ON nba_line_snapshots (game_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_source
    ON nba_line_snapshots (fetch_source);
CREATE INDEX IF NOT EXISTS idx_snapshots_at
    ON nba_line_snapshots (snapshot_at);

COMMENT ON TABLE nba_line_snapshots IS 'Append-only line movement history from all sources (direct + BettingPros)';

-- ==========================================================================
-- ALTER nba_props_xl — add fetch_source tracking columns
-- ==========================================================================
-- These columns track where each prop was sourced from and enable
-- cross-source comparison (direct vs BettingPros).

DO $$
BEGIN
    -- fetch_source: which pipeline sourced this prop
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'nba_props_xl' AND column_name = 'fetch_source'
    ) THEN
        ALTER TABLE nba_props_xl ADD COLUMN fetch_source VARCHAR(50) DEFAULT 'bettingpros';
    END IF;

    -- bp_reported_line: what BettingPros reports for the same book
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'nba_props_xl' AND column_name = 'bp_reported_line'
    ) THEN
        ALTER TABLE nba_props_xl ADD COLUMN bp_reported_line DECIMAL(5,2);
    END IF;

    -- bp_discrepancy: direct_line - bp_line (positive = BP lagging behind)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'nba_props_xl' AND column_name = 'bp_discrepancy'
    ) THEN
        ALTER TABLE nba_props_xl ADD COLUMN bp_discrepancy DECIMAL(5,2);
    END IF;
END $$;

-- Index on fetch_source for filtering
CREATE INDEX IF NOT EXISTS idx_props_xl_fetch_source ON nba_props_xl (fetch_source);

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON nba_line_snapshots TO mlb_user;
GRANT USAGE, SELECT ON SEQUENCE nba_line_snapshots_id_seq TO mlb_user;
