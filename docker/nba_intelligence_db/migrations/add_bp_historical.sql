-- Migration: Add BP historical analytics + DVP tables for V4 training
-- Date: 2026-03-16
-- Purpose: Store backfilled BettingPros analytics data for V4 feature extraction
--
-- Run on nba_intelligence (port 5539):
--   PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence -f add_bp_historical.sql

-- ==========================================================================
-- BP HISTORICAL ANALYTICS (per-prop analytics from /v3/props API)
-- ==========================================================================
-- One row per (player, game_date, stat_type).
-- Contains BP's projection, hit rates, EV, bet rating, opposition rank
-- at the time the prop was offered. Used for V4 training features.

CREATE TABLE IF NOT EXISTS bp_historical_analytics (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    -- Projection
    bp_projection DECIMAL(5,2),
    bp_projection_diff DECIMAL(5,2),
    bp_probability DECIMAL(5,4),
    bp_expected_value DECIMAL(6,4),
    bp_bet_rating INTEGER,
    bp_recommended_side VARCHAR(10),
    -- Opposition
    bp_opposition_rank INTEGER,
    bp_opposition_value DECIMAL(6,2),
    -- Hit Rates (over rate at time of prop)
    bp_hit_rate_L1 DECIMAL(4,3),
    bp_hit_rate_L5 DECIMAL(4,3),
    bp_hit_rate_L10 DECIMAL(4,3),
    bp_hit_rate_L15 DECIMAL(4,3),
    bp_hit_rate_L20 DECIMAL(4,3),
    bp_hit_rate_season DECIMAL(4,3),
    -- Lines
    bp_over_line DECIMAL(5,2),
    bp_consensus_line DECIMAL(5,2),
    bp_over_odds INTEGER,
    bp_consensus_odds INTEGER,
    -- Actual (from BP scoring)
    bp_actual_value DECIMAL(5,2),
    bp_is_scored BOOLEAN DEFAULT FALSE,
    -- Metadata
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (player_name, game_date, stat_type)
);

CREATE INDEX IF NOT EXISTS idx_bp_hist_date ON bp_historical_analytics (game_date);
CREATE INDEX IF NOT EXISTS idx_bp_hist_player ON bp_historical_analytics (player_name);
CREATE INDEX IF NOT EXISTS idx_bp_hist_stat ON bp_historical_analytics (stat_type);
CREATE INDEX IF NOT EXISTS idx_bp_hist_player_date ON bp_historical_analytics (player_name, game_date);

COMMENT ON TABLE bp_historical_analytics IS 'BettingPros historical prop analytics backfilled from /v3/props API for V4 training';

-- ==========================================================================
-- BP DVP HISTORICAL (defense vs position per season)
-- ==========================================================================
-- One row per (season, team, position, stat_name).
-- Contains how many points/rebounds/etc each team allows per position.
-- Scraped from bettingpros.com/nba/defense-vs-position/?season=XXXX

CREATE TABLE IF NOT EXISTS bp_dvp_historical (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    team VARCHAR(10) NOT NULL,
    position VARCHAR(5) NOT NULL,
    stat_name VARCHAR(30) NOT NULL,
    value DECIMAL(6,2),
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (season, team, position, stat_name)
);

CREATE INDEX IF NOT EXISTS idx_dvp_season ON bp_dvp_historical (season);
CREATE INDEX IF NOT EXISTS idx_dvp_team ON bp_dvp_historical (team);
CREATE INDEX IF NOT EXISTS idx_dvp_team_season ON bp_dvp_historical (team, season);

COMMENT ON TABLE bp_dvp_historical IS 'BettingPros DVP (defense vs position) stats per season for V4 training';

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON bp_historical_analytics TO mlb_user;
GRANT USAGE, SELECT ON SEQUENCE bp_historical_analytics_id_seq TO mlb_user;
GRANT ALL PRIVILEGES ON bp_dvp_historical TO mlb_user;
GRANT USAGE, SELECT ON SEQUENCE bp_dvp_historical_id_seq TO mlb_user;
