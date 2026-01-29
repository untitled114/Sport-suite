-- NBA Intelligence Database Schema
-- Created: October 22, 2025
-- Port: 5539
-- Purpose: Prop lines, picks tracking, injury reports, lineup intelligence

-- ==========================================================================
-- INJURY REPORT
-- ==========================================================================
CREATE TABLE injury_report (
    injury_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    report_date DATE NOT NULL,
    status VARCHAR(50),  -- 'Out', 'Doubtful', 'Questionable', 'Probable', 'GTD'
    injury_type VARCHAR(100),
    expected_return DATE,
    confidence DECIMAL(3,2),  -- 0.0 to 1.0 (e.g., 0.8 for Probable)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_injury_player ON injury_report(player_id);
CREATE INDEX idx_injury_date ON injury_report(report_date);
CREATE INDEX idx_injury_status ON injury_report(status);

COMMENT ON TABLE injury_report IS 'NBA player injury status and expected return dates';

-- ==========================================================================
-- LINEUP INTELLIGENCE
-- ==========================================================================
CREATE TABLE lineup_intel (
    lineup_id SERIAL PRIMARY KEY,
    game_id VARCHAR(20) NOT NULL,
    team_abbrev VARCHAR(10) NOT NULL,
    starting_lineup TEXT[],  -- Array of player_ids
    confirmed BOOLEAN DEFAULT FALSE,
    minutes_projection JSONB,  -- {"player_id": minutes_estimate}
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lineup_game ON lineup_intel(game_id);
CREATE INDEX idx_lineup_team ON lineup_intel(team_abbrev);

COMMENT ON TABLE lineup_intel IS 'NBA starting lineups and minute projections';

-- ==========================================================================
-- PROP LINES (Unified from PrizePicks/Underdog)
-- ==========================================================================
CREATE TABLE nba_prop_lines (
    prop_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    game_id VARCHAR(20),
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,  -- 'POINTS', 'REBOUNDS', 'ASSISTS', 'PRA', '3PM', etc.
    line DECIMAL(5,2) NOT NULL,
    line_type VARCHAR(20),  -- 'main', 'demon', 'goblin'
    source VARCHAR(50) NOT NULL,  -- 'prizepicks', 'underdog', 'oddsjam'
    odds_over DECIMAL(6,2),
    odds_under DECIMAL(6,2),
    ingested_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prop_player ON nba_prop_lines(player_id);
CREATE INDEX idx_prop_date ON nba_prop_lines(game_date);
CREATE INDEX idx_prop_stat_type ON nba_prop_lines(stat_type);
CREATE INDEX idx_prop_source ON nba_prop_lines(source);

COMMENT ON TABLE nba_prop_lines IS 'NBA player prop lines from multiple sportsbooks';

-- ==========================================================================
-- PICKS PLACED (Tracking System)
-- ==========================================================================
CREATE TABLE nba_picks_placed (
    pick_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    line DECIMAL(5,2) NOT NULL,
    pick_direction VARCHAR(10) NOT NULL,  -- 'OVER', 'UNDER'
    model_prediction DECIMAL(5,2),
    edge DECIMAL(5,2),  -- Model prediction - line
    confidence DECIMAL(5,3),  -- Probability (0-1)
    bet_size DECIMAL(10,2),  -- Dollar amount
    source VARCHAR(50),
    odds DECIMAL(6,2),
    -- Results (populated after game)
    actual_value DECIMAL(5,2),
    result VARCHAR(10),  -- 'WIN', 'LOSS', 'PUSH', 'PENDING'
    profit_loss DECIMAL(10,2),
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at TIMESTAMP
);

CREATE INDEX idx_picks_date ON nba_picks_placed(game_date);
CREATE INDEX idx_picks_player ON nba_picks_placed(player_id);
CREATE INDEX idx_picks_result ON nba_picks_placed(result);
CREATE INDEX idx_picks_stat_type ON nba_picks_placed(stat_type);

COMMENT ON TABLE nba_picks_placed IS 'NBA picks tracking with results and P&L';

-- ==========================================================================
-- PLAYER FORM (Recent performance trends)
-- ==========================================================================
CREATE TABLE player_form (
    player_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    games_l3 INTEGER,
    games_l5 INTEGER,
    games_l10 INTEGER,
    ppg_l3 DECIMAL(5,2),
    ppg_l5 DECIMAL(5,2),
    ppg_l10 DECIMAL(5,2),
    rpg_l5 DECIMAL(5,2),
    apg_l5 DECIMAL(5,2),
    minutes_trend VARCHAR(20),  -- 'increasing', 'stable', 'decreasing'
    usage_rate_l5 DECIMAL(5,2),
    hot_streak BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, as_of_date)
);

CREATE INDEX idx_form_date ON player_form(as_of_date);
CREATE INDEX idx_form_player ON player_form(player_id);

COMMENT ON TABLE player_form IS 'NBA player recent form and trend analysis';

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlb_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlb_user;
