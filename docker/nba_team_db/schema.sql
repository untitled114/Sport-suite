-- NBA Team Database Schema
-- Created: October 22, 2025
-- Port: 5538
-- Purpose: Team profiles, season stats, defensive ratings by position

-- ==========================================================================
-- TEAMS
-- ==========================================================================
CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_abbrev VARCHAR(10) UNIQUE NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    conference VARCHAR(10),  -- 'East', 'West'
    division VARCHAR(20),    -- 'Atlantic', 'Central', 'Southeast', 'Northwest', 'Pacific', 'Southwest'
    arena_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_team_abbrev ON teams(team_abbrev);
CREATE INDEX idx_conference ON teams(conference);

COMMENT ON TABLE teams IS 'NBA team profiles and conference/division info';

-- ==========================================================================
-- TEAM SEASON STATS
-- ==========================================================================
CREATE TABLE team_season_stats (
    team_abbrev VARCHAR(10) NOT NULL,
    season INTEGER NOT NULL,
    games_played INTEGER,
    wins INTEGER,
    losses INTEGER,
    pace DECIMAL(5,2),
    offensive_rating DECIMAL(5,2),
    defensive_rating DECIMAL(5,2),
    net_rating DECIMAL(5,2),
    -- Defensive ratings by position (opponent stats allowed)
    def_rating_vs_pg DECIMAL(5,2),
    def_rating_vs_sg DECIMAL(5,2),
    def_rating_vs_sf DECIMAL(5,2),
    def_rating_vs_pf DECIMAL(5,2),
    def_rating_vs_c DECIMAL(5,2),
    -- Pace-neutral ratings (per-100 possessions)
    pace_neutral_off_rating DECIMAL(5,2),
    pace_neutral_def_rating DECIMAL(5,2),
    true_shooting_pct DECIMAL(5,3),
    rebounding_pct DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_abbrev, season)
);

CREATE INDEX idx_team_season ON team_season_stats(season);

COMMENT ON TABLE team_season_stats IS 'NBA team season aggregates with pace-adjusted and position-specific defensive metrics';

-- ==========================================================================
-- TEAM ROLLING STATS
-- ==========================================================================
CREATE TABLE team_rolling_stats (
    team_abbrev VARCHAR(10) NOT NULL,
    as_of_date DATE NOT NULL,
    window_size INTEGER NOT NULL,  -- 5, 10, 20 games
    games_in_window INTEGER,
    -- Rolling averages
    avg_points DECIMAL(5,2),
    avg_points_allowed DECIMAL(5,2),
    avg_pace DECIMAL(5,2),
    avg_offensive_rating DECIMAL(5,2),
    avg_defensive_rating DECIMAL(5,2),
    net_rating DECIMAL(5,2),
    -- Win/loss record in window
    wins_in_window INTEGER,
    losses_in_window INTEGER,
    win_pct DECIMAL(5,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_abbrev, as_of_date, window_size)
);

CREATE INDEX idx_rolling_team_date ON team_rolling_stats(as_of_date);
CREATE INDEX idx_rolling_team_abbrev ON team_rolling_stats(team_abbrev);

COMMENT ON TABLE team_rolling_stats IS 'NBA team rolling statistics for recent form analysis';

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlb_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlb_user;
