-- Teams schema (was nba_team, port 5538)
SET search_path TO teams, public;

-- ==========================================================================
-- TEAMS
-- ==========================================================================
CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_abbrev VARCHAR(10) UNIQUE NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    conference VARCHAR(10),
    division VARCHAR(20),
    arena_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_teams_abbrev ON teams(team_abbrev);
CREATE INDEX idx_teams_conference ON teams(conference);

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
    def_rating_vs_pg DECIMAL(5,2),
    def_rating_vs_sg DECIMAL(5,2),
    def_rating_vs_sf DECIMAL(5,2),
    def_rating_vs_pf DECIMAL(5,2),
    def_rating_vs_c DECIMAL(5,2),
    pace_neutral_off_rating DECIMAL(5,2),
    pace_neutral_def_rating DECIMAL(5,2),
    true_shooting_pct DECIMAL(5,3),
    rebounding_pct DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (team_abbrev, season)
);

CREATE INDEX idx_tss_season ON team_season_stats(season);

-- ==========================================================================
-- TEAM ROLLING STATS
-- ==========================================================================
CREATE TABLE team_rolling_stats (
    team_abbrev VARCHAR(10) NOT NULL,
    as_of_date DATE NOT NULL,
    window_size INTEGER NOT NULL,
    games_in_window INTEGER,
    avg_points DECIMAL(5,2),
    avg_points_allowed DECIMAL(5,2),
    avg_pace DECIMAL(5,2),
    avg_offensive_rating DECIMAL(5,2),
    avg_defensive_rating DECIMAL(5,2),
    net_rating DECIMAL(5,2),
    wins_in_window INTEGER,
    losses_in_window INTEGER,
    win_pct DECIMAL(5,3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (team_abbrev, as_of_date, window_size)
);

CREATE INDEX idx_trs_date ON team_rolling_stats(as_of_date);
CREATE INDEX idx_trs_team ON team_rolling_stats(team_abbrev);

-- ==========================================================================
-- TEAM BETTING PERFORMANCE
-- ==========================================================================
CREATE TABLE team_betting_performance (
    id SERIAL PRIMARY KEY,
    season VARCHAR(10) NOT NULL,
    team_abbrev VARCHAR(10) NOT NULL,
    ats_pct DECIMAL(5,3),
    ou_pct DECIMAL(5,3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, team_abbrev)
);

CREATE INDEX idx_tbp_season ON team_betting_performance(season);
