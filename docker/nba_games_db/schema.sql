-- NBA Games Database Schema
-- Created: October 22, 2025
-- Port: 5537
-- Purpose: Game schedules, box scores, team game logs

-- ==========================================================================
-- GAMES
-- ==========================================================================
CREATE TABLE games (
    game_id VARCHAR(20) PRIMARY KEY,
    game_date DATE NOT NULL,
    season INTEGER NOT NULL,
    home_team VARCHAR(10) NOT NULL,
    away_team VARCHAR(10) NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    total_possessions INTEGER,
    pace DECIMAL(5,2),
    vegas_total DECIMAL(5,1),
    vegas_spread DECIMAL(5,1),
    game_status VARCHAR(20),  -- 'scheduled', 'in_progress', 'final'
    blowout_flag BOOLEAN GENERATED ALWAYS AS (
        CASE
            WHEN home_score IS NOT NULL AND away_score IS NOT NULL
            THEN ABS(home_score - away_score) > 20
            ELSE FALSE
        END
    ) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_game_date ON games(game_date);
CREATE INDEX idx_season ON games(season);
CREATE INDEX idx_home_team ON games(home_team);
CREATE INDEX idx_away_team ON games(away_team);
CREATE INDEX idx_blowout ON games(blowout_flag) WHERE blowout_flag = TRUE;

COMMENT ON TABLE games IS 'NBA game schedules and basic box scores';

-- ==========================================================================
-- TEAM GAME LOGS
-- ==========================================================================
CREATE TABLE team_game_logs (
    game_log_id SERIAL PRIMARY KEY,
    team_abbrev VARCHAR(10) NOT NULL,
    game_id VARCHAR(20) NOT NULL,
    game_date DATE NOT NULL,
    season INTEGER NOT NULL,
    opponent VARCHAR(10) NOT NULL,
    is_home BOOLEAN NOT NULL,
    points INTEGER,
    possessions INTEGER,
    pace DECIMAL(5,2),
    offensive_rating DECIMAL(5,2),
    defensive_rating DECIMAL(5,2),
    fg_made INTEGER,
    fg_attempted INTEGER,
    three_pt_made INTEGER,
    three_pt_attempted INTEGER,
    ft_made INTEGER,
    ft_attempted INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    turnovers INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX idx_team_logs_date ON team_game_logs(game_date);
CREATE INDEX idx_team_logs_team ON team_game_logs(team_abbrev);
CREATE INDEX idx_team_logs_season ON team_game_logs(season);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('team_game_logs', 'game_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

COMMENT ON TABLE team_game_logs IS 'NBA team game-by-game statistics (TimescaleDB hypertable)';

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlb_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlb_user;
