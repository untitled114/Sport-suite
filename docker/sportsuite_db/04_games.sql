-- Games schema (was nba_games, port 5537)
SET search_path TO games, public;

-- ==========================================================================
-- GAMES
-- ==========================================================================
CREATE TABLE games (
    game_id VARCHAR(50) PRIMARY KEY,
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
    game_status VARCHAR(20),
    blowout_flag BOOLEAN GENERATED ALWAYS AS (
        CASE
            WHEN home_score IS NOT NULL AND away_score IS NOT NULL
            THEN ABS(home_score - away_score) > 20
            ELSE FALSE
        END
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_season ON games(season);
CREATE INDEX idx_games_home ON games(home_team);
CREATE INDEX idx_games_away ON games(away_team);
CREATE INDEX idx_games_blowout ON games(blowout_flag) WHERE blowout_flag = TRUE;

-- ==========================================================================
-- TEAM GAME LOGS (TimescaleDB hypertable)
-- ==========================================================================
CREATE TABLE team_game_logs (
    game_log_id SERIAL,
    team_abbrev VARCHAR(10) NOT NULL,
    game_id VARCHAR(50) NOT NULL,
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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX idx_tgl_date ON team_game_logs(game_date);
CREATE INDEX idx_tgl_team ON team_game_logs(team_abbrev);
CREATE INDEX idx_tgl_season ON team_game_logs(season);

SELECT create_hypertable('team_game_logs', 'game_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);
