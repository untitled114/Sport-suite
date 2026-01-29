-- NBA Players Database Schema
-- Created: October 22, 2025
-- Port: 5536
-- Purpose: Player profiles, season stats, game logs, rolling stats

-- ==========================================================================
-- PLAYER PROFILE
-- ==========================================================================
CREATE TABLE player_profile (
    player_id INTEGER PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    position VARCHAR(10),
    height_inches INTEGER,
    weight_lbs INTEGER,
    draft_year INTEGER,
    team_abbrev VARCHAR(10),
    birth_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_position ON player_profile(position);
CREATE INDEX idx_team ON player_profile(team_abbrev);
CREATE INDEX idx_name ON player_profile(full_name);

COMMENT ON TABLE player_profile IS 'NBA player static information and current team';

-- ==========================================================================
-- PLAYER SEASON STATS
-- ==========================================================================
CREATE TABLE player_season_stats (
    player_id INTEGER,
    season INTEGER,
    games_played INTEGER,
    minutes_per_game DECIMAL(5,2),
    ppg DECIMAL(5,2),
    rpg DECIMAL(5,2),
    apg DECIMAL(5,2),
    spg DECIMAL(5,2),  -- steals per game
    bpg DECIMAL(5,2),  -- blocks per game
    tpg DECIMAL(5,2),  -- turnovers per game
    fg_pct DECIMAL(5,3),
    three_pt_pct DECIMAL(5,3),
    ft_pct DECIMAL(5,3),
    usage_rate DECIMAL(5,2),
    true_shooting_pct DECIMAL(5,3),
    per DECIMAL(5,2),  -- player efficiency rating
    -- Per-100 possessions (pace-adjusted)
    ppg_per100 DECIMAL(5,2),
    rpg_per100 DECIMAL(5,2),
    apg_per100 DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_season ON player_season_stats(season);

COMMENT ON TABLE player_season_stats IS 'NBA player season aggregates with pace-adjusted metrics';

-- ==========================================================================
-- PLAYER GAME LOGS
-- ==========================================================================
CREATE TABLE player_game_logs (
    game_log_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    game_id VARCHAR(20) NOT NULL,
    game_date DATE NOT NULL,
    season INTEGER NOT NULL,
    team_abbrev VARCHAR(10),
    opponent_abbrev VARCHAR(10),
    is_home BOOLEAN,
    minutes_played INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    three_pointers_made INTEGER,
    fg_made INTEGER,
    fg_attempted INTEGER,
    three_pt_attempted INTEGER,
    ft_made INTEGER,
    ft_attempted INTEGER,
    plus_minus INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_game_date ON player_game_logs(game_date);
CREATE INDEX idx_player_id_game ON player_game_logs(player_id);
CREATE INDEX idx_game_id ON player_game_logs(game_id);
CREATE INDEX idx_season_logs ON player_game_logs(season);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('player_game_logs', 'game_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

COMMENT ON TABLE player_game_logs IS 'NBA player game-by-game box scores (TimescaleDB hypertable)';

-- ==========================================================================
-- PLAYER ROLLING STATS
-- ==========================================================================
CREATE TABLE player_rolling_stats (
    player_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    window_size INTEGER NOT NULL,  -- 3, 5, 10, 20 games
    -- EMA-based rolling averages
    ema_points DECIMAL(5,2),
    ema_rebounds DECIMAL(5,2),
    ema_assists DECIMAL(5,2),
    ema_minutes DECIMAL(5,2),
    ema_fg_pct DECIMAL(5,3),
    ema_three_pt_pct DECIMAL(5,3),
    ema_usage_rate DECIMAL(5,2),
    games_in_window INTEGER,
    is_hot_streak BOOLEAN,  -- Points > average + 1 std dev
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, as_of_date, window_size),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_rolling_date ON player_rolling_stats(as_of_date);
CREATE INDEX idx_rolling_player ON player_rolling_stats(player_id);

COMMENT ON TABLE player_rolling_stats IS 'NBA player rolling statistics with EMA smoothing';

-- ==========================================================================
-- GRANTS
-- ==========================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlb_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlb_user;
