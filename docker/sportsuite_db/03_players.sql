-- Players schema (was nba_players, port 5536)
SET search_path TO players, public;

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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_players_position ON player_profile(position);
CREATE INDEX idx_players_team ON player_profile(team_abbrev);
CREATE INDEX idx_players_name ON player_profile(full_name);

-- ==========================================================================
-- PLAYER SEASON STATS
-- ==========================================================================
CREATE TABLE player_season_stats (
    player_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    games_played INTEGER,
    minutes_per_game DECIMAL(5,2),
    ppg DECIMAL(5,2),
    rpg DECIMAL(5,2),
    apg DECIMAL(5,2),
    spg DECIMAL(5,2),
    bpg DECIMAL(5,2),
    tpg DECIMAL(5,2),
    fg_pct DECIMAL(5,3),
    three_pt_pct DECIMAL(5,3),
    ft_pct DECIMAL(5,3),
    usage_rate DECIMAL(5,2),
    true_shooting_pct DECIMAL(5,3),
    per DECIMAL(5,2),
    ppg_per100 DECIMAL(5,2),
    rpg_per100 DECIMAL(5,2),
    apg_per100 DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_players_season ON player_season_stats(season);

-- ==========================================================================
-- PLAYER GAME LOGS (TimescaleDB hypertable)
-- ==========================================================================
CREATE TABLE player_game_logs (
    game_log_id SERIAL,
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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_pgl_game_date ON player_game_logs(game_date);
CREATE INDEX idx_pgl_player_id ON player_game_logs(player_id);
CREATE INDEX idx_pgl_game_id ON player_game_logs(game_id);
CREATE INDEX idx_pgl_season ON player_game_logs(season);

SELECT create_hypertable('player_game_logs', 'game_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ==========================================================================
-- PLAYER ROLLING STATS
-- ==========================================================================
CREATE TABLE player_rolling_stats (
    player_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    window_size INTEGER NOT NULL,
    ema_points DECIMAL(5,2),
    ema_rebounds DECIMAL(5,2),
    ema_assists DECIMAL(5,2),
    ema_minutes DECIMAL(5,2),
    ema_fg_pct DECIMAL(5,3),
    ema_three_pt_pct DECIMAL(5,3),
    ema_usage_rate DECIMAL(5,2),
    games_in_window INTEGER,
    is_hot_streak BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, as_of_date, window_size),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_rolling_date ON player_rolling_stats(as_of_date);
CREATE INDEX idx_rolling_player ON player_rolling_stats(player_id);

-- ==========================================================================
-- PLAYER MINUTES PROJECTIONS
-- ==========================================================================
CREATE TABLE player_minutes_projections (
    player_id INTEGER PRIMARY KEY,
    projected_mpg DECIMAL(5,2),
    confidence DECIMAL(3,2),
    method VARCHAR(50),
    prev_season_mpg DECIMAL(5,2),
    curr_season_mpg DECIMAL(5,2),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);
