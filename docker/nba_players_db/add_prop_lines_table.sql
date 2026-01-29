-- NBA Prop Lines staging table
-- Stores prop lines from PrizePicks, Underdog, OddsJam

CREATE TABLE IF NOT EXISTS nba_prop_lines (
    line_id SERIAL PRIMARY KEY,
    player_id INTEGER,
    player_name VARCHAR(255) NOT NULL,
    team_abbrev VARCHAR(10),
    opponent_abbrev VARCHAR(10),
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,  -- POINTS, REBOUNDS, ASSISTS, THREES, PTS_REBS_ASTS, etc.
    line_value DECIMAL(5,1) NOT NULL,  -- e.g., 24.5 points
    over_odds INTEGER,  -- e.g., -110
    under_odds INTEGER,  -- e.g., -110
    sportsbook VARCHAR(50) NOT NULL,  -- PrizePicks, Underdog, OddsJam
    fetched_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE SET NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_nba_prop_player ON nba_prop_lines(player_id);
CREATE INDEX IF NOT EXISTS idx_nba_prop_game_date ON nba_prop_lines(game_date);
CREATE INDEX IF NOT EXISTS idx_nba_prop_stat_type ON nba_prop_lines(stat_type);
CREATE INDEX IF NOT EXISTS idx_nba_prop_sportsbook ON nba_prop_lines(sportsbook);
CREATE INDEX IF NOT EXISTS idx_nba_prop_active ON nba_prop_lines(is_active);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_nba_prop_lookup ON nba_prop_lines(game_date, player_id, stat_type);

COMMENT ON TABLE nba_prop_lines IS 'Staging table for NBA prop lines from multiple sportsbooks';
COMMENT ON COLUMN nba_prop_lines.stat_type IS 'Standardized stat types: POINTS, REBOUNDS, ASSISTS, THREES, BLOCKS, STEALS, PTS_REBS_ASTS, etc.';
