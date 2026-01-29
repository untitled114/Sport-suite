-- Add injury tracking table to NBA players database
-- Run with: PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5536 -U mlb_user -d nba_players -f add_injury_table.sql

-- Drop table if exists (for development)
DROP TABLE IF EXISTS player_injury_status CASCADE;

-- Create injury status table
CREATE TABLE player_injury_status (
    player_id INTEGER NOT NULL,
    injury_status VARCHAR(20) NOT NULL,  -- Out, Questionable, Doubtful, Probable, GTD, Healthy
    injury_description TEXT,
    return_date DATE,
    games_missed INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data_source VARCHAR(50) DEFAULT 'ESPN',

    PRIMARY KEY (player_id),
    FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX idx_injury_status ON player_injury_status(injury_status);
CREATE INDEX idx_injury_updated ON player_injury_status(last_updated);

-- Comments
COMMENT ON TABLE player_injury_status IS 'Current injury status for all NBA players';
COMMENT ON COLUMN player_injury_status.injury_status IS 'Out, Questionable, Doubtful, Probable, GTD (Game-Time Decision), Healthy';
COMMENT ON COLUMN player_injury_status.injury_description IS 'Description of injury (e.g., Left Ankle Sprain, Rest)';
COMMENT ON COLUMN player_injury_status.return_date IS 'Expected return date (if known)';
COMMENT ON COLUMN player_injury_status.games_missed IS 'Number of consecutive games missed';
COMMENT ON COLUMN player_injury_status.data_source IS 'Source of injury data (ESPN, RotoWire, etc.)';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON player_injury_status TO mlb_user;

-- Sample data structure
-- INSERT INTO player_injury_status (player_id, injury_status, injury_description, games_missed)
-- VALUES (2544, 'Healthy', NULL, 0);

COMMIT;
