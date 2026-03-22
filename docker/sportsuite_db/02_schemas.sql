-- Create all schemas and grant access
-- Each schema replaces what was a separate database container.

CREATE SCHEMA IF NOT EXISTS players;      -- was nba_players (port 5536)
CREATE SCHEMA IF NOT EXISTS games;        -- was nba_games (port 5537)
CREATE SCHEMA IF NOT EXISTS teams;        -- was nba_team (port 5538)
CREATE SCHEMA IF NOT EXISTS intelligence; -- was nba_intelligence (port 5539)
CREATE SCHEMA IF NOT EXISTS axiom;        -- was cephalon_axiom (port 5541)
CREATE SCHEMA IF NOT EXISTS features;     -- NEW: feature store

-- Grant schema access to the application user
DO $$
DECLARE
    s TEXT;
BEGIN
    FOREACH s IN ARRAY ARRAY['players','games','teams','intelligence','axiom','features']
    LOOP
        EXECUTE format('GRANT ALL ON SCHEMA %I TO mlb_user', s);
        EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA %I GRANT ALL ON TABLES TO mlb_user', s);
        EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA %I GRANT ALL ON SEQUENCES TO mlb_user', s);
    END LOOP;
END $$;
