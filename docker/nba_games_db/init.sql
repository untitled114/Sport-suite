-- NBA Games Database Initialization
-- Created: October 22, 2025
-- Purpose: Enable TimescaleDB extension for time-series game data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Set timezone
SET timezone = 'UTC';

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'NBA Games Database initialized successfully';
    RAISE NOTICE 'TimescaleDB extension enabled';
END $$;
