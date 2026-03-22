-- Sport-Suite Consolidated Database Initialization
-- Single PostgreSQL 16 + TimescaleDB instance
-- Replaces 6 separate containers with schema-based isolation

-- Extensions (database-wide)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS unaccent;

SET timezone = 'America/New_York';

DO $$
BEGIN
    RAISE NOTICE 'Sport-Suite database initialized — extensions enabled';
END $$;
