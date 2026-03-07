-- Cephalon Axiom Database Initialization
-- Purpose: Axiom's own DB for pipeline auditing, conviction scoring, and persistent memory

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SET timezone = 'UTC';

DO $$
BEGIN
    RAISE NOTICE 'Cephalon Axiom database initialized';
END $$;
