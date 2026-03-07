-- Cephalon Axiom Schema
-- Every table here is designed to be readable by Atlas and Claude Code on escalation.
-- The goal: understand exactly what happened without SSHing into the server.

SET timezone = 'UTC';

-- ─────────────────────────────────────────────────────────────────
-- axiom_pipeline_audit
-- One row per pipeline run. First thing to check on any escalation.
-- Did data arrive? How many props? Any errors? How long did it take?
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS axiom_pipeline_audit (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    run_number          INTEGER NOT NULL,         -- 1=2AM, 2=5AM, 3=8AM, 4=11AM, 5=2PM, 6=5PM
    run_type            TEXT NOT NULL,            -- 'full' (run 1) or 'refresh' (runs 2-6)
    run_timestamp       TIMESTAMPTZ NOT NULL,
    status              TEXT NOT NULL,            -- 'success', 'partial', 'failed'

    -- Data freshness
    props_fetched       INTEGER,                  -- total props from BettingPros
    books_available     INTEGER,                  -- how many sportsbooks returned data
    injuries_updated    BOOLEAN,                  -- did injury report refresh succeed?
    games_found         INTEGER,                  -- how many games scheduled today

    -- Timing
    duration_seconds    INTEGER,

    -- Prediction output
    picks_generated     INTEGER,                  -- total picks this run
    xl_picks            INTEGER,
    v3_picks            INTEGER,

    -- Error capture
    error_message       TEXT,
    error_traceback     TEXT,

    -- Anomalies flagged during this run
    -- e.g. {"warning": "BetMGM returned 0 props", "stale_books": ["ESPNBet"]}
    anomalies           JSONB,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_date, run_number)
);

-- ─────────────────────────────────────────────────────────────────
-- nba_prediction_history
-- One row per (run, player, stat_type, model_version).
-- Conviction engine derives everything from this table.
-- Outcomes are backfilled post-game by the validation pipeline.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nba_prediction_history (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    run_number          INTEGER NOT NULL,
    run_timestamp       TIMESTAMPTZ NOT NULL,
    player_name         TEXT NOT NULL,
    stat_type           TEXT NOT NULL,            -- POINTS, REBOUNDS
    model_version       TEXT NOT NULL,            -- xl, v3
    tier                TEXT,
    line                NUMERIC(5,1),
    p_over              NUMERIC(5,4),
    edge                NUMERIC(6,3),
    spread              NUMERIC(5,1),
    book                TEXT,
    game_time           TIMESTAMPTZ,
    opponent_team       TEXT,
    is_home             BOOLEAN,

    -- Context at prediction time: injuries active, rest days, back-to-back, etc.
    -- e.g. {"injured_teammates": ["Simmons"], "b2b": true, "opponent_def_rank": 4}
    context_snapshot    JSONB,

    -- Outcome — backfilled post-game
    actual_result       NUMERIC(5,1),
    is_hit              BOOLEAN,
    result_source       TEXT,                     -- 'espn', 'nba_api', 'manual'
    result_recorded_at  TIMESTAMPTZ,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────
-- axiom_conviction
-- Recomputed after every pipeline run.
-- Captures the full narrative of each pick's day.
-- This is what Axiom reads when composing the daily card,
-- and what Claude/Atlas check first on escalation.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS axiom_conviction (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    player_name         TEXT NOT NULL,
    stat_type           TEXT NOT NULL,

    -- Conviction
    conviction          NUMERIC(5,3) NOT NULL,    -- 0.0 - 1.0
    conviction_label    TEXT NOT NULL,            -- LOCKED, STRONG, WATCH, SKIP

    -- Appearance tracking
    appearances         INTEGER NOT NULL,         -- how many runs included this pick
    total_runs          INTEGER NOT NULL,         -- how many runs have run today so far
    entry_run           INTEGER NOT NULL,         -- first run it appeared (1-6)
    last_seen_run       INTEGER NOT NULL,         -- most recent run it appeared
    is_active           BOOLEAN NOT NULL,         -- still in the latest run?
    run_pattern         TEXT NOT NULL,            -- e.g. "1,2,3,4" or "1,3,5" (scattered)

    -- Line movement
    line_at_entry       NUMERIC(5,1),             -- line when first seen
    line_latest         NUMERIC(5,1),             -- line in most recent run
    line_movement       NUMERIC(5,1),             -- line_latest - line_at_entry (+ rising, - softening)
    line_direction      TEXT,                     -- 'rising', 'falling', 'stable'
    book_latest         TEXT,

    -- Signal strength
    p_over_at_entry     NUMERIC(5,4),
    p_over_latest       NUMERIC(5,4),
    p_over_trend        NUMERIC(5,4),             -- p_over_latest - p_over_at_entry
    p_over_std          NUMERIC(5,4),             -- std dev across runs (lower = stable)

    -- Pick lifecycle status
    -- active:     still in latest run
    -- dropped:    appeared then vanished (injury, news, stat correction)
    -- evaporated: was strong but line moved unfavorably — edge gone
    status              TEXT NOT NULL,

    -- Context when conviction was last computed
    -- e.g. {"active_injuries": ["Simmons OUT"], "game_time": "7:30PM ET", "b2b": false}
    context_snapshot    JSONB,

    -- Plain English summary of this pick's day.
    -- Written by the conviction engine. Read by Atlas and Claude on escalation.
    -- e.g. "Present all 6 runs, line held at 26.5 @ BetRivers. Signal consistent."
    --      "Strong runs 1-3, line jumped 24.5→27.5 by run 4. Sharp money — value gone."
    --      "Appeared run 4 after Simmons injury report. 3/3 recent runs. Fresh but contextual."
    narrative           TEXT,

    computed_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_date, player_name, stat_type)
);

-- ─────────────────────────────────────────────────────────────────
-- axiom_posts
-- Every message Axiom has sent. Prevents duplicates.
-- Atlas reads this to know what was communicated and when.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS axiom_posts (
    id              SERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    post_type       TEXT NOT NULL,            -- 'daily_card', 'alert', 'recap'
    trigger         TEXT,                     -- 'scheduled', 'manual', 'alert'
    picks_sent      JSONB,
    channel_id      TEXT,
    message_id      TEXT,                     -- Discord message ID
    sent_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────
-- axiom_memory
-- Persistent key-value store for Axiom's AI context.
-- Survives bot restarts. Atlas reads this for fleet context.
-- Keys: "history:{user_id}", "bankroll", "recent_win_rate", etc.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS axiom_memory (
    key             TEXT PRIMARY KEY,
    value           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pipeline_audit_date       ON axiom_pipeline_audit (run_date, run_number);
CREATE INDEX IF NOT EXISTS idx_pipeline_audit_status     ON axiom_pipeline_audit (run_date, status);
CREATE INDEX IF NOT EXISTS idx_prediction_history_date   ON nba_prediction_history (run_date, run_number);
CREATE INDEX IF NOT EXISTS idx_prediction_history_player ON nba_prediction_history (run_date, player_name, stat_type);
CREATE INDEX IF NOT EXISTS idx_prediction_outcome        ON nba_prediction_history (run_date, is_hit) WHERE is_hit IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conviction_date           ON axiom_conviction (run_date);
CREATE INDEX IF NOT EXISTS idx_conviction_status         ON axiom_conviction (run_date, status);
CREATE INDEX IF NOT EXISTS idx_conviction_label          ON axiom_conviction (run_date, conviction_label);
CREATE INDEX IF NOT EXISTS idx_axiom_posts_date          ON axiom_posts (run_date, post_type);
