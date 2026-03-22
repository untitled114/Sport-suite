-- Feature Store schema
-- Decouples feature computation from both training and inference.
-- Both pipelines read from the same versioned feature store.
SET search_path TO features, public;

-- ==========================================================================
-- FEATURE SETS — versioned definitions of which features exist
-- ==========================================================================
CREATE TABLE feature_sets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,       -- 'xl_v1', 'v3_v1', 'v4_v1'
    version INTEGER NOT NULL DEFAULT 1,
    feature_count INTEGER NOT NULL,
    feature_names TEXT[] NOT NULL,           -- ordered list of feature column names
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed the current feature sets
INSERT INTO feature_sets (name, version, feature_count, feature_names, description) VALUES
    ('xl_v1', 1, 102, ARRAY[
        'is_home','line','opponent_team',
        'ema_points_L3','ema_points_L5','ema_points_L10','ema_points_L20',
        'ema_rebounds_L3','ema_rebounds_L5','ema_rebounds_L10','ema_rebounds_L20',
        'ema_assists_L3','ema_assists_L5','ema_assists_L10','ema_assists_L20',
        'ema_threes_L3','ema_threes_L5','ema_threes_L10','ema_threes_L20',
        'ema_steals_L5','ema_blocks_L5','ema_turnovers_L5',
        'ema_minutes_L3','ema_minutes_L5','ema_minutes_L10','ema_minutes_L20',
        'ema_fg_pct_L5','ema_fg_pct_L10',
        'team_pace','team_off_rating','team_def_rating','projected_possessions',
        'rest_days','is_back_to_back','travel_distance','altitude','season_phase_encoded',
        'is_starter','bench_points_ratio','position_encoded','avg_teammate_usage',
        'injured_teammates_count',
        'h2h_avg','h2h_std','matchup_advantage_score',
        'points_per_minute_L5','player_last_game_minutes','days_since_last_30pt_game',
        'line_spread','consensus_line','line_std','num_books','coeff_of_variation',
        'dk_deviation','fd_deviation','mgm_deviation','caesars_deviation',
        'bet365_deviation','betrivers_deviation','espnbet_deviation','fanatics_deviation',
        'softest_book_id','hardest_book_id','line_spread_percentile',
        'books_agree','books_disagree',
        'softest_vs_consensus','hardest_vs_consensus','min_line','max_line',
        'expected_diff'
    ], 'XL model feature set (102 features, Dec 2025)'),
    ('v3_v1', 1, 136, ARRAY[
        -- Same 102 as XL plus 34 additional V3 features
        'is_home','line','opponent_team',
        'ema_points_L3','ema_points_L5','ema_points_L10','ema_points_L20',
        'days_into_season','season_phase_encoded','is_early_season','is_mid_season','is_late_season','is_playoffs',
        'stat_std_L5','stat_std_L10','minutes_std_L5','minutes_std_L10','fga_std_L5',
        'stat_trend_ratio','minutes_trend_ratio','usage_volatility_score',
        'h2h_decayed_avg','h2h_trend','h2h_recency_adjusted','h2h_time_decay_factor','h2h_reliability',
        'line_std','softest_book_hit_rate','softest_book_soft_rate','softest_book_line_bias','line_source_reliability',
        'line_delta','line_movement_std','consensus_strength','snapshot_count','hours_tracked',
        'efficiency_vs_context','game_velocity','season_phase','resistance_adjusted_L3','volume_proxy','momentum_short_term'
    ], 'V3 model feature set (136 features, Feb 2026)')
ON CONFLICT (name) DO NOTHING;

-- ==========================================================================
-- COMPUTED FEATURES — materialized feature vectors per player/game/date
-- ==========================================================================
CREATE TABLE computed_features (
    id BIGSERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    player_id INTEGER,
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    feature_set_name VARCHAR(50) NOT NULL REFERENCES feature_sets(name),
    feature_values JSONB NOT NULL,           -- {feature_name: value, ...}
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (player_name, game_date, stat_type, feature_set_name)
);

CREATE INDEX idx_cf_player_date ON computed_features(player_name, game_date);
CREATE INDEX idx_cf_date ON computed_features(game_date);
CREATE INDEX idx_cf_fset ON computed_features(feature_set_name);

-- ==========================================================================
-- FEATURE METADATA — tracks drift baselines and importance
-- ==========================================================================
CREATE TABLE feature_metadata (
    id SERIAL PRIMARY KEY,
    feature_set_name VARCHAR(50) NOT NULL REFERENCES feature_sets(name),
    feature_name VARCHAR(100) NOT NULL,
    -- Distribution baseline (from training data)
    train_mean DECIMAL(10,4),
    train_std DECIMAL(10,4),
    train_min DECIMAL(10,4),
    train_max DECIMAL(10,4),
    train_median DECIMAL(10,4),
    -- Importance
    shap_importance DECIMAL(8,6),
    importance_rank INTEGER,
    -- Drift monitoring
    last_drift_check TIMESTAMPTZ,
    last_psi DECIMAL(8,4),
    drift_alert BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (feature_set_name, feature_name)
);

CREATE INDEX idx_fm_fset ON feature_metadata(feature_set_name);
CREATE INDEX idx_fm_drift ON feature_metadata(drift_alert) WHERE drift_alert = TRUE;

-- ==========================================================================
-- CLV TRACKING — closing line value per pick (Claude 3 will populate this)
-- ==========================================================================
CREATE TABLE clv_tracking (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER,                   -- FK to axiom.nba_prediction_history.id
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    book_name VARCHAR(100),
    -- Lines
    opening_line DECIMAL(5,2),
    closing_line DECIMAL(5,2),
    model_line DECIMAL(5,2),                 -- our predicted value
    -- Probabilities
    opening_implied_prob DECIMAL(5,4),
    closing_implied_prob DECIMAL(5,4),
    model_prob DECIMAL(5,4),
    -- CLV metrics
    clv_cents DECIMAL(8,4),                  -- closing_implied - opening_implied (in our direction)
    beat_closing_line BOOLEAN,               -- did our model beat the close?
    -- Result
    actual_value DECIMAL(5,2),
    is_hit BOOLEAN,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (player_name, game_date, stat_type, book_name)
);

CREATE INDEX idx_clv_date ON clv_tracking(game_date);
CREATE INDEX idx_clv_player ON clv_tracking(player_name, game_date);
CREATE INDEX idx_clv_beat ON clv_tracking(beat_closing_line) WHERE beat_closing_line IS NOT NULL;

-- ==========================================================================
-- PERFORMANCE METRICS — daily/rolling aggregates (Claude 3 will populate)
-- ==========================================================================
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    period VARCHAR(20) NOT NULL,             -- 'daily', '7d', '30d', 'season'
    market VARCHAR(20),                      -- NULL = all markets
    tier VARCHAR(20),                        -- NULL = all tiers
    model_version VARCHAR(10),               -- NULL = all versions
    total_picks INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    pushes INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(5,4),
    roi DECIMAL(8,4),
    avg_edge DECIMAL(6,4),
    avg_clv DECIMAL(8,4),
    clv_positive_rate DECIMAL(5,4),          -- % of picks that beat closing line
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (metric_date, period, COALESCE(market, ''), COALESCE(tier, ''), COALESCE(model_version, ''))
);

CREATE INDEX idx_pm_date ON performance_metrics(metric_date, period);
