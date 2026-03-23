-- Intelligence schema (was nba_intelligence, port 5539)
SET search_path TO intelligence, public;

-- ==========================================================================
-- INJURY REPORT
-- ==========================================================================
CREATE TABLE injury_report (
    injury_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    report_date DATE NOT NULL,
    status VARCHAR(50),
    injury_type VARCHAR(100),
    expected_return DATE,
    confidence DECIMAL(3,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_injury_player ON injury_report(player_id);
CREATE INDEX idx_injury_date ON injury_report(report_date);
CREATE INDEX idx_injury_status ON injury_report(status);

-- ==========================================================================
-- LINEUP INTELLIGENCE
-- ==========================================================================
CREATE TABLE lineup_intel (
    lineup_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    team_abbrev VARCHAR(10) NOT NULL,
    starting_lineup TEXT[],
    confirmed BOOLEAN DEFAULT FALSE,
    minutes_projection JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lineup_game ON lineup_intel(game_id);
CREATE INDEX idx_lineup_team ON lineup_intel(team_abbrev);

-- ==========================================================================
-- LEGACY PROP LINES (PrizePicks/Underdog)
-- ==========================================================================
CREATE TABLE nba_prop_lines (
    prop_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    game_id VARCHAR(50),
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    line DECIMAL(5,2) NOT NULL,
    line_type VARCHAR(20),
    source VARCHAR(50) NOT NULL,
    odds_over DECIMAL(6,2),
    odds_under DECIMAL(6,2),
    ingested_at_utc TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_prop_player ON nba_prop_lines(player_id);
CREATE INDEX idx_prop_date ON nba_prop_lines(game_date);
CREATE INDEX idx_prop_stat ON nba_prop_lines(stat_type);
CREATE INDEX idx_prop_source ON nba_prop_lines(source);

-- ==========================================================================
-- NBA PROPS XL (Multi-Book Production Props)
-- ==========================================================================
CREATE TABLE nba_props_xl (
    id SERIAL PRIMARY KEY,
    prop_uuid UUID,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    book_name VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    game_time TIME,
    over_line DECIMAL(5,2),
    over_odds INTEGER DEFAULT -110,
    under_line DECIMAL(5,2),
    under_odds INTEGER DEFAULT -110,
    game_id VARCHAR(50),
    player_team VARCHAR(50),
    opponent_team VARCHAR(50),
    is_home BOOLEAN,
    consensus_line DECIMAL(5,2),
    min_over_line DECIMAL(5,2),
    max_over_line DECIMAL(5,2),
    line_spread DECIMAL(5,2),
    num_books INTEGER,
    bp_reported_line DECIMAL(5,2),
    bp_discrepancy DECIMAL(5,2),
    actual_value DECIMAL(5,2),
    softest_book VARCHAR(100),
    hardest_book VARCHAR(100),
    softest_vs_consensus DECIMAL(5,2),
    hardest_vs_consensus DECIMAL(5,2),
    source_url VARCHAR(500),
    source VARCHAR(50) DEFAULT 'bettingpros',
    fetch_source VARCHAR(50) DEFAULT 'bettingpros',
    is_active BOOLEAN DEFAULT TRUE,
    odds_type VARCHAR(50),
    trending_count INTEGER,
    board_time TIMESTAMPTZ,
    pp_updated_at TIMESTAMPTZ,
    adjusted_odds BOOLEAN,
    projection_id VARCHAR(100),
    is_promo BOOLEAN,
    flash_sale_line DECIMAL(5,2),
    line_movement JSONB,
    fetch_timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (player_id, game_date, stat_type, book_name, fetch_timestamp)
);

CREATE INDEX idx_pxl_player ON nba_props_xl(player_id);
CREATE INDEX idx_pxl_date ON nba_props_xl(game_date);
CREATE INDEX idx_pxl_stat ON nba_props_xl(stat_type);
CREATE INDEX idx_pxl_book ON nba_props_xl(book_name);
CREATE INDEX idx_pxl_active ON nba_props_xl(is_active);
CREATE INDEX idx_pxl_source ON nba_props_xl(fetch_source);

-- ==========================================================================
-- LINE SNAPSHOTS (append-only line movement history)
-- ==========================================================================
CREATE TABLE nba_line_snapshots (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    book_name VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    over_line DECIMAL(5,2),
    over_odds INTEGER,
    under_line DECIMAL(5,2),
    under_odds INTEGER,
    fetch_source VARCHAR(50) NOT NULL DEFAULT 'bettingpros',
    snapshot_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_snap_player_date ON nba_line_snapshots(player_name, game_date);
CREATE INDEX idx_snap_book_date ON nba_line_snapshots(book_name, game_date);
CREATE INDEX idx_snap_date ON nba_line_snapshots(game_date);
CREATE INDEX idx_snap_source ON nba_line_snapshots(fetch_source);
CREATE INDEX idx_snap_at ON nba_line_snapshots(snapshot_at);

-- ==========================================================================
-- PICKS PLACED (Tracking System)
-- ==========================================================================
CREATE TABLE nba_picks_placed (
    pick_id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    line DECIMAL(5,2) NOT NULL,
    pick_direction VARCHAR(10) NOT NULL,
    model_prediction DECIMAL(5,2),
    edge DECIMAL(5,2),
    confidence DECIMAL(5,3),
    bet_size DECIMAL(10,2),
    source VARCHAR(50),
    odds DECIMAL(6,2),
    actual_value DECIMAL(5,2),
    result VARCHAR(10),
    profit_loss DECIMAL(10,2),
    placed_at TIMESTAMPTZ DEFAULT NOW(),
    graded_at TIMESTAMPTZ
);

CREATE INDEX idx_picks_date ON nba_picks_placed(game_date);
CREATE INDEX idx_picks_player ON nba_picks_placed(player_id);
CREATE INDEX idx_picks_result ON nba_picks_placed(result);
CREATE INDEX idx_picks_stat ON nba_picks_placed(stat_type);

-- ==========================================================================
-- PLAYER FORM
-- ==========================================================================
CREATE TABLE player_form (
    player_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    games_l3 INTEGER,
    games_l5 INTEGER,
    games_l10 INTEGER,
    ppg_l3 DECIMAL(5,2),
    ppg_l5 DECIMAL(5,2),
    ppg_l10 DECIMAL(5,2),
    rpg_l5 DECIMAL(5,2),
    apg_l5 DECIMAL(5,2),
    minutes_trend VARCHAR(20),
    usage_rate_l5 DECIMAL(5,2),
    hot_streak BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, as_of_date)
);

CREATE INDEX idx_form_date ON player_form(as_of_date);
CREATE INDEX idx_form_player ON player_form(player_id);

-- ==========================================================================
-- MATCHUP HISTORY
-- ==========================================================================
CREATE TABLE matchup_history (
    player_name VARCHAR(255) NOT NULL,
    opponent_team VARCHAR(10) NOT NULL,
    stat_type VARCHAR(20) NOT NULL,
    games_played INT NOT NULL DEFAULT 0,
    avg_points DECIMAL(5,2),
    avg_rebounds DECIMAL(5,2),
    avg_assists DECIMAL(5,2),
    avg_threes DECIMAL(5,2),
    std_points DECIMAL(5,2),
    std_rebounds DECIMAL(5,2),
    std_assists DECIMAL(5,2),
    std_threes DECIMAL(5,2),
    L3_points DECIMAL(5,2), L5_points DECIMAL(5,2), L10_points DECIMAL(5,2), L20_points DECIMAL(5,2),
    L3_rebounds DECIMAL(5,2), L5_rebounds DECIMAL(5,2), L10_rebounds DECIMAL(5,2), L20_rebounds DECIMAL(5,2),
    L3_assists DECIMAL(5,2), L5_assists DECIMAL(5,2), L10_assists DECIMAL(5,2), L20_assists DECIMAL(5,2),
    L3_threes DECIMAL(5,2), L5_threes DECIMAL(5,2), L10_threes DECIMAL(5,2), L20_threes DECIMAL(5,2),
    home_avg_points DECIMAL(5,2), away_avg_points DECIMAL(5,2), home_away_split_points DECIMAL(5,2),
    home_avg_rebounds DECIMAL(5,2), away_avg_rebounds DECIMAL(5,2), home_away_split_rebounds DECIMAL(5,2),
    home_avg_assists DECIMAL(5,2), away_avg_assists DECIMAL(5,2), home_away_split_assists DECIMAL(5,2),
    home_avg_threes DECIMAL(5,2), away_avg_threes DECIMAL(5,2), home_away_split_threes DECIMAL(5,2),
    last_matchup_date DATE,
    days_since_last INT,
    sample_quality DECIMAL(3,2),
    recency_weight DECIMAL(3,2),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_name, opponent_team, stat_type)
);

CREATE INDEX idx_matchup_player ON matchup_history(player_name);
CREATE INDEX idx_matchup_opponent ON matchup_history(opponent_team);

-- ==========================================================================
-- CHEATSHEET DATA (BettingPros projections)
-- ==========================================================================
CREATE TABLE cheatsheet_data (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    game_date DATE NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    projection_diff DECIMAL(5,2),
    bet_rating INTEGER,
    ev_pct DECIMAL(10,4),
    probability DECIMAL(5,4),
    opp_rank INTEGER,
    hit_rate_l5 DECIMAL(4,3),
    hit_rate_l15 DECIMAL(4,3),
    hit_rate_season DECIMAL(4,3),
    fetch_timestamp TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (player_name, game_date, stat_type)
);

CREATE INDEX idx_cs_player_date ON cheatsheet_data(player_name, game_date);
CREATE INDEX idx_cs_date ON cheatsheet_data(game_date);

-- ==========================================================================
-- BP HISTORICAL ANALYTICS (per-prop analytics for V4 training)
-- ==========================================================================
CREATE TABLE bp_historical_analytics (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    bp_projection DECIMAL(5,2),
    bp_projection_diff DECIMAL(5,2),
    bp_probability DECIMAL(5,4),
    bp_expected_value DECIMAL(6,4),
    bp_bet_rating INTEGER,
    bp_recommended_side VARCHAR(10),
    bp_opposition_rank INTEGER,
    bp_opposition_value DECIMAL(6,2),
    bp_hit_rate_L1 DECIMAL(4,3),
    bp_hit_rate_L5 DECIMAL(4,3),
    bp_hit_rate_L10 DECIMAL(4,3),
    bp_hit_rate_L15 DECIMAL(4,3),
    bp_hit_rate_L20 DECIMAL(4,3),
    bp_hit_rate_season DECIMAL(4,3),
    bp_over_line DECIMAL(5,2),
    bp_consensus_line DECIMAL(5,2),
    bp_over_odds INTEGER,
    bp_consensus_odds INTEGER,
    bp_actual_value DECIMAL(5,2),
    bp_is_scored BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (player_name, game_date, stat_type)
);

CREATE INDEX idx_bpha_date ON bp_historical_analytics(game_date);
CREATE INDEX idx_bpha_player ON bp_historical_analytics(player_name);
CREATE INDEX idx_bpha_player_date ON bp_historical_analytics(player_name, game_date);

-- ==========================================================================
-- BP DVP HISTORICAL (defense vs position per season)
-- ==========================================================================
CREATE TABLE bp_dvp_historical (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    team VARCHAR(10) NOT NULL,
    position VARCHAR(5) NOT NULL,
    stat_name VARCHAR(30) NOT NULL,
    value DECIMAL(6,2),
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, team, position, stat_name)
);

CREATE INDEX idx_dvp_season ON bp_dvp_historical(season);
CREATE INDEX idx_dvp_team ON bp_dvp_historical(team);
CREATE INDEX idx_dvp_team_season ON bp_dvp_historical(team, season);

-- ==========================================================================
-- BOOK HISTORICAL ACCURACY
-- ==========================================================================
CREATE TABLE book_historical_accuracy (
    id SERIAL PRIMARY KEY,
    book_name VARCHAR(50) NOT NULL,
    market VARCHAR(20) NOT NULL,
    player_id INTEGER,
    player_name VARCHAR(100),
    metric_window VARCHAR(20) NOT NULL,
    as_of_date DATE NOT NULL,
    season VARCHAR(10),
    total_props INTEGER NOT NULL DEFAULT 0,
    total_games INTEGER,
    hit_rate NUMERIC(5,4),
    avg_residual NUMERIC(8,4),
    median_residual NUMERIC(8,4),
    line_bias NUMERIC(8,4),
    avg_absolute_error NUMERIC(8,4),
    rmse NUMERIC(8,4),
    volatility NUMERIC(8,4),
    percentile_25 NUMERIC(8,4),
    percentile_75 NUMERIC(8,4),
    edge_vs_consensus NUMERIC(8,4),
    soft_line_rate NUMERIC(5,4),
    hard_line_rate NUMERIC(5,4),
    avg_line_spread NUMERIC(8,4),
    max_line_spread NUMERIC(8,4),
    over_rate_vs_consensus NUMERIC(8,4),
    sharpe_ratio NUMERIC(8,4),
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (market IN ('POINTS','REBOUNDS','ASSISTS','THREES')),
    CHECK (metric_window IN ('L30','season','all_time')),
    CHECK (total_props >= 0)
);

CREATE UNIQUE INDEX idx_bha_unique
    ON book_historical_accuracy (book_name, market, COALESCE(player_id, -1), metric_window, as_of_date);
CREATE INDEX idx_bha_lookup
    ON book_historical_accuracy (book_name, market, metric_window, as_of_date DESC);

-- ==========================================================================
-- PROP PERFORMANCE HISTORY
-- ==========================================================================
CREATE TABLE prop_performance_history (
    player_name VARCHAR(100) NOT NULL,
    stat_type VARCHAR(20) NOT NULL,
    line_center NUMERIC(4,1) NOT NULL,
    season INTEGER NOT NULL,
    total_props INTEGER NOT NULL,
    props_l20 INTEGER NOT NULL,
    props_l10 INTEGER NOT NULL,
    hit_rate_all NUMERIC(5,4),
    hit_rate_l20 NUMERIC(5,4),
    hit_rate_l10 NUMERIC(5,4),
    hit_rate_home NUMERIC(5,4),
    hit_rate_away NUMERIC(5,4),
    hit_rate_vs_top10_def NUMERIC(5,4),
    hit_rate_vs_bottom10_def NUMERIC(5,4),
    hit_rate_rested NUMERIC(5,4),
    hit_rate_b2b NUMERIC(5,4),
    n_home INTEGER,
    n_away INTEGER,
    n_vs_top10_def INTEGER,
    n_vs_bottom10_def INTEGER,
    n_rested INTEGER,
    n_b2b INTEGER,
    line_vs_season_avg NUMERIC(5,2),
    line_percentile NUMERIC(5,4),
    days_since_last_prop INTEGER,
    days_since_last_hit INTEGER,
    consecutive_overs INTEGER,
    max_streak_overs INTEGER,
    max_streak_unders INTEGER,
    sample_quality_score NUMERIC(5,4),
    bayesian_prior_weight NUMERIC(5,4),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_name, stat_type, line_center, season)
);

CREATE INDEX idx_pph_lookup ON prop_performance_history(player_name, stat_type, line_center);
CREATE INDEX idx_pph_season ON prop_performance_history(season, player_name, stat_type);
CREATE INDEX idx_pph_sample ON prop_performance_history(total_props DESC);
