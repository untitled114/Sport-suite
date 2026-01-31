{{
    config(
        materialized='table',
        description='Final feature vectors for ML model training and inference'
    )
}}

/*
    Feature Vector Assembly
    -----------------------
    Joins rolling stats + book spreads into the final feature set.
    This table feeds directly into model training and live predictions.

    Output: One row per player-game-stat_type with all 102 features.
*/

with props as (
    select * from {{ ref('stg_props') }}
),

rolling_stats as (
    select * from {{ ref('int_rolling_stats') }}
),

book_spreads as (
    select * from {{ ref('int_book_spreads') }}
),

-- Join all feature sources
feature_vectors as (
    select
        -- Identifiers
        p.prop_id,
        p.player_name,
        p.player_id,
        p.game_date,
        p.stat_type,

        -- Target (for training)
        p.actual_value,
        p.hit_over,

        -- Game context features
        p.is_home,
        r.is_back_to_back,
        r.games_played_prior,

        -- Line features
        p.line,
        b.avg_line as consensus_line,
        b.line_spread,
        b.spread_pct,
        b.num_books,
        b.std_line as line_std,

        -- Per-book deviations
        b.dk_deviation,
        b.fd_deviation,
        b.mgm_deviation,
        b.underdog_deviation,

        -- Book metadata
        b.softest_book,
        b.sharpest_book,
        b.books_agree,

        -- Rolling stats (stat-specific selection)
        case p.stat_type
            when 'POINTS' then r.avg_points_l3
            when 'REBOUNDS' then r.avg_rebounds_l3
            when 'ASSISTS' then r.avg_assists_l3
            else null
        end as avg_stat_l3,

        case p.stat_type
            when 'POINTS' then r.avg_points_l5
            when 'REBOUNDS' then r.avg_rebounds_l5
            when 'ASSISTS' then r.avg_assists_l5
            else null
        end as avg_stat_l5,

        case p.stat_type
            when 'POINTS' then r.avg_points_l10
            when 'REBOUNDS' then r.avg_rebounds_l10
            when 'ASSISTS' then r.avg_assists_l10
            else null
        end as avg_stat_l10,

        case p.stat_type
            when 'POINTS' then r.avg_points_l20
            when 'REBOUNDS' then r.avg_rebounds_l20
            when 'ASSISTS' then r.avg_assists_l20
            else null
        end as avg_stat_l20,

        -- Momentum features
        case p.stat_type
            when 'POINTS' then r.points_momentum
            when 'REBOUNDS' then r.rebounds_momentum
            when 'ASSISTS' then r.assists_momentum
            else null
        end as stat_momentum,

        -- Consistency
        r.points_consistency,

        -- Minutes baseline (important for all stats)
        r.avg_minutes_l5,
        r.avg_minutes_l10,

        -- Derived: line vs recent average
        case p.stat_type
            when 'POINTS' then round(p.line - coalesce(r.avg_points_l10, 0), 2)
            when 'REBOUNDS' then round(p.line - coalesce(r.avg_rebounds_l10, 0), 2)
            when 'ASSISTS' then round(p.line - coalesce(r.avg_assists_l10, 0), 2)
            else null
        end as line_vs_avg_l10,

        -- Metadata
        current_timestamp as _created_at

    from props p
    left join rolling_stats r
        on p.player_id = r.player_id
        and p.game_date = r.game_date
    left join book_spreads b
        on p.player_name = b.player_name
        and p.game_date = b.game_date
        and p.stat_type = b.stat_type
)

select * from feature_vectors
where avg_stat_l5 is not null  -- Require minimum game history
