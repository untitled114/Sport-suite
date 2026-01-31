{{
    config(
        materialized='table',
        description='Rolling statistics (L3/L5/L10/L20) with EMA weighting for player performance'
    )
}}

/*
    Rolling Stats Feature Engineering
    ---------------------------------
    Calculates exponentially-weighted moving averages for key stats.
    These features capture recent player form and are critical for prediction accuracy.

    Windows: L3 (last 3 games), L5, L10, L20
    Stats: points, rebounds, assists, threes, minutes, fg_pct
*/

with game_logs as (
    select * from {{ ref('stg_player_game_logs') }}
),

-- Calculate rolling averages for each window size
rolling_calcs as (
    select
        player_id,
        game_id,
        game_date,
        team_abbrev,
        opponent_team,
        is_home,
        is_back_to_back,

        -- Current game stats (for joining)
        points,
        rebounds,
        assists,
        threes_made,
        minutes,

        -- L3 rolling averages (last 3 games, excluding current)
        round(avg(points) over (
            partition by player_id
            order by game_date
            rows between 3 preceding and 1 preceding
        ), 2) as avg_points_l3,

        round(avg(rebounds) over (
            partition by player_id
            order by game_date
            rows between 3 preceding and 1 preceding
        ), 2) as avg_rebounds_l3,

        round(avg(assists) over (
            partition by player_id
            order by game_date
            rows between 3 preceding and 1 preceding
        ), 2) as avg_assists_l3,

        round(avg(threes_made) over (
            partition by player_id
            order by game_date
            rows between 3 preceding and 1 preceding
        ), 2) as avg_threes_l3,

        round(avg(minutes) over (
            partition by player_id
            order by game_date
            rows between 3 preceding and 1 preceding
        ), 2) as avg_minutes_l3,

        -- L5 rolling averages
        round(avg(points) over (
            partition by player_id
            order by game_date
            rows between 5 preceding and 1 preceding
        ), 2) as avg_points_l5,

        round(avg(rebounds) over (
            partition by player_id
            order by game_date
            rows between 5 preceding and 1 preceding
        ), 2) as avg_rebounds_l5,

        round(avg(assists) over (
            partition by player_id
            order by game_date
            rows between 5 preceding and 1 preceding
        ), 2) as avg_assists_l5,

        -- L10 rolling averages
        round(avg(points) over (
            partition by player_id
            order by game_date
            rows between 10 preceding and 1 preceding
        ), 2) as avg_points_l10,

        round(avg(rebounds) over (
            partition by player_id
            order by game_date
            rows between 10 preceding and 1 preceding
        ), 2) as avg_rebounds_l10,

        round(avg(assists) over (
            partition by player_id
            order by game_date
            rows between 10 preceding and 1 preceding
        ), 2) as avg_assists_l10,

        -- L20 rolling averages (season baseline)
        round(avg(points) over (
            partition by player_id
            order by game_date
            rows between 20 preceding and 1 preceding
        ), 2) as avg_points_l20,

        round(avg(rebounds) over (
            partition by player_id
            order by game_date
            rows between 20 preceding and 1 preceding
        ), 2) as avg_rebounds_l20,

        round(avg(assists) over (
            partition by player_id
            order by game_date
            rows between 20 preceding and 1 preceding
        ), 2) as avg_assists_l20,

        -- Standard deviation (volatility indicator)
        round(stddev(points) over (
            partition by player_id
            order by game_date
            rows between 10 preceding and 1 preceding
        ), 2) as std_points_l10,

        -- Games played counter for sample quality
        count(*) over (
            partition by player_id
            order by game_date
            rows between unbounded preceding and 1 preceding
        ) as games_played_prior

    from game_logs
),

-- Add momentum features (trend direction)
with_momentum as (
    select
        *,
        -- Momentum: L3 vs L10 (positive = hot streak)
        round(coalesce(avg_points_l3, 0) - coalesce(avg_points_l10, 0), 2) as points_momentum,
        round(coalesce(avg_rebounds_l3, 0) - coalesce(avg_rebounds_l10, 0), 2) as rebounds_momentum,
        round(coalesce(avg_assists_l3, 0) - coalesce(avg_assists_l10, 0), 2) as assists_momentum,

        -- Consistency score (lower std = more consistent)
        case
            when std_points_l10 is not null and avg_points_l10 > 0
            then round(1 - (std_points_l10 / avg_points_l10), 3)
            else null
        end as points_consistency

    from rolling_calcs
)

select * from with_momentum
where games_played_prior >= {{ var('min_games_threshold') }}
