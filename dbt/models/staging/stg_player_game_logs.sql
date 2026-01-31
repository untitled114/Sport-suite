{{
    config(
        materialized='view',
        description='Staged player game logs with cleaned fields and derived metrics'
    )
}}

with source as (
    select * from {{ source('nba_players', 'player_game_logs') }}
),

cleaned as (
    select
        -- Identifiers
        player_id,
        game_id,
        game_date,
        team_abbrev,
        opponent_team,

        -- Game context
        is_home,
        case
            when game_date - lag(game_date) over (
                partition by player_id order by game_date
            ) = 1 then true
            else false
        end as is_back_to_back,

        -- Core stats
        coalesce(points, 0) as points,
        coalesce(rebounds, 0) as rebounds,
        coalesce(assists, 0) as assists,
        coalesce(threes_made, 0) as threes_made,
        coalesce(steals, 0) as steals,
        coalesce(blocks, 0) as blocks,
        coalesce(turnovers, 0) as turnovers,
        coalesce(minutes, 0) as minutes,

        -- Shooting
        coalesce(fg_made, 0) as fg_made,
        coalesce(fg_attempted, 0) as fg_attempted,
        coalesce(ft_made, 0) as ft_made,
        coalesce(ft_attempted, 0) as ft_attempted,

        -- Derived metrics
        case
            when coalesce(fg_attempted, 0) > 0
            then round(fg_made::numeric / fg_attempted, 3)
            else null
        end as fg_pct,

        case
            when coalesce(minutes, 0) > 0
            then round(points::numeric / minutes, 3)
            else null
        end as points_per_minute,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where minutes > 0  -- Exclude DNPs
)

select * from cleaned
