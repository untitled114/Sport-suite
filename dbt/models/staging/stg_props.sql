{{
    config(
        materialized='view',
        description='Staged prop lines with normalized book names and line values'
    )
}}

with source as (
    select * from {{ source('nba_intelligence', 'nba_props_xl') }}
),

cleaned as (
    select
        -- Identifiers
        prop_id,
        player_name,
        player_id,
        game_date,
        stat_type,

        -- Game context
        team as player_team,
        opponent_team,
        is_home,

        -- Line data
        line,
        book_id,
        {{ map_book_name('book_id') }} as book_name,

        -- Consensus metrics (pre-computed)
        consensus_line,
        line_spread,
        num_books,

        -- Outcome (for training/validation)
        actual_value,
        case
            when actual_value is not null and actual_value > line then 1
            when actual_value is not null and actual_value <= line then 0
            else null
        end as hit_over,

        -- Timestamps
        created_at,
        current_timestamp as _loaded_at

    from source
    where line is not null
      and stat_type in ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')
)

select * from cleaned
