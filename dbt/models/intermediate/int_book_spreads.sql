{{
    config(
        materialized='table',
        description='Book disagreement features - spread analysis across sportsbooks'
    )
}}

/*
    Book Disagreement Feature Engineering
    -------------------------------------
    When sportsbooks disagree on a line, it often signals edge opportunities.
    Large spreads between books indicate market inefficiency.

    Key features:
    - line_spread: max - min across books
    - consensus_deviation: how far each book is from consensus
    - softest_book: which book has the most favorable line
*/

with props as (
    select * from {{ ref('stg_props') }}
),

-- Pivot to get each book's line in columns
book_lines as (
    select
        player_name,
        player_id,
        game_date,
        stat_type,
        player_team,
        opponent_team,
        is_home,

        -- Individual book lines
        max(case when book_name = 'DraftKings' then line end) as dk_line,
        max(case when book_name = 'FanDuel' then line end) as fd_line,
        max(case when book_name = 'BetMGM' then line end) as mgm_line,
        max(case when book_name = 'Caesars' then line end) as caesars_line,
        max(case when book_name = 'BetRivers' then line end) as betrivers_line,
        max(case when book_name = 'ESPNBet' then line end) as espnbet_line,
        max(case when book_name = 'Underdog' then line end) as underdog_line,

        -- Aggregate metrics
        count(distinct book_id) as num_books,
        min(line) as min_line,
        max(line) as max_line,
        round(avg(line), 2) as avg_line,
        round(stddev(line), 3) as std_line

    from props
    group by 1, 2, 3, 4, 5, 6, 7
),

-- Calculate spread and deviation features
with_spreads as (
    select
        *,

        -- Core spread metrics
        round(max_line - min_line, 2) as line_spread,
        round((max_line - min_line) / nullif(avg_line, 0) * 100, 2) as spread_pct,

        -- Per-book deviations from consensus
        round(dk_line - avg_line, 2) as dk_deviation,
        round(fd_line - avg_line, 2) as fd_deviation,
        round(mgm_line - avg_line, 2) as mgm_deviation,
        round(underdog_line - avg_line, 2) as underdog_deviation,

        -- Identify softest line (highest for OVER bets)
        case
            when underdog_line = max_line then 'Underdog'
            when dk_line = max_line then 'DraftKings'
            when fd_line = max_line then 'FanDuel'
            when mgm_line = max_line then 'BetMGM'
            when caesars_line = max_line then 'Caesars'
            when betrivers_line = max_line then 'BetRivers'
            when espnbet_line = max_line then 'ESPNBet'
            else 'Unknown'
        end as softest_book,

        -- Identify sharpest line (lowest for OVER bets)
        case
            when dk_line = min_line then 'DraftKings'
            when fd_line = min_line then 'FanDuel'
            when mgm_line = min_line then 'BetMGM'
            when caesars_line = min_line then 'Caesars'
            when betrivers_line = min_line then 'BetRivers'
            when espnbet_line = min_line then 'ESPNBet'
            when underdog_line = min_line then 'Underdog'
            else 'Unknown'
        end as sharpest_book,

        -- Agreement flag (books within 0.5 of each other)
        case when (max_line - min_line) <= 0.5 then true else false end as books_agree

    from book_lines
    where num_books >= 3  -- Require at least 3 books for meaningful spread
)

select * from with_spreads
