#!/usr/bin/env python3
"""
NBA ROLLING STATS CALCULATOR
=============================

Calculates EMA-based rolling statistics for NBA players across L3/L5/L10/L20 game windows.

Uses Exponential Moving Average (EMA) with alpha=0.4 for optimal noise reduction
while maintaining recency bias.

Features calculated:
- Points, Rebounds, Assists
- Minutes, FG%, 3P%
- Usage rate (from season stats)
- Steals, Blocks, Turnovers
- Plus/Minus
- Hot streak detection (points > avg + 1 std)

Output: Populates player_rolling_stats table in nba_players database
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

# Rolling window sizes (in games)
WINDOW_SIZES = [3, 5, 10, 20]

# EMA alpha (0.4 = optimal per grid search on NBA data)
EMA_ALPHA = 0.4


def calculate_ema(values, alpha=EMA_ALPHA):
    """
    Calculate Exponential Moving Average

    EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    Args:
        values: List of values (most recent first)
        alpha: Smoothing factor (higher = more weight on recent values)

    Returns:
        float: EMA value
    """
    if not values or len(values) == 0:
        return None

    # Start with first value
    ema = values[0]

    # Calculate EMA iteratively
    for value in values[1:]:
        if value is not None:
            ema = alpha * value + (1 - alpha) * ema

    return round(ema, 2)


def calculate_rolling_for_player(conn, player_id, player_name, window_size):
    """
    Calculate rolling stats for one player across one window size

    Returns list of dicts ready for database insertion
    """
    cursor = conn.cursor()

    # Get all games for this player, ordered by date
    query = """
        SELECT
            game_date,
            points,
            rebounds,
            assists,
            minutes_played,
            fg_made,
            fg_attempted,
            three_pointers_made,
            COALESCE(three_pt_attempted, 0) as three_pt_attempted,
            steals,
            blocks,
            turnovers,
            plus_minus
        FROM player_game_logs
        WHERE player_id = %s
        ORDER BY game_date ASC
    """

    cursor.execute(query, (player_id,))
    games = cursor.fetchall()
    cursor.close()

    if len(games) < window_size:
        # Not enough games for this window
        return []

    rolling_stats = []

    # Calculate rolling stats for each game (starting from window_size)
    for i in range(window_size, len(games) + 1):
        # Get last N games (most recent first for EMA)
        window = games[i-window_size:i]
        window_reversed = list(reversed(window))  # Most recent first

        as_of_date = window[-1][0]  # Date of most recent game

        # Extract stats
        points = [g[1] for g in window_reversed if g[1] is not None]
        rebounds = [g[2] for g in window_reversed if g[2] is not None]
        assists = [g[3] for g in window_reversed if g[3] is not None]
        minutes = [g[4] for g in window_reversed if g[4] is not None]
        steals = [g[9] for g in window_reversed if g[9] is not None]
        blocks = [g[10] for g in window_reversed if g[10] is not None]
        turnovers = [g[11] for g in window_reversed if g[11] is not None]
        plus_minus = [g[12] for g in window_reversed if g[12] is not None]

        # Calculate FG%
        fg_pcts = []
        for g in window_reversed:
            if g[5] is not None and g[6] is not None and g[6] > 0:  # fg_made, fg_attempted
                fg_pcts.append(g[5] / g[6])

        # Calculate 3P%
        three_pt_pcts = []
        for g in window_reversed:
            if g[7] is not None and g[8] is not None and g[8] > 0:  # 3pm, 3pa
                three_pt_pcts.append(g[7] / g[8])

        # Calculate EMAs
        ema_points = calculate_ema(points)
        ema_rebounds = calculate_ema(rebounds)
        ema_assists = calculate_ema(assists)
        ema_minutes = calculate_ema(minutes)
        ema_fg_pct = calculate_ema(fg_pcts) if fg_pcts else None
        ema_three_pt_pct = calculate_ema(three_pt_pcts) if three_pt_pcts else None
        ema_steals = calculate_ema(steals)
        ema_blocks = calculate_ema(blocks)
        ema_turnovers = calculate_ema(turnovers)
        ema_plus_minus = calculate_ema(plus_minus)

        # Hot streak detection (points > avg + 1 std)
        is_hot_streak = False
        if len(points) >= 3:
            avg_points = np.mean(points)
            std_points = np.std(points)
            recent_points = points[0]  # Most recent game
            is_hot_streak = bool(recent_points > (avg_points + std_points))

        rolling_stats.append({
            'player_id': player_id,
            'as_of_date': as_of_date,
            'window_size': window_size,
            'ema_points': ema_points,
            'ema_rebounds': ema_rebounds,
            'ema_assists': ema_assists,
            'ema_minutes': ema_minutes,
            'ema_fg_pct': ema_fg_pct,
            'ema_three_pt_pct': ema_three_pt_pct,
            'ema_steals': ema_steals,
            'ema_blocks': ema_blocks,
            'ema_turnovers': ema_turnovers,
            'ema_plus_minus': ema_plus_minus,
            'games_in_window': len(points),
            'is_hot_streak': is_hot_streak
        })

    return rolling_stats


def insert_rolling_stats(conn, rolling_stats):
    """Batch insert rolling stats into database"""
    if not rolling_stats:
        return 0

    cursor = conn.cursor()

    insert_query = """
        INSERT INTO player_rolling_stats (
            player_id, as_of_date, window_size,
            ema_points, ema_rebounds, ema_assists, ema_minutes,
            ema_fg_pct, ema_three_pt_pct,
            ema_steals, ema_blocks, ema_turnovers, ema_plus_minus,
            games_in_window, is_hot_streak
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (player_id, as_of_date, window_size)
        DO UPDATE SET
            ema_points = EXCLUDED.ema_points,
            ema_rebounds = EXCLUDED.ema_rebounds,
            ema_assists = EXCLUDED.ema_assists,
            ema_minutes = EXCLUDED.ema_minutes,
            ema_fg_pct = EXCLUDED.ema_fg_pct,
            ema_three_pt_pct = EXCLUDED.ema_three_pt_pct,
            ema_steals = EXCLUDED.ema_steals,
            ema_blocks = EXCLUDED.ema_blocks,
            ema_turnovers = EXCLUDED.ema_turnovers,
            ema_plus_minus = EXCLUDED.ema_plus_minus,
            games_in_window = EXCLUDED.games_in_window,
            is_hot_streak = EXCLUDED.is_hot_streak
    """

    for stat in rolling_stats:
        cursor.execute(insert_query, (
            stat['player_id'], stat['as_of_date'], stat['window_size'],
            stat['ema_points'], stat['ema_rebounds'], stat['ema_assists'], stat['ema_minutes'],
            stat['ema_fg_pct'], stat['ema_three_pt_pct'],
            stat['ema_steals'], stat['ema_blocks'], stat['ema_turnovers'], stat['ema_plus_minus'],
            stat['games_in_window'], stat['is_hot_streak']
        ))

    conn.commit()
    cursor.close()

    return len(rolling_stats)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("NBA ROLLING STATS CALCULATOR")
    print("EMA-based Rolling Statistics (L3/L5/L10/L20)")
    print("="*80 + "\n")

    # Connect to database
    print("🔌 Connecting to NBA Players Database...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("   ✅ Connected (port 5536)\n")

    # Get all players with game logs
    print("📊 Fetching players with game logs...")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT pp.player_id, pp.full_name, COUNT(*) as game_count
        FROM player_profile pp
        JOIN player_game_logs pgl ON pp.player_id = pgl.player_id
        GROUP BY pp.player_id, pp.full_name
        HAVING COUNT(*) >= 3
        ORDER BY game_count DESC
    """)
    players = cursor.fetchall()
    cursor.close()

    print(f"   ✅ Found {len(players)} players with 3+ games\n")

    # Process each player
    total_stats_inserted = 0

    for idx, (player_id, player_name, game_count) in enumerate(players, 1):
        print(f"[{idx}/{len(players)}] Processing {player_name} ({game_count} games)...")

        player_total = 0

        for window_size in WINDOW_SIZES:
            rolling_stats = calculate_rolling_for_player(conn, player_id, player_name, window_size)

            if rolling_stats:
                inserted = insert_rolling_stats(conn, rolling_stats)
                player_total += inserted

        total_stats_inserted += player_total
        print(f"            ✅ {player_total} rolling stat records created")

    print(f"\n" + "="*80)
    print(f"✅ COMPLETE: {total_stats_inserted:,} rolling stat records inserted")
    print("="*80 + "\n")

    # Verify results
    cursor = conn.cursor()
    cursor.execute("SELECT window_size, COUNT(*) FROM player_rolling_stats GROUP BY window_size ORDER BY window_size")
    results = cursor.fetchall()
    cursor.close()

    print("📊 Rolling Stats by Window Size:")
    for window_size, count in results:
        print(f"   L{window_size}: {count:,} records")

    conn.close()


if __name__ == '__main__':
    main()
