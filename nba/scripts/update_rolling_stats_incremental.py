#!/usr/bin/env python3
"""
Incremental Rolling Stats Update
Updates only players with new games since last calculation.

This script efficiently updates rolling statistics by:
1. Checking the last update date from player_rolling_stats
2. Finding players with games after that date
3. Recalculating only those players' rolling stats

Usage:
    python3 update_rolling_stats_incremental.py
    python3 update_rolling_stats_incremental.py --force-date 2025-11-01
"""
import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PLAYERS = {
    'host': os.getenv('NBA_PLAYERS_DB_HOST', 'localhost'),
    'port': int(os.getenv('NBA_PLAYERS_DB_PORT', 5536)),  # LEGACY: nba_players database
    'user': os.getenv('NBA_PLAYERS_DB_USER', os.getenv('NBA_DB_USER', os.getenv('DB_USER', 'nba_user'))),
    'password': os.getenv('NBA_PLAYERS_DB_PASSWORD', os.getenv('NBA_DB_PASSWORD', os.getenv('DB_PASSWORD'))),
    'database': os.getenv('NBA_PLAYERS_DB_NAME', 'nba_players')  # LEGACY DATABASE
}

DB_TEAM = {
    'host': os.getenv('NBA_TEAM_DB_HOST', 'localhost'),
    'port': int(os.getenv('NBA_TEAM_DB_PORT', 5538)),
    'user': os.getenv('NBA_TEAM_DB_USER', os.getenv('NBA_DB_USER', os.getenv('DB_USER', 'nba_user'))),
    'password': os.getenv('NBA_TEAM_DB_PASSWORD', os.getenv('NBA_DB_PASSWORD', os.getenv('DB_PASSWORD'))),
    'database': os.getenv('NBA_TEAM_DB_NAME', 'nba_team')
}

def get_current_season():
    """
    Calculate current NBA season based on date.
    NBA season uses END year (2025-26 season = 2026).
    Season starts in October, so Oct-Dec uses next year's number.
    """
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year

CURRENT_SEASON = get_current_season()  # Dynamic based on current date


def get_last_update_date(conn):
    """Get max as_of_date from player_rolling_stats"""
    query = """
        SELECT MAX(as_of_date)
        FROM player_rolling_stats;
    """
    result = pd.read_sql_query(query, conn)
    last_date = result.iloc[0, 0]

    if last_date is None:
        # No rolling stats yet - return season start date
        logger.warning("No rolling stats found - full rebuild required")
        return None

    return last_date


def get_players_with_new_games(conn, since_date):
    """Find players with games after since_date"""
    query = """
        SELECT DISTINCT g.player_id, p.full_name as player_name
        FROM player_game_logs g
        JOIN player_profile p ON g.player_id = p.player_id
        WHERE g.season = %s
          AND g.game_date > %s
        ORDER BY p.full_name;
    """
    df = pd.read_sql_query(query, conn, params=(CURRENT_SEASON, since_date))
    return df


def calculate_ema(values, window):
    """Calculate Exponential Moving Average"""
    if len(values) == 0:
        return 0.0

    # Use pandas ewm for EMA calculation
    series = pd.Series(values)
    ema = series.ewm(span=window, adjust=False).mean().iloc[-1]
    return float(ema)


def load_team_pace() -> Dict[str, float]:
    """
    Load team pace values keyed by team_abbrev.
    Falls back to latest available season if current season pace is missing.
    """
    try:
        team_conn = psycopg2.connect(**DB_TEAM)
    except Exception as e:
        logger.warning(f"Could not connect to team DB for pace lookup: {e}")
        return {}

    with team_conn.cursor() as cur:
        cur.execute("SELECT MAX(season) FROM team_season_stats;")
        latest_season = cur.fetchone()[0]

        if latest_season is None:
            logger.warning("No team pace data found in team_season_stats")
            return {}

        cur.execute(
            """
            SELECT team_abbrev, pace
            FROM team_season_stats
            WHERE season = %s
            """,
            (latest_season,)
        )
        pace_map = {row[0]: float(row[1]) for row in cur.fetchall() if row[1] is not None}

    team_conn.close()

    if latest_season != CURRENT_SEASON:
        logger.info(f"Using team pace from season {latest_season} (current season pace not available)")

    return pace_map


def compute_usage_rate(row: pd.Series, team_pace: Dict[str, float]) -> Optional[float]:
    """
    Approximate usage rate for one game using shot volume and turnovers scaled by team pace.

    Usage% ≈ 100 * (FGA + 0.44 * FTA + TOV) * 48 / (minutes_played * team_pace)
    """
    minutes = row.get('minutes')
    if minutes is None or minutes <= 0:
        return None

    pace = team_pace.get(row.get('team_abbrev'))
    if pace is None or pace <= 0:
        return None

    fg_attempted = row.get('field_goals_attempted') or 0
    ft_attempted = row.get('free_throws_attempted') or 0
    turnovers = row.get('turnovers') or 0

    usage_events = fg_attempted + 0.44 * ft_attempted + turnovers
    usage_rate = (usage_events * 48.0) / (minutes * pace) * 100.0
    return float(usage_rate)


def update_player_rolling_stats(conn, player_id, player_name, from_date, team_pace_map):
    """
    Recalculate rolling stats for one player from a specific date forward.

    This uses Exponential Moving Average (EMA) for L3/L5/L10/L20 windows.
    """
    logger.info(f"Updating {player_name} from {from_date}")

    # Get all games for this player (need history for EMA calculation)
    query = """
        SELECT
            game_date,
            team_abbrev,
            points, rebounds, assists, three_pointers_made as threes_made,
            steals, blocks, turnovers, minutes_played as minutes,
            fg_made as field_goals_made, fg_attempted as field_goals_attempted,
            ft_attempted as free_throws_attempted,
            plus_minus,
            COALESCE(three_pt_attempted, 0) as three_pt_attempted
        FROM player_game_logs
        WHERE player_id = %s
          AND season = %s
        ORDER BY game_date ASC;
    """

    games_df = pd.read_sql_query(query, conn, params=(player_id, CURRENT_SEASON))

    if len(games_df) == 0:
        logger.warning(f"No games found for {player_name}")
        return 0

    # Calculate rolling stats for each date
    stats_to_insert = []

    for idx in range(len(games_df)):
        game_date = games_df.iloc[idx]['game_date']

        # Only process dates after from_date
        if game_date <= from_date:
            continue

        # Get games up to this point (for EMA calculation)
        games_so_far = games_df.iloc[:idx+1]

        # Calculate EMA for different windows (create separate row for each window)
        stats_cols = ['points', 'rebounds', 'assists', 'threes_made', 'steals',
                     'blocks', 'turnovers', 'minutes', 'plus_minus']
        windows = [3, 5, 10, 20]

        for window in windows:
            rolling_stats = {
                'player_id': player_id,
                'as_of_date': game_date,
                'window_size': window,
                'games_in_window': min(len(games_so_far), window)
            }

            # Get last N games
            recent_games = games_so_far.tail(window)

            # Calculate usage_rate for each of the recent games
            usage_values = []
            for _, recent_game in recent_games.iterrows():
                usage = compute_usage_rate(recent_game, team_pace_map)
                if usage is not None:
                    usage_values.append(usage)

            # Calculate EMA for each stat
            for stat in stats_cols:
                if len(recent_games) > 0:
                    ema_value = calculate_ema(recent_games[stat].values, window)
                    # Map threes_made to column name (no ema_threes_made column, skipped)
                    if stat == 'threes_made':
                        continue  # Skip - no ema_threes_made column
                    if stat == 'plus_minus':
                        rolling_stats['ema_plus_minus'] = ema_value
                        continue
                    rolling_stats[f'ema_{stat}'] = ema_value
                else:
                    if stat == 'plus_minus':
                        rolling_stats['ema_plus_minus'] = 0.0
                    elif stat != 'threes_made':
                        rolling_stats[f'ema_{stat}'] = 0.0

            # Usage rate EMA (computed separately from shot/turnover volume)
            if len(usage_values) > 0:
                rolling_stats['ema_usage_rate'] = calculate_ema(usage_values, window)
            else:
                rolling_stats['ema_usage_rate'] = 0.0

            # Calculate FG%
            if len(recent_games) > 0:
                total_made = recent_games['field_goals_made'].sum()
                total_attempted = recent_games['field_goals_attempted'].sum()

                if total_attempted > 0:
                    rolling_stats['ema_fg_pct'] = float(total_made / total_attempted)
                else:
                    rolling_stats['ema_fg_pct'] = 0.0
            else:
                rolling_stats['ema_fg_pct'] = 0.0

            # Calculate 3PT% (NEW - fixes NULL ema_three_pt_pct issue)
            if len(recent_games) > 0:
                total_3pm = recent_games['threes_made'].fillna(0).sum()
                total_3pa = recent_games['three_pt_attempted'].fillna(0).sum()

                if total_3pa > 0:
                    rolling_stats['ema_three_pt_pct'] = float(total_3pm / total_3pa)
                else:
                    rolling_stats['ema_three_pt_pct'] = 0.0
            else:
                rolling_stats['ema_three_pt_pct'] = 0.0

            # Calculate is_hot_streak (NEW - fixes NULL is_hot_streak issue)
            # Hot streak: recent L5 points > 1.2 * season average OR > season_avg + std
            if len(games_so_far) >= 5:
                season_avg_points = games_so_far['points'].mean()
                season_std_points = games_so_far['points'].std()
                l5_avg_points = games_so_far.tail(5)['points'].mean()

                if season_avg_points > 0:
                    is_hot = (l5_avg_points > season_avg_points * 1.2) or \
                             (season_std_points > 0 and l5_avg_points > season_avg_points + season_std_points)
                    rolling_stats['is_hot_streak'] = bool(is_hot)
                else:
                    rolling_stats['is_hot_streak'] = False
            else:
                rolling_stats['is_hot_streak'] = False

            stats_to_insert.append(rolling_stats)

    if len(stats_to_insert) == 0:
        logger.info(f"No new rolling stats to insert for {player_name}")
        return 0

    # Insert/update rolling stats
    cursor = conn.cursor()

    for stats in stats_to_insert:
        # Build column lists and convert numpy types to Python native types
        columns = list(stats.keys())
        values = []
        for col in columns:
            val = stats[col]
            # Convert numpy types to Python native types
            if hasattr(val, 'item'):  # numpy scalar
                val = val.item()
            values.append(val)
        placeholders = ', '.join(['%s'] * len(columns))

        # Delete existing record for this date (if any)
        delete_query = """
            DELETE FROM player_rolling_stats
            WHERE player_id = %s
              AND as_of_date = %s
              AND window_size = %s;
        """
        cursor.execute(delete_query, (player_id, stats['as_of_date'], stats['window_size']))

        # Insert new record
        insert_query = f"""
            INSERT INTO player_rolling_stats ({', '.join(columns)})
            VALUES ({placeholders});
        """
        cursor.execute(insert_query, values)

    conn.commit()
    logger.info(f"✅ Inserted {len(stats_to_insert)} rolling stat records for {player_name}")

    return len(stats_to_insert)


def main(force_date=None):
    """Main execution"""
    logger.info("="*80)
    logger.info("NBA INCREMENTAL ROLLING STATS UPDATE")
    logger.info("="*80)

    conn = psycopg2.connect(**DB_PLAYERS)
    team_pace_map = load_team_pace()

    try:
        # Get last update date
        if force_date:
            last_update = pd.to_datetime(force_date).date()
            logger.info(f"FORCED update from: {last_update}")
        else:
            last_update = get_last_update_date(conn)

            if last_update is None:
                logger.error("No rolling stats found - run full calculation first")
                logger.error("Use calculate_rolling_stats.py to do initial build")
                return

            logger.info(f"Last rolling stats update: {last_update}")

        # Find players with new games
        players_to_update = get_players_with_new_games(conn, last_update)
        logger.info(f"Found {len(players_to_update)} players with new games")

        if len(players_to_update) == 0:
            logger.info("✅ No updates needed - rolling stats are current")
            return

        # Update each player
        total_records = 0
        for idx, row in players_to_update.iterrows():
            records = update_player_rolling_stats(
                conn,
                row['player_id'],
                row['player_name'],
                last_update,
                team_pace_map
            )
            total_records += records

            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx+1}/{len(players_to_update)} players")

        logger.info("="*80)
        logger.info(f"✅ Updated rolling stats for {len(players_to_update)} players")
        logger.info(f"✅ Inserted {total_records} total records")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error during update: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Incrementally update NBA rolling stats'
    )
    parser.add_argument(
        '--force-date',
        help='Force update from this date (YYYY-MM-DD)',
        default=None
    )

    args = parser.parse_args()
    main(force_date=args.force_date)
