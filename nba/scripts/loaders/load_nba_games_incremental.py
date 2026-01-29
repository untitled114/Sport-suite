#!/usr/bin/env python3
"""
NBA Games Data Loader (Incremental)
====================================
Fast incremental loader for daily pipeline.
Only processes games newer than the latest in database.

Uses LeagueGameFinder which returns ALL season games in ONE API call,
then filters to only insert new records.

Usage:
    # Daily pipeline (auto-detects season)
    python load_nba_games_incremental.py

    # Specific season
    python load_nba_games_incremental.py --season 2025-26
"""

import sys
import os
import argparse
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))
from nba_api_wrapper import NBAApiWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection params
DB_CONFIG = {
    'host': 'localhost',
    'port': 5537,
    'database': 'nba_games',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}


def get_current_season() -> str:
    """Get current NBA season in API format (e.g., '2025-26')"""
    now = datetime.now()
    # NBA season starts in October
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[-2:]}"


def get_latest_game_date(conn) -> str:
    """Get the most recent game_date in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_date) FROM games;")
    result = cursor.fetchone()[0]
    cursor.close()

    if result:
        return result.strftime('%Y-%m-%d')
    return '2000-01-01'  # If empty, load everything


def load_games_incremental(season: str = None) -> int:
    """
    Load only NEW games since last update.

    Returns:
        Number of new games loaded
    """
    if season is None:
        season = get_current_season()

    logger.info(f"=== Incremental Games Loader for {season} ===")

    conn = psycopg2.connect(**DB_CONFIG)

    try:
        # Check latest data
        latest_date = get_latest_game_date(conn)
        logger.info(f"Latest game in database: {latest_date}")

        # Single API call to get ALL season games
        api = NBAApiWrapper(requests_per_minute=20)
        logger.info(f"Fetching season games from NBA API (single call)...")
        games_df = api.get_season_games(season=season)

        if games_df.empty:
            logger.warning(f"No games found for {season}")
            return 0

        # Deduplicate (API returns one row per team)
        games_unique = games_df.drop_duplicates(subset=['GAME_ID']).copy()
        logger.info(f"API returned {len(games_unique)} unique games for season")

        # Filter to only new games
        games_unique['GAME_DATE'] = pd.to_datetime(games_unique['GAME_DATE'])
        new_games = games_unique[games_unique['GAME_DATE'] > latest_date]

        if len(new_games) == 0:
            logger.info("✅ Database is up-to-date, no new games to load")
            return 0

        logger.info(f"Found {len(new_games)} new games to load")

        # Parse season year
        season_year = int(season.split('-')[0]) + 1  # 2025-26 -> 2026

        # Prepare games data
        games_insert = []
        team_logs_insert = []

        for _, row in games_df.iterrows():
            try:
                game_date = pd.to_datetime(row.get('GAME_DATE'))

                # Skip old games
                if game_date.strftime('%Y-%m-%d') <= latest_date:
                    continue

                # Parse matchup
                matchup = row.get('MATCHUP', '')
                team_abbrev = row.get('TEAM_ABBREVIATION', '')
                is_home = 'vs.' in matchup

                # Get opponent
                if '@' in matchup:
                    opponent = matchup.split('@')[-1].strip()
                    home_team = opponent
                    away_team = team_abbrev
                elif 'vs.' in matchup:
                    opponent = matchup.split('vs.')[-1].strip()
                    home_team = team_abbrev
                    away_team = opponent
                else:
                    continue

                pts = int(row.get('PTS', 0))
                game_id = row.get('GAME_ID')

                # Add to games insert (home perspective only to avoid duplicates)
                if is_home:
                    games_insert.append((
                        game_id,
                        game_date,
                        season_year,
                        home_team,
                        away_team,
                        pts,  # home_score
                        None,  # away_score (will be updated from away team row)
                        None,  # total_possessions
                        None,  # pace
                        None,  # vegas_total
                        None   # vegas_spread
                    ))

                # Add to team_game_logs (both teams)
                team_logs_insert.append((
                    team_abbrev,
                    game_id,
                    game_date,
                    season_year,
                    opponent,
                    is_home,
                    pts,
                    None,  # possessions
                    None,  # pace
                    None,  # offensive_rating
                    None   # defensive_rating
                ))

            except Exception as e:
                logger.warning(f"Skipping game: {e}")
                continue

        cursor = conn.cursor()

        # Insert games
        if games_insert:
            games_query = """
                INSERT INTO games
                (game_id, game_date, season, home_team, away_team, home_score, away_score,
                 total_possessions, pace, vegas_total, vegas_spread)
                VALUES %s
                ON CONFLICT (game_id)
                DO UPDATE SET
                    home_score = COALESCE(EXCLUDED.home_score, games.home_score),
                    away_score = COALESCE(EXCLUDED.away_score, games.away_score)
            """
            execute_values(cursor, games_query, games_insert)
            logger.info(f"✅ Inserted {len(games_insert)} games")

        # Insert team game logs
        if team_logs_insert:
            logs_query = """
                INSERT INTO team_game_logs
                (team_abbrev, game_id, game_date, season, opponent, is_home, points,
                 possessions, pace, offensive_rating, defensive_rating)
                VALUES %s
                ON CONFLICT (team_abbrev, game_id)
                DO UPDATE SET
                    points = EXCLUDED.points,
                    season = EXCLUDED.season
            """
            execute_values(cursor, logs_query, team_logs_insert)
            logger.info(f"✅ Inserted {len(team_logs_insert)} team game logs")

        conn.commit()
        cursor.close()

        return len(new_games)

    except Exception as e:
        logger.error(f"Error loading games: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Incremental NBA games loader')
    parser.add_argument('--season', type=str, default=None,
                       help='NBA season (e.g., 2025-26). Auto-detects if not specified.')

    args = parser.parse_args()

    try:
        new_games = load_games_incremental(args.season)
        if new_games > 0:
            logger.info(f"🎉 Loaded {new_games} new games!")
        else:
            logger.info("✅ No new games to load")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
