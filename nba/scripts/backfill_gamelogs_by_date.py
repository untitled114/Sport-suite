#!/usr/bin/env python3
"""
Backfill Player Game Logs for Specific Dates
==============================================
Fetches and loads player game logs for specific dates using NBA Stats API.

Usage:
    python3 backfill_gamelogs_by_date.py --dates 2025-11-06 2025-11-07 2025-11-08
    python3 backfill_gamelogs_by_date.py --start-date 2025-11-06 --end-date 2025-11-08
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5536,
    "database": "nba_players",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

# NBA Stats API config
NBA_STATS_API = "https://stats.nba.com/stats/playergamelogs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def fetch_gamelogs_for_date_range(date_from, date_to, season="2025-26"):
    """
    Fetch player game logs for a date range from NBA Stats API

    Args:
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        season: NBA season (e.g., '2025-26')

    Returns:
        DataFrame with game logs
    """
    logger.info(f"Fetching gamelogs from {date_from} to {date_to}...")

    params = {
        "DateFrom": date_from,
        "DateTo": date_to,
        "GameSegment": "",
        "LastNGames": 0,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "Totals",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "VsConference": "",
        "VsDivision": "",
    }

    max_retries = 5
    base_delay = 3

    for attempt in range(max_retries):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)

            resp = session.get(NBA_STATS_API, params=params, timeout=60)
            resp.raise_for_status()

            data = resp.json()

            if "resultSets" not in data or len(data["resultSets"]) == 0:
                logger.error(f"No data returned from API")
                return pd.DataFrame()

            result_set = data["resultSets"][0]
            headers = result_set["headers"]
            rows = result_set["rowSet"]

            logger.info(f"✅ Received {len(rows)} game logs")

            df = pd.DataFrame(rows, columns=headers)
            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            delay = base_delay * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s..."
            )
            time.sleep(delay)
            continue
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()

    logger.error(f"All {max_retries} attempts failed")
    return pd.DataFrame()


def load_gamelogs_to_db(df, conn):
    """
    Load game logs DataFrame to database

    Args:
        df: DataFrame with game logs from NBA API
        conn: Database connection

    Returns:
        Number of records loaded
    """
    if df.empty:
        logger.warning("No data to load")
        return 0

    # First, upsert any missing players into player_profile
    cur = conn.cursor()
    unique_players = df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()

    for _, player in unique_players.iterrows():
        player_id = int(player["PLAYER_ID"])
        player_name = str(player["PLAYER_NAME"])

        # Upsert player profile (insert if not exists)
        cur.execute(
            """
            INSERT INTO player_profile (player_id, full_name)
            VALUES (%s, %s)
            ON CONFLICT (player_id) DO NOTHING
        """,
            (player_id, player_name),
        )

    conn.commit()
    logger.info(f"Upserted {len(unique_players)} players to player_profile")

    # Prepare data for insertion
    insert_data = []

    for _, row in df.iterrows():
        try:
            # Parse matchup (e.g., "LAL vs. BOS" or "LAL @ BOS")
            matchup = row.get("MATCHUP", "")
            is_home = "vs." in matchup

            # Extract team_abbrev and opponent from matchup
            team_abbrev = None
            opponent_abbrev = None

            if "@" in matchup:
                # Away game: "LAL @ BOS"
                parts = matchup.split("@")
                team_abbrev = parts[0].strip()
                opponent_abbrev = parts[-1].strip()
            elif "vs." in matchup:
                # Home game: "LAL vs. BOS"
                parts = matchup.split("vs.")
                team_abbrev = parts[0].strip()
                opponent_abbrev = parts[-1].strip()

            # Parse game date
            game_date = pd.to_datetime(row.get("GAME_DATE"))

            # Extract season year from game date
            year = game_date.year
            month = game_date.month
            if month >= 10:  # Oct-Dec are from first year of season
                season_year = year
            else:  # Jan-Apr are from second year of season
                season_year = year - 1

            insert_data.append(
                (
                    int(row.get("PLAYER_ID")),
                    str(row.get("GAME_ID", "")),
                    game_date,
                    season_year,
                    team_abbrev,
                    opponent_abbrev,
                    is_home,
                    (
                        int(row.get("MIN", 0))
                        if pd.notna(row.get("MIN")) and str(row.get("MIN", "")).strip()
                        else 0
                    ),
                    int(row.get("PTS", 0)),
                    int(row.get("REB", 0)),
                    int(row.get("AST", 0)),
                    int(row.get("STL", 0)),
                    int(row.get("BLK", 0)),
                    int(row.get("TOV", 0)),
                    int(row.get("FG3M", 0)),
                    int(row.get("FGM", 0)),
                    int(row.get("FGA", 0)),
                    int(row.get("FG3A", 0)),
                    int(row.get("FTM", 0)),
                    int(row.get("FTA", 0)),
                    int(row.get("PLUS_MINUS", 0)) if pd.notna(row.get("PLUS_MINUS")) else 0,
                )
            )
        except Exception as e:
            logger.warning(f"Skipping row: {e}")
            continue

    if not insert_data:
        logger.warning("No valid data to insert")
        return 0

    # Insert with ON CONFLICT to handle duplicates
    cur = conn.cursor()

    insert_query = """
        INSERT INTO player_game_logs
        (player_id, game_id, game_date, season, team_abbrev, opponent_abbrev,
         is_home, minutes_played, points, rebounds, assists, steals, blocks,
         turnovers, three_pointers_made, fg_made, fg_attempted,
         three_pt_attempted, ft_made, ft_attempted, plus_minus)
        VALUES %s
        ON CONFLICT (player_id, game_id)
        DO UPDATE SET
            minutes_played = EXCLUDED.minutes_played,
            points = EXCLUDED.points,
            rebounds = EXCLUDED.rebounds,
            assists = EXCLUDED.assists,
            steals = EXCLUDED.steals,
            blocks = EXCLUDED.blocks,
            turnovers = EXCLUDED.turnovers,
            three_pointers_made = EXCLUDED.three_pointers_made,
            fg_made = EXCLUDED.fg_made,
            fg_attempted = EXCLUDED.fg_attempted,
            three_pt_attempted = EXCLUDED.three_pt_attempted,
            ft_made = EXCLUDED.ft_made,
            ft_attempted = EXCLUDED.ft_attempted,
            plus_minus = EXCLUDED.plus_minus
    """

    try:
        execute_values(cur, insert_query, insert_data)
        conn.commit()
        logger.info(f"✅ Loaded {len(insert_data)} game logs to database")
        return len(insert_data)
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Insert failed: {e}")
        return 0


def backfill_dates(dates, season="2025-26"):
    """
    Backfill game logs for specific dates

    Args:
        dates: List of dates (YYYY-MM-DD format)
        season: NBA season (e.g., '2025-26')
    """
    logger.info("=" * 80)
    logger.info("BACKFILL PLAYER GAME LOGS BY DATE")
    logger.info("=" * 80)
    logger.info(f"Dates to backfill: {dates}")
    logger.info(f"Season: {season}")
    logger.info("=" * 80)

    conn = psycopg2.connect(**DB_CONFIG)

    try:
        total_loaded = 0

        for date_str in dates:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {date_str}")
            logger.info(f"{'='*80}")

            # Fetch game logs for this date
            df = fetch_gamelogs_for_date_range(date_str, date_str, season)

            if df.empty:
                logger.warning(f"No game logs found for {date_str}")
                continue

            logger.info(f"Players in games: {df['PLAYER_NAME'].nunique()}")
            logger.info(f"Total game logs: {len(df)}")

            # Load to database
            loaded = load_gamelogs_to_db(df, conn)
            total_loaded += loaded

            # Rate limiting
            time.sleep(0.6)

        logger.info(f"\n{'='*80}")
        logger.info(f"BACKFILL COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total game logs loaded: {total_loaded}")
        logger.info(f"{'='*80}")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill NBA player game logs for specific dates")
    parser.add_argument(
        "--dates", type=str, nargs="+", help="Specific dates to backfill (YYYY-MM-DD)"
    )
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--season", type=str, default="2025-26", help="NBA season (default: 2025-26)"
    )

    args = parser.parse_args()

    dates = []

    if args.dates:
        dates = args.dates
    elif args.start_date and args.end_date:
        # Generate date range
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        logger.error("Must provide either --dates or --start-date and --end-date")
        return

    backfill_dates(dates, args.season)


if __name__ == "__main__":
    main()
