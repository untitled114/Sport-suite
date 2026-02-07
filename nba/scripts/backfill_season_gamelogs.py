#!/usr/bin/env python3
"""
Backfill entire season of game logs at once (faster than date-by-date)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

from nba.config.database import get_players_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = get_players_db_config()

NBA_STATS_API = "https://stats.nba.com/stats/playergamelogs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def fetch_season_gamelogs(season):
    """Fetch ALL game logs for entire season at once"""
    params = {
        "DateFrom": "",
        "DateTo": "",
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
        "TeamID": 0,
        "VsConference": "",
        "VsDivision": "",
    }

    for attempt in range(5):
        try:
            logger.info(f"Fetching {season} game logs (attempt {attempt+1})...")
            session = requests.Session()
            session.headers.update(HEADERS)
            resp = session.get(NBA_STATS_API, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if "resultSets" not in data:
                return pd.DataFrame()

            result_set = data["resultSets"][0]
            df = pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
            logger.info(f"✅ Received {len(df)} game logs for {season}")
            return df

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (2**attempt))

    return pd.DataFrame()


def load_to_db(df, conn):
    """Load game logs to database"""
    if df.empty:
        return 0

    cur = conn.cursor()

    # Upsert players first
    players = df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    for _, p in players.iterrows():
        cur.execute(
            """
            INSERT INTO player_profile (player_id, full_name)
            VALUES (%s, %s) ON CONFLICT (player_id) DO NOTHING
        """,
            (int(p["PLAYER_ID"]), str(p["PLAYER_NAME"])),
        )
    conn.commit()
    logger.info(f"Upserted {len(players)} players")

    # Insert game logs
    insert_data = []
    for _, row in df.iterrows():
        matchup = row.get("MATCHUP", "")
        is_home = "vs." in matchup
        team = opponent = None
        if "@" in matchup:
            parts = matchup.split("@")
            team, opponent = parts[0].strip(), parts[-1].strip()
        elif "vs." in matchup:
            parts = matchup.split("vs.")
            team, opponent = parts[0].strip(), parts[-1].strip()

        game_date = pd.to_datetime(row.get("GAME_DATE"))
        season_year = game_date.year if game_date.month >= 10 else game_date.year - 1

        insert_data.append(
            (
                int(row["PLAYER_ID"]),
                str(row.get("GAME_ID", "")),
                game_date,
                season_year,
                team,
                opponent,
                is_home,
                int(row.get("MIN", 0)) if pd.notna(row.get("MIN")) else 0,
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

    execute_values(
        cur,
        """
        INSERT INTO player_game_logs
        (player_id, game_id, game_date, season, team_abbrev, opponent_abbrev,
         is_home, minutes_played, points, rebounds, assists, steals, blocks,
         turnovers, three_pointers_made, fg_made, fg_attempted,
         three_pt_attempted, ft_made, ft_attempted, plus_minus)
        VALUES %s
        ON CONFLICT (player_id, game_id) DO UPDATE SET
            points = EXCLUDED.points, rebounds = EXCLUDED.rebounds
    """,
        insert_data,
    )
    conn.commit()
    logger.info(f"✅ Loaded {len(insert_data)} game logs")
    return len(insert_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25"])
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    total = 0

    for season in args.seasons:
        logger.info(f"\n{'='*60}\nProcessing {season}\n{'='*60}")
        df = fetch_season_gamelogs(season)
        if not df.empty:
            total += load_to_db(df, conn)
        time.sleep(5)  # Rate limit between seasons

    conn.close()
    logger.info(f"\n✅ Total loaded: {total} game logs")


if __name__ == "__main__":
    main()
