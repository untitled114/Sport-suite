#!/usr/bin/env python3
"""
Backfill plus_minus for player_game_logs rows where it's NULL.

Re-fetches boxscores from ESPN API and updates the plus_minus column.
Only touches rows with plus_minus IS NULL, doesn't modify other columns.

Usage:
    python3 nba/scripts/backfill_plus_minus.py
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.config.database import get_players_db_config
from nba.utils.team_utils import normalize_team_abbrev

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_BOXSCORE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})


def get_null_dates():
    """Get distinct game dates where plus_minus is NULL."""
    conn = psycopg2.connect(**get_players_db_config())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT game_date FROM player_game_logs WHERE plus_minus IS NULL ORDER BY game_date"
        )
        dates = [row[0] for row in cur.fetchall()]
    conn.close()
    return dates


def fetch_boxscore_pm(game_id):
    """Fetch plus_minus values from ESPN boxscore API."""
    try:
        resp = session.get(ESPN_BOXSCORE_URL, params={"event": game_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch boxscore {game_id}: {e}")
        return {}

    pm_map = {}  # player_name -> plus_minus
    boxscore = data.get("boxscore", {})
    for team_data in boxscore.get("players", []):
        for stat_group in team_data.get("statistics", []):
            labels = stat_group.get("labels", [])
            for player in stat_group.get("athletes", []):
                name = player.get("athlete", {}).get("displayName", "")
                stats_array = player.get("stats", [])
                stats_dict = dict(zip(labels, stats_array)) if labels and stats_array else {}
                pm_raw = stats_dict.get("+/-", stats_dict.get("PLUSMINUS"))
                if pm_raw is not None and name:
                    try:
                        pm_map[name.lower()] = int(float(pm_raw))
                    except (ValueError, TypeError):
                        pass
    return pm_map


def main():
    dates = get_null_dates()
    if not dates:
        logger.info("No NULL plus_minus rows found - nothing to backfill")
        return

    logger.info(f"Found {len(dates)} dates with NULL plus_minus ({dates[0]} to {dates[-1]})")
    conn = psycopg2.connect(**get_players_db_config())
    total_updated = 0

    for date_obj in dates:
        date_str = date_obj.strftime("%Y-%m-%d") if hasattr(date_obj, "strftime") else str(date_obj)
        espn_date = date_str.replace("-", "")

        # Get games for this date
        try:
            resp = session.get(
                ESPN_SCOREBOARD_URL, params={"dates": espn_date, "limit": 50}, timeout=15
            )
            resp.raise_for_status()
            events = resp.json().get("events", [])
        except Exception as e:
            logger.warning(f"Failed to fetch scoreboard for {date_str}: {e}")
            continue

        date_pm = {}  # player_name_lower -> plus_minus
        for event in events:
            game_id = event.get("id")
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                continue

            pm_map = fetch_boxscore_pm(game_id)
            date_pm.update(pm_map)
            time.sleep(0.3)

        if not date_pm:
            logger.info(f"  {date_str}: no PM data found")
            continue

        # Update rows for this date
        with conn.cursor() as cur:
            # Get NULL rows for this date with player names
            cur.execute(
                """
                SELECT pgl.player_id, pp.full_name
                FROM player_game_logs pgl
                JOIN player_profile pp ON pgl.player_id = pp.player_id
                WHERE pgl.game_date = %s AND pgl.plus_minus IS NULL
                """,
                (date_str,),
            )
            null_rows = cur.fetchall()

            updated = 0
            for player_id, full_name in null_rows:
                pm_val = date_pm.get(full_name.lower())
                if pm_val is not None:
                    cur.execute(
                        "UPDATE player_game_logs SET plus_minus = %s WHERE player_id = %s AND game_date = %s",
                        (pm_val, player_id, date_str),
                    )
                    updated += 1

            conn.commit()
            total_updated += updated
            logger.info(
                f"  {date_str}: updated {updated}/{len(null_rows)} rows ({len(date_pm)} PM values from ESPN)"
            )

    conn.close()
    logger.info(f"Done: {total_updated} rows backfilled across {len(dates)} dates")


if __name__ == "__main__":
    main()
