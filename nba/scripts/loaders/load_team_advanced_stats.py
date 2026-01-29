#!/usr/bin/env python3
"""
Load Team Advanced Stats (Pace, Ratings) from NBA API
======================================================
Uses LeagueDashTeamStats endpoint to get real PACE, OFF_RATING, DEF_RATING
for all teams in a single API call.

This is much more accurate than calculating from game logs.

Usage:
    python load_team_advanced_stats.py
    python load_team_advanced_stats.py --season 2025-26
"""

import sys
import os
import argparse
import logging
import time
import psycopg2
from datetime import datetime

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as static_teams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
TEAM_DB = {
    'host': 'localhost',
    'port': 5538,
    'database': 'nba_team',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}


def get_current_season() -> str:
    """Get current NBA season in API format (e.g., '2025-26')"""
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[-2:]}"


def get_season_year(season: str) -> int:
    """Convert season string to year (e.g., '2025-26' -> 2026)"""
    return int(season.split('-')[0]) + 1


def load_team_advanced_stats(season: str = None) -> int:
    """
    Load real PACE, OFF_RATING, DEF_RATING from NBA API.

    Returns:
        Number of teams updated
    """
    if season is None:
        season = get_current_season()

    season_year = get_season_year(season)

    logger.info(f"=== Loading Team Advanced Stats for {season} (season {season_year}) ===")

    # Build team ID to abbreviation mapping
    nba_teams = static_teams.get_teams()
    team_id_to_abbrev = {t['id']: t['abbreviation'] for t in nba_teams}
    nba_team_ids = set(team_id_to_abbrev.keys())

    # Fetch from NBA API (single call for all teams)
    logger.info("Fetching from LeagueDashTeamStats API...")

    try:
        time.sleep(0.6)  # Rate limiting
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]

        if df.empty:
            logger.warning("No data returned from API")
            return 0

        logger.info(f"API returned stats for {len(df)} entries")

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return 0

    # Connect to database
    conn = psycopg2.connect(**TEAM_DB)
    cursor = conn.cursor()

    updated = 0

    try:
        for _, row in df.iterrows():
            team_id = row.get('TEAM_ID')

            # Skip non-NBA teams (G-League, WNBA, etc.)
            if team_id not in nba_team_ids:
                continue

            team_abbrev = team_id_to_abbrev[team_id]
            pace = row.get('PACE')
            off_rating = row.get('OFF_RATING')
            def_rating = row.get('DEF_RATING')

            if pace is None:
                continue

            # Update team_season_stats
            cursor.execute("""
                UPDATE team_season_stats
                SET pace = %s,
                    offensive_rating = %s,
                    defensive_rating = %s
                WHERE team_abbrev = %s AND season = %s
            """, (
                round(float(pace), 2),
                round(float(off_rating), 2) if off_rating else None,
                round(float(def_rating), 2) if def_rating else None,
                team_abbrev,
                season_year
            ))

            if cursor.rowcount > 0:
                logger.info(f"  {team_abbrev}: PACE={pace:.1f}, OFF={off_rating:.1f}, DEF={def_rating:.1f}")
                updated += 1

        conn.commit()
        logger.info(f"✅ Updated {updated} teams with real pace/ratings")

    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

    return updated


def main():
    parser = argparse.ArgumentParser(description='Load team advanced stats from NBA API')
    parser.add_argument('--season', type=str, default=None,
                       help='NBA season (e.g., 2025-26). Auto-detects if not specified.')

    args = parser.parse_args()

    try:
        updated = load_team_advanced_stats(args.season)
        if updated > 0:
            logger.info(f"🎉 Loaded real pace/ratings for {updated} teams!")
        else:
            logger.warning("No teams updated")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
