#!/usr/bin/env python3
"""
Backfill Player Positions from NBA API
=======================================

Fetches player positions from the NBA Stats API and updates player_profile table.

The NBA Stats API provides position data via the commonplayerinfo endpoint.

Usage:
    python backfill_player_positions.py
    python backfill_player_positions.py --dry-run
    python backfill_player_positions.py --batch-size 50
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import psycopg2
import requests
from tqdm import tqdm


def normalize_name(name: str) -> str:
    """Normalize player name for matching.

    Handles:
    - Accented characters (é -> e, ć -> c)
    - Suffixes (Jr., Jr, III, II, IV)
    - Special characters and punctuation
    """
    # Convert accented characters to ASCII equivalents
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ASCII", "ignore").decode("ASCII")

    # Lowercase and remove suffixes
    clean = ascii_name.lower()
    clean = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv)\s*$", "", clean, flags=re.IGNORECASE)

    # Remove punctuation and extra spaces
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nba.config.database import get_players_db_config

# Database configuration
DB_CONFIG = get_players_db_config()

# NBA Stats API endpoint
NBA_API_URL = "https://stats.nba.com/stats/commonplayerinfo"

# Headers to mimic browser request
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

# Fallback: ESPN API for position data
ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"


def get_nba_player_position(player_id: int) -> str | None:
    """Fetch player position from NBA Stats API"""
    params = {"PlayerID": player_id}

    try:
        response = requests.get(NBA_API_URL, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse response - position is in CommonPlayerInfo resultSet
        result_sets = data.get("resultSets", [])
        for rs in result_sets:
            if rs.get("name") == "CommonPlayerInfo":
                headers = rs.get("headers", [])
                rows = rs.get("rowSet", [])
                if rows and "POSITION" in headers:
                    pos_idx = headers.index("POSITION")
                    return rows[0][pos_idx]
        return None
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"  Warning: Failed to fetch position for player {player_id}: {e}")
        return None


def get_espn_roster() -> dict[str, str]:
    """Fetch all player positions from ESPN roster API"""
    positions = {}

    # ESPN team IDs (1-30)
    for team_id in range(1, 31):
        try:
            url = ESPN_API_URL.format(team_id=team_id)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            athletes = data.get("athletes", [])
            for athlete in athletes:
                name = athlete.get("fullName", "")
                position = athlete.get("position", {}).get("abbreviation", "")
                if name and position:
                    positions[name] = position

            time.sleep(0.3)  # Rate limit
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  Warning: Failed to fetch ESPN roster for team {team_id}: {e}")
            continue

    return positions


def get_balldontlie_positions() -> dict[str, str]:
    """Fetch player positions from balldontlie API (requires auth now)"""
    print("Note: balldontlie API now requires authentication. Skipping.")
    return {}


def get_nba_all_players() -> dict[str, str]:
    """Fetch all historical players from NBA Stats API"""
    positions = {}
    url = "https://stats.nba.com/stats/commonallplayers"

    params = {
        "IsOnlyCurrentSeason": 0,  # Include historical players
        "LeagueID": "00",
        "Season": "2024-25",
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        result_sets = data.get("resultSets", [])
        for rs in result_sets:
            if rs.get("name") == "CommonAllPlayers":
                headers = rs.get("headers", [])
                rows = rs.get("rowSet", [])

                # Find column indices
                name_idx = (
                    headers.index("DISPLAY_FIRST_LAST") if "DISPLAY_FIRST_LAST" in headers else None
                )

                if name_idx is not None:
                    for row in rows:
                        name = row[name_idx]
                        # This endpoint doesn't include position, but we can use it
                        # to get player IDs for the commonplayerinfo endpoint
                        if name:
                            positions[name] = ""  # Placeholder

    except Exception as e:
        print(f"Error fetching from NBA Stats API: {e}")

    return positions


def fetch_players_without_position() -> list[tuple]:
    """Get all players from database missing position"""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT player_id, full_name
                FROM player_profile
                WHERE position IS NULL
                ORDER BY full_name
            """
            )
            return cur.fetchall()


def update_player_position(player_id: int, position: str, dry_run: bool = False):
    """Update player position in database"""
    if dry_run:
        return True

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE player_profile
                SET position = %s, updated_at = NOW()
                WHERE player_id = %s
            """,
                (position, player_id),
            )
        conn.commit()
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill player positions from NBA API")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--batch-size", type=int, default=100, help="Players per batch")
    parser.add_argument(
        "--source",
        choices=["espn", "balldontlie", "nba"],
        default="balldontlie",
        help="API source (default: balldontlie)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PLAYER POSITION BACKFILL")
    print("=" * 70)

    # Get players missing position
    players = fetch_players_without_position()
    print(f"Players missing position: {len(players)}")

    if not players:
        print("All players have positions. Nothing to do.")
        return

    # Build lookup dict from API
    print(f"\nFetching positions from {args.source} API...")

    if args.source == "espn":
        position_lookup = get_espn_roster()
    elif args.source == "balldontlie":
        position_lookup = get_balldontlie_positions()
    else:
        # NBA API requires looking up player IDs first - use balldontlie instead
        print("Note: NBA API requires player IDs. Using balldontlie as fallback.")
        position_lookup = get_balldontlie_positions()

    print(f"Fetched positions for {len(position_lookup)} players")

    # Match and update
    updated = 0
    not_found = []

    # Build normalized lookup for fuzzy matching
    normalized_lookup = {}
    for api_name, pos in position_lookup.items():
        norm = normalize_name(api_name)
        normalized_lookup[norm] = pos

    for player_id, name in tqdm(players, desc="Updating"):
        # Try exact match first
        position = position_lookup.get(name)

        # Try normalized match (handles accents, suffixes)
        if not position:
            norm_name = normalize_name(name)
            position = normalized_lookup.get(norm_name)

        # Try fuzzy match (first name + last name only)
        if not position:
            name_parts = norm_name.split()
            if len(name_parts) >= 2:
                # Try "First Last" format (skip middle names)
                simple_name = f"{name_parts[0]} {name_parts[-1]}"
                position = normalized_lookup.get(simple_name)

        if position:
            update_player_position(player_id, position, args.dry_run)
            updated += 1
        else:
            not_found.append(name)

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Updated: {updated}")
    print(f"Not found: {len(not_found)}")

    if args.dry_run:
        print("\n[DRY RUN - no changes made]")

    if not_found and len(not_found) <= 20:
        print("\nPlayers not found:")
        for name in not_found[:20]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
