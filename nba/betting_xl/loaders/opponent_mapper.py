#!/usr/bin/env python3
"""
OpponentMapper - Enriches props with opponent_team and is_home
===============================================================

Fetches NBA schedule from ESPN API and maps players to their opponents.
Used by load_props_to_db.py to enrich props before database insertion.

Usage:
    from opponent_mapper import OpponentMapper

    mapper = OpponentMapper(verbose=True)
    enriched_props = mapper.enrich_props(props)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
import requests

from nba.utils.team_utils import normalize_team_abbrev

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database config for player team lookup
# Note: Production uses mlb_user for all databases (legacy naming)
DB_PLAYERS = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv(
        "NBA_PLAYERS_DB_USER", os.getenv("NBA_DB_USER", os.getenv("DB_USER", "mlb_user"))
    ),
    "password": os.getenv(
        "NBA_PLAYERS_DB_PASSWORD", os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    ),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


class OpponentMapper:
    """
    Maps players to their opponents using ESPN schedule data.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._schedule_cache: Dict[str, Dict[str, Dict]] = {}  # {date: {team: {opponent, is_home}}}
        self._player_teams_cache: Dict[str, str] = {}  # {player_name: team_abbrev}
        self._conn = None

    def _get_db_connection(self):
        """Get database connection for player team lookup."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**DB_PLAYERS)
        return self._conn

    def _close_connection(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def fetch_schedule(self, game_date: str) -> Dict[str, Dict]:
        """
        Fetch NBA schedule from ESPN API for a given date.

        Args:
            game_date: Date string in YYYY-MM-DD format

        Returns:
            Dict mapping team_abbrev -> {opponent: str, is_home: bool}
        """
        # Check cache first
        if game_date in self._schedule_cache:
            return self._schedule_cache[game_date]

        # Format date for ESPN API (YYYYMMDD)
        date_param = game_date.replace("-", "")
        url = f"{ESPN_SCOREBOARD_URL}?dates={date_param}"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            games_map = {}
            events = data.get("events", [])

            if not events:
                if self.verbose:
                    logger.warning(f"No games found for {game_date}")
                return {}

            for event in events:
                # Parse from shortName format: "HOU @ MIL"
                short_name = event.get("shortName", "")

                if " @ " not in short_name:
                    continue

                parts = short_name.split(" @ ")
                if len(parts) != 2:
                    continue

                away_team = normalize_team_abbrev(parts[0].strip())
                home_team = normalize_team_abbrev(parts[1].strip())

                if away_team and home_team:
                    games_map[away_team] = {"opponent": home_team, "is_home": False}
                    games_map[home_team] = {"opponent": away_team, "is_home": True}

            if self.verbose:
                logger.info(f"Fetched {len(games_map) // 2} games for {game_date}")

            # Cache the result
            self._schedule_cache[game_date] = games_map
            return games_map

        except Exception as e:
            logger.error(f"Failed to fetch ESPN schedule for {game_date}: {e}")
            return {}

    def batch_load_player_teams(self, player_names: List[str]) -> None:
        """
        Batch load player teams into cache for efficiency.
        Players not found are marked as None in cache to prevent repeated lookups.

        Args:
            player_names: List of player names to lookup
        """
        # Filter out already cached players
        names_to_lookup = [n for n in player_names if n and n not in self._player_teams_cache]
        if not names_to_lookup:
            return

        try:
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                # Batch query - get all players at once
                cur.execute(
                    """
                    SELECT full_name, team_abbrev FROM player_profile
                    WHERE full_name = ANY(%s)
                """,
                    (names_to_lookup,),
                )
                rows = cur.fetchall()

                for name, team in rows:
                    if team:
                        self._player_teams_cache[name] = normalize_team_abbrev(team)

                # For any remaining players not found, check game logs
                still_missing = [n for n in names_to_lookup if n not in self._player_teams_cache]
                if still_missing:
                    cur.execute(
                        """
                        SELECT DISTINCT ON (pp.full_name) pp.full_name, pgl.team_abbrev
                        FROM player_game_logs pgl
                        JOIN player_profile pp ON pgl.player_id = pp.player_id
                        WHERE pp.full_name = ANY(%s)
                        ORDER BY pp.full_name, pgl.game_date DESC
                    """,
                        (still_missing,),
                    )
                    rows = cur.fetchall()

                    for name, team in rows:
                        if team:
                            self._player_teams_cache[name] = normalize_team_abbrev(team)

                # Mark all remaining players as "not found" to prevent individual lookups
                for name in names_to_lookup:
                    if name not in self._player_teams_cache:
                        self._player_teams_cache[name] = None

        except Exception as e:
            logger.warning(f"Batch player team lookup failed: {e}")
            self._close_connection()

    def get_player_team(self, player_name: str) -> Optional[str]:
        """
        Get player's current team from database.

        Args:
            player_name: Player name to lookup

        Returns:
            Team abbreviation or None if not found
        """
        # Check cache first (including "not found" markers which are None)
        if player_name in self._player_teams_cache:
            return self._player_teams_cache[player_name]  # May be None if not found

        try:
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                # Try player_profile first (column is 'full_name')
                cur.execute(
                    """
                    SELECT team_abbrev FROM player_profile
                    WHERE full_name = %s
                    LIMIT 1
                """,
                    (player_name,),
                )
                row = cur.fetchone()

                if row and row[0]:
                    team = normalize_team_abbrev(row[0])
                    self._player_teams_cache[player_name] = team
                    return team

                # Fallback to most recent game log (join to player_profile for name)
                cur.execute(
                    """
                    SELECT pgl.team_abbrev
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE pp.full_name = %s
                    ORDER BY pgl.game_date DESC
                    LIMIT 1
                """,
                    (player_name,),
                )
                row = cur.fetchone()

                if row and row[0]:
                    team = normalize_team_abbrev(row[0])
                    self._player_teams_cache[player_name] = team
                    return team

        except Exception as e:
            logger.warning(f"Failed to lookup team for {player_name}: {e}")
            # Reset connection on error
            self._close_connection()

        return None

    def enrich_props(self, props: List[Dict]) -> List[Dict]:
        """
        Enrich props with opponent_team and is_home fields.

        Args:
            props: List of prop dicts with at least 'player_name' and 'game_date'

        Returns:
            Props list with 'opponent_team' and 'is_home' added
        """
        if not props:
            return props

        enriched_count = 0
        total_count = len(props)

        # Group by game_date for efficient schedule fetching
        dates = set(p.get("game_date") for p in props if p.get("game_date"))

        # Pre-fetch schedules for all dates
        if self.verbose:
            logger.info(f"Fetching schedules for {len(dates)} date(s)...")
        for date_str in dates:
            if isinstance(date_str, datetime):
                date_str = date_str.strftime("%Y-%m-%d")
            self.fetch_schedule(date_str)

        # Collect unique players to lookup (more efficient than per-prop)
        unique_players = list(set(p.get("player_name") for p in props if p.get("player_name")))
        if self.verbose:
            logger.info(f"Batch loading teams for {len(unique_players)} unique players...")

        # Batch load all player teams in a single query (much faster)
        self.batch_load_player_teams(unique_players)

        if self.verbose:
            cached = len([p for p in unique_players if p in self._player_teams_cache])
            logger.info(
                f"Player team lookup complete: {cached}/{len(unique_players)} found. Enriching {total_count} props..."
            )

        for prop in props:
            player_name = prop.get("player_name")
            game_date = prop.get("game_date")

            if not player_name or not game_date:
                continue

            # Normalize date format
            if isinstance(game_date, datetime):
                game_date = game_date.strftime("%Y-%m-%d")

            # Get player's team - prefer prop's player_team, fallback to database lookup
            player_team = prop.get("player_team")
            if player_team:
                player_team = normalize_team_abbrev(player_team)

            # Fallback to database if prop doesn't have team
            if not player_team:
                player_team = self.get_player_team(player_name)

            if not player_team:
                if self.verbose:
                    logger.debug(f"No team found for {player_name}")
                continue

            # Get schedule for this date
            schedule = self._schedule_cache.get(game_date, {})

            if player_team in schedule:
                game_info = schedule[player_team]
                prop["opponent_team"] = game_info["opponent"]
                prop["is_home"] = game_info["is_home"]
                enriched_count += 1
            else:
                if self.verbose:
                    logger.debug(f"No game found for {player_name} ({player_team}) on {game_date}")

        # Close connection when done
        self._close_connection()

        if self.verbose:
            pct = (enriched_count / total_count * 100) if total_count > 0 else 0
            logger.info(
                f"Enriched {enriched_count}/{total_count} props ({pct:.1f}%) with opponent data"
            )

        return props


def main():
    """Test the OpponentMapper"""
    import argparse

    parser = argparse.ArgumentParser(description="Test OpponentMapper")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Game date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    mapper = OpponentMapper(verbose=True)

    # Test schedule fetch
    print(f"\n=== Schedule for {args.date} ===")
    schedule = mapper.fetch_schedule(args.date)

    if schedule:
        seen_games = set()
        for team, info in schedule.items():
            game_key = tuple(sorted([team, info["opponent"]]))
            if game_key not in seen_games:
                if info["is_home"]:
                    print(f"  {info['opponent']} @ {team}")
                seen_games.add(game_key)
    else:
        print("  No games found")

    # Test with sample props
    print(f"\n=== Testing prop enrichment ===")
    sample_props = [
        {"player_name": "LeBron James", "game_date": args.date, "stat_type": "POINTS"},
        {"player_name": "Stephen Curry", "game_date": args.date, "stat_type": "THREES"},
        {"player_name": "Nikola Jokic", "game_date": args.date, "stat_type": "REBOUNDS"},
    ]

    enriched = mapper.enrich_props(sample_props)
    for prop in enriched:
        print(
            f"  {prop['player_name']}: opponent={prop.get('opponent_team')}, is_home={prop.get('is_home')}"
        )


if __name__ == "__main__":
    main()
