#!/usr/bin/env python3
"""
Enrich nba_props_xl with Game Context
======================================
Fetches today's NBA games from BettingPros /v3/offers endpoint and enriches
props with opponent_team and is_home fields.

THIS SCRIPT IS CRITICAL FOR PREDICTION ACCURACY. Without opponent_team and is_home,
the feature extractor falls back to league averages instead of matchup-specific stats,
reducing prediction power by 20-30%.

Strategy:
1. Fetch today's games (away_team @ home_team) from BettingPros
2. Query player_profile to map player_name -> team_abbrev
3. For each player, find their game and determine:
   - opponent_team (the other team in their game)
   - is_home (True if player's team == home_team)
4. Fallback: If API fails or coverage insufficient, enrich from player_game_logs
5. UPDATE nba_props_xl SET opponent_team, is_home

Usage:
    python3 enrich_props_with_matchups.py --date 2025-11-07
    python3 enrich_props_with_matchups.py --date 2025-11-07 --dry-run

Exit codes:
    0 = Success (95%+ coverage achieved)
    1 = Failed (< 95% coverage - cannot guarantee prediction quality)
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import psycopg2
import requests

from nba.betting_xl.utils.logging_config import add_logging_args, get_logger, setup_logging
from nba.utils.team_utils import normalize_team_abbrev

# Logger will be configured in main()
logger = get_logger(__name__)


def get_current_season():
    """
    Calculate current NBA season based on date.
    NBA season uses END year (2025-26 season = 2026).
    Season starts in October, so Oct-Dec uses next year's number.
    """
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year


# Database configs - Use environment variables with fallback defaults
# This allows the script to work both standalone and when called from pipeline
DB_DEFAULT_USER = os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
DB_DEFAULT_PASSWORD = os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))

DB_INTELLIGENCE = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv("NBA_INT_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_INT_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}

DB_PLAYERS = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv("NBA_PLAYERS_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_PLAYERS_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}

# BettingPros API - Use PREMIUM credentials that work with /v3/events
BETTINGPROS_BASE_URL = "https://api.bettingpros.com/v3"
BETTINGPROS_PREMIUM_HEADERS = {
    "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
    "x-level": "cHJlbWl1bQ==",  # base64 for "premium"
    "accept": "application/json",
}

# Coverage threshold for production readiness
COVERAGE_THRESHOLD = 95.0


class PropsMatchupEnricher:
    """Enriches props with game matchup data"""

    def __init__(self, game_date: str):
        self.game_date = game_date
        self.conn_intelligence = None
        self.conn_players = None
        self.games_map = {}  # {team_abbrev: {opponent, is_home}}
        self.player_teams = {}  # {player_name: team_abbrev}

    def connect(self):
        """Connect to databases"""
        self.conn_intelligence = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        logger.info("[OK] Connected to databases")

    def close(self):
        """Close database connections"""
        if self.conn_intelligence:
            self.conn_intelligence.close()
        if self.conn_players:
            self.conn_players.close()

    def fetch_todays_games(self) -> Dict[str, Dict]:
        """
        Fetch today's NBA games from BettingPros /v3/events endpoint.

        Returns:
            Dict mapping team_abbrev -> {opponent, is_home}
            Example: {'WAS': {'opponent': 'CLE', 'is_home': False},
                     'CLE': {'opponent': 'WAS', 'is_home': True}}
        """
        logger.info(f"Fetching today's games for {self.game_date}...")

        # Use /v3/events endpoint to get today's games
        url = f"{BETTINGPROS_BASE_URL}/events"
        params = {"sport": "NBA", "date": self.game_date, "limit": "50"}  # Fetch up to 50 games

        try:
            response = requests.get(
                url, headers=BETTINGPROS_PREMIUM_HEADERS, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            games_map = {}

            # Check if response has 'events' key (v3/events endpoint)
            if "events" in data:
                for event in data["events"]:
                    participants = event.get("participants", [])
                    if len(participants) < 2:
                        continue

                    # participants[0] = away team, participants[1] = home team
                    away_team = participants[0].get("abbreviation") or participants[0].get(
                        "team", {}
                    ).get("abbreviation")
                    home_team = participants[1].get("abbreviation") or participants[1].get(
                        "team", {}
                    ).get("abbreviation")

                    if not away_team or not home_team:
                        continue

                    # Map both teams
                    games_map[away_team] = {"opponent": home_team, "is_home": False}
                    games_map[home_team] = {"opponent": away_team, "is_home": True}

            # Or if response has 'offers' key (v3/offers endpoint)
            elif "offers" in data:
                for offer in data["offers"]:
                    participants = offer.get("participants", [])
                    if len(participants) < 2:
                        continue

                    # participants[0] = away team, participants[1] = home team
                    away_team = participants[0]["team"]["abbreviation"]
                    home_team = participants[1]["team"]["abbreviation"]

                    # Map both teams
                    games_map[away_team] = {"opponent": home_team, "is_home": False}
                    games_map[home_team] = {"opponent": away_team, "is_home": True}
            else:
                logger.error(f"Unexpected API response structure (no 'events' or 'offers')")
                return {}

            logger.info(f"[OK] Found {len(games_map) // 2} games ({len(games_map)} team entries)")

            # Log games
            games_list = []
            seen_games = set()
            for team, info in games_map.items():
                game_key = tuple(sorted([team, info["opponent"]]))
                if game_key not in seen_games:
                    if info["is_home"]:
                        games_list.append(f"{info['opponent']} @ {team}")
                    seen_games.add(game_key)

            for game in sorted(games_list):
                logger.info(f"  {game}")

            return games_map

        except Exception as e:
            logger.error(f"Failed to fetch games from BettingPros API: {e}")
            logger.info("Trying ESPN API fallback...")
            return self._fetch_games_from_espn()

    def _fetch_games_from_espn(self) -> Dict[str, Dict]:
        """
        Fetch today's NBA games from ESPN Scoreboard API (fallback).

        IMPORTANT: Uses scoreboard API with date filter to get SCHEDULED games,
        not header API which returns mixed completed/scheduled games.

        Returns:
            Dict mapping team_abbrev -> {opponent, is_home}
        """
        # Format date as YYYYMMDD for ESPN API
        date_param = self.game_date.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_param}"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            games_map = {}

            # Get events from scoreboard
            events = data.get("events", [])

            if not events:
                logger.error(f"ESPN Scoreboard API: No events for date {self.game_date}")
                return {}

            for event in events:
                # Parse from shortName format: "HOU @ MIL" or "MEM VS ORL" (neutral site games)
                short_name = event.get("shortName", "")

                # Handle both formats: "@" for regular games, "VS"/"vs." for neutral site games
                if " @ " in short_name:
                    parts = short_name.split(" @ ")
                    is_neutral_site = False
                elif " VS " in short_name.upper():
                    # Neutral site game (e.g., Berlin games) - use VS/vs./vs format
                    parts = re.split(r"\s+(?:VS\.?|vs\.?)\s+", short_name, flags=re.IGNORECASE)
                    is_neutral_site = True
                else:
                    continue

                if len(parts) != 2:
                    continue

                away_team = parts[0].strip()
                home_team = parts[1].strip()

                # For neutral site games, the format is still "AWAY VS HOME" per ESPN convention
                # Log neutral site games for visibility
                if is_neutral_site:
                    logger.info(f"  [Neutral site] {away_team} vs {home_team}")

                # Normalize team abbreviations using centralized utility
                away_team = normalize_team_abbrev(away_team)
                home_team = normalize_team_abbrev(home_team)

                games_map[away_team] = {"opponent": home_team, "is_home": False}
                games_map[home_team] = {"opponent": away_team, "is_home": True}

            logger.info(
                f"[OK] ESPN Scoreboard API: Found {len(games_map) // 2} games ({len(games_map)} team entries)"
            )

            # Log games
            games_list = []
            seen_games = set()
            for team, info in games_map.items():
                game_key = tuple(sorted([team, info["opponent"]]))
                if game_key not in seen_games:
                    if info["is_home"]:
                        games_list.append(f"{info['opponent']} @ {team}")
                    seen_games.add(game_key)

            for game in sorted(games_list):
                logger.info(f"  {game}")

            return games_map

        except Exception as e:
            logger.error(f"ESPN Scoreboard API failed: {e}")
            logger.warning("All APIs failed - enrichment will be incomplete")
            return {}

    def _load_games_from_cache(self) -> Dict[str, Dict]:
        """Load games from cached bp-games.json file (fallback)"""
        import json
        import os

        cache_file = os.path.join(os.path.dirname(__file__), "..", "..", "bp-games.json")

        if not os.path.exists(cache_file):
            logger.error(f"Cache file not found: {cache_file}")
            return {}

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            games_map = {}

            # Handle /v3/events format (simpler structure)
            if "events" in data:
                for event in data["events"]:
                    home_team = event.get("home")
                    away_team = event.get("visitor")

                    if not home_team or not away_team:
                        continue

                    games_map[away_team] = {"opponent": home_team, "is_home": False}
                    games_map[home_team] = {"opponent": away_team, "is_home": True}

            # Handle /v3/offers format (nested participants)
            elif "offers" in data:
                for offer in data["offers"]:
                    participants = offer.get("participants", [])
                    if len(participants) < 2:
                        continue

                    away_team = participants[0]["team"]["abbreviation"]
                    home_team = participants[1]["team"]["abbreviation"]

                    games_map[away_team] = {"opponent": home_team, "is_home": False}
                    games_map[home_team] = {"opponent": away_team, "is_home": True}

            logger.info(f"[OK] Loaded {len(games_map) // 2} games from cache")

            # Log games for verification
            games_list = []
            seen_games = set()
            for team, info in games_map.items():
                game_key = tuple(sorted([team, info["opponent"]]))
                if game_key not in seen_games:
                    if info["is_home"]:
                        games_list.append(f"{info['opponent']} @ {team}")
                    seen_games.add(game_key)

            for game in sorted(games_list):
                logger.info(f"  {game}")

            return games_map

        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return {}

    def normalize_name(self, name: str) -> str:
        """
        Normalize player name for matching.
        Handles variations like "Jr" vs "Jr.", "III" vs "III.", special characters, etc.

        Args:
            name: Player name

        Returns:
            Normalized name
        """
        if not name:
            return name

        import unicodedata

        # Remove periods from suffixes
        normalized = name.replace(" Jr.", " Jr").replace(" Sr.", " Sr")
        normalized = normalized.replace(" II.", " II").replace(" III.", " III")
        normalized = normalized.replace(" IV.", " IV").replace(" V.", " V")

        # Remove accents and special characters for better matching
        # Convert "Porziņģis" → "Porzingis", "Nurkić" → "Nurkic"
        normalized = "".join(
            c for c in unicodedata.normalize("NFD", normalized) if unicodedata.category(c) != "Mn"
        )

        return normalized.strip()

    def get_base_name(self, name: str) -> str:
        """
        Get base name without suffix for fuzzy matching.
        "Russell Westbrook III" → "Russell Westbrook"

        Args:
            name: Player name

        Returns:
            Base name without suffix
        """
        # Remove common suffixes
        for suffix in [
            " Jr",
            " Jr.",
            " Sr",
            " Sr.",
            " II",
            " II.",
            " III",
            " III.",
            " IV",
            " IV.",
            " V",
            " V.",
        ]:
            if name.endswith(suffix):
                return name[: -len(suffix)].strip()
        return name

    def load_player_teams(self):
        """Load player_name -> team_abbrev mapping from player_profile"""
        logger.info("Loading player teams from player_profile...")

        # Use DISTINCT ON to get one entry per player (handles duplicate player_ids)
        # Prefer entries WITH team_abbrev over NULL entries
        query = """
        SELECT DISTINCT ON (full_name) full_name, team_abbrev
        FROM player_profile
        WHERE team_abbrev IS NOT NULL AND team_abbrev != ''
        ORDER BY full_name, player_id
        """

        try:
            cur = self.conn_players.cursor()
            cur.execute(query)
            rows = cur.fetchall()

            for full_name, team_abbrev in rows:
                # Store multiple name variations for matching
                # 1. Original name
                self.player_teams[full_name] = team_abbrev

                # 2. Normalized name (Jr. → Jr, remove accents)
                normalized = self.normalize_name(full_name)
                if normalized != full_name:
                    self.player_teams[normalized] = team_abbrev

                # 3. Base name without suffix (for "Russell Westbrook" matching "Russell Westbrook III")
                base = self.get_base_name(normalized)
                if base != normalized:
                    self.player_teams[base] = team_abbrev

            logger.info(
                f"[OK] Loaded {len(rows)} players -> {len(self.player_teams)} name mappings (with normalization)"
            )

        except Exception as e:
            logger.error(f"Failed to load player teams: {e}")

    def enrich_from_game_logs(self) -> int:
        """
        Fallback: Enrich props from player_game_logs database.

        Returns:
            Number of props enriched
        """
        logger.info("\n🔄 Fallback: Enriching from player_game_logs database...")

        # Find props still missing matchup data
        query = """
        SELECT DISTINCT player_name
        FROM nba_props_xl
        WHERE game_date = %s
          AND (opponent_team IS NULL OR opponent_team = '' OR is_home IS NULL)
        """

        cur = self.conn_intelligence.cursor()
        cur.execute(query, (self.game_date,))
        players_needing_enrichment = [row[0] for row in cur.fetchall()]

        if len(players_needing_enrichment) == 0:
            logger.info("  No props need enrichment from game logs")
            return 0

        logger.info(f"  Found {len(players_needing_enrichment)} players still needing enrichment")

        # Enrich from player_game_logs (JOIN with player_profile to match by name)
        enriched_count = 0

        cur_players = self.conn_players.cursor()

        for player_name in players_needing_enrichment:
            # Query player_game_logs for recent games to determine current team
            # Note: We can't query today's game (hasn't been played yet!)
            # Instead, query most recent games to find which team the player plays for
            normalized_name = self.normalize_name(player_name)
            base_name = self.get_base_name(normalized_name)

            # First, get player's current team from recent game logs
            current_season = get_current_season()
            team_query = """
            SELECT pgl.team_abbrev
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE pgl.season = %s
              AND (
                pp.full_name = %s
                OR REPLACE(REPLACE(pp.full_name, ' Jr.', ' Jr'), ' III.', ' III') = %s
                OR REPLACE(REPLACE(pp.full_name, ' Jr.', ''), ' III.', '') = %s
              )
            ORDER BY pgl.game_date DESC
            LIMIT 1
            """

            cur_players.execute(
                team_query, (current_season, player_name, normalized_name, base_name)
            )
            team_result = cur_players.fetchone()

            if not team_result or not team_result[0]:
                continue  # No recent games found

            player_team = team_result[0]

            # Normalize team abbreviation using centralized utility
            player_team = normalize_team_abbrev(player_team)

            # Now check if this team has a game today
            if player_team not in self.games_map:
                continue  # Player's team doesn't have a game today

            # Get opponent and home/away from games_map
            game_info = self.games_map[player_team]
            opponent_abbrev = game_info["opponent"]
            is_home = game_info["is_home"]

            # Update props
            update_query = """
            UPDATE nba_props_xl
            SET opponent_team = %s,
                is_home = %s
            WHERE player_name = %s
              AND game_date = %s
              AND (opponent_team IS NULL OR opponent_team = '' OR is_home IS NULL)
            """

            cur.execute(update_query, (opponent_abbrev, is_home, player_name, self.game_date))
            enriched_count += 1

        self.conn_intelligence.commit()
        logger.info(f"  [OK] Enriched {enriched_count} players from player_game_logs")

        return enriched_count

    def enrich_props(self, dry_run: bool = False):
        """
        Enrich nba_props_xl with opponent_team and is_home.

        Args:
            dry_run: If True, only show what would be updated
        """
        logger.info(f"\nEnriching props for {self.game_date}...")

        # Query props needing enrichment
        query = """
        SELECT DISTINCT player_name
        FROM nba_props_xl
        WHERE game_date = %s
          AND (opponent_team IS NULL OR opponent_team = '' OR is_home IS NULL)
        """

        cur = self.conn_intelligence.cursor()
        cur.execute(query, (self.game_date,))
        players_needing_enrichment = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(players_needing_enrichment)} players needing enrichment")

        # Enrich each player
        enriched_count = 0
        missing_team_count = 0
        missing_game_count = 0

        for player_name in players_needing_enrichment:
            # Get player's team (try multiple variations)
            # 1. Try original name
            player_team = self.player_teams.get(player_name)

            # 2. Try normalized name (remove periods, accents)
            if not player_team:
                normalized_name = self.normalize_name(player_name)
                player_team = self.player_teams.get(normalized_name)

            # 3. Try base name without suffix (for "Russell Westbrook III" → "Russell Westbrook")
            if not player_team:
                base_name = self.get_base_name(self.normalize_name(player_name))
                player_team = self.player_teams.get(base_name)

            if not player_team:
                missing_team_count += 1
                logger.debug(f"[WARN]  {player_name}: No team in player_profile")
                continue

            # Get player's game
            game_info = self.games_map.get(player_team)

            if not game_info:
                missing_game_count += 1
                logger.debug(f"[WARN]  {player_name} ({player_team}): No game today")
                continue

            opponent_team = game_info["opponent"]
            is_home = game_info["is_home"]

            if dry_run:
                logger.debug(
                    f"{player_name} ({player_team}): "
                    f"opponent={opponent_team}, is_home={is_home}"
                )
            else:
                # Update database
                update_query = """
                UPDATE nba_props_xl
                SET opponent_team = %s,
                    is_home = %s
                WHERE player_name = %s
                  AND game_date = %s
                """

                cur.execute(update_query, (opponent_team, is_home, player_name, self.game_date))

            enriched_count += 1

        if not dry_run:
            self.conn_intelligence.commit()
            logger.info(f"\n[OK] Enriched {enriched_count} players")
        else:
            logger.info(f"\n[OK] Would enrich {enriched_count} players (dry run)")

        if missing_team_count > 0:
            logger.warning(f"[WARN]  {missing_team_count} players have no team in player_profile")

        if missing_game_count > 0:
            logger.warning(f"[WARN]  {missing_game_count} teams have no game today")

    def verify_enrichment(self) -> bool:
        """
        Verify enrichment was successful.

        Returns:
            True if coverage >= threshold, False otherwise
        """
        logger.info("\n" + "=" * 80)
        logger.info("MATCHUP DATA COVERAGE VERIFICATION")
        logger.info("=" * 80)

        query = """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN opponent_team IS NOT NULL AND opponent_team != '' THEN 1 END) as has_opponent,
            COUNT(CASE WHEN is_home IS NOT NULL THEN 1 END) as has_is_home,
            COUNT(CASE WHEN opponent_team IS NOT NULL AND opponent_team != ''
                        AND is_home IS NOT NULL THEN 1 END) as fully_enriched
        FROM nba_props_xl
        WHERE game_date = %s
        """

        cur = self.conn_intelligence.cursor()
        cur.execute(query, (self.game_date,))
        row = cur.fetchone()

        total, has_opponent, has_is_home, fully_enriched = row

        opponent_coverage = (has_opponent / total * 100) if total > 0 else 0
        is_home_coverage = (has_is_home / total * 100) if total > 0 else 0
        full_coverage = (fully_enriched / total * 100) if total > 0 else 0

        logger.info(f"\nProps for {self.game_date}:")
        logger.info(f"  Total props:           {total}")
        logger.info(f"  With opponent_team:    {has_opponent:>4} ({opponent_coverage:>5.1f}%)")
        logger.info(f"  With is_home:          {has_is_home:>4} ({is_home_coverage:>5.1f}%)")
        logger.info(f"  Fully enriched:        {fully_enriched:>4} ({full_coverage:>5.1f}%)")

        logger.info(f"\nCoverage Threshold:    {COVERAGE_THRESHOLD}%")

        # Determine pass/fail
        # If no props exist for this date, consider it a pass (nothing to enrich)
        if total == 0:
            logger.info(f"\n[OK] NO PROPS FOR THIS DATE - nothing to enrich")
            logger.info("   This is normal for future dates without published lines")
            logger.info("=" * 80 + "\n")
            return True

        coverage_ok = full_coverage >= COVERAGE_THRESHOLD

        if coverage_ok:
            logger.info(
                f"\n[OK] COVERAGE CHECK PASSED ({full_coverage:.1f}% ≥ {COVERAGE_THRESHOLD}%)"
            )
            logger.info("   System ready for production predictions")
        else:
            logger.error(
                f"\n[ERROR] COVERAGE CHECK FAILED ({full_coverage:.1f}% < {COVERAGE_THRESHOLD}%)"
            )
            logger.error("   Insufficient matchup data - predictions will be degraded")
            logger.error(f"   Missing data for {total - fully_enriched} props")

            if total - has_opponent > 0:
                logger.error(f"   - {total - has_opponent} props missing opponent_team")
            if total - has_is_home > 0:
                logger.error(f"   - {total - has_is_home} props missing is_home")

        logger.info("=" * 80 + "\n")

        return coverage_ok

    def run(self, dry_run: bool = False) -> bool:
        """
        Main execution.

        Returns:
            True if coverage threshold met, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("NBA PROPS MATCHUP ENRICHMENT")
            logger.info("=" * 80)
            logger.info(f"Date: {self.game_date}")
            logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
            logger.info("=" * 80 + "\n")

            self.connect()

            if not dry_run:
                # Fetch today's games from API
                self.games_map = self.fetch_todays_games()

                if self.games_map:
                    # Load player teams
                    self.load_player_teams()

                    # Enrich props from API
                    self.enrich_props(dry_run=False)
                else:
                    logger.warning("[WARN]  No games found from API, skipping API enrichment")

                # Fallback: Enrich from player_game_logs if needed
                self.enrich_from_game_logs()

                # Verify coverage
                coverage_ok = self.verify_enrichment()

                if coverage_ok:
                    logger.info("[OK] Enrichment complete - system ready for predictions!")
                else:
                    logger.error("[ERROR] Enrichment failed to meet coverage threshold")
                    logger.error("   Cannot generate production-quality predictions")

                return coverage_ok

            else:
                # Dry run mode - just show what would be done
                self.games_map = self.fetch_todays_games()
                if self.games_map:
                    self.load_player_teams()
                    self.enrich_props(dry_run=True)
                logger.info("\n[OK] Dry run complete!")
                return True

        except Exception as e:
            logger.error(f"[ERROR] Enrichment failed with error: {e}")
            if self.conn_intelligence:
                self.conn_intelligence.rollback()
            raise

        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Enrich NBA props with game matchups",
        epilog="Exit codes: 0 = Success (≥95%% coverage), 1 = Failed (< 95%% coverage)",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Game date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be updated without making changes"
    )
    add_logging_args(parser)  # Adds --debug and --quiet flags

    args = parser.parse_args()

    # Setup unified logging
    setup_logging("enrich_matchups", debug=args.debug, quiet=args.quiet)
    logger.info(f"Enriching matchups for {args.date}")

    enricher = PropsMatchupEnricher(game_date=args.date)

    try:
        success = enricher.run(dry_run=args.dry_run)

        # Exit with appropriate code
        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
