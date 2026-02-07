#!/usr/bin/env python3
"""
BettingPros Vegas Lines Fetcher for NBA XL System
==================================================
Fetches game-level vegas lines (spread, total, moneyline) from BettingPros API.

This fetcher uses:
- /v3/events: Get list of games for a date
- /v3/offers: Get spread (market_id=129) and total (market_id=128) for each game

Usage:
    python fetch_vegas_lines.py --date 2026-01-07
    python fetch_vegas_lines.py --date 2026-01-07 --save-to-db
    python fetch_vegas_lines.py --backfill --start 2025-10-22 --end 2026-01-07
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# BettingPros API
BETTINGPROS_BASE_URL = "https://api.bettingpros.com/v3"
PREMIUM_HEADERS = {
    "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
    "x-level": "cHJlbWl1bQ==",
    "accept": "application/json",
}

# Market IDs for game-level bets
MARKET_IDS = {
    "spread": 129,
    "total": 128,
    "moneyline": 127,
}

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nba.config.database import get_games_db_config

# Database config (centralized)
DB_CONFIG = get_games_db_config()

# Team abbreviation mapping (API -> Database)
# BettingPros API uses different abbreviations than our database
TEAM_ABBREV_MAP = {
    "NOR": "NOP",  # New Orleans Pelicans
    "UTH": "UTA",  # Utah Jazz
    "PHO": "PHX",  # Phoenix Suns
    "BRK": "BKN",  # Brooklyn Nets
    "SA": "SAS",  # San Antonio Spurs
    "GS": "GSW",  # Golden State Warriors
    "NY": "NYK",  # New York Knicks
    "NO": "NOP",  # New Orleans (alternate)
}

# ESPN uses slightly different abbreviations
ESPN_TEAM_MAP = {
    "WSH": "WAS",
    "SA": "SAS",
    "GS": "GSW",
    "UTAH": "UTA",
    "NY": "NYK",
    "NO": "NOP",
    "PHO": "PHX",
    "BRK": "BKN",
    "NOR": "NOP",
}

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# The Odds API (premium key for historical, fallback for live)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com"

# The Odds API uses full team names — map to our DB abbreviations
ODDS_API_TEAM_MAP = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


class VegasLinesFetcher:
    """Fetches game-level vegas lines from BettingPros API."""

    def __init__(self, date: str = None, rate_limit: float = 2.5):
        """
        Initialize fetcher.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            rate_limit: Seconds between API requests.
        """
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.rate_limit = rate_limit
        self.events = []

    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting."""
        try:
            response = requests.get(url, headers=PREMIUM_HEADERS, params=params, timeout=30)

            if response.status_code == 403:
                logger.warning(f"403 Forbidden - API rate limited or key issue")
                return None

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" not in content_type:
                logger.warning(f"Invalid content type: {content_type}")
                return None

            return response.json()

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Request failed: {e}")
            return None

    def fetch_events(self) -> List[Dict]:
        """
        Fetch NBA events (games) for the date.

        Returns:
            List of event dictionaries with id, home, visitor, etc.
        """
        logger.info(f"Fetching events for {self.date}...")

        data = self._make_request(
            f"{BETTINGPROS_BASE_URL}/events", {"sport": "NBA", "date": self.date, "limit": 50}
        )

        if not data:
            return []

        events = data.get("events", [])
        logger.info(f"Found {len(events)} games")

        self.events = events
        return events

    def fetch_game_lines(self, event_id: int, market_type: str = "spread") -> Optional[Dict]:
        """
        Fetch lines for a specific game and market.

        Args:
            event_id: BettingPros event ID
            market_type: 'spread', 'total', or 'moneyline'

        Returns:
            Dict with consensus line, opening line, and book-specific lines
        """
        market_id = MARKET_IDS.get(market_type)
        if not market_id:
            logger.error(f"Unknown market type: {market_type}")
            return None

        data = self._make_request(
            f"{BETTINGPROS_BASE_URL}/offers",
            {"sport": "NBA", "market_id": market_id, "event_id": event_id},
        )

        if not data or not data.get("offers"):
            return None

        offer = data["offers"][0]
        selections = offer.get("selections", [])

        if market_type == "total":
            # Total has over/under selections
            return self._parse_total(offer, selections)
        else:
            # Spread and moneyline have team selections
            return self._parse_spread_or_ml(offer, selections, market_type)

    def _parse_spread_or_ml(self, offer: Dict, selections: List, market_type: str) -> Dict:
        """Parse spread or moneyline data."""
        result = {
            "event_id": offer.get("event_id"),
            "market_type": market_type,
            "home_team": None,
            "away_team": None,
            "consensus_line": None,
            "opening_line": None,
            "best_line": None,
            "books": [],
        }

        participants = offer.get("participants", [])
        if len(participants) >= 2:
            # participants[0] = home (DET), participants[1] = away (CHI) based on API structure
            # But actually looking at the data, it seems reversed - let's check team abbreviations
            result["home_team"] = participants[0].get("id") or participants[0].get("team", {}).get(
                "abbreviation"
            )
            result["away_team"] = participants[1].get("id") or participants[1].get("team", {}).get(
                "abbreviation"
            )

        for selection in selections:
            team = selection.get("participant")
            opening = selection.get("opening_line", {})
            books = selection.get("books", [])

            # Find consensus (book_id=0)
            for book in books:
                if book.get("id") == 0:
                    lines = book.get("lines", [])
                    if lines:
                        consensus = lines[0]
                        result["consensus_line"] = consensus.get("line")
                        break

            # Store opening line
            if opening and result["opening_line"] is None:
                result["opening_line"] = opening.get("line")

            # Find best line
            for book in books:
                lines = book.get("lines", [])
                for line in lines:
                    if line.get("best"):
                        result["best_line"] = line.get("line")
                        result["best_book_id"] = book.get("id")
                        break

            # Collect book-specific lines
            for book in books:
                book_id = book.get("id")
                if book_id == 0:  # Skip consensus
                    continue
                lines = book.get("lines", [])
                if lines:
                    line_data = lines[0]
                    result["books"].append(
                        {
                            "book_id": book_id,
                            "line": line_data.get("line"),
                            "odds": line_data.get("cost"),
                            "updated": line_data.get("updated"),
                        }
                    )

        return result

    def _parse_total(self, offer: Dict, selections: List) -> Dict:
        """Parse total (O/U) data."""
        result = {
            "event_id": offer.get("event_id"),
            "market_type": "total",
            "consensus_total": None,
            "opening_total": None,
            "best_total": None,
            "books": [],
        }

        participants = offer.get("participants", [])
        if len(participants) >= 2:
            result["home_team"] = participants[0].get("id") or participants[0].get("team", {}).get(
                "abbreviation"
            )
            result["away_team"] = participants[1].get("id") or participants[1].get("team", {}).get(
                "abbreviation"
            )

        # For totals, selections are "over" and "under" - we just need the line value
        for selection in selections:
            books = selection.get("books", [])
            opening = selection.get("opening_line", {})

            # Find consensus
            for book in books:
                if book.get("id") == 0:
                    lines = book.get("lines", [])
                    if lines:
                        line_val = abs(lines[0].get("line", 0))
                        if line_val > 100:  # NBA totals are typically 200-250
                            result["consensus_total"] = line_val
                            break

            # Store opening
            if opening and result["opening_total"] is None:
                opening_val = abs(opening.get("line", 0))
                if opening_val > 100:
                    result["opening_total"] = opening_val

            # Find best
            for book in books:
                lines = book.get("lines", [])
                for line in lines:
                    if line.get("best"):
                        line_val = abs(line.get("line", 0))
                        if line_val > 100:
                            result["best_total"] = line_val
                            result["best_book_id"] = book.get("id")
                            break

            # Only process once (over/under have same line)
            if result["consensus_total"]:
                break

        return result

    def fetch_from_odds_api(self, historical: bool = False) -> List[Dict]:
        """
        Fetch vegas lines from The Odds API (premium).
        Supports both live and historical endpoints.

        Args:
            historical: If True, use /v4/historical/ endpoint for past dates.

        Returns:
            List of game line dictionaries (same format as fetch_all_lines)
        """
        if not ODDS_API_KEY:
            logger.warning("ODDS_API_KEY not set, skipping Odds API")
            return []

        if historical:
            url = f"{ODDS_API_BASE}/v4/historical/sports/basketball_nba/odds"
            # Historical endpoint wants a UTC timestamp
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads,totals",
                "date": f"{self.date}T18:00:00Z",
            }
        else:
            url = f"{ODDS_API_BASE}/v4/sports/basketball_nba/odds"
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american",
            }

        logger.info(
            f"Fetching from Odds API ({'historical' if historical else 'live'}) for {self.date}..."
        )

        try:
            response = requests.get(url, params=params, timeout=30)
            remaining = response.headers.get("x-requests-remaining", "?")
            logger.info(f"Odds API credits remaining: {remaining}")

            if response.status_code == 401:
                logger.warning("Odds API: unauthorized or plan limitation")
                return []
            response.raise_for_status()
            raw = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Odds API failed: {e}")
            return []

        # Historical wraps data in {"data": [...], "timestamp": ...}
        games = raw.get("data", raw) if historical else raw
        if not isinstance(games, list):
            games = []

        if not games:
            logger.warning("Odds API returned no games")
            return []

        logger.info(f"Odds API: Found {len(games)} games")
        all_lines = []

        for game in games:
            home_full = game.get("home_team", "")
            away_full = game.get("away_team", "")
            home_team = ODDS_API_TEAM_MAP.get(home_full, home_full)
            away_team = ODDS_API_TEAM_MAP.get(away_full, away_full)

            # For live endpoint, filter to today's games only
            if not historical:
                commence = game.get("commence_time", "")
                if commence:
                    try:
                        utc_time = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                        est_time = utc_time - timedelta(hours=5)
                        game_date = est_time.strftime("%Y-%m-%d")
                        if game_date != self.date:
                            continue
                    except (ValueError, TypeError):
                        pass

            spread_val = None
            total_val = None

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_full:
                                try:
                                    spread_val = float(outcome.get("point", 0))
                                except (TypeError, ValueError):
                                    pass
                                break
                    elif market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                try:
                                    total_val = float(outcome.get("point", 0))
                                except (TypeError, ValueError):
                                    pass
                                break
                # Use first bookmaker with data
                if spread_val is not None or total_val is not None:
                    break

            game_lines = {
                "event_id": game.get("id"),
                "game_date": self.date,
                "home_team": home_team,
                "away_team": away_team,
                "scheduled": game.get("commence_time"),
                "status": None,
                "spread_consensus": spread_val,
                "spread_opening": None,
                "spread_best": None,
                "spread_books": [],
                "total_consensus": total_val,
                "total_opening": None,
                "total_best": None,
                "total_books": [],
                "fetch_timestamp": datetime.now().isoformat(),
                "source": "odds_api",
            }

            all_lines.append(game_lines)

            spread_str = f"{spread_val}" if spread_val is not None else "N/A"
            total_str = f"{total_val}" if total_val is not None else "N/A"
            logger.info(f"  {away_team} @ {home_team}: Spread={spread_str}, Total={total_str}")

        logger.info(f"Odds API: {len(all_lines)} games with lines")
        return all_lines

    def fetch_from_espn(self) -> List[Dict]:
        """
        Fetch vegas lines from ESPN scoreboard API (free, no key required).
        Used as fallback when BettingPros returns 403.

        Returns:
            List of game line dictionaries (same format as fetch_all_lines)
        """
        date_param = self.date.replace("-", "")
        url = f"{ESPN_SCOREBOARD_URL}?dates={date_param}"
        logger.info(f"Fetching from ESPN fallback for {self.date}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error(f"ESPN fallback failed: {e}")
            return []

        events = data.get("events", [])
        if not events:
            logger.warning("ESPN returned no events")
            return []

        logger.info(f"ESPN: Found {len(events)} games")
        all_lines = []

        for event in events:
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            # competitors: homeAway == "home" or "away"
            home_team = None
            away_team = None
            for team_entry in competitors:
                abbrev = team_entry.get("team", {}).get("abbreviation", "")
                abbrev = ESPN_TEAM_MAP.get(abbrev, abbrev)
                if team_entry.get("homeAway") == "home":
                    home_team = abbrev
                else:
                    away_team = abbrev

            if not home_team or not away_team:
                continue

            # Extract odds
            spread_val = None
            total_val = None
            odds_list = comp.get("odds", [])
            if odds_list:
                odds = odds_list[0]
                # spread is from home team perspective (negative = home favored)
                try:
                    spread_val = float(odds.get("spread", 0)) or None
                except (TypeError, ValueError):
                    spread_val = None
                try:
                    total_val = float(odds.get("overUnder", 0)) or None
                except (TypeError, ValueError):
                    total_val = None

            game_lines = {
                "event_id": event.get("id"),
                "game_date": self.date,
                "home_team": home_team,
                "away_team": away_team,
                "scheduled": event.get("date"),
                "status": event.get("status", {}).get("type", {}).get("name"),
                "spread_consensus": spread_val,
                "spread_opening": None,
                "spread_best": None,
                "spread_books": [],
                "total_consensus": total_val,
                "total_opening": None,
                "total_best": None,
                "total_books": [],
                "fetch_timestamp": datetime.now().isoformat(),
                "source": "espn",
            }

            all_lines.append(game_lines)

            spread_str = f"{spread_val}" if spread_val else "N/A"
            total_str = f"{total_val}" if total_val else "N/A"
            logger.info(f"  {away_team} @ {home_team}: Spread={spread_str}, Total={total_str}")

        logger.info(f"ESPN fallback: {len(all_lines)} games with lines")
        return all_lines

    def fetch_all_lines(self) -> List[Dict]:
        """
        Fetch all vegas lines for all games on the date.
        Fallback chain: BettingPros -> Odds API (live) -> ESPN.
        For historical dates, uses Odds API historical endpoint.

        Returns:
            List of game line dictionaries
        """
        if not self.events:
            self.fetch_events()

        if not self.events:
            # Check if this is a past date (use historical endpoint)
            try:
                target = datetime.strptime(self.date, "%Y-%m-%d").date()
                is_historical = target < datetime.now().date()
            except ValueError:
                is_historical = False

            logger.info("BettingPros unavailable, trying Odds API fallback...")
            lines = self.fetch_from_odds_api(historical=is_historical)
            if lines:
                return lines

            logger.info("Odds API unavailable, trying ESPN fallback...")
            return self.fetch_from_espn()

        all_lines = []

        for event in self.events:
            event_id = event.get("id")
            # Normalize team abbreviations to match database
            home_team = TEAM_ABBREV_MAP.get(event.get("home"), event.get("home"))
            visitor = TEAM_ABBREV_MAP.get(event.get("visitor"), event.get("visitor"))

            logger.info(f"Fetching lines for {visitor} @ {home_team} (event_id={event_id})...")

            # Fetch spread
            time.sleep(self.rate_limit)
            spread_data = self.fetch_game_lines(event_id, "spread")

            # Fetch total
            time.sleep(self.rate_limit)
            total_data = self.fetch_game_lines(event_id, "total")

            # Combine into single record
            game_lines = {
                "event_id": event_id,
                "game_date": self.date,
                "home_team": home_team,
                "away_team": visitor,
                "scheduled": event.get("scheduled"),
                "status": event.get("status"),
                # Spread data
                "spread_consensus": spread_data.get("consensus_line") if spread_data else None,
                "spread_opening": spread_data.get("opening_line") if spread_data else None,
                "spread_best": spread_data.get("best_line") if spread_data else None,
                "spread_books": spread_data.get("books", []) if spread_data else [],
                # Total data
                "total_consensus": total_data.get("consensus_total") if total_data else None,
                "total_opening": total_data.get("opening_total") if total_data else None,
                "total_best": total_data.get("best_total") if total_data else None,
                "total_books": total_data.get("books", []) if total_data else [],
                # Metadata
                "fetch_timestamp": datetime.now().isoformat(),
            }

            all_lines.append(game_lines)

            # Log summary
            spread_str = (
                f"{game_lines['spread_consensus']}" if game_lines["spread_consensus"] else "N/A"
            )
            total_str = (
                f"{game_lines['total_consensus']}" if game_lines["total_consensus"] else "N/A"
            )
            logger.info(f"  Spread: {spread_str}, Total: {total_str}")

        return all_lines

    def save_to_json(self, lines: List[Dict]) -> str:
        """Save lines to JSON file."""
        output_dir = Path(__file__).parent.parent / "lines"
        output_dir.mkdir(exist_ok=True)

        filename = f"vegas_lines_{self.date}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {"date": self.date, "fetch_timestamp": datetime.now().isoformat(), "games": lines},
                f,
                indent=2,
            )

        logger.info(f"Saved to {filepath}")
        return str(filepath)

    def save_to_database(self, lines: List[Dict]) -> int:
        """
        Save vegas lines to games table.

        Uses UPDATE-then-INSERT pattern:
        1. Try to UPDATE existing game record
        2. If no match, INSERT new record with generated game_id

        Returns:
            Number of games saved
        """
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        updated = 0
        inserted = 0

        for game in lines:
            if not game["spread_consensus"] and not game["total_consensus"]:
                continue

            try:
                # Try UPDATE first (for existing games)
                cursor.execute(
                    """
                    UPDATE games
                    SET vegas_spread = %s, vegas_total = %s
                    WHERE game_date = %s
                      AND home_team = %s AND away_team = %s
                """,
                    (
                        game["spread_consensus"],
                        game["total_consensus"],
                        game["game_date"],
                        game["home_team"],
                        game["away_team"],
                    ),
                )

                if cursor.rowcount > 0:
                    updated += 1
                    logger.info(
                        f"Updated {game['away_team']} @ {game['home_team']}: "
                        f"spread={game['spread_consensus']}, total={game['total_consensus']}"
                    )
                    conn.commit()
                else:
                    # INSERT new game with generated game_id
                    # Format: VEG + YYYYMMDD + home + away (e.g., VEG20260107DETCHI)
                    date_part = game["game_date"].replace("-", "")
                    game_id = f"VEG{date_part}{game['home_team']}{game['away_team']}"

                    # Calculate NBA season (Oct-Dec = next year's season)
                    game_dt = datetime.strptime(game["game_date"], "%Y-%m-%d")
                    season = game_dt.year + 1 if game_dt.month >= 10 else game_dt.year

                    cursor.execute(
                        """
                        INSERT INTO games (game_id, game_date, season, home_team, away_team, vegas_spread, vegas_total)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            game_id,
                            game["game_date"],
                            season,
                            game["home_team"],
                            game["away_team"],
                            game["spread_consensus"],
                            game["total_consensus"],
                        ),
                    )
                    inserted += 1
                    logger.info(
                        f"Inserted {game['away_team']} @ {game['home_team']}: "
                        f"spread={game['spread_consensus']}, total={game['total_consensus']}"
                    )
                    conn.commit()

            except (requests.RequestException, KeyError, ValueError, TypeError) as e:
                conn.rollback()
                logger.warning(f"Failed {game['away_team']} @ {game['home_team']}: {e}")
        cursor.close()
        conn.close()

        logger.info(f"Database: {updated} updated, {inserted} inserted")
        return updated + inserted


def backfill_date_range(start_date: str, end_date: str, save_to_db: bool = False):
    """
    Backfill vegas lines for a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        save_to_db: Whether to update database
    """
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_games = 0
    total_updated = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {date_str}")
        logger.info(f"{'='*60}")

        fetcher = VegasLinesFetcher(date=date_str)
        lines = fetcher.fetch_all_lines()

        if lines:
            fetcher.save_to_json(lines)
            total_games += len(lines)

            if save_to_db:
                updated = fetcher.save_to_database(lines)
                total_updated += updated

        current += timedelta(days=1)

        # Extra delay between days
        time.sleep(5)

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total games fetched: {total_games}")
    if save_to_db:
        logger.info(f"Total games updated in DB: {total_updated}")


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA vegas lines from BettingPros")
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD)")
    parser.add_argument("--save-to-db", action="store_true", help="Update database")
    parser.add_argument("--backfill", action="store_true", help="Backfill date range")
    parser.add_argument("--start", type=str, help="Start date for backfill")
    parser.add_argument("--end", type=str, help="End date for backfill")

    args = parser.parse_args()

    if args.backfill:
        if not args.start or not args.end:
            print("Error: --backfill requires --start and --end dates")
            sys.exit(1)
        backfill_date_range(args.start, args.end, args.save_to_db)
    else:
        fetcher = VegasLinesFetcher(date=args.date)
        lines = fetcher.fetch_all_lines()

        if lines:
            fetcher.save_to_json(lines)

            if args.save_to_db:
                fetcher.save_to_database(lines)

            # Print summary
            print(f"\n{'='*60}")
            print(f"VEGAS LINES SUMMARY - {fetcher.date}")
            print(f"{'='*60}")
            print(f"{'Game':<25} {'Spread':>10} {'Total':>10}")
            print(f"{'-'*60}")
            for game in lines:
                matchup = f"{game['away_team']} @ {game['home_team']}"
                spread = game["spread_consensus"] or "N/A"
                total = game["total_consensus"] or "N/A"
                print(f"{matchup:<25} {spread:>10} {total:>10}")
            print(f"{'='*60}")
        else:
            print("No games found for this date")


if __name__ == "__main__":
    main()
