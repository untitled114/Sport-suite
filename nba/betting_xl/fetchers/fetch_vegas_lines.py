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

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5537,
    "database": "nba_games",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

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

    def fetch_all_lines(self) -> List[Dict]:
        """
        Fetch all vegas lines for all games on the date.

        Returns:
            List of game line dictionaries
        """
        if not self.events:
            self.fetch_events()

        if not self.events:
            logger.warning("No events found")
            return []

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
