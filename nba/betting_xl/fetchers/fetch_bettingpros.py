#!/usr/bin/env python3
"""
BettingPros Premium Multi-Book Fetcher for NBA XL System
==========================================================
Fetches props from BettingPros Premium /v3/props endpoint.

With Premium membership, we can fetch individual sportsbook lines using book_id parameter.
This enables true multi-book line shopping (the key to 60%+ win rates).

Supported sportsbooks (18 total):
- DraftKings, FanDuel, BetMGM, Caesars, bet365, BetRivers, ESPN Bet, and more

Usage:
    python fetch_bettingpros.py --books all          # Fetch all 9 priority books
    python fetch_bettingpros.py --books consensus    # Consensus only (old behavior)
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

# Configure logging
logger = logging.getLogger(__name__)


class BettingProsFetcher(BaseFetcher):
    """Fetches NBA props from BettingPros Premium API with multi-book support"""

    # BettingPros API market IDs
    # FIXED 2026-01-03: assists was 160 (STEALS), correct is 151
    MARKETS = {
        "points": 156,
        "rebounds": 157,
        "assists": 151,  # FIXED: Was 160 (steals)
        "threes": 162,
    }

    # Priority sportsbooks (9 active books for line shopping)
    # NOTE: bet365 (24) tested 2026-01-30 - API works but returns 0 props (they don't offer NBA player props)
    PRIORITY_BOOKS = {
        12: "draftkings",
        10: "fanduel",
        19: "betmgm",
        13: "caesars",
        14: "fanatics",  # Re-added 2026-01-30 - now working (117 props)
        18: "betrivers",
        33: "espnbet",
        36: "underdog",  # DFS site - typically softer lines
        37: "prizepicks",  # DFS site - added 2026-02-01, historical data back to Oct 2023
    }

    # Markets not supported by certain books (skip to avoid noise)
    BOOK_MARKET_EXCLUSIONS = {
        33: {"threes"},  # espnbet doesn't offer THREES
        36: {"threes"},  # underdog doesn't offer THREES
    }

    # Additional books (lower priority)
    ALL_BOOKS = {
        **PRIORITY_BOOKS,
        15: "sugarhouse",
        27: "partycasino",
        36: "hardrock",
        37: "unibet",
        38: "sisportsbook",
        39: "fliff",
        49: "betway",
        60: "superbook",
        63: "action247",
    }

    API_BASE_URL = "https://api.bettingpros.com/v3/props"

    # Premium authentication (from existing codebase)
    PREMIUM_HEADERS = {
        "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
        "x-level": "cHJlbWl1bQ==",  # base64 for "premium"
        "accept": "application/json",
    }

    def __init__(self, date: str = None, books: str = "priority", verbose: bool = True):
        """
        Initialize BettingPros Premium fetcher.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            books: Which books to fetch ('priority', 'all', 'consensus')
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name="bettingpros",
            rate_limit=2.5,  # 2.5 seconds between requests (prevents rate limiting)
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )

        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.books_mode = books

        # Select which books to fetch
        if books == "all":
            self.books_to_fetch = self.ALL_BOOKS
        elif books == "priority":
            self.books_to_fetch = self.PRIORITY_BOOKS
        elif books == "consensus":
            self.books_to_fetch = {}  # No book_id filter = consensus
        else:
            self.books_to_fetch = self.PRIORITY_BOOKS

    def fetch_market(
        self, market_name: str, market_id: int, book_id: int = None, book_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all props for a single market and optionally a specific sportsbook.

        Args:
            market_name: Market name (e.g., 'points')
            market_id: BettingPros market ID
            book_id: Optional sportsbook ID (for multi-book fetching)
            book_name: Optional book name (for tagging results)

        Returns:
            List of prop dictionaries
        """
        props = []
        page = 1

        while True:
            params = {
                "sport": "NBA",
                "date": self.date,
                "market_id": market_id,
                "limit": 500,  # Max per page
                "page": page,
            }

            # Add book_id filter if specified (Premium feature)
            if book_id is not None:
                params["book_id"] = book_id

            response = self._make_request(
                url=self.API_BASE_URL,
                method="GET",
                params=params,
                headers=self.PREMIUM_HEADERS,  # Use premium auth
            )

            if not response:
                break

            # Validate response before JSON parsing
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" not in content_type:
                logger.warning(
                    f"Invalid content-type for {market_name}: {content_type}. "
                    f"Response size: {len(response.content)} bytes. "
                    f"Likely HTML error page (rate limit or API issue). Skipping market."
                )
                if self.verbose:
                    print(
                        f"  [WARN]  Invalid response type (not JSON, got {content_type}). Skipping."
                    )
                break

            # Check for suspiciously small response (likely error page)
            if len(response.content) < 50:
                logger.warning(
                    f"Empty/tiny response for {market_name} book_id={book_id}: "
                    f"{len(response.content)} bytes. Skipping."
                )
                if self.verbose:
                    print(f"  [WARN]  Empty response ({len(response.content)} bytes). Skipping.")
                break

            try:
                data = response.json()
            except (requests.RequestException, KeyError, ValueError, TypeError) as e:
                # Log error without binary garbage - check if response is printable
                try:
                    preview = response.text[:100] if response.text else "(empty)"
                    # Only show preview if it's printable ASCII
                    if all(c.isprintable() or c.isspace() for c in preview):
                        response_preview = preview
                    else:
                        response_preview = f"(binary data, {len(response.content)} bytes)"
                except (AttributeError, TypeError, UnicodeDecodeError):
                    response_preview = f"(unreadable, {len(response.content)} bytes)"
                logger.error(
                    f"Failed to parse JSON for {market_name} (book_id={book_id}): {e}. "
                    f"Response: {response_preview}"
                )
                if self.verbose:
                    print(f"  [WARN] JSON parse error: {e}")
                break

            # Extract props from response
            page_props = data.get("props", [])
            if not page_props:
                break

            # Parse each prop
            for prop in page_props:
                parsed_prop = self._parse_prop(prop, market_name, book_name)
                if parsed_prop:
                    props.append(parsed_prop)

            if self.verbose:
                book_label = f" [{book_name}]" if book_name else ""
                print(
                    f"  {market_name}{book_label} Page {page}: +{len(page_props)} props", flush=True
                )

            # Check pagination
            pagination = data.get("_pagination", {})
            total_pages = pagination.get("total_pages", 1)

            if page >= total_pages:
                break

            page += 1

        return props

    def _parse_prop(
        self, raw_prop: Dict[str, Any], market_name: str, book_name: str = None
    ) -> Dict[str, Any]:
        """
        Parse raw prop from API response.

        Args:
            raw_prop: Raw prop data from API
            market_name: Market name
            book_name: Optional book name (if fetching specific book)

        Returns:
            Parsed prop dictionary or None if invalid
        """
        try:
            # Get player name
            player_name = raw_prop.get("participant", {}).get("name", "")
            if not player_name:
                return None

            # Normalize player name
            player_name = self.normalize_player_name(player_name)

            # Get over/under data
            over_data = raw_prop.get("over", {})
            under_data = raw_prop.get("under", {})

            # If book_name specified, use actual book line; otherwise use consensus
            if book_name:
                # Book-specific line (from premium book_id filter)
                over_line = over_data.get("line")
                over_odds = over_data.get("odds", -110)
                under_line = under_data.get("line")
                under_odds = under_data.get("odds", -110)

                # Lines should match for over/under on same prop
                line = over_line if over_line is not None else under_line

                # Also capture consensus for reference
                consensus_line = over_data.get("consensus_line")
                consensus_odds_over = over_data.get("consensus_odds")
                consensus_odds_under = under_data.get("consensus_odds")
            else:
                # Consensus mode (no book filter)
                consensus_line = over_data.get("consensus_line")
                consensus_odds_over = over_data.get("consensus_odds", -110)
                consensus_odds_under = under_data.get("consensus_odds", -110)

                line = consensus_line
                over_odds = consensus_odds_over
                under_odds = consensus_odds_under

            if line is None:
                return None

            # Get game info
            game_info = raw_prop.get("game", {})
            game_id = game_info.get("id", "")
            game_time = game_info.get("start", "")

            # Parse game date/time
            # IMPORTANT: Use API's actual game date, then filter by requested date later
            if game_time:
                try:
                    game_dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                    game_date = game_dt.strftime("%Y-%m-%d")  # API's actual game date
                    game_time_str = game_dt.strftime("%H:%M:%S")
                except (ValueError, AttributeError):
                    game_date = self.date
                    game_time_str = None
            else:
                game_date = self.date
                game_time_str = None

            # Get opponent info
            participant_info = raw_prop.get("participant", {}).get("player", {})
            player_team = participant_info.get("team", "")

            # Determine opponent (home or away team, whichever is not player's team)
            home_team = game_info.get("home_team", "")
            away_team = game_info.get("away_team", "")

            if player_team == home_team:
                opponent = away_team
                is_home = True
            elif player_team == away_team:
                opponent = home_team
                is_home = False
            else:
                opponent = home_team or away_team
                is_home = None

            # Normalize stat type
            stat_type = self.normalize_stat_type(market_name)

            # Build standardized prop
            prop = {
                "player_name": player_name,
                "player_team": player_team,  # Store for date filtering
                "stat_type": stat_type,
                "line": float(line),
                "over_line": (
                    float(over_line) if book_name and over_line is not None else float(line)
                ),
                "over_odds": over_odds,
                "under_line": (
                    float(under_line) if book_name and under_line is not None else float(line)
                ),
                "under_odds": under_odds,
                "book_name": book_name or "consensus",
                "game_id": game_id,
                "game_date": game_date,
                "game_time": game_time_str,
                "opponent_team": opponent,
                "is_home": is_home,
                "fetch_timestamp": datetime.now().isoformat(),
                "source": self.source_name,
            }

            # Add consensus data if fetching specific book (for comparison)
            if book_name and consensus_line is not None:
                prop["consensus_line"] = float(consensus_line)
                prop["consensus_odds_over"] = consensus_odds_over
                prop["consensus_odds_under"] = consensus_odds_under

            return prop

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            if self.verbose:
                print(f"  Error parsing prop: {e}")
            return None

    def fetch_todays_schedule(self) -> Dict[str, str]:
        """
        Fetch today's NBA schedule from ESPN to filter props by actual game date.

        Returns:
            Dict mapping team_abbrev -> game_date for teams playing on requested date
        """
        import requests

        # Convert date to ESPN format (YYYYMMDD)
        date_param = self.date.replace("-", "")

        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_param}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch ESPN schedule: HTTP {response.status_code}")
                return {}

            data = response.json()
            events = data.get("events", [])

            # Build mapping: team -> date
            team_schedule = {}
            for event in events:
                competitions = event.get("competitions", [])
                if not competitions:
                    continue

                competition = competitions[0]
                competitors = competition.get("competitors", [])

                # Get game date from event
                game_date_str = event.get("date", "")
                if game_date_str:
                    try:
                        game_date = datetime.fromisoformat(
                            game_date_str.replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        game_date = self.date
                else:
                    game_date = self.date

                # Add both teams to schedule
                for competitor in competitors:
                    team_abbrev = competitor.get("team", {}).get("abbreviation", "")
                    if team_abbrev:
                        # Normalize team abbreviation
                        team_abbrev = self._normalize_team_abbrev(team_abbrev)
                        team_schedule[team_abbrev] = game_date

            logger.info(f"Fetched schedule: {len(team_schedule)} teams playing on {self.date}")
            return team_schedule

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to fetch ESPN schedule: {e}")
            return {}

    def _normalize_team_abbrev(self, abbrev: str) -> str:
        """Normalize team abbreviation to canonical format"""
        mapping = {
            "GS": "GSW",
            "NO": "NOP",
            "NOR": "NOP",
            "SA": "SAS",
            "WSH": "WAS",
            "NY": "NYK",
            "UTAH": "UTA",
        }
        return mapping.get(abbrev, abbrev)

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA props from BettingPros Premium API.

        Supports multi-book fetching based on self.books_mode:
        - 'consensus': Fetch consensus lines only (old behavior)
        - 'priority': Fetch from 8 priority sportsbooks
        - 'all': Fetch from all 17 sportsbooks

        Returns:
            List of prop dictionaries with book-specific lines
        """
        print("\n" + "=" * 70)
        print(f"FETCHING BETTINGPROS PREMIUM NBA PROPS")
        print("=" * 70)
        print(f"Date: {self.date}")
        print(f"Markets: {', '.join(self.MARKETS.keys())}")
        print(f"Mode: {self.books_mode}")

        if self.books_to_fetch:
            print(f"Books: {len(self.books_to_fetch)} sportsbooks")
            print(f"  {', '.join(sorted(set(self.books_to_fetch.values())))}")
        else:
            print(f"Books: Consensus only")

        print("=" * 70 + "\n", flush=True)

        # Fetch today's schedule to filter props by actual game date
        print("Fetching today's schedule from ESPN...", flush=True)
        team_schedule = self.fetch_todays_schedule()
        teams_playing_today = set(team_schedule.keys())

        if teams_playing_today:
            print(f"[OK] {len(teams_playing_today)} teams playing on {self.date}", flush=True)
            print(f"   Teams: {', '.join(sorted(teams_playing_today))}\n", flush=True)
        else:
            print(
                f"[WARN]  WARNING: Could not fetch schedule - will accept all props\n", flush=True
            )

        all_props = []

        # If consensus mode, fetch without book filter
        if self.books_mode == "consensus":
            for market_name, market_id in self.MARKETS.items():
                print(f"Fetching {market_name.upper()} (consensus)...", flush=True)
                props = self.fetch_market(market_name, market_id)
                all_props.extend(props)
                print(f"  Total: {len(props)} props\n", flush=True)
        else:
            # Multi-book mode: fetch each book separately
            for book_id, book_name in self.books_to_fetch.items():
                print(f"\nFetching from {book_name.upper()} (book_id={book_id})...", flush=True)

                book_props = []
                for market_name, market_id in self.MARKETS.items():
                    # Skip markets that this book doesn't support
                    excluded_markets = self.BOOK_MARKET_EXCLUSIONS.get(book_id, set())
                    if market_name in excluded_markets:
                        continue  # Skip silently - this book doesn't offer this market

                    props = self.fetch_market(
                        market_name, market_id, book_id=book_id, book_name=book_name
                    )
                    book_props.extend(props)

                all_props.extend(book_props)
                print(f"  {book_name}: {len(book_props)} total props", flush=True)

        # Filter props to only teams playing on requested date
        # CRITICAL: Require successful schedule fetch to prevent wrong-date props
        if not teams_playing_today:
            error_msg = (
                f"[ERROR] CRITICAL: Cannot verify game dates - ESPN schedule fetch failed for {self.date}.\n"
                f"   Without schedule verification, props may be for wrong dates.\n"
                f"   Refusing to proceed. Please retry or check ESPN API status."
            )
            print(f"\n{error_msg}", flush=True)
            raise RuntimeError(error_msg)

        filtered_props = []
        rejected_count = 0
        date_mismatch_count = 0

        for prop in all_props:
            player_team = prop.get("player_team", "")
            normalized_team = self._normalize_team_abbrev(player_team)

            # Only accept props for teams playing today
            if normalized_team not in teams_playing_today:
                rejected_count += 1
                continue

            # CRITICAL: Verify prop's game_date matches requested date
            prop_game_date = prop.get("game_date", "")
            if prop_game_date != self.date:
                date_mismatch_count += 1
                continue

            # Date is already validated and correct - don't override it!
            # (team_schedule may contain wrong dates if ESPN lists future games)
            filtered_props.append(prop)

        print(f"\n🔍 Date filtering: {len(filtered_props)} props kept", flush=True)
        print(
            f"   Rejected: {rejected_count} (team not playing today) + {date_mismatch_count} (wrong game date)",
            flush=True,
        )
        all_props = filtered_props

        # Validate and deduplicate
        valid_props = [p for p in all_props if self.validate_prop(p)]
        deduped_props = self.deduplicate_props(valid_props)

        # Summary
        print("\n" + "=" * 70)
        print("BETTINGPROS PREMIUM FETCH SUMMARY")
        print("=" * 70)
        print(f"Total props fetched: {len(all_props)}")
        print(f"Valid props: {len(valid_props)}")
        print(f"After deduplication: {len(deduped_props)}")
        print()

        # Breakdown by market
        print("Breakdown by market:")
        for market_name in self.MARKETS.keys():
            stat_type = self.normalize_stat_type(market_name)
            market_props = [p for p in deduped_props if p["stat_type"] == stat_type]
            count = len(market_props)

            if count > 0:
                avg_line = sum(p["line"] for p in market_props) / count
                print(f"  {market_name.upper():10s}: {count:4d} props (avg line: {avg_line:.2f})")

        # Breakdown by book (if multi-book mode)
        if self.books_mode != "consensus":
            print("\nBreakdown by sportsbook:")
            book_counts = {}
            for prop in deduped_props:
                book = prop.get("book_name", "unknown")
                book_counts[book] = book_counts.get(book, 0) + 1

            for book in sorted(book_counts.keys()):
                print(f"  {book:15s}: {book_counts[book]:4d} props")

        print("=" * 70 + "\n", flush=True)

        return deduped_props


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA props from BettingPros Premium API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch from all 8 priority sportsbooks (default)
  python fetch_bettingpros.py

  # Fetch from all 17 sportsbooks
  python fetch_bettingpros.py --books all

  # Fetch consensus only (old behavior)
  python fetch_bettingpros.py --books consensus

  # Fetch for specific date
  python fetch_bettingpros.py --date 2025-11-06

Premium Features:
  With BettingPros Premium membership, fetches individual sportsbook lines
  for line shopping. This is the key to achieving 60%+ win rates.

  Priority sportsbooks (9):
    DraftKings, FanDuel, BetMGM, Caesars, bet365, BetRivers, ESPN Bet, Fanatics, Underdog

  All sportsbooks (17):
    All priority books plus SugarHouse, PartyCasino, Hard Rock, Unibet, etc.
        """,
    )
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--books",
        type=str,
        choices=["priority", "all", "consensus"],
        default="priority",
        help="Which books to fetch (default: priority)",
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less verbose)")

    args = parser.parse_args()

    # Create fetcher
    with BettingProsFetcher(date=args.date, books=args.books, verbose=not args.quiet) as fetcher:
        # Fetch props
        props = fetcher.fetch()

        # Save to JSON
        if props:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")

            # Show line shopping opportunities (if multi-book mode)
            if args.books != "consensus" and len(props) > 0:
                print("=" * 70)
                print("LINE SHOPPING OPPORTUNITIES (Sample)")
                print("=" * 70)

                # Group by player + stat
                from collections import defaultdict

                player_stats = defaultdict(list)

                for prop in props:
                    key = (prop["player_name"], prop["stat_type"])
                    player_stats[key].append(prop)

                # Find best line shopping opportunities (high spread)
                opportunities = []
                for (player, stat), props_list in player_stats.items():
                    if len(props_list) >= 3:  # Need at least 3 books
                        lines = [p["line"] for p in props_list]
                        spread = max(lines) - min(lines)
                        if spread >= 1.5:  # Significant spread
                            opportunities.append(
                                {
                                    "player": player,
                                    "stat": stat,
                                    "spread": spread,
                                    "softest": min(lines),
                                    "hardest": max(lines),
                                    "books": len(props_list),
                                    "props": props_list,
                                }
                            )

                # Show top 5 opportunities
                opportunities.sort(key=lambda x: x["spread"], reverse=True)
                for i, opp in enumerate(opportunities[:5], 1):
                    print(f"\n{i}. {opp['player']} - {opp['stat']}")
                    print(
                        f"   Spread: {opp['spread']:.1f} points ({opp['softest']:.1f} to {opp['hardest']:.1f})"
                    )
                    print(f"   Books: {opp['books']} sportsbooks")
                    print(f"   Lines: ", end="")
                    book_lines = [(p["book_name"], p["line"]) for p in opp["props"]]
                    book_lines.sort(key=lambda x: x[1])
                    print(", ".join([f"{book}={line:.1f}" for book, line in book_lines[:4]]))

                print("\n" + "=" * 70 + "\n")

        else:
            print("\n[WARN]  No props fetched!\n")


if __name__ == "__main__":
    main()
