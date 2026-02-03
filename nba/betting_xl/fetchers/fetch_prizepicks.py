#!/usr/bin/env python3
"""
PrizePicks Fetcher via The Odds API
====================================
Fetches NBA player props from PrizePicks using The Odds API.

PrizePicks is a DFS platform with three pick types:

  STANDARD (-137 in API): Core picks at normal multiplier
  GOBLINS  (+100 in API): Easier picks (lower lines) - REDUCED payout/multiplier
  DEMONS   (+100 in API): Harder picks (higher lines) - INCREASED payout/multiplier

Note: The odds from the API (-137, +100) are just representations.
      Actual payouts depend on flex play multipliers when combining picks.

Line Shopping Value:
  - PrizePicks often has SOFTER (higher) standard lines than sportsbooks
  - Compare PP standard lines vs sportsbook lines to find OVER value
  - Goblins can indicate where "easy" money is (low lines that books also have low)

Usage:
    python fetch_prizepicks.py                    # Fetch today's props
    python fetch_prizepicks.py --date 2026-01-30  # Specific date

API Key (Premium 20K plan):
    Set THEODDSAPI_KEY environment variable or it will read from odds-prem-api.txt
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

# Configure logging
logger = logging.getLogger(__name__)


class PrizePicksFetcher(BaseFetcher):
    """Fetches NBA props from PrizePicks via The Odds API"""

    # The Odds API endpoints
    API_BASE_URL = "https://api.the-odds-api.com/v4"

    # NBA sport key
    SPORT_KEY = "basketball_nba"

    # DFS platform bookmaker keys (The Odds API)
    # Available: prizepicks, underdog, pick6 (DK Pick6), betr_us_dfs
    BOOKMAKER = "prizepicks"

    # All supported DFS platforms
    ALL_DFS_BOOKMAKERS = {
        "prizepicks": "PrizePicks",
        "underdog": "Underdog Fantasy",
        "pick6": "DraftKings Pick6",
        "betr_us_dfs": "Betr Picks",
    }

    # Standard player prop markets (core picks at -137 odds)
    MARKETS = {
        "player_points": "POINTS",
        "player_rebounds": "REBOUNDS",
        "player_assists": "ASSISTS",
        "player_threes": "THREES",
    }

    # Alternate markets (demon/goblin picks at +100 odds - higher lines)
    ALTERNATE_MARKETS = {
        "player_points_alternate": "POINTS",
        "player_rebounds_alternate": "REBOUNDS",
        "player_assists_alternate": "ASSISTS",
        "player_threes_alternate": "THREES",
    }

    def __init__(self, date: str = None, verbose: bool = True):
        """
        Initialize PrizePicks fetcher.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name="prizepicks",
            rate_limit=1.0,  # The Odds API is generous with rate limits
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )

        self.date = date or datetime.now().strftime("%Y-%m-%d")

        # Get API key from environment or file
        self.api_key = os.getenv("THEODDSAPI_KEY")
        if not self.api_key:
            # Try reading from root file (Sport-suite/odds-prem-api.txt)
            api_file = Path(__file__).parents[3] / "odds-prem-api.txt"
            if api_file.exists():
                self.api_key = api_file.read_text().strip()

        if not self.api_key:
            raise ValueError(
                "The Odds API key not found. Set THEODDSAPI_KEY env var or "
                "create odds-prem-api.txt in project root."
            )

    def get_todays_events(self) -> List[Dict[str, Any]]:
        """
        Get NBA events (games) for the target date.

        Note: The Odds API returns times in UTC. A game at 2026-02-01T00:10:00Z
        is actually Jan 31 at 7:10 PM EST. We filter based on US Eastern Time
        date to match when NBA games are actually played.

        Returns:
            List of event dictionaries with id, commence_time, teams
        """
        from zoneinfo import ZoneInfo

        url = f"{self.API_BASE_URL}/sports/{self.SPORT_KEY}/events"

        params = {
            "apiKey": self.api_key,
            "dateFormat": "iso",
        }

        response = self._make_request(url, params=params)
        if not response:
            return []

        try:
            events = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Failed to parse events response: {e}")
            return []

        # Filter to events on target date (using US Eastern Time)
        target_date = datetime.strptime(self.date, "%Y-%m-%d").date()
        eastern = ZoneInfo("America/New_York")
        filtered = []

        for event in events:
            commence_time = event.get("commence_time", "")
            if commence_time:
                try:
                    # Parse UTC time and convert to Eastern
                    event_dt_utc = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    event_dt_et = event_dt_utc.astimezone(eastern)
                    event_date_et = event_dt_et.date()

                    if event_date_et == target_date:
                        filtered.append(event)
                except (ValueError, AttributeError):
                    pass

        logger.info(f"Found {len(filtered)} NBA events on {self.date}")
        return filtered

    def fetch_event_props(
        self, event_id: str, event_info: Dict, include_alternates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch player props for a single event from PrizePicks.

        Args:
            event_id: The Odds API event ID
            event_info: Event dict with team info
            include_alternates: Whether to fetch alternate (demon) lines

        Returns:
            List of prop dictionaries
        """
        url = f"{self.API_BASE_URL}/sports/{self.SPORT_KEY}/events/{event_id}/odds"

        # Request all supported markets at once (standard + alternates)
        all_markets = list(self.MARKETS.keys())
        if include_alternates:
            all_markets.extend(self.ALTERNATE_MARKETS.keys())
        markets = ",".join(all_markets)

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": markets,
            "bookmakers": self.BOOKMAKER,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        response = self._make_request(url, params=params)
        if not response:
            return []

        try:
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Failed to parse props response for event {event_id}: {e}")
            return []

        props = []

        # Extract team info
        home_team = event_info.get("home_team", "")
        away_team = event_info.get("away_team", "")
        commence_time = event_info.get("commence_time", "")

        # Parse game date/time (convert UTC to Eastern for consistency with BettingPros)
        from zoneinfo import ZoneInfo

        eastern = ZoneInfo("America/New_York")

        if commence_time:
            try:
                game_dt_utc = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                game_dt_et = game_dt_utc.astimezone(eastern)
                game_date = game_dt_et.strftime("%Y-%m-%d")  # Use EST date
                game_time = game_dt_et.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                game_date = self.date
                game_time = None
        else:
            game_date = self.date
            game_time = None

        # Extract bookmaker data
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return []

        # Find PrizePicks bookmaker
        prizepicks_data = None
        for bm in bookmakers:
            if bm.get("key") == self.BOOKMAKER:
                prizepicks_data = bm
                break

        if not prizepicks_data:
            logger.debug(f"No PrizePicks data for event {event_id}")
            return []

        # Combined market mapping (standard + alternates)
        all_market_mapping = {**self.MARKETS, **self.ALTERNATE_MARKETS}

        # Process each market
        for market in prizepicks_data.get("markets", []):
            market_key = market.get("key", "")
            stat_type = all_market_mapping.get(market_key)

            if not stat_type:
                continue

            # Check if this is an alternate market
            is_alternate = "_alternate" in market_key

            # Process outcomes (each player's line)
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "")
                if not player_name:
                    continue

                # Normalize player name
                player_name = self.normalize_player_name(player_name)

                # Get line and odds
                line = outcome.get("point")
                odds = outcome.get("price", -110)
                bet_type = outcome.get("name", "").lower()  # "Over" or "Under"

                if line is None:
                    continue

                # Determine player's team (heuristic based on name matching)
                player_team = self._guess_player_team(player_name, home_team, away_team)
                is_home = player_team == home_team if player_team else None
                opponent = away_team if is_home else home_team

                # Build prop dict
                # For alternates, tag with prizepicks_alt book name
                book_name = "prizepicks_alt" if is_alternate else self.BOOKMAKER

                prop = {
                    "player_name": player_name,
                    "player_team": player_team or "",
                    "stat_type": stat_type,
                    "line": float(line),
                    "over_line": float(line),
                    "over_odds": odds if bet_type == "over" else -110,
                    "under_line": float(line),
                    "under_odds": odds if bet_type == "under" else -110,
                    "book_name": book_name,
                    "is_alternate": is_alternate,
                    "game_id": event_id,
                    "game_date": game_date,
                    "game_time": game_time,
                    "opponent_team": opponent or "",
                    "is_home": is_home,
                    "fetch_timestamp": datetime.now().isoformat(),
                    "source": self.source_name,
                }

                props.append(prop)

        return props

    def _guess_player_team(self, player_name: str, home_team: str, away_team: str) -> Optional[str]:
        """
        Try to guess which team a player is on.

        Note: This is a heuristic. For more accurate mapping, use
        a player database lookup.

        Args:
            player_name: Player's name
            home_team: Home team name
            away_team: Away team name

        Returns:
            Team name or None if unknown
        """
        # TODO: Implement player-to-team lookup from database
        # For now, return None and let downstream handle it
        return None

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA props from PrizePicks via The Odds API.

        Returns:
            List of prop dictionaries
        """
        print("\n" + "=" * 70)
        print("FETCHING PRIZEPICKS NBA PROPS (via The Odds API)")
        print("=" * 70)
        print(f"Date: {self.date}")
        print(f"Markets: {', '.join(self.MARKETS.values())}")
        print(f"Bookmaker: {self.BOOKMAKER}")
        print("=" * 70 + "\n", flush=True)

        # Get today's events
        print("Fetching NBA events...", flush=True)
        events = self.get_todays_events()

        if not events:
            print(f"[WARN] No NBA events found for {self.date}", flush=True)
            return []

        print(f"[OK] Found {len(events)} games on {self.date}", flush=True)
        for event in events:
            home = event.get("home_team", "Unknown")
            away = event.get("away_team", "Unknown")
            print(f"   • {away} @ {home}", flush=True)

        # Fetch props for each event
        all_props = []

        for event in events:
            event_id = event.get("id")
            home = event.get("home_team", "Unknown")
            away = event.get("away_team", "Unknown")

            print(f"\nFetching props: {away} @ {home}...", flush=True)

            props = self.fetch_event_props(event_id, event)
            all_props.extend(props)

            print(f"   {len(props)} props found", flush=True)

        # Validate and deduplicate
        valid_props = [p for p in all_props if self.validate_prop(p)]
        deduped_props = self.deduplicate_props(valid_props)

        # Summary
        print("\n" + "=" * 70)
        print("PRIZEPICKS FETCH SUMMARY")
        print("=" * 70)
        print(f"Total props fetched: {len(all_props)}")
        print(f"Valid props: {len(valid_props)}")
        print(f"After deduplication: {len(deduped_props)}")
        print()

        # Count standard vs alternate (goblins + demons)
        standard_props = [p for p in deduped_props if not p.get("is_alternate", False)]
        alt_props = [p for p in deduped_props if p.get("is_alternate", False)]
        print(f"Standard picks: {len(standard_props)} (normal multiplier)")
        print(f"Alternate picks: {len(alt_props)} (goblins=less payout, demons=more payout)")
        print()

        # Breakdown by market (standard lines only)
        print("Standard lines by market:")
        for stat_type in self.MARKETS.values():
            market_props = [p for p in standard_props if p["stat_type"] == stat_type]
            count = len(market_props)

            if count > 0:
                avg_line = sum(p["line"] for p in market_props) / count
                print(f"  {stat_type:10s}: {count:4d} props (avg line: {avg_line:.2f})")

        # Breakdown by market (alternate lines - goblins vs demons)
        print("\nAlternate lines by market (goblins < standard < demons):")
        for stat_type in self.ALTERNATE_MARKETS.values():
            # Get standard avg for this stat
            std_market = [p for p in standard_props if p["stat_type"] == stat_type]
            std_avg = sum(p["line"] for p in std_market) / len(std_market) if std_market else 0

            market_alts = [p for p in alt_props if p["stat_type"] == stat_type]
            if market_alts:
                goblins = [p["line"] for p in market_alts if p["line"] < std_avg]
                demons = [p["line"] for p in market_alts if p["line"] >= std_avg]

                goblin_str = f"{len(goblins)} goblins" if goblins else "0 goblins"
                demon_str = f"{len(demons)} demons" if demons else "0 demons"
                print(f"  {stat_type:10s}: {goblin_str}, {demon_str}")

        print("=" * 70 + "\n", flush=True)

        return deduped_props


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA props from PrizePicks via The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's props
  python fetch_prizepicks.py

  # Fetch for specific date
  python fetch_prizepicks.py --date 2026-01-30

Note:
  Requires The Odds API key (Premium 20K plan recommended for full access).
  Set THEODDSAPI_KEY environment variable or create odds-prem-api.txt in project root.
        """,
    )
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less verbose)")

    args = parser.parse_args()

    # Create fetcher
    with PrizePicksFetcher(date=args.date, verbose=not args.quiet) as fetcher:
        # Fetch props
        props = fetcher.fetch()

        # Save to JSON
        if props:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")
        else:
            print("\n[WARN] No props fetched!\n")


if __name__ == "__main__":
    main()
