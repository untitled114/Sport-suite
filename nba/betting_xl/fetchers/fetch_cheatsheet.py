#!/usr/bin/env python3
"""
BettingPros Cheat Sheet Fetcher for NBA XL System
===================================================
Fetches FULL cheat sheet data from BettingPros API including:
- Projections (value, diff)
- Bet ratings (1-5 stars)
- Expected Value (EV)
- Hit rates (L-5, L-15, Season)
- Opposition rank
- Recommended side

This provides all the analytics data shown on the BettingPros cheat sheet UI,
which can significantly improve model decision-making.

Supports filtering by book for DFS platforms:
- Underdog (book_id=36)
- PrizePicks (via generic filtering)

Usage:
    python fetch_cheatsheet.py --platform underdog
    python fetch_cheatsheet.py --platform prizepicks
    python fetch_cheatsheet.py --platform all
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

# Configure logging
logger = logging.getLogger(__name__)


class CheatSheetFetcher(BaseFetcher):
    """
    Fetches full BettingPros cheat sheet data including projections, EV, and hit rates.

    This extracts the same analytics data shown on the BettingPros UI:
    - Projection value and diff
    - Bet rating (1-5 stars)
    - Expected Value percentage
    - Hit rates (L-5, L-15, Season)
    - Opposition rank
    """

    # BettingPros API market IDs
    # Verified 2026-01-08 from /v3/markets endpoint
    MARKETS = {
        # Single stats
        "points": 156,
        "rebounds": 157,
        "assists": 151,
        "threes": 162,  # 3-pointers made
        "steals": 160,
        "blocks": 152,
        # Combo stats (CORRECT IDs - verified 2026-01-08)
        "pts_ast": 335,  # Points + Assists (PA)
        "pts_reb": 336,  # Points + Rebounds (PR)
        "reb_ast": 337,  # Rebounds + Assists (RA)
        "pts_reb_ast": 338,  # Points + Rebounds + Assists (PRA)
    }

    # Stat type mapping for combo markets
    COMBO_STAT_TYPES = {
        "pts_ast": "PA",  # Points + Assists
        "pts_reb": "PR",  # Points + Rebounds
        "reb_ast": "RA",  # Rebounds + Assists
        "pts_reb_ast": "PRA",  # Points + Rebounds + Assists
    }

    # Platform book IDs
    PLATFORM_BOOKS = {
        "underdog": 36,
        "draftkings": 12,
        "fanduel": 10,
        "betmgm": 19,
        "caesars": 13,
        "bet365": 24,
        "betrivers": 18,
        "espnbet": 33,
    }

    # Which markets each platform offers
    # Underdog has: points, rebounds, assists, steals, blocks + combos (PA, PR, RA)
    # NOTE: PRA and THREES are on other books but NOT Underdog
    PLATFORM_MARKETS = {
        "underdog": [
            "points",
            "rebounds",
            "assists",  # Core single stats
            "pts_ast",
            "pts_reb",
            "reb_ast",  # Combo stats (PA, PR, RA)
        ],
        "all": [
            "points",
            "rebounds",
            "assists",
            "threes",  # Single stats
            "pts_ast",
            "pts_reb",
            "reb_ast",
            "pts_reb_ast",  # Combo stats
        ],
    }

    API_BASE_URL = "https://api.bettingpros.com/v3/props"

    # Premium authentication
    PREMIUM_HEADERS = {
        "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
        "x-level": "cHJlbWl1bQ==",
        "accept": "application/json",
    }

    def __init__(
        self,
        date: str = None,
        platform: str = "all",
        include_combos: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize cheat sheet fetcher.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            platform: Platform to fetch for ('underdog', 'prizepicks', 'all')
            include_combos: Include combo markets (Pts+Ast, etc.)
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name=f"cheatsheet_{platform}",
            rate_limit=2.5,  # 2.5 seconds between requests
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )

        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.platform = platform.lower()
        self.include_combos = include_combos

        # Get book_id for platform (None = no filter, gets all books)
        self.book_id = self.PLATFORM_BOOKS.get(self.platform)

        # Get markets for this platform
        if self.platform in self.PLATFORM_MARKETS:
            self.markets_to_fetch = self.PLATFORM_MARKETS[self.platform]
        else:
            self.markets_to_fetch = list(self.MARKETS.keys())

        # Filter out combo markets if not wanted
        if not include_combos:
            self.markets_to_fetch = [
                m
                for m in self.markets_to_fetch
                if m in ["points", "rebounds", "assists", "threes", "steals", "blocks"]
            ]

        # Teams playing on requested date (for date filtering)
        self.teams_playing_today: Set[str] = set()

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
            "PHX": "PHO",
            "BKN": "BRK",
        }
        return mapping.get(abbrev.upper(), abbrev.upper())

    def fetch_todays_schedule(self) -> Set[str]:
        """
        Fetch today's NBA schedule from ESPN to filter props by actual game date.

        CRITICAL: This prevents mixing props from different dates.

        Returns:
            Set of team abbreviations playing on the requested date
        """
        # Convert date to ESPN format (YYYYMMDD)
        date_param = self.date.replace("-", "")

        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_param}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch ESPN schedule: HTTP {response.status_code}")
                return set()

            data = response.json()
            events = data.get("events", [])

            # Build set of teams playing today
            teams_today = set()
            for event in events:
                competitions = event.get("competitions", [])
                if not competitions:
                    continue

                competition = competitions[0]
                competitors = competition.get("competitors", [])

                # Get game date from event and verify it matches requested date
                game_date_str = event.get("date", "")
                if game_date_str:
                    try:
                        game_date = datetime.fromisoformat(
                            game_date_str.replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d")
                        # Only include teams if game is on requested date
                        if game_date != self.date:
                            continue
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Could not parse game date '{game_date_str}': {e}")

                # Add both teams
                for competitor in competitors:
                    team_abbrev = competitor.get("team", {}).get("abbreviation", "")
                    if team_abbrev:
                        team_abbrev = self._normalize_team_abbrev(team_abbrev)
                        teams_today.add(team_abbrev)

            logger.info(f"Fetched schedule: {len(teams_today)} teams playing on {self.date}")
            return teams_today

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to fetch ESPN schedule: {e}")
            return set()

    def fetch_market(self, market_name: str, market_id: int) -> List[Dict[str, Any]]:
        """
        Fetch all props for a single market with full cheat sheet data.

        Args:
            market_name: Market name (e.g., 'points')
            market_id: BettingPros market ID

        Returns:
            List of prop dictionaries with full analytics data
        """
        props = []
        page = 1

        while True:
            params = {
                "sport": "NBA",
                "date": self.date,
                "market_id": market_id,
                "limit": 500,
                "page": page,
            }

            # Add book filter if fetching for specific platform
            if self.book_id is not None:
                params["book_id"] = self.book_id

            response = self._make_request(
                url=self.API_BASE_URL, method="GET", params=params, headers=self.PREMIUM_HEADERS
            )

            if not response:
                break

            # Validate response
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" not in content_type:
                logger.warning(f"Invalid content-type for {market_name}: {content_type}")
                break

            if len(response.content) < 50:
                logger.warning(f"Empty response for {market_name}: {len(response.content)} bytes")
                break

            try:
                data = response.json()
            except (requests.RequestException, KeyError, ValueError, TypeError) as e:
                logger.error(f"JSON parse error for {market_name}: {e}")
                break

            page_props = data.get("props", [])
            if not page_props:
                break

            # Parse each prop with full cheat sheet data
            for prop in page_props:
                parsed = self._parse_cheatsheet_prop(prop, market_name)
                if parsed:
                    props.append(parsed)

            if self.verbose:
                print(f"  {market_name} Page {page}: +{len(page_props)} props", flush=True)

            # Check pagination
            pagination = data.get("_pagination", {})
            total_pages = pagination.get("total_pages", 1)

            if page >= total_pages:
                break

            page += 1

        return props

    def _parse_cheatsheet_prop(
        self, raw_prop: Dict[str, Any], market_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse raw prop with FULL cheat sheet data.

        Extracts:
        - Basic prop info (player, line, odds)
        - Projection data (value, diff, rating, EV)
        - Hit rates (L-5, L-15, Season)
        - Opposition rank

        Args:
            raw_prop: Raw prop data from API
            market_name: Market name

        Returns:
            Parsed prop dictionary with full analytics or None if invalid
        """
        try:
            # Get player info
            participant = raw_prop.get("participant", {})
            player_name = participant.get("name", "")
            if not player_name:
                return None

            player_name = self.normalize_player_name(player_name)
            player_info = participant.get("player", {})
            player_team = player_info.get("team", "")

            # Get over/under data
            over_data = raw_prop.get("over", {})
            under_data = raw_prop.get("under", {})

            # Get line
            line = over_data.get("line") or over_data.get("consensus_line")
            if line is None:
                return None

            # Get projection data (KEY CHEAT SHEET DATA)
            projection = raw_prop.get("projection", {})
            proj_value = projection.get("value")
            proj_diff = projection.get("diff")
            bet_rating = projection.get("bet_rating", 0)
            expected_value = projection.get("expected_value", 0)
            probability = projection.get("probability", 0.5)
            recommended_side = projection.get("recommended_side", "")

            # Get opposition rank (OPP VS PROP)
            extra = raw_prop.get("extra", {})
            opp_rank_data = extra.get("opposition_rank", {})
            opp_rank = opp_rank_data.get("rank")
            opp_value = opp_rank_data.get("value")

            # Get hit rates (PERFORMANCE DATA)
            performance = raw_prop.get("performance", {})

            def calc_hit_rate(perf_data: Dict) -> Optional[float]:
                """Calculate hit rate from performance data"""
                over = perf_data.get("over", 0)
                under = perf_data.get("under", 0)
                total = over + under
                if total > 0:
                    return round(over / total, 3)
                return None

            hit_rate_l5 = calc_hit_rate(performance.get("last_5", {}))
            hit_rate_l15 = calc_hit_rate(performance.get("last_15", {}))
            hit_rate_season = calc_hit_rate(performance.get("season", {}))

            # Get raw counts for analysis
            l5_data = performance.get("last_5", {})
            l15_data = performance.get("last_15", {})
            season_data = performance.get("season", {})

            # Get game info
            game_info = raw_prop.get("game", {})
            if not game_info:
                game_info = {}
            game_id = game_info.get("id", "")
            game_time = game_info.get("start", "")

            # Parse game date/time
            if game_time:
                try:
                    game_dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                    game_date = game_dt.strftime("%Y-%m-%d")
                    game_time_str = game_dt.strftime("%H:%M:%S")
                except (ValueError, AttributeError):
                    game_date = self.date
                    game_time_str = None
            else:
                game_date = self.date
                game_time_str = None

            # Get opponent and home/away
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
            if market_name in self.COMBO_STAT_TYPES:
                stat_type = self.COMBO_STAT_TYPES[market_name]
            else:
                stat_type = self.normalize_stat_type(market_name)

            # Build comprehensive prop with all cheat sheet data
            prop = {
                # Basic prop info
                "player_name": player_name,
                "player_team": player_team,
                "stat_type": stat_type,
                "line": float(line),
                "over_line": float(over_data.get("line", line)),
                "over_odds": over_data.get("odds", -110),
                "under_line": float(under_data.get("line", line)),
                "under_odds": under_data.get("odds", -110),
                "book_name": self.platform if self.book_id else "consensus",
                # Game info
                "game_id": game_id,
                "game_date": game_date,
                "game_time": game_time_str,
                "opponent_team": opponent,
                "is_home": is_home,
                # CHEAT SHEET DATA - Projection
                "projection": proj_value,
                "projection_diff": proj_diff,
                "bet_rating": bet_rating,  # 1-5 stars
                "expected_value": expected_value,  # EV as decimal
                "ev_pct": round(expected_value * 100, 2) if expected_value else None,  # EV as %
                "probability": probability,  # P(over)
                "recommended_side": recommended_side,  # 'over' or 'under'
                # CHEAT SHEET DATA - Opposition
                "opp_rank": opp_rank,  # OPP VS PROP rank
                "opp_value": opp_value,  # Opponent's average
                # CHEAT SHEET DATA - Hit Rates
                "hit_rate_l5": hit_rate_l5,
                "hit_rate_l15": hit_rate_l15,
                "hit_rate_season": hit_rate_season,
                # Raw hit rate data (for analysis)
                "l5_over": l5_data.get("over", 0),
                "l5_under": l5_data.get("under", 0),
                "l15_over": l15_data.get("over", 0),
                "l15_under": l15_data.get("under", 0),
                "season_over": season_data.get("over", 0),
                "season_under": season_data.get("under", 0),
                # Metadata
                "fetch_timestamp": datetime.now().isoformat(),
                "source": f"bettingpros_{self.platform}",
            }

            # Add consensus data if available
            consensus_line = over_data.get("consensus_line")
            if consensus_line is not None and consensus_line != line:
                prop["consensus_line"] = float(consensus_line)
                prop["line_vs_consensus"] = float(line) - float(consensus_line)

            return prop

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            if self.verbose:
                logger.warning(f"Error parsing cheat sheet prop: {e}")
            return None

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all cheat sheet props for configured platform.

        Returns:
            List of prop dictionaries with full cheat sheet analytics
        """
        print("\n" + "=" * 70)
        print(f"FETCHING BETTINGPROS CHEAT SHEET DATA")
        print("=" * 70)
        print(f"Date: {self.date}")
        print(f"Platform: {self.platform.upper()}")
        print(f"Book ID: {self.book_id or 'All books'}")
        print(f"Markets: {', '.join(self.markets_to_fetch)}")
        print(f"Include Combos: {self.include_combos}")
        print("=" * 70 + "\n", flush=True)

        all_props = []

        for market_name in self.markets_to_fetch:
            market_id = self.MARKETS.get(market_name)
            if market_id is None:
                logger.warning(f"Unknown market: {market_name}")
                continue

            print(f"Fetching {market_name.upper()}...", flush=True)
            props = self.fetch_market(market_name, market_id)
            all_props.extend(props)
            print(f"  Total: {len(props)} props\n", flush=True)

        # Filter props by game_date to ensure we only get requested date
        # (BettingPros API date param should handle this, but double-check)
        filtered_props = []
        rejected_date = 0

        for prop in all_props:
            prop_game_date = prop.get("game_date", "")
            if prop_game_date and prop_game_date != self.date:
                rejected_date += 1
                continue
            filtered_props.append(prop)

        if rejected_date > 0:
            print(f"[DATE FILTER] Rejected {rejected_date} props with wrong date", flush=True)

        # Validate and deduplicate
        valid_props = [p for p in filtered_props if self.validate_prop(p)]
        deduped_props = self.deduplicate_props(valid_props)

        # Summary
        print("\n" + "=" * 70)
        print(f"CHEAT SHEET FETCH SUMMARY ({self.platform.upper()})")
        print("=" * 70)
        print(f"Total props: {len(deduped_props)}")

        # Breakdown by stat type
        print("\nBy stat type:")
        stat_counts = {}
        for prop in deduped_props:
            st = prop["stat_type"]
            stat_counts[st] = stat_counts.get(st, 0) + 1
        for st, count in sorted(stat_counts.items()):
            print(f"  {st:25s}: {count:4d} props")

        # Sample high-rated picks
        high_rated = [p for p in deduped_props if p.get("bet_rating", 0) >= 4]
        if high_rated:
            print(f"\nHigh-rated picks (4+ stars): {len(high_rated)}")
            for prop in sorted(high_rated, key=lambda x: x.get("expected_value", 0), reverse=True)[
                :5
            ]:
                ev_pct = prop.get("ev_pct", 0) or 0
                print(
                    f"  {prop['player_name']:20s} {prop['stat_type']:10s} "
                    f"{prop['recommended_side'].upper():5s} {prop['line']:.1f} "
                    f"(Proj: {prop.get('projection', 0):.1f}, EV: {ev_pct:+.1f}%, "
                    f"Rating: {prop.get('bet_rating', 0)} stars)"
                )

        print("=" * 70 + "\n", flush=True)

        return deduped_props


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch BettingPros cheat sheet data with projections and analytics"
    )
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD)")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["underdog", "all", "draftkings", "fanduel"],  # PrizePicks removed - no book_id
        default="all",
        help="Platform to fetch for",
    )
    parser.add_argument("--no-combos", action="store_true", help="Skip combo markets")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    with CheatSheetFetcher(
        date=args.date,
        platform=args.platform,
        include_combos=not args.no_combos,
        verbose=not args.quiet,
    ) as fetcher:
        props = fetcher.fetch()

        if props:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props with cheat sheet data to: {output_file}\n")
        else:
            print("\n[WARN] No props fetched!\n")


if __name__ == "__main__":
    main()
