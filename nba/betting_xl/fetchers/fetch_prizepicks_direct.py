#!/usr/bin/env python3
"""
PrizePicks Direct Fetcher
=========================
Fetches NBA player props directly from PrizePicks API.

Captures all valuable fields discovered from API analysis:
- odds_type: standard/goblin/demon
- line_score: the actual line
- description: opponent team
- updated_at: when line was last updated
- trending_count: pick popularity (when present)
- adjusted_odds: null/true/false
- board_time: when line was first posted

Usage:
    python fetch_prizepicks_direct.py                    # Fetch today's props
    python fetch_prizepicks_direct.py --state FL         # Specific state
    python fetch_prizepicks_direct.py --include-combos   # Include combo props
    python fetch_prizepicks_direct.py --all-stats        # Include all stat types
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)


class PrizePicksDirectFetcher(BaseFetcher):
    """Fetches NBA props directly from PrizePicks API"""

    # PrizePicks API
    API_URL = "https://api.prizepicks.com/projections"

    # NBA league ID in PrizePicks
    LEAGUE_ID = 7

    # Stat type ID mappings (from API analysis)
    STAT_TYPE_ID_MAP = {
        # Core single stats
        "19": "POINTS",
        "22": "REBOUNDS",
        "20": "ASSISTS",
        "31": "THREES",
        "23": "STEALS",
        "21": "BLOCKS",
        "24": "TURNOVERS",
        "68": "FREE_THROWS_MADE",
        # Combo stats
        "106": "PRA",  # Pts+Rebs+Asts
        "243": "PR",  # Pts+Rebs
        "244": "PA",  # Pts+Asts
        "245": "RA",  # Rebs+Asts
        "105": "BLKS_STLS",  # Blks+Stls
        # Other stats
        "14": "FANTASY_SCORE",
        "117": "FG_MADE",
        "386": "FG_ATTEMPTED",
        "392": "THREES_ATTEMPTED",
        "385": "DUNKS",
        "403": "OFFENSIVE_REBOUNDS",
        "404": "DEFENSIVE_REBOUNDS",
        "890": "FREE_THROWS_ATTEMPTED",
        # Combo player stats
        "425": "POINTS_COMBO",
        "426": "ASSISTS_COMBO",
        "427": "REBOUNDS_COMBO",
        "428": "THREES_COMBO",
        # Period stats
        "1078": "QUARTERS_3_PLUS_PTS",
        "1080": "QUARTERS_5_PLUS_PTS",
        "1140": "ASSISTS_FIRST_3_MIN",
        "1141": "POINTS_FIRST_3_MIN",
        "1142": "REBOUNDS_FIRST_3_MIN",
    }

    # Core stats we care about for betting
    CORE_STATS = {"POINTS", "REBOUNDS", "ASSISTS", "THREES"}

    # Extended stats (combos)
    EXTENDED_STATS = {"PRA", "PR", "PA", "RA", "BLKS_STLS", "STEALS", "BLOCKS"}

    def __init__(
        self,
        state_code: str = "FL",
        include_combos: bool = False,
        all_stats: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize PrizePicks direct fetcher.

        Args:
            state_code: US state code (affects available props)
            include_combos: Whether to include combo player props
            all_stats: Whether to include all stat types (not just core)
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name="prizepicks_direct",
            rate_limit=2.0,
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )

        self.state_code = state_code
        self.include_combos = include_combos
        self.all_stats = all_stats

    def _fetch_projections(self) -> Optional[Dict]:
        """
        Fetch projections from PrizePicks API.

        Returns:
            Raw API response dict or None on error
        """
        params = {
            "league_id": self.LEAGUE_ID,
            "per_page": 250,
            "state_code": self.state_code,
        }

        try:
            response = requests.get(
                self.API_URL,
                params=params,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch PrizePicks projections: {e}")
            return None

    def _build_lookups(self, included: List[Dict]) -> Dict[str, Dict]:
        """
        Build lookup dictionaries from included data.

        Args:
            included: The 'included' array from API response

        Returns:
            Dict with players, stat_types, games, teams lookups
        """
        lookups = {
            "players": {},
            "stat_types": {},
            "games": {},
            "teams": {},
        }

        for item in included:
            item_type = item.get("type")
            item_id = item.get("id")
            attrs = item.get("attributes", {})
            rels = item.get("relationships", {})

            if item_type == "new_player":
                lookups["players"][item_id] = {
                    "name": attrs.get("name", ""),
                    "display_name": attrs.get("display_name", ""),
                    "team": attrs.get("team", ""),
                    "team_name": attrs.get("team_name", ""),
                    "position": attrs.get("position", ""),
                    "is_combo": attrs.get("combo", False),
                    "market": attrs.get("market", ""),
                    "jersey_number": attrs.get("jersey_number", ""),
                    "image_url": attrs.get("image_url", ""),
                }
            elif item_type == "stat_type":
                lookups["stat_types"][item_id] = {
                    "name": attrs.get("name", ""),
                    "display_name": attrs.get("display_name", ""),
                }
            elif item_type == "game":
                metadata = attrs.get("metadata", {})
                game_info = metadata.get("game_info", {})
                teams = game_info.get("teams", {})

                lookups["games"][item_id] = {
                    "external_game_id": attrs.get("external_game_id", ""),
                    "start_time": attrs.get("start_time", ""),
                    "status": attrs.get("status", ""),
                    "is_live": attrs.get("is_live", False),
                    "home_team": teams.get("home", {}).get("abbreviation", ""),
                    "away_team": teams.get("away", {}).get("abbreviation", ""),
                }
            elif item_type == "team":
                lookups["teams"][item_id] = {
                    "abbreviation": attrs.get("abbreviation", ""),
                    "name": attrs.get("name", ""),
                    "market": attrs.get("market", ""),
                }

        return lookups

    def _parse_projection(
        self,
        proj: Dict,
        lookups: Dict[str, Dict],
    ) -> Optional[Dict]:
        """
        Parse a single projection into our prop format.

        Args:
            proj: Projection dict from API
            lookups: Lookup dictionaries

        Returns:
            Prop dict or None if should be skipped
        """
        attrs = proj.get("attributes", {})
        rels = proj.get("relationships", {})

        # Get player info
        player_id = rels.get("new_player", {}).get("data", {}).get("id")
        player = lookups["players"].get(player_id, {})

        # Skip combos unless requested
        if player.get("is_combo", False) and not self.include_combos:
            return None

        # Get stat type
        stat_type_id = rels.get("stat_type", {}).get("data", {}).get("id")
        stat_type = self.STAT_TYPE_ID_MAP.get(stat_type_id, attrs.get("stat_type", "UNKNOWN"))

        # Filter stats unless all_stats is True
        if not self.all_stats:
            if stat_type not in self.CORE_STATS and stat_type not in self.EXTENDED_STATS:
                return None

        # Get game info
        game_id = rels.get("game", {}).get("data", {}).get("id")
        game = lookups["games"].get(game_id, {})

        # Get odds type (standard, goblin, demon)
        odds_type = attrs.get("odds_type", "standard")

        # Build book name based on odds type
        if odds_type == "standard":
            book_name = "prizepicks"
        elif odds_type == "goblin":
            book_name = "prizepicks_goblin"
        elif odds_type == "demon":
            book_name = "prizepicks_demon"
        else:
            book_name = f"prizepicks_{odds_type}"

        # Get line
        line = attrs.get("line_score")
        if line is None:
            return None

        # Parse timestamps
        start_time = attrs.get("start_time", "")
        board_time = attrs.get("board_time", "")
        updated_at = attrs.get("updated_at", "")

        game_date = None
        game_time = None

        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                game_date = dt.strftime("%Y-%m-%d")
                game_time = dt.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                game_date = datetime.now().strftime("%Y-%m-%d")
        else:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Normalize player name
        player_name = player.get("name", player.get("display_name", "Unknown"))
        player_name = self.normalize_player_name(player_name)

        # Get opponent team from description field
        opponent_team = attrs.get("description", "")

        # Determine home/away
        player_team = player.get("team", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        is_home = None
        if player_team and home_team:
            is_home = player_team == home_team

        # Build prop dict with all discovered fields
        prop = {
            # Core identifiers
            "projection_id": proj.get("id", ""),
            "player_name": player_name,
            "player_id": player_id,
            "player_team": player_team,
            "player_position": player.get("position", ""),
            "is_combo": player.get("is_combo", False),
            # Stat info
            "stat_type": stat_type,
            "stat_type_id": stat_type_id,
            "stat_display_name": attrs.get("stat_display_name", ""),
            # Line info
            "line": float(line),
            "over_line": float(line),
            "under_line": float(line),
            "over_odds": -110,  # PrizePicks doesn't expose odds
            "under_odds": -110,
            # Odds type (KEY FIELD)
            "odds_type": odds_type,  # standard, goblin, demon
            "book_name": book_name,
            # Timing fields
            "board_time": board_time,  # When line was first posted
            "updated_at": updated_at,  # When line was last updated
            "start_time": start_time,  # Game start time
            # Game info
            "game_id": attrs.get("game_id", ""),
            "pp_game_id": game_id,  # PrizePicks internal game ID
            "game_date": game_date,
            "game_time": game_time,
            "opponent_team": opponent_team,  # From description field
            "home_team": home_team,
            "away_team": away_team,
            "is_home": is_home,
            # Popularity/trending (VALUABLE)
            "trending_count": attrs.get("trending_count"),  # None if not present
            "rank": attrs.get("rank"),
            # Status flags
            "adjusted_odds": attrs.get("adjusted_odds"),  # null/true/false
            "is_promo": attrs.get("is_promo", False),
            "flash_sale_line_score": attrs.get("flash_sale_line_score"),
            "refundable": attrs.get("refundable", True),
            "status": attrs.get("status", ""),
            "is_live": attrs.get("is_live", False),
            "in_game": attrs.get("in_game", False),
            # Meta
            "fetch_timestamp": datetime.now().isoformat(),
            "source": "prizepicks_direct",
        }

        return prop

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA props from PrizePicks.

        Returns:
            List of prop dictionaries
        """
        print("\n" + "=" * 70)
        print("FETCHING PRIZEPICKS NBA PROPS (Direct API)")
        print("=" * 70)
        print(f"State: {self.state_code}")
        print(f"Include combos: {self.include_combos}")
        print(f"All stats: {self.all_stats}")
        print("=" * 70 + "\n", flush=True)

        # Fetch raw data
        print("Fetching projections...", flush=True)
        response = self._fetch_projections()

        if not response:
            print("[WARN] Failed to fetch PrizePicks data", flush=True)
            return []

        # Extract data and included
        projections = response.get("data", [])
        included = response.get("included", [])

        print(f"[OK] Retrieved {len(projections)} total projections", flush=True)

        # Build lookups
        lookups = self._build_lookups(included)
        print(
            f"[OK] Found {len(lookups['players'])} players, {len(lookups['games'])} games",
            flush=True,
        )

        # Parse projections
        props = []
        skipped_combos = 0
        skipped_stats = 0

        for proj in projections:
            # Check if combo
            player_id = (
                proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
            )
            player = lookups["players"].get(player_id, {})

            if player.get("is_combo", False) and not self.include_combos:
                skipped_combos += 1
                continue

            prop = self._parse_projection(proj, lookups)
            if prop:
                props.append(prop)
            else:
                skipped_stats += 1

        # Summary
        print("\n" + "=" * 70)
        print("PRIZEPICKS DIRECT FETCH SUMMARY")
        print("=" * 70)
        print(f"Total projections: {len(projections)}")
        print(f"Skipped (combos): {skipped_combos}")
        print(f"Skipped (filtered stats): {skipped_stats}")
        print(f"Parsed props: {len(props)}")
        print()

        # Count by odds type
        standard = [p for p in props if p["odds_type"] == "standard"]
        goblins = [p for p in props if p["odds_type"] == "goblin"]
        demons = [p for p in props if p["odds_type"] == "demon"]

        print(f"Standard lines: {len(standard)}")
        print(f"Goblin lines:   {len(goblins)} (easier, less payout)")
        print(f"Demon lines:    {len(demons)} (harder, more payout)")
        print()

        # Count trending
        trending_props = [p for p in props if p.get("trending_count") is not None]
        if trending_props:
            print(f"Props with trending_count: {len(trending_props)}")
            top_trending = sorted(
                trending_props, key=lambda x: x.get("trending_count", 0), reverse=True
            )[:5]
            print("Top 5 trending:")
            for p in top_trending:
                print(
                    f"  {p['player_name']:25s} | {p['stat_type']:10s} | {p['odds_type']:8s} | {p['trending_count']:,} picks"
                )
            print()

        # Breakdown by stat (standard only)
        print("Standard lines by stat type:")
        stat_counts = {}
        for p in standard:
            stat = p["stat_type"]
            stat_counts[stat] = stat_counts.get(stat, 0) + 1

        for stat, count in sorted(stat_counts.items(), key=lambda x: -x[1]):
            stat_props = [p for p in standard if p["stat_type"] == stat]
            avg_line = sum(p["line"] for p in stat_props) / len(stat_props)
            print(f"  {stat:15s}: {count:4d} props (avg line: {avg_line:.1f})")

        print("=" * 70 + "\n", flush=True)

        return props


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA props directly from PrizePicks API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's props (Florida)
  python fetch_prizepicks_direct.py

  # Different state
  python fetch_prizepicks_direct.py --state NY

  # Include combo props
  python fetch_prizepicks_direct.py --include-combos

  # Include all stat types
  python fetch_prizepicks_direct.py --all-stats

  # Save to JSON
  python fetch_prizepicks_direct.py --save
        """,
    )
    parser.add_argument("--state", type=str, default="FL", help="US state code (default: FL)")
    parser.add_argument("--include-combos", action="store_true", help="Include combo player props")
    parser.add_argument("--all-stats", action="store_true", help="Include all stat types")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--save", action="store_true", help="Save to JSON file")

    args = parser.parse_args()

    # Create fetcher
    with PrizePicksDirectFetcher(
        state_code=args.state,
        include_combos=args.include_combos,
        all_stats=args.all_stats,
        verbose=not args.quiet,
    ) as fetcher:
        # Fetch props
        props = fetcher.fetch()

        # Save to JSON if requested
        if props and args.save:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")
        elif not props:
            print("\n[WARN] No props fetched!\n")
        else:
            # Show sample
            print("\n=== SAMPLE PROPS ===\n")
            for prop in props[:10]:
                trending = (
                    f" | {prop['trending_count']:,} picks" if prop.get("trending_count") else ""
                )
                print(
                    f"{prop['player_name']:25s} | {prop['stat_type']:10s} | {prop['line']:5.1f} | {prop['odds_type']:8s}{trending}"
                )


if __name__ == "__main__":
    main()
