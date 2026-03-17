#!/usr/bin/env python3
"""
Underdog Fantasy Direct Fetcher
=================================
Fetches NBA player props directly from the Underdog Fantasy API.

Underdog Fantasy is a DFS (Daily Fantasy Sports) platform that typically offers
softer lines than traditional sportsbooks, making it valuable for line shopping.

API endpoints:
- v5: https://api.underdogfantasy.com/beta/v5/over_under_lines
- v7: https://api.underdogfantasy.com/beta/v7/over_under_lines

Lines are DFS-style (no traditional odds), so we default to -110/-110.

Usage:
    python fetch_underdog_direct.py                    # Core stats only
    python fetch_underdog_direct.py --include-combos   # Include PRA, PR, PA, RA
    python fetch_underdog_direct.py --save             # Save to JSON
    python fetch_underdog_direct.py --quiet            # Minimal output
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

_EST = ZoneInfo("America/New_York")


class UnderdogDirectFetcher(BaseFetcher):
    """Fetches NBA player props directly from Underdog Fantasy API."""

    # API endpoints (v7 preferred, v5 fallback)
    API_URL_V7 = "https://api.underdogfantasy.com/beta/v7/over_under_lines"
    API_URL_V5 = "https://api.underdogfantasy.com/beta/v5/over_under_lines"

    # Underdog stat name -> canonical stat type
    STAT_TYPE_MAP = {
        # Core single stats
        "Points": "POINTS",
        "Rebounds": "REBOUNDS",
        "Assists": "ASSISTS",
        "3-Pointers Made": "THREES",
        "Steals": "STEALS",
        "Blocks": "BLOCKS",
        "Turnovers": "TURNOVERS",
        # Combo stats
        "Pts + Rebs + Asts": "PRA",
        "Pts + Rebs": "PR",
        "Pts + Asts": "PA",
        "Rebs + Asts": "RA",
        # Alternative spellings seen in wild
        "Three Pointers Made": "THREES",
        "3-Pt Made": "THREES",
        "3PM": "THREES",
        "Fantasy Points": "FANTASY_POINTS",
        "Double Double": "DOUBLE_DOUBLE",
        "Triple Double": "TRIPLE_DOUBLE",
    }

    # Core stats we care about for betting models
    CORE_STATS = {"POINTS", "REBOUNDS", "ASSISTS", "THREES"}

    # Combo/extended stats (included with --include-combos)
    COMBO_STATS = {"PRA", "PR", "PA", "RA"}

    # Extended single stats (always included when recognized)
    EXTENDED_STATS = {"STEALS", "BLOCKS", "TURNOVERS"}

    # Underdog team abbreviation -> standard NBA abbreviation
    TEAM_ABBREV_MAP = {
        # Non-standard abbreviations Underdog may use
        "PHO": "PHX",
        "PHOE": "PHX",
        "PHOENIX": "PHX",
        "GS": "GSW",
        "GOLDEN STATE": "GSW",
        "SA": "SAS",
        "SAN ANTONIO": "SAS",
        "NO": "NOP",
        "NOR": "NOP",
        "NEW ORLEANS": "NOP",
        "WSH": "WAS",
        "WASH": "WAS",
        "WASHINGTON": "WAS",
        "NY": "NYK",
        "BKN": "BKN",
        "BK": "BKN",
        "BROOKLYN": "BKN",
        "CHA": "CHA",
        "CHAR": "CHA",
        "CHARLOTTE": "CHA",
        "UTH": "UTA",
        "UTAH": "UTA",
        # Standard abbreviations (pass-through)
        "ATL": "ATL",
        "BOS": "BOS",
        "CHI": "CHI",
        "CLE": "CLE",
        "DAL": "DAL",
        "DEN": "DEN",
        "DET": "DET",
        "GSW": "GSW",
        "HOU": "HOU",
        "IND": "IND",
        "LAC": "LAC",
        "LAL": "LAL",
        "MEM": "MEM",
        "MIA": "MIA",
        "MIL": "MIL",
        "MIN": "MIN",
        "NOP": "NOP",
        "NYK": "NYK",
        "OKC": "OKC",
        "ORL": "ORL",
        "PHI": "PHI",
        "PHX": "PHX",
        "POR": "POR",
        "SAC": "SAC",
        "SAS": "SAS",
        "TOR": "TOR",
        "UTA": "UTA",
        "WAS": "WAS",
    }

    def __init__(
        self,
        include_combos: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Underdog Fantasy direct fetcher.

        Args:
            include_combos: Whether to include combo stats (PRA, PR, PA, RA)
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name="underdog_direct",
            rate_limit=2.0,
            max_retries=3,
            timeout=30,
            verbose=verbose,
            proxy_profile="sportsbooks",
        )

        self.include_combos = include_combos

    def _normalize_team_abbrev(self, abbrev: str) -> Optional[str]:
        """
        Normalize a team abbreviation to standard NBA format.

        Args:
            abbrev: Raw team abbreviation from Underdog API

        Returns:
            Standard NBA abbreviation or None if not recognized
        """
        if not abbrev:
            return None

        abbrev_upper = abbrev.strip().upper()
        normalized = self.TEAM_ABBREV_MAP.get(abbrev_upper)

        if normalized:
            return normalized

        # If 3-letter code not in map, pass through (might be standard already)
        if len(abbrev_upper) == 3 and abbrev_upper.isalpha():
            return abbrev_upper

        return None

    def _resolve_stat_type(self, stat_name: str) -> Optional[str]:
        """
        Resolve Underdog stat name to canonical stat type.

        Args:
            stat_name: Raw stat name from Underdog API

        Returns:
            Canonical stat type string or None if unrecognized
        """
        if not stat_name:
            return None

        # Direct lookup
        canonical = self.STAT_TYPE_MAP.get(stat_name)
        if canonical:
            return canonical

        # Fallback: try the base class normalizer
        normalized = self.normalize_stat_type(stat_name)
        if normalized and normalized != stat_name.upper():
            return normalized

        return stat_name.upper()

    def _should_include_stat(self, stat_type: str) -> bool:
        """
        Check if a stat type should be included based on current filters.

        Args:
            stat_type: Canonical stat type string

        Returns:
            True if the stat should be included
        """
        if stat_type in self.CORE_STATS:
            return True

        if stat_type in self.EXTENDED_STATS:
            return True

        if stat_type in self.COMBO_STATS and self.include_combos:
            return True

        return False

    def _fetch_api_data(self) -> Optional[Dict]:
        """
        Fetch over/under lines from Underdog Fantasy API.

        Tries v7 endpoint first with curl_cffi browser impersonation,
        falls back to v5, then to standard requests.

        Returns:
            Raw API response dict or None on failure
        """
        params = {"sport": "NBA"}

        # Strategy 1: curl_cffi with browser impersonation (v7)
        if HAS_CURL_CFFI:
            data = self._fetch_with_curl_cffi(self.API_URL_V7, params)
            if data is not None:
                return data

            # curl_cffi fallback to v5
            data = self._fetch_with_curl_cffi(self.API_URL_V5, params)
            if data is not None:
                return data

        # Strategy 2: standard requests with proxy (v7 then v5)
        data = self._fetch_with_requests(self.API_URL_V7, params)
        if data is not None:
            return data

        data = self._fetch_with_requests(self.API_URL_V5, params)
        if data is not None:
            return data

        logger.error("[underdog_direct] All fetch strategies exhausted")
        return None

    def _fetch_with_curl_cffi(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Fetch using curl_cffi with Chrome TLS impersonation.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Parsed JSON response or None on failure
        """
        proxies = None
        if self.proxy_profile:
            from nba.betting_xl.fetchers.proxy_manager import get_proxy_manager

            pm = get_proxy_manager()
            proxies = pm.get_proxies_dict(self.proxy_profile) or None

        for attempt in range(self.max_retries):
            delay = random.uniform(0.5, 1.5)
            time.sleep(delay)

            try:
                if self.verbose:
                    logger.info(
                        f"[underdog_direct] curl_cffi: {url} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )

                response = cffi_requests.get(
                    url,
                    params=params,
                    impersonate="chrome",
                    timeout=self.timeout,
                    proxies=proxies,
                    headers={
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate",
                        "Referer": "https://underdogfantasy.com/",
                        "Origin": "https://underdogfantasy.com",
                    },
                )

                self._api_calls += 1
                self._bytes_transferred += len(response.content)

                if response.status_code == 403:
                    logger.warning(
                        f"[underdog_direct] 403 Forbidden via curl_cffi "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                    continue

                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"[underdog_direct] Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict) and "error" in data:
                    logger.warning(f"[underdog_direct] API returned error: {data.get('error')}")
                    continue

                if self.verbose:
                    logger.info(
                        f"[underdog_direct] curl_cffi success " f"({len(response.content)} bytes)"
                    )
                return data

            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(
                    f"[underdog_direct] curl_cffi failed "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def _fetch_with_requests(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Fallback fetch using standard requests library (with session proxy).

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Parsed JSON response or None on failure
        """
        headers = {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://underdogfantasy.com/",
            "Origin": "https://underdogfantasy.com",
        }

        if self.verbose:
            logger.info(f"[underdog_direct] requests fallback: {url}")

        response = self._make_request(url, params=params, headers=headers)

        if response is None:
            return None

        try:
            data = response.json()

            if isinstance(data, dict) and "error" in data:
                logger.warning(f"[underdog_direct] API returned error: {data.get('error')}")
                return None

            return data

        except json.JSONDecodeError:
            logger.error(
                f"[underdog_direct] Invalid JSON response " f"({len(response.content)} bytes)"
            )
            self._error_count += 1
            self._last_error = "Invalid JSON response"
            return None

    def _build_lookups(self, data: Dict) -> Dict[str, Dict]:
        """
        Build lookup dictionaries from the top-level arrays.

        The Underdog v5 API uses flat parallel arrays with UUID references:
        - players[]: id, first_name, last_name, team_id
        - appearances[]: id, player_id, match_id, team_id
        - games[]: id, abbreviated_title, home_team_id, away_team_id, scheduled_at

        Args:
            data: Full API response

        Returns:
            Dict with 'players', 'appearances', 'games' lookup dicts
        """
        lookups = {
            "players": {},
            "appearances": {},
            "games": {},
            "teams": {},
            "nba_game_ids": set(),
        }

        for player in data.get("players", []):
            pid = player.get("id")
            if pid:
                lookups["players"][pid] = player

        for app in data.get("appearances", []):
            aid = app.get("id")
            if aid:
                lookups["appearances"][aid] = app

        for game in data.get("games", []):
            gid = game.get("id")
            if gid:
                lookups["games"][gid] = game
                # Track NBA game IDs for filtering
                if game.get("sport_id") == "NBA":
                    lookups["nba_game_ids"].add(gid)

        # Build team lookup from games (home_team_id / away_team_id -> abbrev from title)
        for game in data.get("games", []):
            title = game.get("abbreviated_title", "")  # e.g. "ORL @ ATL"
            if " @ " in title:
                parts = title.split(" @ ")
                away_abbrev = self._normalize_team_abbrev(parts[0].strip())
                home_abbrev = self._normalize_team_abbrev(parts[1].strip())
                home_id = game.get("home_team_id")
                away_id = game.get("away_team_id")
                if home_id and home_abbrev:
                    lookups["teams"][home_id] = home_abbrev
                if away_id and away_abbrev:
                    lookups["teams"][away_id] = away_abbrev

        return lookups

    def _parse_line(self, line_data: Dict, lookups: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """
        Parse a single over_under_line entry into standardized prop format.

        Uses lookup dictionaries to resolve UUID references between
        lines -> appearances -> players/games.

        Args:
            line_data: Single entry from over_under_lines array
            lookups: Pre-built lookup dictionaries

        Returns:
            Standardized prop dict or None if should be skipped
        """
        # Filter to NBA only — resolve appearance -> match_id -> check sport_id
        over_under = line_data.get("over_under", {})
        if not isinstance(over_under, dict):
            return None

        appearance_stat = over_under.get("appearance_stat", {})
        if not isinstance(appearance_stat, dict):
            return None

        appearance_id = appearance_stat.get("appearance_id", "")
        appearance = lookups["appearances"].get(appearance_id, {})
        match_id = appearance.get("match_id")
        if match_id not in lookups["nba_game_ids"]:
            return None  # Not an NBA game — skip

        # The stat name lives in appearance_stat.stat (e.g. "points", "rebounds")
        stat_name = appearance_stat.get("stat", "") or appearance_stat.get("display_stat", "")
        stat_type = self._resolve_stat_type(stat_name)
        if stat_type is None:
            return None

        # Filter by stat type
        if not self._should_include_stat(stat_type):
            return None

        # Extract the line value
        stat_value = line_data.get("stat_value")
        if stat_value is None:
            return None

        try:
            line = float(stat_value)
        except (ValueError, TypeError):
            return None

        if line <= 0 or line > 200:
            return None

        # Resolve player via: appearance_stat.appearance_id -> appearances -> player_id -> players
        appearance_id = appearance_stat.get("appearance_id", "")
        appearance = lookups["appearances"].get(appearance_id, {})

        player_id = appearance.get("player_id", "")
        player = lookups["players"].get(player_id, {})

        first_name = player.get("first_name", "")
        last_name = player.get("last_name", "")
        player_name = f"{first_name} {last_name}".strip()
        if not player_name:
            return None
        player_name = self.normalize_player_name(player_name)

        # Resolve game via: appearance.match_id -> games
        match_id = appearance.get("match_id")
        game = lookups["games"].get(match_id, {})
        game_id = str(match_id) if match_id else ""

        # Parse game time
        scheduled_at = game.get("scheduled_at", "")
        game_date = datetime.now(_EST).strftime("%Y-%m-%d")
        game_time = None
        if scheduled_at:
            try:
                dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
                est_dt = dt.astimezone(_EST)
                game_date = est_dt.strftime("%Y-%m-%d")
                game_time = est_dt.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                pass

        # Resolve teams via team_id lookups
        player_team_id = appearance.get("team_id") or player.get("team_id")
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")

        player_team = lookups["teams"].get(player_team_id)
        home_team = lookups["teams"].get(home_team_id)
        away_team = lookups["teams"].get(away_team_id)

        is_home = None
        opponent_team = None
        if player_team_id and home_team_id and away_team_id:
            if player_team_id == home_team_id:
                is_home = True
                opponent_team = away_team
            elif player_team_id == away_team_id:
                is_home = False
                opponent_team = home_team

        # Extract odds from options (Higher/Lower)
        over_odds = -110
        under_odds = -110
        options = line_data.get("options", [])
        for opt in options:
            choice = opt.get("choice", "")
            price = opt.get("american_price")
            if price:
                try:
                    odds_val = int(float(str(price).replace("+", "")))
                except (ValueError, TypeError):
                    odds_val = -110
                if choice == "higher":
                    over_odds = odds_val
                elif choice == "lower":
                    under_odds = odds_val

        now_iso = datetime.now(_EST).isoformat()

        return {
            "player_name": player_name,
            "player_team": player_team,
            "stat_type": stat_type,
            "line": line,
            "over_line": line,
            "under_line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "book_name": "underdog_direct",
            "game_date": game_date,
            "game_time": game_time,
            "game_id": game_id,
            "opponent_team": opponent_team,
            "is_home": is_home,
            "fetch_timestamp": now_iso,
            "source": "underdog_direct",
            "fetch_source": "direct",
        }

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA player props from Underdog Fantasy.

        Returns:
            List of standardized prop dictionaries
        """
        print("\n" + "=" * 70)
        print("FETCHING UNDERDOG FANTASY NBA PROPS (Direct API)")
        print("=" * 70)
        print(f"Include combos: {self.include_combos}")
        print(f"Proxy profile: {self.proxy_profile or 'none'}")
        print("=" * 70 + "\n", flush=True)

        # Fetch raw data
        print("Fetching over/under lines...", flush=True)
        response_data = self._fetch_api_data()

        if not response_data:
            print("[WARN] Failed to fetch Underdog Fantasy data", flush=True)
            return []

        # Extract over_under_lines array
        over_under_lines = response_data.get("over_under_lines", [])

        if not over_under_lines:
            # Some API versions nest differently
            over_under_lines = response_data.get("data", [])

        print(
            f"[OK] Retrieved {len(over_under_lines)} total over/under lines",
            flush=True,
        )

        if not over_under_lines:
            print("[WARN] No over/under lines in response", flush=True)
            return []

        # Build lookup dictionaries from top-level arrays
        lookups = self._build_lookups(response_data)
        print(
            f"[OK] Lookups: {len(lookups['players'])} players, "
            f"{len(lookups['appearances'])} appearances, "
            f"{len(lookups['games'])} games, "
            f"{len(lookups['teams'])} teams",
            flush=True,
        )

        # Parse all lines
        props = []
        skipped_stat_filter = 0
        skipped_parse_error = 0
        skipped_validation = 0

        for line_data in over_under_lines:
            try:
                prop = self._parse_line(line_data, lookups)

                if prop is None:
                    skipped_stat_filter += 1
                    continue

                # Validate the prop
                if not self.validate_prop(prop):
                    skipped_validation += 1
                    continue

                props.append(prop)

            except Exception as e:
                skipped_parse_error += 1
                logger.debug(f"[underdog_direct] Parse error: {e}")

        # Deduplicate
        pre_dedup_count = len(props)
        props = self.deduplicate_props(props)
        dedup_removed = pre_dedup_count - len(props)

        # Print summary
        self._print_summary(
            total_lines=len(over_under_lines),
            props=props,
            skipped_stat_filter=skipped_stat_filter,
            skipped_parse_error=skipped_parse_error,
            skipped_validation=skipped_validation,
            dedup_removed=dedup_removed,
        )

        # Log to Atlas data registry
        self._log_registry(props)

        return props

    def _print_summary(
        self,
        total_lines: int,
        props: List[Dict],
        skipped_stat_filter: int,
        skipped_parse_error: int,
        skipped_validation: int,
        dedup_removed: int,
    ) -> None:
        """Print fetch summary to stdout."""
        print("\n" + "=" * 70)
        print("UNDERDOG FANTASY DIRECT FETCH SUMMARY")
        print("=" * 70)
        print(f"Total over/under lines: {total_lines}")
        print(f"Skipped (stat filter):  {skipped_stat_filter}")
        print(f"Skipped (parse errors): {skipped_parse_error}")
        print(f"Skipped (validation):   {skipped_validation}")
        print(f"Duplicates removed:     {dedup_removed}")
        print(f"Final parsed props:     {len(props)}")
        print()

        if not props:
            print("[WARN] No props parsed!")
            print("=" * 70 + "\n", flush=True)
            return

        # Breakdown by stat type
        stat_counts: Dict[str, int] = {}
        for p in props:
            stat = p["stat_type"]
            stat_counts[stat] = stat_counts.get(stat, 0) + 1

        print("Lines by stat type:")
        for stat, count in sorted(stat_counts.items(), key=lambda x: -x[1]):
            stat_props = [p for p in props if p["stat_type"] == stat]
            avg_line = sum(p["line"] for p in stat_props) / len(stat_props)
            print(f"  {stat:15s}: {count:4d} props (avg line: {avg_line:.1f})")

        print()

        # Team coverage
        teams_seen = set()
        for p in props:
            if p.get("player_team"):
                teams_seen.add(p["player_team"])
        print(f"Teams with props: {len(teams_seen)}")
        if teams_seen:
            print(f"  {', '.join(sorted(teams_seen))}")

        # Game date distribution
        date_counts: Dict[str, int] = {}
        for p in props:
            d = p.get("game_date", "unknown")
            date_counts[d] = date_counts.get(d, 0) + 1
        if date_counts:
            print()
            print("Props by game date:")
            for d, count in sorted(date_counts.items()):
                print(f"  {d}: {count} props")

        print("=" * 70 + "\n", flush=True)

    def _log_registry(self, props: List[Dict]) -> None:
        """Log fetch results to Atlas data registry (fire-and-forget)."""
        try:
            from nba.core.data_registry import log_ingestion

            stats = self.get_registry_stats()
            log_ingestion(
                "underdog_direct",
                "fetch",
                "success" if props else "empty",
                records_fetched=len(props),
                api_calls_made=stats["api_calls_made"],
                bytes_transferred=stats["bytes_transferred"],
                error_count=stats["error_count"],
                error_message=stats["error_message"],
                metadata={
                    "include_combos": self.include_combos,
                    "proxy_profile": self.proxy_profile,
                },
            )
        except Exception:
            pass


def main():
    """Main execution with argparse CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA player props directly from Underdog Fantasy API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch core stats (POINTS, REBOUNDS, ASSISTS, THREES)
  python fetch_underdog_direct.py

  # Include combo stats (PRA, PR, PA, RA)
  python fetch_underdog_direct.py --include-combos

  # Save to JSON file
  python fetch_underdog_direct.py --save

  # Quiet mode (minimal output)
  python fetch_underdog_direct.py --quiet

  # Combine flags
  python fetch_underdog_direct.py --include-combos --save --quiet
        """,
    )
    parser.add_argument(
        "--include-combos",
        action="store_true",
        help="Include combo stats (PRA, PR, PA, RA)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save props to JSON file in lines/ directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (suppress verbose logging)",
    )

    args = parser.parse_args()

    with UnderdogDirectFetcher(
        include_combos=args.include_combos,
        verbose=not args.quiet,
    ) as fetcher:
        props = fetcher.fetch()

        if props and args.save:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")
        elif not props:
            print("\n[WARN] No props fetched!\n")
        else:
            # Show sample props
            print("\n=== SAMPLE PROPS ===\n")
            for prop in props[:10]:
                team_str = prop.get("player_team") or "???"
                opp_str = prop.get("opponent_team") or "???"
                home_str = (
                    "H" if prop.get("is_home") else "A" if prop.get("is_home") is False else "?"
                )
                print(
                    f"{prop['player_name']:25s} | {team_str:3s} {home_str} vs {opp_str:3s} | "
                    f"{prop['stat_type']:10s} | {prop['line']:5.1f}"
                )

            remaining = len(props) - 10
            if remaining > 0:
                print(f"\n  ... and {remaining} more props")

        # Log Atlas registry stats
        stats = fetcher.get_registry_stats()
        print(f"\n--- Registry Stats ---")
        print(f"API calls: {stats['api_calls_made']}")
        print(f"Bytes transferred: {stats['bytes_transferred']:,}")
        print(f"Errors: {stats['error_count']}")
        if stats["error_message"]:
            print(f"Last error: {stats['error_message']}")


if __name__ == "__main__":
    main()
