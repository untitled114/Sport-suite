#!/usr/bin/env python3
"""
BetMGM Direct Fetcher for NBA XL System
=========================================
Fetches NBA player props directly from BetMGM (Entain/bwin CDS platform).

API Flow:
1. GET /fixtures — list today's NBA game fixtures (sportIds=7, competitionIds=6004)
2. GET /fixture-view?fixtureIds={id} — all markets for a single game fixture
3. Parse optionMarkets[] for player O/U props (MarketType=Over/Under, Period=FullTime)

Player prop structure in optionMarkets:
- Market name: "Jalen Johnson - Assists" (rsplit on " - " for hyphenated names)
- Line: parameters[] where key="DecimalValue"
- Stat: parameters[] where key="Happening" (Point, Rebound, Assist, ThreePointer, etc.)
- Options: options[].parameters.optionTypes=["Over"] or ["Under"]
- Odds: options[].price.americanOdds (already American format)

Usage:
    python fetch_betmgm_direct.py               # Fetch today's props
    python fetch_betmgm_direct.py --save         # Save to JSON
    python fetch_betmgm_direct.py --quiet        # Quiet mode
    python fetch_betmgm_direct.py --include-combos  # Include combo markets
"""

import logging
import os
import random
import time
from datetime import datetime
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
EST = ZoneInfo("America/New_York")

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
BETMGM_CDS_BASE = "https://sports.co.betmgm.com/cds-api/bettingoffer"
BETMGM_FIXTURES_URL = f"{BETMGM_CDS_BASE}/fixtures"
BETMGM_FIXTURE_VIEW_URL = f"{BETMGM_CDS_BASE}/fixture-view"
ACCESS_ID = os.getenv("BETMGM_ACCESS_ID", "OTU4NDk3MzEtOTAyNS00MjQzLWIxNWEtNTI2MjdhNWM3Zjk3")

TEAM_MAP = {
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

TEAM_NAME_MAP = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

# Map BetMGM "Happening" parameter values to canonical stat types
HAPPENING_MAP = {
    "Point": "POINTS",
    "Rebound": "REBOUNDS",
    "Assist": "ASSISTS",
    "ThreePointer": "THREES",
    "Steal": "STEALS",
    "Block": "BLOCKS",
    "Turnover": "TURNOVERS",
}

# Map market name suffixes (after " - ") to canonical stat types
MARKET_NAME_STAT_MAP = {
    "points": "POINTS",
    "rebounds": "REBOUNDS",
    "assists": "ASSISTS",
    "threes": "THREES",
    "three pointers": "THREES",
    "3-pointers": "THREES",
    "steals": "STEALS",
    "blocks": "BLOCKS",
    "turnovers": "TURNOVERS",
    "pts + reb + ast": "PRA",
    "points + rebounds + assists": "PRA",
    "pts + reb": "PR",
    "points + rebounds": "PR",
    "pts + ast": "PA",
    "points + assists": "PA",
    "reb + ast": "RA",
    "rebounds + assists": "RA",
}

COMBO_MARKETS = {"PRA", "PR", "PA", "RA"}


class BetMGMDirectFetcher(BaseFetcher):
    """Fetches NBA player props directly from BetMGM (Entain CDS platform).

    Two-step flow:
    1. Fetch NBA fixture list to get fixture IDs and game metadata.
    2. For each fixture, fetch fixture-view to get all optionMarkets (player props).
    """

    def __init__(self, verbose: bool = True, include_combos: bool = False):
        super().__init__(
            source_name="betmgm_direct",
            rate_limit=2.5,
            max_retries=3,
            timeout=30,
            verbose=verbose,
            proxy_profile="sportsbooks",
        )
        self.include_combos = include_combos
        self.today = datetime.now(EST).strftime("%Y-%m-%d")
        self._schedule: Dict[str, Dict] = {}

    # ── ESPN schedule (for home/away + opponent enrichment) ──────────

    def _fetch_schedule(self) -> List[Dict]:
        """Fetch today's NBA schedule from ESPN scoreboard."""
        url = f"{ESPN_SCOREBOARD_URL}?dates={self.today.replace('-', '')}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"[{self.source_name}] ESPN schedule failed: {e}")
            return []

        games = []
        for event in data.get("events", []):
            comps = event.get("competitions", [{}])
            home = away = None
            for t in comps[0].get("competitors", []):
                ab = TEAM_MAP.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
                if t.get("homeAway") == "home":
                    home = ab
                else:
                    away = ab
            if home and away:
                eid = event.get("id", "")
                games.append(
                    {
                        "event_id": eid,
                        "home_team": home,
                        "away_team": away,
                        "start_time": event.get("date", ""),
                    }
                )
                self._schedule[home] = {
                    "opponent": away,
                    "is_home": True,
                    "game_id": eid,
                    "start_time": event.get("date", ""),
                }
                self._schedule[away] = {
                    "opponent": home,
                    "is_home": False,
                    "game_id": eid,
                    "start_time": event.get("date", ""),
                }
        return games

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _cffi_fetch(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch using curl_cffi with browser TLS impersonation."""
        if not HAS_CURL_CFFI:
            return None
        headers = {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Referer": "https://sports.co.betmgm.com/",
            "Origin": "https://sports.co.betmgm.com",
        }
        proxies = dict(self.session.proxies) if self.session.proxies else None

        for attempt in range(self.max_retries):
            time.sleep(random.uniform(1.0, 2.5))
            try:
                r = cffi_requests.get(
                    url,
                    params=params,
                    headers=headers,
                    impersonate="chrome",
                    timeout=self.timeout,
                    proxies=proxies,
                )
                if r.status_code in (403, 429):
                    wait = (5 if r.status_code == 403 else 10) * (attempt + 1)
                    time.sleep(wait)
                    continue
                if r.status_code != 200 or len(r.content) < 50:
                    continue
                self._api_calls += 1
                self._bytes_transferred += len(r.content)
                return r.json()
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
        return None

    def _requests_fetch(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fallback fetch using standard requests session."""
        headers = {
            "Accept": "application/json",
            "Referer": "https://sports.co.betmgm.com/",
            "Origin": "https://sports.co.betmgm.com",
        }
        response = self._make_request(url, method="GET", params=params, headers=headers)
        if not response or len(response.content) < 50:
            return None
        if "json" not in response.headers.get("content-type", "").lower():
            return None
        try:
            return response.json()
        except (ValueError, TypeError):
            return None

    def _api_fetch(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch with curl_cffi -> requests fallback chain."""
        if HAS_CURL_CFFI:
            result = self._cffi_fetch(url, params)
            if result is not None:
                return result
        return self._requests_fetch(url, params)

    # ── BetMGM API: Step 1 — fixture list ────────────────────────────

    def _fetch_fixtures(self) -> List[Dict]:
        """Fetch today's NBA fixtures from BetMGM CDS API.

        Returns list of fixture dicts with id, name, startDate, participants.
        """
        params = {
            "x-bwin-accessid": ACCESS_ID,
            "lang": "en-us",
            "country": "US",
            "userCountry": "US",
            "fixtureTypes": "Standard",
            "state": "Latest",
            "sportIds": "7",
            "competitionIds": "6004",
            "offerMapping": "Filtered",
            "offerCategories": "Gridable",
            "sortBy": "Tags",
        }
        data = self._api_fetch(BETMGM_FIXTURES_URL, params)
        if not data or not isinstance(data, dict):
            return []
        return data.get("fixtures", [])

    # ── BetMGM API: Step 2 — fixture-view per game ──────────────────

    def _fetch_fixture_view(self, fixture_id: int) -> Optional[Dict]:
        """Fetch full fixture-view (all markets) for a single game fixture.

        Returns the fixture dict containing optionMarkets[] and participants[].
        """
        params = {
            "x-bwin-accessid": ACCESS_ID,
            "offerMapping": "All",
            "lang": "en-us",
            "country": "US",
            "fixtureIds": str(fixture_id),
        }
        data = self._api_fetch(BETMGM_FIXTURE_VIEW_URL, params)
        if not data or not isinstance(data, dict):
            return None
        # Response has fixture at top level or inside fixtures[]
        if "optionMarkets" in data:
            return data
        fixtures = data.get("fixtures", [])
        if fixtures:
            return fixtures[0]
        # Sometimes returned as {"fixture": {...}}
        return data.get("fixture")

    # ── Parsing helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_param(parameters: List[Dict], key: str) -> Optional[str]:
        """Extract a parameter value from BetMGM's parameters[] array.

        Parameters look like: [{"key": "DecimalValue", "value": "7.5000"}, ...]
        """
        if not parameters:
            return None
        for p in parameters:
            if isinstance(p, dict) and p.get("key") == key:
                return p.get("value")
        return None

    @staticmethod
    def _get_option_type(option: Dict) -> Optional[str]:
        """Extract 'Over' or 'Under' from an option's parameters.optionTypes."""
        params = option.get("parameters", {})
        if not params:
            return None
        option_types = params.get("optionTypes", [])
        if isinstance(option_types, list):
            for ot in option_types:
                if ot in ("Over", "Under"):
                    return ot
        return None

    def _resolve_stat_from_happening(self, happening: Optional[str]) -> Optional[str]:
        """Map BetMGM 'Happening' parameter to canonical stat type."""
        if not happening:
            return None
        return HAPPENING_MAP.get(happening)

    def _resolve_stat_from_market_name(self, market_name: str) -> Optional[str]:
        """Extract stat type from market name suffix (after ' - ')."""
        parts = market_name.rsplit(" - ", 1)
        if len(parts) < 2:
            return None
        stat_part = parts[1].strip().lower()
        return MARKET_NAME_STAT_MAP.get(stat_part)

    def _extract_player_from_market_name(self, market_name: str) -> Optional[str]:
        """Extract player name from market name like 'Jalen Johnson - Assists'.

        Uses rsplit to handle hyphenated player names correctly:
        'Nickeil Alexander-Walker - Rebounds' -> 'Nickeil Alexander-Walker'
        """
        parts = market_name.rsplit(" - ", 1)
        if len(parts) < 2:
            return None
        name = parts[0].strip()
        if len(name) < 3:
            return None
        return self.normalize_player_name(name)

    def _parse_fixture_home_away(self, fixture_name: str) -> tuple:
        """Parse 'Orlando Magic at Atlanta Hawks' into (away_team, home_team) codes.

        BetMGM uses ' at ' as separator (not '@').
        Returns (away_code, home_code) or (None, None).
        """
        parts = fixture_name.split(" at ", 1)
        if len(parts) != 2:
            return None, None
        away_name = parts[0].strip().lower()
        home_name = parts[1].strip().lower()
        away_code = TEAM_NAME_MAP.get(away_name)
        home_code = TEAM_NAME_MAP.get(home_name)
        return away_code, home_code

    # ── Parse option markets from fixture-view ───────────────────────

    def _parse_option_markets(
        self,
        option_markets: List[Dict],
        fixture_name: str,
        fixture_id: str,
        game_time: Optional[str],
    ) -> List[Dict]:
        """Parse optionMarkets[] from a fixture-view into prop dicts.

        Filters to player Over/Under props with FullTime period.
        """
        props = []
        now_iso = datetime.now(EST).isoformat()
        away_code, home_code = self._parse_fixture_home_away(fixture_name)

        for market in option_markets:
            market_name = market.get("name", {})
            if isinstance(market_name, dict):
                market_name = market_name.get("value", "")
            if not isinstance(market_name, str):
                market_name = str(market_name) if market_name else ""

            params_list = market.get("parameters", [])

            # Filter: must be Over/Under market
            market_type = self._get_param(params_list, "MarketType")
            if market_type != "Over/Under":
                continue

            # Filter: must be FullTime period
            period = self._get_param(params_list, "Period")
            if period and period != "FullTime":
                continue

            # Must have exactly 2 options (Over + Under)
            options = market.get("options", [])
            if len(options) != 2:
                continue

            # Determine if this is a player prop — market name must contain " - "
            if " - " not in market_name:
                continue

            # Extract player name
            player_name = self._extract_player_from_market_name(market_name)
            if not player_name:
                continue

            # Resolve stat type — prefer Happening parameter, fallback to market name
            happening = self._get_param(params_list, "Happening")
            stat = self._resolve_stat_from_happening(happening)
            if not stat:
                stat = self._resolve_stat_from_market_name(market_name)
            if not stat:
                continue

            # Skip combo markets unless requested
            if stat in COMBO_MARKETS and not self.include_combos:
                continue

            # Extract line from DecimalValue parameter
            decimal_value = self._get_param(params_list, "DecimalValue")
            if not decimal_value:
                continue
            try:
                line = float(decimal_value)
            except (ValueError, TypeError):
                continue

            # Parse Over/Under options for odds
            over_odds = -110
            under_odds = -110

            for option in options:
                option_type = self._get_option_type(option)
                price = option.get("price", {})
                american_odds = price.get("americanOdds")

                if american_odds is not None:
                    try:
                        odds_val = int(american_odds)
                    except (ValueError, TypeError):
                        odds_val = -110
                else:
                    odds_val = -110

                if option_type == "Over":
                    over_odds = odds_val
                elif option_type == "Under":
                    under_odds = odds_val

            # Build prop dict
            prop = {
                "player_name": player_name,
                "stat_type": stat,
                "line": line,
                "over_line": line,
                "under_line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "book_name": "betmgm_direct",
                "game_date": self.today,
                "game_time": game_time,
                "game_id": str(fixture_id),
                "home_team": home_code,
                "away_team": away_code,
                "opponent_team": None,
                "is_home": None,
                "fetch_timestamp": now_iso,
                "source": "betmgm_direct",
                "fetch_source": "direct",
            }

            # Enrich home/away from ESPN schedule
            for team_code in [home_code, away_code]:
                if team_code and team_code in self._schedule:
                    sched = self._schedule[team_code]
                    prop["game_id"] = prop["game_id"] or sched.get("game_id", "")

            if self.validate_prop(prop):
                props.append(prop)

        return props

    # ── Main fetch ───────────────────────────────────────────────────

    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch all NBA player props from BetMGM.

        Flow:
        1. Fetch ESPN schedule for home/away enrichment.
        2. Fetch BetMGM fixture list (NBA, competition 6004).
        3. For each fixture, fetch fixture-view with all option markets.
        4. Parse player O/U props from optionMarkets[].
        """
        print("\n" + "=" * 70)
        print("FETCHING BETMGM NBA PLAYER PROPS (Direct CDS API)")
        print("=" * 70)
        print(f"Date: {self.today} | curl_cffi: {'yes' if HAS_CURL_CFFI else 'no'}")
        print("=" * 70 + "\n", flush=True)

        # Step 0: ESPN schedule for home/away enrichment
        games = self._fetch_schedule()
        if not games:
            print("[WARN] No NBA games found for today", flush=True)
            return []
        print(f"[OK] {len(games)} games on the schedule\n", flush=True)

        # Step 1: Fetch BetMGM fixture list
        print("Step 1: Fetching BetMGM fixture list...", flush=True)
        fixtures = self._fetch_fixtures()
        if not fixtures:
            print("[WARN] No BetMGM fixtures returned\n", flush=True)
            return []
        print(f"[OK] {len(fixtures)} fixtures retrieved\n", flush=True)

        # Step 2: Fetch fixture-view for each game and parse props
        all_props: List[Dict] = []
        for i, fx in enumerate(fixtures):
            fixture_id = fx.get("id")
            fixture_name = fx.get("name", {})
            if isinstance(fixture_name, dict):
                fixture_name = fixture_name.get("value", "")
            if not isinstance(fixture_name, str):
                fixture_name = str(fixture_name) if fixture_name else ""

            # Parse game time from startDate
            game_time = None
            start_date = fx.get("startDate", "")
            if start_date:
                try:
                    dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    game_time = dt.astimezone(EST).strftime("%H:%M:%S")
                except (ValueError, AttributeError):
                    pass

            if not fixture_id:
                continue

            print(
                f"  [{i + 1}/{len(fixtures)}] {fixture_name} (ID: {fixture_id})...",
                end="",
                flush=True,
            )

            self._enforce_rate_limit()
            view = self._fetch_fixture_view(fixture_id)
            if not view:
                print(" FAILED", flush=True)
                continue

            option_markets = view.get("optionMarkets", [])
            props = self._parse_option_markets(
                option_markets, fixture_name, str(fixture_id), game_time
            )
            all_props.extend(props)
            print(
                f" {len(option_markets)} markets -> {len(props)} props",
                flush=True,
            )

        print(flush=True)

        # Deduplicate
        all_props = self.deduplicate_props(all_props)

        # Summary
        print(f"\n{'=' * 70}")
        print("BETMGM DIRECT FETCH SUMMARY")
        print("=" * 70)
        print(f"Total props: {len(all_props)}\n")
        stats: Dict[str, List] = {}
        for p in all_props:
            stats.setdefault(p["stat_type"], []).append(p["line"])
        for st, lines in sorted(stats.items(), key=lambda x: -len(x[1])):
            print(
                f"  {st:15s}: {len(lines):4d} props" f" (avg line: {sum(lines) / len(lines):.1f})"
            )
        print("=" * 70 + "\n", flush=True)

        # Atlas data registry
        try:
            from nba.core.data_registry import log_ingestion

            rs = self.get_registry_stats()
            log_ingestion(
                "betmgm_direct",
                "fetch",
                "success" if all_props else "empty",
                records_fetched=len(all_props),
                api_calls_made=rs["api_calls_made"],
                bytes_transferred=rs["bytes_transferred"],
                error_count=rs["error_count"],
                error_message=rs["error_message"],
                metadata={"game_date": self.today, "games": len(games)},
            )
        except Exception:
            pass
        return all_props


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NBA props from BetMGM directly")
    parser.add_argument("--save", action="store_true", help="Save to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--include-combos", action="store_true", help="Include combo markets")
    args = parser.parse_args()

    with BetMGMDirectFetcher(verbose=not args.quiet, include_combos=args.include_combos) as f:
        props = f.fetch()
        if props and args.save:
            print(f"\n[OK] Saved {len(props)} props to: {f.save_to_json(props)}\n")
        elif not props:
            print("\n[WARN] No props fetched!\n")
        else:
            print("\n=== SAMPLE PROPS ===\n")
            for p in props[:10]:
                h = (
                    " (HOME)"
                    if p.get("is_home") is True
                    else (" (AWAY)" if p.get("is_home") is False else "")
                )
                print(
                    f"{p['player_name']:25s} | {p['stat_type']:10s} | "
                    f"{p['line']:5.1f} | {p['over_odds']:+4d}/{p['under_odds']:+4d} | "
                    f"vs {p.get('opponent_team') or '???'}{h}"
                )


if __name__ == "__main__":
    main()
