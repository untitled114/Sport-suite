#!/usr/bin/env python3
"""
FanDuel Direct Fetcher for NBA XL System
==========================================
Fetches NBA player props directly from the FanDuel sportsbook API.

Two-step approach:
1. Hit content-managed-page with customPageId=nba to discover today's game events.
2. For each game event, hit event-page with per-stat tabs (player-points,
   player-rebounds, etc.) to retrieve runner-level prop lines and odds.

The sbapi.il.sportsbook.fanduel.com host works globally — the .co host
causes SSL handshake failures from all IPs.

Proxy profile: sportsbooks (Colorado residential IP) — FanDuel is
geo-fenced and blocks datacenter IPs.

Fallback chain per request:
1. curl_cffi with Chrome TLS impersonation (fastest, bypasses basic CF)
2. Standard requests via BaseFetcher (slower, may get blocked)

Usage:
    python fetch_fanduel_direct.py                # Fetch today's props
    python fetch_fanduel_direct.py --save          # Fetch and save to JSON
    python fetch_fanduel_direct.py --quiet         # Suppress verbose output
    python fetch_fanduel_direct.py --all-stats     # Include all stat types
"""

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

# EST timezone — standard across the entire pipeline
EST = ZoneInfo("America/New_York")


class FanDuelDirectFetcher(BaseFetcher):
    """Fetches NBA player props directly from FanDuel SBK API."""

    # ── API Configuration ────────────────────────────────────────────────
    # FanDuel uses state-specific API hosts. sbapi.il works globally (not geo-blocked).
    # sbapi.co.fanduel.com causes SSL handshake failures from all IPs.
    BASE_URL = "https://sbapi.il.sportsbook.fanduel.com/api"
    CONTENT_PAGE_URL = f"{BASE_URL}/content-managed-page"
    EVENT_PAGE_URL = f"{BASE_URL}/event-page"
    API_KEY = "FhMFpcPWXMeyZxOx"
    DEFAULT_PARAMS = {
        "_ak": API_KEY,
        "betexRegion": "GBR",
        "capiJurisdiction": "intl",
        "currencyCode": "USD",
        "exchangeLocale": "en_US",
        "language": "en",
        "regionCode": "NAMERICA",
    }

    # Tab names for per-event player prop queries
    STAT_TABS = {
        "POINTS": "player-points",
        "REBOUNDS": "player-rebounds",
        "ASSISTS": "player-assists",
        "THREES": "player-threes",
    }

    # ── Market type mapping ──────────────────────────────────────────────
    # FanDuel market types (from attachments.markets[].marketType)
    # Real types seen: PLAYER_F_TOTAL_POINTS, PLAYER_I_ALT_TOTAL_REBOUNDS, etc.
    MARKET_TYPE_MAP = {
        "PLAYER_F_TOTAL_POINTS": "POINTS",
        "PLAYER_I_ALT_TOTAL_POINTS": "POINTS",
        "ALTERNATE_PLAYER_POINTS": "POINTS",
        "PLAYER_POINTS": "POINTS",
        "PLAYER_F_TOTAL_REBOUNDS": "REBOUNDS",
        "PLAYER_I_ALT_TOTAL_REBOUNDS": "REBOUNDS",
        "ALTERNATE_PLAYER_REBOUNDS": "REBOUNDS",
        "PLAYER_REBOUNDS": "REBOUNDS",
        "PLAYER_F_TOTAL_ASSISTS": "ASSISTS",
        "PLAYER_I_ALT_TOTAL_ASSISTS": "ASSISTS",
        "ALTERNATE_PLAYER_ASSISTS": "ASSISTS",
        "PLAYER_ASSISTS": "ASSISTS",
        "PLAYER_A_TOTAL_THREES": "THREES",
        "PLAYER_A_ALT_TOTAL_THREES": "THREES",
        "PLAYER_THREES_MADE": "THREES",
        "PLAYER_THREE_POINTERS_MADE": "THREES",
        "ALTERNATE_PLAYER_THREES_MADE": "THREES",
        "PLAYER_STEALS": "STEALS",
        "PLAYER_BLOCKS": "BLOCKS",
        "PLAYER_TURNOVERS": "TURNOVERS",
        # Combos (tracked but filtered unless --all-stats)
        "PLAYER_E_TOTAL_POINTS_+_REBOUNDS": "PTS_REBS",
        "PLAYER_POINTS_PLUS_REBOUNDS": "PTS_REBS",
        "PLAYER_POINTS_PLUS_ASSISTS": "PTS_ASTS",
        "PLAYER_REBOUNDS_PLUS_ASSISTS": "REBS_ASTS",
        "PLAYER_POINTS_PLUS_REBOUNDS_PLUS_ASSISTS": "PTS_REBS_ASTS",
        "PLAYER_STEALS_PLUS_BLOCKS": "STLS_BLKS",
        "PLAYER_DOUBLE_DOUBLE": "DOUBLE_DOUBLE",
    }

    # Core stat types we care about for the XL model
    CORE_STATS = {"POINTS", "REBOUNDS", "ASSISTS", "THREES"}

    # Extended stats (included with --all-stats)
    EXTENDED_STATS = {
        "STEALS",
        "BLOCKS",
        "TURNOVERS",
        "PTS_REBS",
        "PTS_ASTS",
        "REBS_ASTS",
        "PTS_REBS_ASTS",
        "STLS_BLKS",
    }

    # ── Team abbreviation normalization ──────────────────────────────────
    # FanDuel uses full city names in event names; we parse abbreviations
    # from the event/runner data and normalize to our DB standard.
    TEAM_ABBREV_MAP = {
        "NO": "NOP",
        "NOR": "NOP",
        "SA": "SAS",
        "GS": "GSW",
        "NY": "NYK",
        "BRK": "BKN",
        "PHO": "PHX",
        "UTAH": "UTA",
        "WSH": "WAS",
    }

    # ── Full team name → abbreviation map for event name parsing ─────────
    TEAM_NAME_TO_ABBREV = {
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
        "LA Clippers": "LAC",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "LA Lakers": "LAL",
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

    def __init__(
        self,
        all_stats: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize FanDuel direct fetcher.

        Args:
            all_stats: Include all stat types, not just core 4.
            verbose: Enable verbose logging.
        """
        super().__init__(
            source_name="fanduel_direct",
            rate_limit=2.0,
            max_retries=3,
            timeout=30,
            verbose=verbose,
            proxy_profile="sportsbooks",
        )

        self.all_stats = all_stats

    # ── Odds Conversion ──────────────────────────────────────────────────

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """
        Convert decimal odds to American odds.

        Args:
            decimal_odds: European decimal odds (e.g. 1.91, 2.10).

        Returns:
            American odds integer (e.g. -110, +110).
        """
        if decimal_odds is None or decimal_odds <= 1.0:
            return -110  # Fallback for invalid/missing odds

        if decimal_odds >= 2.0:
            return round((decimal_odds - 1) * 100)
        else:
            return round(-100 / (decimal_odds - 1))

    # ── Team Normalization ───────────────────────────────────────────────

    def _normalize_team(self, abbrev: str) -> str:
        """Normalize a team abbreviation to our DB standard."""
        if not abbrev:
            return ""
        abbrev = abbrev.strip().upper()
        return self.TEAM_ABBREV_MAP.get(abbrev, abbrev)

    def _team_name_to_abbrev(self, team_name: str) -> str:
        """Convert a full team name (e.g. 'Orlando Magic') to abbreviation."""
        abbrev = self.TEAM_NAME_TO_ABBREV.get(team_name, "")
        if not abbrev:
            # Try partial match — some names may differ slightly
            team_lower = team_name.lower()
            for full_name, code in self.TEAM_NAME_TO_ABBREV.items():
                if full_name.lower() in team_lower or team_lower in full_name.lower():
                    return code
        return abbrev

    # ── Request Methods ──────────────────────────────────────────────────

    def _get_fanduel_headers(self) -> Dict[str, str]:
        """Build FanDuel-specific request headers."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://sportsbook.fanduel.com/",
            "Origin": "https://sportsbook.fanduel.com",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

    def _fetch_with_curl_cffi(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Fetch a URL using curl_cffi with Chrome TLS impersonation.

        Uses the sportsbooks proxy profile if configured.

        Args:
            url: Full URL to fetch.
            params: Query parameters for the request.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        proxies = None
        if self.proxy_profile:
            from nba.betting_xl.fetchers.proxy_manager import get_proxy_manager

            pm = get_proxy_manager()
            proxy_dict = pm.get_proxies_dict(self.proxy_profile)
            if proxy_dict:
                proxies = proxy_dict
                if self.verbose:
                    state = pm.PROFILE_STATES.get(self.proxy_profile, self.proxy_profile)
                    logger.info(f"[fanduel_direct] curl_cffi using {state} proxy")

        max_retries = 3
        for attempt in range(max_retries):
            delay = random.uniform(1.0, 2.5)
            time.sleep(delay)

            try:
                response = cffi_requests.get(
                    url,
                    params=params,
                    headers=self._get_fanduel_headers(),
                    impersonate="chrome",
                    timeout=self.timeout,
                    proxies=proxies,
                )

                self._api_calls += 1
                self._bytes_transferred += len(response.content)

                if response.status_code == 403:
                    logger.warning(
                        f"[fanduel_direct] 403 Forbidden " f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                    continue

                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"[fanduel_direct] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                if self.verbose:
                    logger.info(
                        f"[fanduel_direct] curl_cffi success: "
                        f"{response.status_code} ({len(response.content)} bytes)"
                    )

                return data

            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(
                    f"[fanduel_direct] curl_cffi attempt "
                    f"{attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))

        logger.error("[fanduel_direct] curl_cffi: all attempts failed")
        return None

    def _fetch_json(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Fetch JSON from a URL with curl_cffi fallback to standard requests.

        Args:
            url: Full URL to fetch.
            params: Query parameters.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        # ── Attempt 1: curl_cffi ─────────────────────────────────────
        if HAS_CURL_CFFI:
            result = self._fetch_with_curl_cffi(url, params)
            if result is not None:
                return result

        # ── Attempt 2: standard requests via BaseFetcher ─────────────
        logger.info("[fanduel_direct] Falling back to standard requests")
        response = self._make_request(
            url=url,
            params=params,
            headers=self._get_fanduel_headers(),
        )
        if response is not None:
            try:
                return response.json()
            except (ValueError, AttributeError) as e:
                logger.warning(f"[fanduel_direct] Failed to parse JSON: {e}")
        return None

    # ── Step 1: Discover NBA events ──────────────────────────────────────

    def _fetch_nba_events(self) -> Dict[str, Dict]:
        """
        Fetch the NBA content-managed page to discover today's game events.

        Hits CONTENT_PAGE_URL with page=CUSTOM, customPageId=nba.
        Filters to real game events (those with " @ " in the name),
        skipping futures and specials.

        Returns:
            Dict mapping event_id (str) -> event dict with keys:
            {event_id, name, home_team, away_team, game_date, game_time}.
            Empty dict on failure.
        """
        params = {
            "page": "CUSTOM",
            "customPageId": "nba",
            **self.DEFAULT_PARAMS,
        }

        print("Step 1: Fetching NBA event list...", flush=True)
        data = self._fetch_json(self.CONTENT_PAGE_URL, params)

        if not data:
            print("[WARN] Failed to fetch NBA content page", flush=True)
            return {}

        attachments = data.get("attachments", {})
        raw_events = attachments.get("events", {})

        if not raw_events:
            print("[WARN] No events in content page response", flush=True)
            return {}

        # Filter to real game events (contain " @ " in name)
        game_events = {}
        skipped = 0

        for eid, event in raw_events.items():
            name = event.get("name", "")
            if " @ " not in name:
                skipped += 1
                continue

            # Parse "Away Team @ Home Team" from event name
            parts = name.split(" @ ", 1)
            away_name = parts[0].strip()
            home_name = parts[1].strip()

            away_team = self._team_name_to_abbrev(away_name)
            home_team = self._team_name_to_abbrev(home_name)

            # Also try competitors array if name parsing missed
            if not away_team or not home_team:
                competitors = event.get("competitors", [])
                for comp in competitors:
                    abbrev = comp.get("abbreviation", "") or comp.get("teamAbbreviation", "")
                    abbrev = self._normalize_team(abbrev)
                    status = (comp.get("homeAwayStatus", "") or "").upper()
                    if status == "HOME" and not home_team:
                        home_team = abbrev
                    elif status == "AWAY" and not away_team:
                        away_team = abbrev

            # Parse game date/time from openDate
            game_date = None
            game_time = None
            open_date = event.get("openDate", "")
            if open_date:
                try:
                    dt = datetime.fromisoformat(open_date.replace("Z", "+00:00"))
                    dt_est = dt.astimezone(EST)
                    game_date = dt_est.strftime("%Y-%m-%d")
                    game_time = dt_est.strftime("%H:%M:%S")
                except (ValueError, AttributeError):
                    pass

            if not game_date:
                game_date = datetime.now(EST).strftime("%Y-%m-%d")

            eid_str = str(eid)
            game_events[eid_str] = {
                "event_id": eid_str,
                "name": name,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date,
                "game_time": game_time,
            }

        print(
            f"[OK] Found {len(game_events)} game events " f"(skipped {skipped} non-game entries)",
            flush=True,
        )

        for eid, info in game_events.items():
            matchup = f"{info['away_team']} @ {info['home_team']}"
            time_str = info["game_time"] or "TBD"
            print(f"  {matchup:20s}  (event {eid}, tip {time_str})", flush=True)

        return game_events

    # ── Step 2: Fetch per-event prop tabs ────────────────────────────────

    def _fetch_event_tab(self, event_id: str, tab: str) -> Optional[Dict[str, Dict]]:
        """
        Fetch player prop markets for a specific event and stat tab.

        Hits EVENT_PAGE_URL with eventId, tab, and includePrices=true.

        Args:
            event_id: FanDuel event ID.
            tab: Tab name (e.g. "player-points", "player-rebounds").

        Returns:
            Markets dict from response.attachments.markets, or None on failure.
        """
        params = {
            "eventId": event_id,
            "tab": tab,
            "includePrices": "true",
            **self.DEFAULT_PARAMS,
        }

        data = self._fetch_json(self.EVENT_PAGE_URL, params)

        if not data:
            return None

        attachments = data.get("attachments", {})
        markets = attachments.get("markets", {})

        if not markets:
            # Try alternate structure
            markets = data.get("markets", {})

        return markets if markets else None

    # ── Parsing ──────────────────────────────────────────────────────────

    def _extract_american_odds_from_runner(self, runner: Dict) -> int:
        """
        Extract American odds from a runner.

        FanDuel nests American odds at:
            runner.winRunnerOdds.americanDisplayOdds.americanOdds

        Falls back to decimal conversion if American format not available.

        Args:
            runner: A runner dict from a market.

        Returns:
            American odds integer (-110 as fallback).
        """
        try:
            win_odds = runner.get("winRunnerOdds", {})

            # Primary: americanDisplayOdds.americanOdds (string like "-110")
            american_display = win_odds.get("americanDisplayOdds", {})
            american_str = american_display.get("americanOdds")
            if american_str is not None:
                return int(float(american_str))

            # Secondary: direct americanOdds field
            direct_american = win_odds.get("americanOdds")
            if direct_american is not None:
                return int(float(direct_american))

            # Tertiary: convert from decimal
            decimal_val = win_odds.get("trueOdds", {}).get("decimalOdds", {}).get("decimalOdds")
            if decimal_val is not None:
                return self.decimal_to_american(float(decimal_val))

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[fanduel_direct] Odds extraction error: {e}")

        return -110  # Standard fallback

    def _extract_player_name_from_runner(self, runner_name: str) -> Optional[str]:
        """
        Extract player name from a runner's name.

        FanDuel runner names follow the pattern:
            "Jalen Johnson Over" or "Jalen Johnson Under"

        Args:
            runner_name: The runnerName field from a runner.

        Returns:
            Player name string, or None if unparseable.
        """
        if not runner_name:
            return None

        cleaned = runner_name.strip()

        # Strip trailing "Over" or "Under" (with optional line number)
        cleaned = re.sub(r"\s+(Over|Under)\s*[\d.]*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[-–]\s*(Over|Under).*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        # Validate: must be at least two words and not just "Over"/"Under"
        if cleaned and len(cleaned.split()) >= 2:
            if cleaned.lower() not in ("over", "under", "o", "u"):
                return cleaned

        return None

    def _parse_markets_for_stat(
        self,
        markets: Dict[str, Dict],
        stat_type: str,
        event_info: Dict,
    ) -> List[Dict[str, Any]]:
        """
        Parse all markets from an event-tab response into prop dicts.

        Groups runners by player+handicap to pair Over/Under sides.

        Args:
            markets: Markets dict from attachments.markets.
            stat_type: The stat type this tab represents (POINTS, REBOUNDS, etc.).
            event_info: Pre-parsed event info dict.

        Returns:
            List of standardized prop dicts.
        """
        props = []
        now_est = datetime.now(EST)
        fetch_ts = now_est.isoformat()

        home_team = event_info.get("home_team")
        away_team = event_info.get("away_team")
        game_date = event_info.get("game_date", now_est.strftime("%Y-%m-%d"))
        game_time = event_info.get("game_time")
        event_id = event_info.get("event_id", "")

        for market_id, market in markets.items():
            market_type = market.get("marketType", "")

            # Only process PLAYER markets — skip game totals, team totals, etc.
            if "PLAYER" not in market_type:
                continue

            # Resolve stat type from market type if available in our map
            resolved_stat = self.MARKET_TYPE_MAP.get(market_type, stat_type)

            # Filter by stat type preference
            if not self.all_stats and resolved_stat not in self.CORE_STATS:
                continue

            runners = market.get("runners", [])
            if not runners:
                continue

            # Group runners by player name + handicap to pair Over/Under
            runner_groups: Dict[str, Dict[str, Any]] = {}

            for runner in runners:
                runner_name = runner.get("runnerName", "")
                handicap = runner.get("handicap")

                if handicap is None:
                    continue

                player_name = self._extract_player_name_from_runner(runner_name)
                if not player_name:
                    continue

                player_name = self.normalize_player_name(player_name)
                line = float(handicap)

                # Skip "to score any" alt markets (handicap=0)
                if line <= 0:
                    continue

                key = f"{player_name}|{line}"

                if key not in runner_groups:
                    runner_groups[key] = {
                        "player_name": player_name,
                        "line": line,
                        "over_odds": -110,
                        "under_odds": -110,
                    }

                # Determine Over/Under from runner name
                name_lower = runner_name.lower()
                is_over = "over" in name_lower

                odds = self._extract_american_odds_from_runner(runner)
                if is_over:
                    runner_groups[key]["over_odds"] = odds
                else:
                    runner_groups[key]["under_odds"] = odds

            # Build prop dicts from grouped runners
            for _key, group in runner_groups.items():
                prop = {
                    "player_name": group["player_name"],
                    "stat_type": resolved_stat,
                    "line": group["line"],
                    "over_line": group["line"],
                    "under_line": group["line"],
                    "over_odds": group["over_odds"],
                    "under_odds": group["under_odds"],
                    "book_name": "fanduel_direct",
                    "game_date": game_date,
                    "game_time": game_time,
                    "game_id": event_id,
                    "opponent_team": None,
                    "is_home": None,
                    "fetch_timestamp": fetch_ts,
                    "source": "fanduel_direct",
                    "fetch_source": "direct",
                    # Extra context for debugging / line shopping
                    "market_type": market_type,
                    "market_id": str(market_id),
                    "home_team": home_team,
                    "away_team": away_team,
                }

                if self.validate_prop(prop):
                    props.append(prop)

        return props

    # ── Main Fetch ───────────────────────────────────────────────────────

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA player props from FanDuel.

        Two-step approach:
        1. Discover today's game events via content-managed-page.
        2. For each event, fetch each stat tab and parse runner markets.

        Returns:
            List of standardized prop dictionaries.
        """
        now_est = datetime.now(EST)
        today_str = now_est.strftime("%Y-%m-%d")

        print("\n" + "=" * 70)
        print("FETCHING FANDUEL NBA PROPS (Direct API)")
        print("=" * 70)
        print(f"Date: {today_str}")
        print(f"All stats: {self.all_stats}")
        print(f"curl_cffi: {'available' if HAS_CURL_CFFI else 'NOT installed'}")
        print("=" * 70 + "\n", flush=True)

        # ── Step 1: Discover game events ─────────────────────────────
        game_events = self._fetch_nba_events()

        if not game_events:
            print("[WARN] No NBA game events found", flush=True)
            return []

        # ── Step 2: Fetch per-event, per-stat-tab props ──────────────
        all_props = []
        tab_counts: Dict[str, int] = {}

        # Determine which tabs to fetch
        if self.all_stats:
            tabs_to_fetch = dict(self.STAT_TABS)
        else:
            # Only core stats
            tabs_to_fetch = {k: v for k, v in self.STAT_TABS.items() if k in self.CORE_STATS}

        total_requests = len(game_events) * len(tabs_to_fetch)
        request_num = 0

        for eid, event_info in game_events.items():
            matchup = f"{event_info['away_team']} @ {event_info['home_team']}"

            for stat_type, tab_name in tabs_to_fetch.items():
                request_num += 1
                print(
                    f"  [{request_num}/{total_requests}] " f"{matchup} / {stat_type}...",
                    end="",
                    flush=True,
                )

                markets = self._fetch_event_tab(eid, tab_name)

                if not markets:
                    print(" 0 markets", flush=True)
                    continue

                props = self._parse_markets_for_stat(markets, stat_type, event_info)
                all_props.extend(props)

                tab_key = stat_type
                tab_counts[tab_key] = tab_counts.get(tab_key, 0) + len(props)

                print(f" {len(props)} props ({len(markets)} markets)", flush=True)

                # Rate limit between event-tab requests
                time.sleep(1)

        # Deduplicate
        all_props = self.deduplicate_props(all_props)

        # ── Summary ──────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FANDUEL DIRECT FETCH SUMMARY")
        print("=" * 70)
        print(f"Games: {len(game_events)}")
        print(f"Total parsed props: {len(all_props)}")

        print("\nLines by stat type:")
        stat_counts: Dict[str, int] = {}
        for p in all_props:
            st = p["stat_type"]
            stat_counts[st] = stat_counts.get(st, 0) + 1

        for stat, count in sorted(stat_counts.items(), key=lambda x: -x[1]):
            stat_props = [p for p in all_props if p["stat_type"] == stat]
            avg_line = sum(p["line"] for p in stat_props) / max(len(stat_props), 1)
            print(f"  {stat:20s}: {count:4d} props (avg line: {avg_line:.1f})")

        # Breakdown by game
        print("\nLines by game:")
        game_counts: Dict[str, int] = {}
        for p in all_props:
            matchup_parts = []
            if p.get("away_team"):
                matchup_parts.append(p["away_team"])
            matchup_parts.append("@")
            if p.get("home_team"):
                matchup_parts.append(p["home_team"])
            matchup = " ".join(matchup_parts) if len(matchup_parts) > 1 else p["game_id"]
            game_counts[matchup] = game_counts.get(matchup, 0) + 1

        for matchup, count in sorted(game_counts.items(), key=lambda x: -x[1]):
            print(f"  {matchup:25s}: {count:4d} props")

        print("=" * 70 + "\n", flush=True)

        # Atlas data registry — log this ingestion
        try:
            from nba.core.data_registry import log_ingestion

            stats = self.get_registry_stats()
            log_ingestion(
                "fanduel_direct",
                "fetch",
                "success" if all_props else "empty",
                records_fetched=len(all_props),
                api_calls_made=stats["api_calls_made"],
                bytes_transferred=stats["bytes_transferred"],
                error_count=stats["error_count"],
                error_message=stats["error_message"],
                metadata={"game_date": today_str, "all_stats": self.all_stats},
            )
        except Exception:
            pass

        return all_props


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA player props directly from FanDuel API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's props (core stats only)
  python fetch_fanduel_direct.py

  # Include all stat types
  python fetch_fanduel_direct.py --all-stats

  # Save to JSON file
  python fetch_fanduel_direct.py --save

  # Quiet mode (less output)
  python fetch_fanduel_direct.py --quiet
        """,
    )
    parser.add_argument(
        "--all-stats",
        action="store_true",
        help="Include all stat types (not just POINTS/REBOUNDS/ASSISTS/THREES)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")

    args = parser.parse_args()

    with FanDuelDirectFetcher(
        all_stats=args.all_stats,
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
            for prop in props[:15]:
                odds_str = f"O {prop['over_odds']:+d} / U {prop['under_odds']:+d}"
                print(
                    f"{prop['player_name']:25s} | {prop['stat_type']:10s} "
                    f"| {prop['line']:5.1f} | {odds_str}"
                )
            if len(props) > 15:
                print(f"  ... and {len(props) - 15} more")


if __name__ == "__main__":
    main()
