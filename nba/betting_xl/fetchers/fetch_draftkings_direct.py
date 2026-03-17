#!/usr/bin/env python3
"""
DraftKings Direct Fetcher for NBA XL System
=============================================
Fetches NBA player props directly from the DraftKings sportsbook API.

Bypasses BettingPros and fetches lines straight from DraftKings, providing:
- Real-time line values without aggregation delay
- Direct odds (American format) for each prop
- Game context (teams, home/away, game time)

DraftKings API Structure:
- League 42648 = NBA
- Category 1215 = Player Props
- Subcategories: 15001 (Points), 15002 (Assists), 15003 (Rebounds), 15004 (Threes)

Usage:
    python fetch_draftkings_direct.py                # Fetch today's props
    python fetch_draftkings_direct.py --save          # Save to JSON
    python fetch_draftkings_direct.py --quiet         # Quiet mode
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

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

# EST timezone used across the codebase
EST = ZoneInfo("America/New_York")


class DraftKingsDirectFetcher(BaseFetcher):
    """Fetches NBA player props directly from DraftKings sportsbook API."""

    # DraftKings sportsbook API base URL
    # Structure: /leagues/{league}/categories/{cat}/subcategories/{subcat}
    API_BASE = (
        "https://sportsbook-nash.draftkings.com/api/sportscontent"
        "/dkusnj/v1/leagues/42648/categories"
    )

    # NBA league ID on DraftKings
    LEAGUE_ID = 42648

    # DraftKings uses SEPARATE categories per stat type, each with O/U subcategories
    # Category IDs: 1215=Player Points, 1216=Player Rebounds, 1217=Player Assists, 1218=Player Threes
    # Subcategory IDs: 12488=Points O/U, 12492=Rebounds O/U, 12495=Assists O/U, 12497=Threes O/U
    STAT_ENDPOINTS = {
        "POINTS": {"category": 1215, "subcategory": 12488},
        "REBOUNDS": {"category": 1216, "subcategory": 12492},
        "ASSISTS": {"category": 1217, "subcategory": 12495},
        "THREES": {"category": 1218, "subcategory": 12497},
    }

    # Legacy compatibility — maps old subcategory IDs (no longer used in fetch logic)
    SUBCATEGORY_MAP = {
        12488: "POINTS",
        12492: "REBOUNDS",
        12495: "ASSISTS",
        12497: "THREES",
    }

    # Non-standard team abbreviation normalization
    TEAM_ABBREV_MAP = {
        "PHO": "PHX",
        "WSH": "WAS",
        "NO": "NOP",
        "NOR": "NOP",
        "SA": "SAS",
        "GS": "GSW",
        "NY": "NYK",
        "UTAH": "UTA",
        "BRK": "BKN",
    }

    # All valid NBA team abbreviations for validation
    VALID_TEAMS = {
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize DraftKings direct fetcher.

        Args:
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name="draftkings_direct",
            rate_limit=2.0,
            max_retries=3,
            timeout=30,
            verbose=verbose,
            proxy_profile="sportsbooks",
        )

        self._today = datetime.now(EST).strftime("%Y-%m-%d")

    # ── Team Abbreviation Normalization ──────────────────────────────────

    def _normalize_team(self, abbrev: str) -> str:
        """
        Normalize a team abbreviation to canonical 3-letter format.

        Args:
            abbrev: Raw team abbreviation from DraftKings

        Returns:
            Canonical abbreviation (e.g., GSW, WAS, NOP)
        """
        if not abbrev:
            return ""
        abbrev = abbrev.strip().upper()
        return self.TEAM_ABBREV_MAP.get(abbrev, abbrev)

    # ── HTTP Fetching ────────────────────────────────────────────────────

    def _get_dk_headers(self) -> Dict[str, str]:
        """
        Build request headers mimicking a browser hitting DraftKings.

        Returns:
            Headers dict suitable for DraftKings API requests
        """
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Origin": "https://sportsbook.draftkings.com",
            "Referer": "https://sportsbook.draftkings.com/",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

    def _fetch_subcategory_data(
        self, subcategory_id: int, category_id: int = None
    ) -> Optional[Dict]:
        """
        Fetch raw JSON for a category/subcategory combo from DraftKings.

        Tries curl_cffi first (browser TLS impersonation), falls back to
        standard requests if curl_cffi is unavailable or fails.

        Args:
            subcategory_id: DraftKings subcategory ID (e.g. 12488 for Points O/U)
            category_id: DraftKings category ID (e.g. 1215 for Player Points).
                         If None, uses only subcategory in URL.

        Returns:
            Parsed JSON dict or None on failure
        """
        if category_id:
            url = f"{self.API_BASE}/{category_id}/subcategories/{subcategory_id}"
        else:
            url = f"{self.API_BASE}/{subcategory_id}"
        headers = self._get_dk_headers()

        # Try curl_cffi first (better TLS fingerprint)
        if HAS_CURL_CFFI:
            result = self._fetch_with_curl_cffi(url, headers)
            if result is not None:
                return result

        # Fall back to standard requests via BaseFetcher
        response = self._make_request(url, method="GET", headers=headers)
        if response is not None:
            try:
                return response.json()
            except (ValueError, TypeError) as e:
                logger.warning(f"JSON parse error from URL: {e}")

        return None

    def _fetch_with_curl_cffi(self, url: str, headers: Dict) -> Optional[Dict]:
        """
        Fetch using curl_cffi with Chrome TLS impersonation.

        Args:
            url: URL to fetch
            headers: Request headers

        Returns:
            Parsed JSON dict or None on failure
        """
        proxies = None
        if self.session.proxies:
            proxies = dict(self.session.proxies)

        for attempt in range(self.max_retries):
            delay = random.uniform(0.5, 1.5)
            time.sleep(delay)

            try:
                response = cffi_requests.get(
                    url,
                    headers=headers,
                    impersonate="chrome",
                    timeout=self.timeout,
                    proxies=proxies,
                )

                # Track for Atlas registry
                self._api_calls += 1
                self._bytes_transferred += len(response.content)

                if response.status_code == 403:
                    logger.warning(
                        f"DraftKings 403 with curl_cffi (attempt {attempt + 1}/{self.max_retries})"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(3 * (attempt + 1))
                    continue

                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"DraftKings rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.warning(
                        f"DraftKings HTTP {response.status_code} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    continue

                data = response.json()
                if self.verbose:
                    logger.info(
                        f"[draftkings_direct] curl_cffi success: "
                        f"{response.status_code} ({len(response.content)} bytes)"
                    )
                return data

            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(
                    f"curl_cffi request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    # ── Response Parsing ─────────────────────────────────────────────────

    def _extract_game_info(self, event: Dict) -> Dict[str, Any]:
        """
        Extract game metadata from a DraftKings event object.

        Args:
            event: Event dict from DraftKings API

        Returns:
            Dict with game_id, game_date, game_time, home_team, away_team
        """
        event_id = str(event.get("eventId", event.get("id", "")))
        start_date = event.get("startDate", event.get("startTime", ""))

        game_date = self._today
        game_time = None

        if start_date:
            try:
                # DraftKings timestamps are typically ISO 8601 / UTC
                if isinstance(start_date, str):
                    # Strip trailing Z, parse as UTC, convert to EST
                    clean = start_date.replace("Z", "+00:00")
                    dt_utc = datetime.fromisoformat(clean)
                    dt_est = dt_utc.astimezone(EST)
                    game_date = dt_est.strftime("%Y-%m-%d")
                    game_time = dt_est.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                pass

        # Extract teams from event name or teamAbbrev fields
        home_team = None
        away_team = None

        # Try structured team data first
        teams = event.get("teams", event.get("teamAbbreviations", []))
        if isinstance(teams, list) and len(teams) >= 2:
            away_team = self._normalize_team(str(teams[0]))
            home_team = self._normalize_team(str(teams[1]))
        elif isinstance(teams, dict):
            home_team = self._normalize_team(teams.get("home", ""))
            away_team = self._normalize_team(teams.get("away", ""))

        # Fallback: parse event name (e.g. "DAL @ BOS" or "Dallas Mavericks vs Boston Celtics")
        if not home_team or not away_team:
            event_name = event.get("name", event.get("eventName", ""))
            parsed = self._parse_teams_from_name(event_name)
            if parsed:
                away_team = away_team or parsed[0]
                home_team = home_team or parsed[1]

        return {
            "game_id": event_id,
            "game_date": game_date,
            "game_time": game_time,
            "home_team": home_team,
            "away_team": away_team,
        }

    def _parse_teams_from_name(self, name: str) -> Optional[Tuple[str, str]]:
        """
        Parse away and home teams from an event name string.

        Handles formats like:
        - "DAL @ BOS"
        - "Dallas Mavericks @ Boston Celtics"
        - "DAL vs BOS"

        Args:
            name: Event name string

        Returns:
            Tuple of (away_team, home_team) abbreviations, or None
        """
        if not name:
            return None

        # Try "AWAY @ HOME" or "AWAY vs HOME"
        match = re.match(r"^(.+?)\s+(?:@|vs\.?|at)\s+(.+?)$", name, re.IGNORECASE)
        if not match:
            return None

        away_raw = match.group(1).strip()
        home_raw = match.group(2).strip()

        # If already abbreviations (3 chars), normalize directly
        if len(away_raw) <= 4 and len(home_raw) <= 4:
            return (self._normalize_team(away_raw), self._normalize_team(home_raw))

        # Otherwise these are full city/team names — we cannot reliably map
        # all city names to abbreviations without a full lookup table, so
        # return None and let the caller handle it
        return None

    def _parse_offers(
        self,
        data: Dict,
        stat_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse the DraftKings API response into standardized prop dicts.

        The current DraftKings API uses flat parallel arrays:
          response -> events[], markets[], selections[]

        Markets reference events via eventId; selections reference markets
        via marketId. Each market is one player's prop (e.g., "Paolo Banchero
        Points O/U"), and its selections contain the Over/Under pair.

        Falls back to the legacy nested structure (events[].offers[].outcomes[])
        if the flat structure is not found.

        Args:
            data: Raw JSON response from DraftKings
            stat_type: Canonical stat type (POINTS, REBOUNDS, etc.)

        Returns:
            List of standardized prop dicts
        """
        props = []
        now_iso = datetime.now(EST).isoformat()

        # Current DK API: flat arrays (events, markets, selections)
        events = data.get("events", [])
        markets = data.get("markets", [])
        selections = data.get("selections", [])

        if markets and selections:
            return self._parse_flat_structure(events, markets, selections, stat_type, now_iso)

        # Legacy fallback: nested events[].offers[].outcomes[]
        if not events:
            offer_categories = data.get("eventGroup", {}).get("offerCategories", [])
            if offer_categories:
                events = self._extract_events_from_categories(offer_categories, data)

        if not events:
            logger.warning(f"No events found in DraftKings response for {stat_type}")
            return []

        for event in events:
            game_info = self._extract_game_info(event)
            offers = event.get("offers", event.get("displayGroups", []))
            if not offers:
                offer_cats = event.get("offerCategories", [])
                for cat in offer_cats:
                    sub_cats = cat.get("offerSubcategoryDescriptors", [])
                    for sub_cat in sub_cats:
                        offers.extend(sub_cat.get("offerSubcategory", {}).get("offers", []))

            for offer in offers:
                parsed = self._parse_single_offer(offer, stat_type, game_info, now_iso)
                if parsed:
                    props.append(parsed)

        return props

    def _parse_flat_structure(
        self,
        events: List[Dict],
        markets: List[Dict],
        selections: List[Dict],
        stat_type: str,
        fetch_timestamp: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse DraftKings flat parallel-array response (current API format).

        Structure:
        - events[]: game info (id, name with "AWAY @ HOME")
        - markets[]: per-player props (id, name like "Player Name Points O/U", eventId)
        - selections[]: Over/Under entries (marketId, label, points, displayOdds, participants)
        """
        props = []

        # Build event lookup
        event_map = {}
        for ev in events:
            eid = ev.get("id", ev.get("eventId"))
            if eid:
                event_map[str(eid)] = self._extract_game_info(ev)

        # Group selections by market
        sel_by_market: Dict[str, List[Dict]] = {}
        for sel in selections:
            mid = str(sel.get("marketId", ""))
            if mid:
                sel_by_market.setdefault(mid, []).append(sel)

        # Process each market (one market = one player prop)
        for market in markets:
            market_id = str(market.get("id", ""))
            event_id = str(market.get("eventId", ""))
            market_name = market.get("name", "")

            game_info = event_map.get(
                event_id,
                {
                    "game_id": event_id,
                    "game_date": self._today,
                    "game_time": None,
                    "home_team": None,
                    "away_team": None,
                },
            )

            market_sels = sel_by_market.get(market_id, [])
            if not market_sels:
                continue

            # Find Over and Under selections
            over_sel = None
            under_sel = None
            for sel in market_sels:
                label = (sel.get("label", "") or "").upper()
                if "OVER" in label:
                    over_sel = sel
                elif "UNDER" in label:
                    under_sel = sel

            if not over_sel and not under_sel:
                continue

            # Extract line from selection points field
            line = None
            for sel in [over_sel, under_sel]:
                if sel and sel.get("points") is not None:
                    try:
                        line = float(sel["points"])
                        break
                    except (ValueError, TypeError):
                        pass

            if line is None:
                continue

            # Extract player name from market name or selection participants
            player_name = None

            # Market name: "Paolo Banchero Points O/U" → "Paolo Banchero"
            if market_name:
                cleaned = re.sub(
                    r"\s+(?:Points|Rebounds|Assists|Threes|3-Pt|Steals|Blocks)" r"(?:\s+O/U)?$",
                    "",
                    market_name,
                    flags=re.IGNORECASE,
                )
                if cleaned.strip():
                    player_name = cleaned.strip()

            # Fallback: participants array on selections
            if not player_name:
                for sel in [over_sel, under_sel]:
                    if not sel:
                        continue
                    parts = sel.get("participants", [])
                    if parts:
                        player_name = parts[0].get("name", "")
                        break

            if not player_name:
                continue

            player_name = self.normalize_player_name(player_name)

            # Extract odds
            over_odds = -110
            under_odds = -110
            if over_sel:
                raw = over_sel.get("displayOdds", {}).get("american", "")
                try:
                    over_odds = int(str(raw).replace("−", "-").replace("+", ""))
                except (ValueError, TypeError):
                    pass
            if under_sel:
                raw = under_sel.get("displayOdds", {}).get("american", "")
                try:
                    under_odds = int(str(raw).replace("−", "-").replace("+", ""))
                except (ValueError, TypeError):
                    pass

            # Extract player team from participants
            player_team = None
            for sel in [over_sel, under_sel]:
                if not sel:
                    continue
                parts = sel.get("participants", [])
                if parts:
                    venue = parts[0].get("venueRole", "")
                    if "Home" in venue:
                        player_team = game_info.get("home_team")
                    elif "Away" in venue:
                        player_team = game_info.get("away_team")
                    break

            # Determine home/away
            home_team = game_info.get("home_team")
            away_team = game_info.get("away_team")
            is_home = None
            opponent_team = None
            if player_team and home_team and away_team:
                if player_team == home_team:
                    is_home = True
                    opponent_team = away_team
                elif player_team == away_team:
                    is_home = False
                    opponent_team = home_team

            props.append(
                {
                    "player_name": player_name,
                    "player_team": player_team,
                    "stat_type": stat_type,
                    "line": line,
                    "over_line": line,
                    "under_line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "book_name": "draftkings_direct",
                    "game_date": game_info.get("game_date", self._today),
                    "game_time": game_info.get("game_time"),
                    "game_id": game_info.get("game_id", ""),
                    "opponent_team": opponent_team,
                    "is_home": is_home,
                    "fetch_timestamp": fetch_timestamp,
                    "source": "draftkings_direct",
                    "fetch_source": "direct",
                }
            )

        return props

    def _extract_events_from_categories(
        self, offer_categories: List[Dict], data: Dict
    ) -> List[Dict]:
        """
        Extract events from offerCategories structure when events are not
        at the top level.

        Some DraftKings API versions nest offers inside:
          eventGroup -> offerCategories[] -> offerSubcategoryDescriptors[]
            -> offerSubcategory -> offers[]

        This method reconstructs event-like objects so the main parser
        can process them uniformly.

        Args:
            offer_categories: List of offer category dicts
            data: Full API response for event lookups

        Returns:
            List of pseudo-event dicts with offers attached
        """
        # Build event lookup from eventGroup
        event_lookup = {}
        raw_events = data.get("eventGroup", {}).get("events", [])
        for ev in raw_events:
            eid = ev.get("eventId", ev.get("id"))
            if eid:
                event_lookup[eid] = ev

        # Group offers by event
        event_offers: Dict[str, List[Dict]] = {}
        for cat in offer_categories:
            sub_descs = cat.get("offerSubcategoryDescriptors", [])
            for sub_desc in sub_descs:
                offers = sub_desc.get("offerSubcategory", {}).get("offers", [])
                for offer in offers:
                    eid = offer.get("eventId")
                    if eid:
                        event_offers.setdefault(str(eid), []).append(offer)

        # Build pseudo-events
        result = []
        for eid_str, offers in event_offers.items():
            ev = event_lookup.get(int(eid_str), event_lookup.get(eid_str, {}))
            pseudo = dict(ev) if ev else {"eventId": eid_str}
            pseudo["offers"] = offers
            result.append(pseudo)

        return result

    def _parse_single_offer(
        self,
        offer: Dict,
        stat_type: str,
        game_info: Dict[str, Any],
        fetch_timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single DraftKings offer into a standardized prop dict.

        An offer typically contains two outcomes: OVER and UNDER.

        Args:
            offer: Single offer dict from DraftKings
            stat_type: Canonical stat type
            game_info: Pre-extracted game metadata
            fetch_timestamp: ISO timestamp of this fetch

        Returns:
            Standardized prop dict or None if unparseable
        """
        outcomes = offer.get("outcomes", [])
        if not outcomes:
            return None

        # Separate OVER and UNDER outcomes
        over_outcome = None
        under_outcome = None

        for outcome in outcomes:
            label = (outcome.get("label", "") or "").upper()
            if "OVER" in label:
                over_outcome = outcome
            elif "UNDER" in label:
                under_outcome = outcome

        # Need at least one side to extract a line
        if not over_outcome and not under_outcome:
            return None

        # Extract line from whichever side is available
        line = None
        if over_outcome:
            line = over_outcome.get("line", over_outcome.get("oddsLine"))
        if line is None and under_outcome:
            line = under_outcome.get("line", under_outcome.get("oddsLine"))

        if line is None:
            return None

        try:
            line = float(line)
        except (ValueError, TypeError):
            return None

        # Extract player name
        player_name = self._extract_player_name(offer, over_outcome, under_outcome)
        if not player_name:
            return None

        player_name = self.normalize_player_name(player_name)

        # Extract odds (American format)
        over_odds = self._extract_odds(over_outcome) if over_outcome else -110
        under_odds = self._extract_odds(under_outcome) if under_outcome else -110

        # Determine player team and home/away
        player_team = self._extract_player_team(offer, over_outcome, under_outcome)
        player_team = self._normalize_team(player_team) if player_team else None

        home_team = game_info.get("home_team")
        away_team = game_info.get("away_team")

        is_home = None
        opponent_team = None
        if player_team and home_team and away_team:
            if player_team == home_team:
                is_home = True
                opponent_team = away_team
            elif player_team == away_team:
                is_home = False
                opponent_team = home_team

        prop = {
            "player_name": player_name,
            "player_team": player_team,
            "stat_type": stat_type,
            "line": line,
            "over_line": line,
            "under_line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "book_name": "draftkings_direct",
            "game_date": game_info.get("game_date", self._today),
            "game_time": game_info.get("game_time"),
            "game_id": game_info.get("game_id", ""),
            "opponent_team": opponent_team,
            "is_home": is_home,
            "fetch_timestamp": fetch_timestamp,
            "source": "draftkings_direct",
            "fetch_source": "direct",
        }

        return prop

    def _extract_player_name(
        self,
        offer: Dict,
        over_outcome: Optional[Dict],
        under_outcome: Optional[Dict],
    ) -> Optional[str]:
        """
        Extract the player name from offer or outcome data.

        DraftKings puts player names in several possible locations:
        - offer["label"] (e.g. "LeBron James - Points")
        - outcome["participant"] or outcome["participantName"]
        - outcome["label"] (e.g. "Over 25.5 - LeBron James")

        Args:
            offer: The offer dict
            over_outcome: OVER outcome dict (may be None)
            under_outcome: UNDER outcome dict (may be None)

        Returns:
            Player name string or None
        """
        # Try offer-level label first (most reliable)
        offer_label = offer.get("label", "")
        if offer_label:
            # Strip stat suffix: "LeBron James - Points" -> "LeBron James"
            name = re.split(r"\s*[-:]\s*(?:Points|Rebounds|Assists|Threes|3-Pt)", offer_label)
            if name and name[0].strip():
                return name[0].strip()

        # Try participant fields on outcomes
        for outcome in [over_outcome, under_outcome]:
            if outcome is None:
                continue

            participant = outcome.get("participant", outcome.get("participantName", ""))
            if participant:
                return participant

            # Try parsing from outcome label
            label = outcome.get("label", "")
            if label:
                # "Over 25.5 - LeBron James" -> "LeBron James"
                match = re.search(r"(?:Over|Under)\s+[\d.]+\s*[-:]\s*(.+)", label)
                if match:
                    return match.group(1).strip()

        return None

    def _extract_odds(self, outcome: Optional[Dict]) -> int:
        """
        Extract American odds from an outcome dict.

        Args:
            outcome: Outcome dict (OVER or UNDER side)

        Returns:
            American odds integer (e.g. -110, +105). Defaults to -110.
        """
        if not outcome:
            return -110

        # Try oddsAmerican first, then odds, then price
        for key in ("oddsAmerican", "odds", "price"):
            raw = outcome.get(key)
            if raw is not None:
                try:
                    val = str(raw).replace("+", "")
                    return int(float(val))
                except (ValueError, TypeError):
                    continue

        return -110

    def _extract_player_team(
        self,
        offer: Dict,
        over_outcome: Optional[Dict],
        under_outcome: Optional[Dict],
    ) -> Optional[str]:
        """
        Extract the player's team abbreviation from offer or outcome data.

        Args:
            offer: The offer dict
            over_outcome: OVER outcome dict
            under_outcome: UNDER outcome dict

        Returns:
            Team abbreviation or None
        """
        # Check offer-level fields
        for key in ("teamAbbreviation", "team", "teamAbbrev"):
            val = offer.get(key)
            if val:
                return str(val).strip()

        # Check outcome-level fields
        for outcome in [over_outcome, under_outcome]:
            if outcome is None:
                continue
            for key in ("teamAbbreviation", "team", "teamAbbrev"):
                val = outcome.get(key)
                if val:
                    return str(val).strip()

        return None

    # ── Main Fetch Orchestration ─────────────────────────────────────────

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch all NBA player props from DraftKings.

        Iterates over all configured subcategories (stat types), fetches
        raw data, parses into standardized prop format, validates, and
        deduplicates.

        Returns:
            List of standardized prop dictionaries
        """
        print("\n" + "=" * 70)
        print("FETCHING DRAFTKINGS NBA PROPS (Direct API)")
        print("=" * 70)
        print(f"Date: {self._today}")
        print(f"Stat types: {len(self.STAT_ENDPOINTS)}")
        print(f"curl_cffi: {'available' if HAS_CURL_CFFI else 'not installed'}")
        print(f"Proxy: {'sportsbooks (Colorado)' if self.proxy_profile else 'none'}")
        print("=" * 70 + "\n", flush=True)

        all_props = []
        stats_by_type: Dict[str, int] = {}

        for stat_type, endpoint in self.STAT_ENDPOINTS.items():
            cat_id = endpoint["category"]
            subcat_id = endpoint["subcategory"]
            print(f"Fetching {stat_type} (cat {cat_id}, subcat {subcat_id})...", flush=True)

            data = self._fetch_subcategory_data(subcat_id, category_id=cat_id)
            if not data:
                print(f"  [WARN] No data returned for {stat_type}", flush=True)
                continue

            props = self._parse_offers(data, stat_type)
            count = len(props)
            stats_by_type[stat_type] = count
            all_props.extend(props)

            print(f"  [OK] {count} props parsed", flush=True)

        # Filter to today's date only
        today_props = [p for p in all_props if p.get("game_date") == self._today]
        filtered_count = len(all_props) - len(today_props)
        if filtered_count > 0:
            print(f"\nFiltered out {filtered_count} props from other dates", flush=True)
        all_props = today_props

        # Validate
        valid_props = [p for p in all_props if self.validate_prop(p)]
        invalid_count = len(all_props) - len(valid_props)

        # Deduplicate
        deduped_props = self.deduplicate_props(valid_props)

        # Print summary
        print("\n" + "=" * 70)
        print("DRAFTKINGS DIRECT FETCH SUMMARY")
        print("=" * 70)
        print(f"Total parsed:      {len(all_props)}")
        if invalid_count > 0:
            print(f"Invalid (skipped):  {invalid_count}")
        print(f"After dedup:       {len(deduped_props)}")
        print()

        print("Props by stat type:")
        for stat_type in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
            count = stats_by_type.get(stat_type, 0)
            if count > 0:
                type_props = [p for p in deduped_props if p["stat_type"] == stat_type]
                if type_props:
                    avg_line = sum(p["line"] for p in type_props) / len(type_props)
                    print(
                        f"  {stat_type:10s}: {len(type_props):4d} props (avg line: {avg_line:.1f})"
                    )
                else:
                    print(f"  {stat_type:10s}: {count:4d} parsed, 0 after dedup")
            else:
                print(f"  {stat_type:10s}:    0 props")

        # Show sample player names if we have data
        if deduped_props:
            print()
            sample = deduped_props[:5]
            print("Sample props:")
            for p in sample:
                odds_str = f"O {p['over_odds']:+d} / U {p['under_odds']:+d}"
                print(
                    f"  {p['player_name']:25s} | {p['stat_type']:10s} | "
                    f"{p['line']:5.1f} | {odds_str}"
                )

        print("=" * 70 + "\n", flush=True)

        # Atlas data registry -- log this ingestion
        try:
            from nba.core.data_registry import log_ingestion

            stats = self.get_registry_stats()
            log_ingestion(
                "draftkings_direct",
                "fetch",
                "success" if deduped_props else "empty",
                records_fetched=len(deduped_props),
                api_calls_made=stats["api_calls_made"],
                bytes_transferred=stats["bytes_transferred"],
                error_count=stats["error_count"],
                error_message=stats["error_message"],
                metadata={
                    "game_date": self._today,
                    "stats_by_type": stats_by_type,
                },
            )
        except Exception:
            pass

        return deduped_props


def main():
    """Main execution for standalone CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NBA player props directly from DraftKings API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's props and show summary
  python fetch_draftkings_direct.py

  # Save to JSON file
  python fetch_draftkings_direct.py --save

  # Quiet mode (minimal output)
  python fetch_draftkings_direct.py --quiet
        """,
    )
    parser.add_argument("--save", action="store_true", help="Save props to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less verbose)")

    args = parser.parse_args()

    with DraftKingsDirectFetcher(verbose=not args.quiet) as fetcher:
        props = fetcher.fetch()

        if props and args.save:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")
        elif not props:
            print("\n[WARN] No props fetched from DraftKings!\n")
        else:
            # Show detailed sample
            print("\n=== DRAFTKINGS PROPS SAMPLE ===\n")
            for prop in props[:15]:
                team = prop.get("player_team") or "???"
                opp = prop.get("opponent_team") or "???"
                home_str = (
                    "H" if prop.get("is_home") else "A" if prop.get("is_home") is False else "?"
                )
                print(
                    f"  {prop['player_name']:25s} ({team} {home_str} vs {opp}) | "
                    f"{prop['stat_type']:10s} | {prop['line']:5.1f} | "
                    f"O:{prop['over_odds']:+d} U:{prop['under_odds']:+d}"
                )
            if len(props) > 15:
                print(f"\n  ... and {len(props) - 15} more props")
            print()

        # Log Atlas registry stats
        registry = fetcher.get_registry_stats()
        if registry["api_calls_made"] > 0:
            print(
                f"Atlas Registry: {registry['api_calls_made']} API calls, "
                f"{registry['bytes_transferred']:,} bytes transferred"
            )
            if registry["error_count"] > 0:
                print(f"  Errors: {registry['error_count']} ({registry['error_message']})")
            print()


if __name__ == "__main__":
    main()
