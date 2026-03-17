#!/usr/bin/env python3
"""
Hard Rock Bet Direct Fetcher (Kambi Backend)
=============================================
Fetches NBA player props from Hard Rock Bet via the Kambi sportsbook API.

Hard Rock Bet (Seminole Hard Rock Digital) uses the Kambi platform.
Available in Florida — uses the Florida proxy profile.

Kambi encodes odds and lines as integers (multiplied by 1000):
    odds  1850 → 1.850 decimal → -118 American
    line 24500 → 24.5

Rate limiting is critical — Kambi aggressively returns 429 on rapid requests.

Usage:
    python fetch_hardrock_direct.py              # Fetch today's props
    python fetch_hardrock_direct.py --save       # Fetch and save to JSON
    python fetch_hardrock_direct.py --quiet      # Minimal output
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

EST = ZoneInfo("America/New_York")

# Kambi API endpoints for Hard Rock Bet (Seminole operator)
# seminolefl confirmed valid (returns 429, not 404)
KAMBI_ENDPOINTS = [
    "https://eu-offering-api.kambicdn.com/offering/v2018/seminolefl",
    "https://eu-offering.kambicdn.org/offering/v2018/seminolefl",
    "https://us1-api.aws.kambicdn.com/offering/v2018/seminolefl",
]
KAMBI_BASE = KAMBI_ENDPOINTS[0]

# Kambi criterion labels that map to our stat types
STAT_LABEL_MAP = {
    "points": "POINTS",
    "rebounds": "REBOUNDS",
    "assists": "ASSISTS",
    "three": "THREES",
    "3-point": "THREES",
    "3 point": "THREES",
    "steals": "STEALS",
    "blocks": "BLOCKS",
    "turnovers": "TURNOVERS",
}

CORE_STATS = {"POINTS", "REBOUNDS", "ASSISTS", "THREES", "STEALS", "BLOCKS"}


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return -110
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def _extract_stat_type(label: str) -> Optional[str]:
    """Map a Kambi criterion label to a canonical stat type."""
    label_lower = label.lower()
    if "player" not in label_lower:
        return None
    for keyword, stat_type in STAT_LABEL_MAP.items():
        if keyword in label_lower:
            return stat_type
    return None


def _normalize_team(name: str) -> Optional[str]:
    """Convert Kambi participant name to canonical NBA abbreviation."""
    if not name:
        return None
    try:
        from nba.utils.team_utils import team_name_to_abbrev

        return team_name_to_abbrev(name)
    except ImportError:
        return name


class HardRockDirectFetcher(BaseFetcher):
    """Fetches NBA player props from the Kambi API powering Hard Rock Bet."""

    def __init__(self, verbose: bool = True):
        super().__init__(
            source_name="hardrock_direct",
            rate_limit=5.0,  # Kambi Varnish is aggressive
            max_retries=3,
            timeout=30,
            verbose=verbose,
            proxy_profile="prizepicks",  # Hard Rock is Florida-only
        )
        self.today_est = datetime.now(EST).strftime("%Y-%m-%d")
        self._active_base = KAMBI_BASE

    def _kambi_headers(self) -> Dict[str, str]:
        """Headers for Kambi API requests."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

    def _kambi_get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request to the Kambi API with retry + rate-limit handling."""
        headers = self._kambi_headers()

        for attempt in range(self.max_retries):
            self._enforce_rate_limit()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code == 410:
                    logger.warning(f"Kambi 410 Gone (IP banned) on {url.split('/offering')[0]}")
                    for fallback in KAMBI_ENDPOINTS:
                        if fallback not in url:
                            new_url = url.replace(self._active_base, fallback)
                            self._active_base = fallback
                            logger.info(f"Switching to fallback: {fallback}")
                            url = new_url
                            break
                    else:
                        return None
                    continue

                if response.status_code == 429:
                    wait = 10 * (attempt + 1) + random.uniform(2, 5)
                    logger.warning(
                        f"Kambi 429 rate-limited (attempt {attempt + 1}), "
                        f"waiting {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code == 404:
                    logger.debug(f"Kambi 404 for {url}")
                    return None

                response.raise_for_status()
                self._api_calls += 1
                self._bytes_transferred += len(response.content)
                return response.json()

            except requests.exceptions.HTTPError as e:
                self._error_count += 1
                self._last_error = str(e)
                status = getattr(e.response, "status_code", None)
                logger.warning(
                    f"Kambi HTTP {status} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if status and 400 <= status < 500 and status != 429:
                    return None

            except requests.exceptions.RequestException as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(
                    f"Kambi request error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries - 1:
                backoff = 2**attempt + random.uniform(0, 1)
                time.sleep(backoff)

        return None

    def _fetch_events(self) -> List[Dict]:
        """Fetch the NBA event list from Kambi."""
        url = f"{self._active_base}/listView/basketball/nba.json"
        params = {
            "lang": "en_US",
            "market": "US",
            "includeParticipants": "true",
        }

        logger.info("Fetching Kambi NBA event list (Hard Rock)...")
        data = self._kambi_get(url, params=params)

        if not data:
            return []

        events = data.get("events", [])
        logger.info(f"Kambi returned {len(events)} NBA events")
        return events

    def _fetch_event_offers(self, event_id: int) -> List[Dict]:
        """Fetch all bet offers for a specific Kambi event."""
        url = f"{self._active_base}/betoffer/event/{event_id}.json"
        params = {
            "lang": "en_US",
            "includeParticipants": "true",
        }

        data = self._kambi_get(url, params=params)
        if not data:
            return []

        return data.get("betOffers", [])

    def _parse_event_metadata(
        self, event_wrapper: Dict
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract event metadata from a Kambi event wrapper."""
        event = event_wrapper.get("event", event_wrapper)
        event_id = event.get("id")

        start_str = event.get("start", "")
        game_date = None
        game_time = None

        if start_str:
            try:
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(EST)
                game_date = dt.strftime("%Y-%m-%d")
                game_time = dt.strftime("%H:%M:%S")
            except (ValueError, AttributeError, TypeError):
                game_date = self.today_est

        if not game_date:
            game_date = self.today_est

        home_team = None
        away_team = None
        home_name = event.get("homeName", "")
        away_name = event.get("awayName", "")

        if home_name:
            home_team = _normalize_team(home_name)
        if away_name:
            away_team = _normalize_team(away_name)

        if not home_team or not away_team:
            event_name = event.get("name", "") or event.get("englishName", "")
            if " @ " in event_name:
                parts = event_name.split(" @ ", 1)
                if not away_team:
                    away_team = _normalize_team(parts[0].strip())
                if not home_team:
                    home_team = _normalize_team(parts[1].strip())
            elif " vs " in event_name.lower():
                parts = re.split(r"\s+vs\.?\s+", event_name, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    if not home_team:
                        home_team = _normalize_team(parts[0].strip())
                    if not away_team:
                        away_team = _normalize_team(parts[1].strip())

        return event_id, home_team, away_team, game_date, game_time

    def _parse_bet_offers(
        self,
        bet_offers: List[Dict],
        home_team: Optional[str],
        away_team: Optional[str],
        game_date: str,
        game_time: Optional[str],
        event_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Parse Kambi betOffer list into standardized prop dicts."""
        props: List[Dict[str, Any]] = []
        now_iso = datetime.now(EST).isoformat()

        for offer in bet_offers:
            criterion = offer.get("criterion", {})
            label = criterion.get("label", "")
            offer_type = offer.get("betOfferType", {})
            type_name = offer_type.get("name", "")

            if type_name not in ("Over/Under", "Total"):
                continue

            stat_type = _extract_stat_type(label)
            if not stat_type or stat_type not in CORE_STATS:
                continue

            outcomes = offer.get("outcomes", [])
            player_sides: Dict[str, Dict[str, Dict]] = {}

            for outcome in outcomes:
                status = outcome.get("status", "OPEN")
                if status != "OPEN":
                    continue

                side_label = (outcome.get("label") or "").lower()
                if side_label not in ("over", "under"):
                    otype = (outcome.get("type") or "").lower()
                    if otype in ("over", "under"):
                        side_label = otype
                    else:
                        continue

                player_name = outcome.get("participant", "")
                if not player_name:
                    match = re.match(r"^(.+?)\s*[-–]\s*Player", label)
                    if match:
                        player_name = match.group(1).strip()
                    else:
                        continue

                if not player_name:
                    continue

                player_name = self.normalize_player_name(player_name)

                if player_name not in player_sides:
                    player_sides[player_name] = {}
                player_sides[player_name][side_label] = outcome

            for player_name, sides in player_sides.items():
                over_oc = sides.get("over", {})
                under_oc = sides.get("under", {})

                raw_line = over_oc.get("line") or under_oc.get("line")
                if raw_line is None:
                    continue

                try:
                    line = float(raw_line) / 1000.0
                except (ValueError, TypeError):
                    continue

                over_raw = over_oc.get("odds")
                under_raw = under_oc.get("odds")

                over_decimal = float(over_raw) / 1000.0 if over_raw else None
                under_decimal = float(under_raw) / 1000.0 if under_raw else None

                over_odds = _decimal_to_american(over_decimal) if over_decimal else -110
                under_odds = _decimal_to_american(under_decimal) if under_decimal else -110

                prop = {
                    "player_name": player_name,
                    "stat_type": stat_type,
                    "line": line,
                    "over_line": line,
                    "under_line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "book_name": "hardrock_direct",
                    "game_date": game_date,
                    "game_time": game_time,
                    "game_id": str(event_id) if event_id else "",
                    "opponent_team": None,
                    "is_home": None,
                    "fetch_timestamp": now_iso,
                    "source": "hardrock_direct",
                    "fetch_source": "direct",
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_name": label,
                }

                props.append(prop)

        return props

    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch all NBA player props from Hard Rock Bet (Kambi backend)."""
        print("\n" + "=" * 70)
        print("FETCHING HARD ROCK BET NBA PROPS (Kambi Backend)")
        print("=" * 70)
        print(f"Date: {self.today_est}")
        print(f"Operator: seminolefl")
        print(f"Endpoint: {self._active_base}")
        print(f"Proxy: Florida (prizepicks profile)")
        print("=" * 70 + "\n", flush=True)

        # Step 1: Get event list
        print("Step 1: Fetching NBA event list...", flush=True)
        event_list = self._fetch_events()

        if not event_list:
            print("[WARN] No NBA events returned from Kambi", flush=True)
            return []

        # Filter to today's events
        today_events: List[Tuple] = []
        for ew in event_list:
            event_id, home_team, away_team, game_date, game_time = self._parse_event_metadata(ew)
            if game_date != self.today_est:
                continue
            today_events.append((event_id, home_team, away_team, game_date, game_time))

        print(f"[OK] {len(today_events)} events for today ({self.today_est})\n", flush=True)

        if not today_events:
            print("[WARN] No NBA events for today", flush=True)
            return []

        # Step 2: Fetch bet offers per event
        all_props: List[Dict[str, Any]] = []
        events_with_props = 0

        for event_id, home_team, away_team, game_date, game_time in today_events:
            matchup = f"{away_team or '???'} @ {home_team or '???'}"
            print(f"Step 2: Fetching offers for {matchup} (event {event_id})...", flush=True)

            time.sleep(random.uniform(5.0, 8.0))

            offers = self._fetch_event_offers(event_id)
            if not offers:
                print("  [--] No bet offers returned", flush=True)
                continue

            event_props = self._parse_bet_offers(
                offers, home_team, away_team, game_date, game_time, event_id
            )

            if event_props:
                events_with_props += 1
                all_props.extend(event_props)
                print(f"  [OK] {len(event_props)} player props", flush=True)
            else:
                print(
                    f"  [--] {len(offers)} offers, but no player props matched filters",
                    flush=True,
                )

        valid_props = [p for p in all_props if self.validate_prop(p)]
        deduped_props = self.deduplicate_props(valid_props)

        # Summary
        print("\n" + "=" * 70)
        print("HARD ROCK BET (KAMBI) FETCH SUMMARY")
        print("=" * 70)
        print(f"Today's events:      {len(today_events)}")
        print(f"Events with props:   {events_with_props}")
        print(f"Total props parsed:  {len(all_props)}")
        print(f"Valid props:         {len(valid_props)}")
        print(f"After deduplication: {len(deduped_props)}")
        print()

        if deduped_props:
            print("Breakdown by stat type:")
            stat_counts: Dict[str, List[Dict]] = {}
            for p in deduped_props:
                stat = p["stat_type"]
                if stat not in stat_counts:
                    stat_counts[stat] = []
                stat_counts[stat].append(p)

            for stat in sorted(stat_counts.keys()):
                props_list = stat_counts[stat]
                count = len(props_list)
                avg_line = sum(p["line"] for p in props_list) / count
                print(f"  {stat:15s}: {count:4d} props (avg line: {avg_line:.1f})")

        print("=" * 70 + "\n", flush=True)

        # Atlas data registry
        try:
            from nba.core.data_registry import log_ingestion

            stats = self.get_registry_stats()
            log_ingestion(
                "hardrock_direct",
                "fetch",
                "success" if deduped_props else "empty",
                records_fetched=len(deduped_props),
                api_calls_made=stats["api_calls_made"],
                bytes_transferred=stats["bytes_transferred"],
                error_count=stats["error_count"],
                error_message=stats["error_message"],
                metadata={"game_date": self.today_est, "events": len(today_events)},
            )
        except Exception:
            pass

        return deduped_props


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Fetch NBA player props from Hard Rock Bet (Kambi backend)",
    )
    parser.add_argument("--save", action="store_true", help="Save to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    with HardRockDirectFetcher(verbose=not args.quiet) as fetcher:
        props = fetcher.fetch()

        if props and args.save:
            output_file = fetcher.save_to_json(props)
            print(f"\n[OK] Saved {len(props)} props to: {output_file}\n")
        elif not props:
            print("\n[WARN] No props fetched!\n")
        else:
            print("\n=== SAMPLE PROPS ===\n")
            for prop in props[:15]:
                print(
                    f"{prop['player_name']:25s} | {prop['stat_type']:10s} "
                    f"| {prop['line']:5.1f} | O {prop['over_odds']:+4d} "
                    f"| U {prop['under_odds']:+4d}"
                )
            if len(props) > 15:
                print(f"  ... and {len(props) - 15} more")


if __name__ == "__main__":
    main()
