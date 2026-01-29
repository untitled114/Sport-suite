#!/usr/bin/env python3
"""
Pick6 (DraftKings DFS) Fetcher via The Odds API
================================================
Fetches Pick6 multipliers for NBA player props.

Key insight: Pick6 multipliers are confidence signals:
- mult < 1.0 = Platform thinks likely to hit (reduced payout)
- mult = 1.0 = Standard (50/50)
- mult > 5.0 = TRAP - Platform thinks unlikely (0% WR on 71 props)

Part of Odds API V3 replacement.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

# API Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com"

# Player prop markets we care about (main lines)
PLAYER_MARKETS = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    # Combo markets
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
    'player_rebounds_assists',
]

# Alternate markets - these have real Pick6 multiplier variance (0.7-10+)
# Main markets always return mult=1.0; alternates have contest multipliers
ALTERNATE_MARKETS = [
    'player_points_alternate',
    'player_rebounds_alternate',
    'player_assists_alternate',
    'player_threes_alternate',
    'player_points_rebounds_assists_alternate',
    'player_points_rebounds_alternate',
    'player_points_assists_alternate',
    'player_rebounds_assists_alternate',
]

# Stat type mapping from Odds API to our format
STAT_TYPE_MAP = {
    'player_points': 'POINTS',
    'player_rebounds': 'REBOUNDS',
    'player_assists': 'ASSISTS',
    'player_threes': 'THREES',
    'player_points_rebounds_assists': 'PRA',
    'player_points_rebounds': 'PR',
    'player_points_assists': 'PA',
    'player_rebounds_assists': 'RA',
    # Alternate markets map to same stat types
    'player_points_alternate': 'POINTS',
    'player_rebounds_alternate': 'REBOUNDS',
    'player_assists_alternate': 'ASSISTS',
    'player_threes_alternate': 'THREES',
    'player_points_rebounds_assists_alternate': 'PRA',
    'player_points_rebounds_alternate': 'PR',
    'player_points_assists_alternate': 'PA',
    'player_rebounds_assists_alternate': 'RA',
}


class Pick6Fetcher(BaseFetcher):
    """Fetcher for Pick6 multipliers via The Odds API"""

    def __init__(self, api_key: str = None, verbose: bool = True):
        """
        Initialize Pick6 fetcher.

        Args:
            api_key: Odds API key (default: from constant)
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name='pick6',
            rate_limit=1.0,  # 1 second between requests
            max_retries=3,
            timeout=30,
            verbose=verbose
        )
        self.api_key = api_key or ODDS_API_KEY

    def get_nba_events(self) -> List[Dict]:
        """
        Get current NBA events (games).

        Returns:
            List of event dictionaries with event_id, teams, commence_time
        """
        url = f"{BASE_URL}/v4/sports/basketball_nba/events"
        params = {
            "apiKey": self.api_key,
        }

        response = self._make_request(url, params=params)
        if response is None:
            logger.error("Failed to fetch NBA events")
            return []

        try:
            events = response.json()
            if self.verbose:
                logger.info(f"Found {len(events)} NBA events")
            return events
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse events response: {e}")
            return []

    def get_event_player_props(self, event_id: str) -> List[Dict]:
        """
        Get player props with Pick6 multipliers for a specific event.

        Makes two API calls:
        1. Main markets (player_points, etc.) - lines with mult=1.0
        2. Alternate markets (player_points_alternate, etc.) - contest multipliers

        The alternate markets contain the real Pick6 contest multipliers (0.7-10+)
        where all players share a common line and get different multipliers based
        on platform confidence. These multipliers are merged back into main props.

        Args:
            event_id: The Odds API event ID

        Returns:
            List of prop dictionaries with real multipliers
        """
        url = f"{BASE_URL}/v4/sports/basketball_nba/events/{event_id}/odds"

        # Request 1: Main markets (get player lines)
        main_params = {
            "apiKey": self.api_key,
            "regions": "us_dfs",
            "bookmakers": "pick6",
            "markets": ','.join(PLAYER_MARKETS),
            "includeMultipliers": "true",
        }

        main_response = self._make_request(url, params=main_params)
        if main_response is None:
            logger.warning(f"No Pick6 data for event {event_id}")
            return []

        try:
            main_data = main_response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse main props response for event {event_id}: {e}")
            return []

        # Request 2: Alternate markets (get real contest multipliers)
        alt_params = {
            "apiKey": self.api_key,
            "regions": "us_dfs",
            "bookmakers": "pick6",
            "markets": ','.join(ALTERNATE_MARKETS),
            "includeMultipliers": "true",
        }

        alt_response = self._make_request(url, params=alt_params)
        alt_data = None
        if alt_response is not None:
            try:
                alt_data = alt_response.json()
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse alternate props for event {event_id}")

        return self._parse_event_props(main_data, alt_data, event_id)

    def _extract_alternate_multipliers(self, data: Dict) -> Dict[tuple, Dict]:
        """
        Extract real multipliers from alternate market data.

        Alternate markets have contest lines where all players share a common
        line value with varying multipliers. Entries with multiplier=None are
        individual alternate lines without Pick6 contest signal — we skip those.

        Args:
            data: Raw API response for alternate markets

        Returns:
            Dict mapping (player_name_lower, stat_type) to multiplier info
        """
        multipliers = {}

        for bookmaker in data.get('bookmakers', []):
            if bookmaker.get('key') != 'pick6':
                continue

            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                stat_type = STAT_TYPE_MAP.get(market_key)
                if not stat_type:
                    continue

                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    side = outcome.get('name', '')
                    multiplier = outcome.get('multiplier')
                    alt_line = outcome.get('point')

                    # Skip non-OVER, missing data, and None multipliers
                    # (None means individual alternate line, not contest)
                    if side.upper() != 'OVER' or not player_name or multiplier is None:
                        continue

                    key = (self.normalize_player_name(player_name).lower(), stat_type)

                    # Keep the entry with the lowest (best) multiplier per player
                    if key not in multipliers or multiplier < multipliers[key]['multiplier']:
                        multipliers[key] = {
                            'multiplier': float(multiplier),
                            'alt_line': float(alt_line) if alt_line is not None else None,
                        }

        return multipliers

    def _parse_event_props(self, main_data: Dict, alt_data: Optional[Dict], event_id: str) -> List[Dict]:
        """
        Parse event odds response into standardized prop format, enriching
        main market props with real contest multipliers from alternates.

        Args:
            main_data: Raw API response for main markets
            alt_data: Raw API response for alternate markets (may be None)
            event_id: Event ID for reference

        Returns:
            List of standardized prop dictionaries with real multipliers
        """
        props = []

        # Extract alternate multipliers first
        alt_multipliers = {}
        if alt_data:
            alt_multipliers = self._extract_alternate_multipliers(alt_data)
            if self.verbose and alt_multipliers:
                real_mults = [v['multiplier'] for v in alt_multipliers.values() if v['multiplier'] != 1.0]
                logger.debug(
                    f"  Alternate multipliers: {len(alt_multipliers)} players, "
                    f"{len(real_mults)} with non-1.0 mult"
                )

        # Get event info
        home_team = main_data.get('home_team', '')
        away_team = main_data.get('away_team', '')
        commence_time = main_data.get('commence_time', '')

        # Parse commence time to game date (EST)
        game_date = None
        if commence_time:
            try:
                utc_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                est_time = utc_time - timedelta(hours=5)
                game_date = est_time.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Failed to parse commence_time: {e}")

        # Parse main market props
        for bookmaker in main_data.get('bookmakers', []):
            if bookmaker.get('key') != 'pick6':
                continue

            markets = bookmaker.get('markets', [])

            for market in markets:
                market_key = market.get('key', '')
                stat_type = STAT_TYPE_MAP.get(market_key)

                if not stat_type:
                    continue

                outcomes = market.get('outcomes', [])

                for outcome in outcomes:
                    player_name = outcome.get('description', '')
                    line = outcome.get('point')
                    side = outcome.get('name', '')

                    if not player_name or line is None:
                        continue

                    if side.upper() != 'OVER':
                        continue

                    normalized_name = self.normalize_player_name(player_name)

                    # Look up real multiplier from alternates
                    alt_key = (normalized_name.lower(), stat_type)
                    alt_info = alt_multipliers.get(alt_key)

                    if alt_info:
                        multiplier = alt_info['multiplier']
                        alt_line = alt_info.get('alt_line')
                    else:
                        # No alternate data — use main market mult (1.0)
                        multiplier = float(outcome.get('multiplier', 1.0))
                        alt_line = None

                    prop = {
                        'player_name': normalized_name,
                        'stat_type': stat_type,
                        'line': float(line),
                        'side': 'OVER',
                        'pick6_multiplier': multiplier,
                        'book_name': 'pick6',
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_date': game_date,
                        'event_id': event_id,
                        'fetch_timestamp': datetime.now().isoformat(),
                    }

                    # Include alternate contest line if available
                    if alt_line is not None:
                        prop['pick6_contest_line'] = alt_line

                    props.append(prop)

        # Also add players who ONLY appear in alternates (not in main market)
        # These are additional players with contest data
        main_keys = {(p['player_name'].lower(), p['stat_type']) for p in props}

        for (name_lower, stat_type), alt_info in alt_multipliers.items():
            if (name_lower, stat_type) not in main_keys:
                # Player only in alternates — use alternate line as their line
                if alt_info.get('alt_line') is not None:
                    prop = {
                        'player_name': name_lower.title(),  # Best effort capitalization
                        'stat_type': stat_type,
                        'line': alt_info['alt_line'],
                        'side': 'OVER',
                        'pick6_multiplier': alt_info['multiplier'],
                        'book_name': 'pick6',
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_date': game_date,
                        'event_id': event_id,
                        'fetch_timestamp': datetime.now().isoformat(),
                        'pick6_contest_line': alt_info['alt_line'],
                        'alternate_only': True,
                    }
                    props.append(prop)

        return props

    def fetch(self) -> List[Dict]:
        """
        Fetch all Pick6 props for today's NBA games.

        Returns:
            List of prop dictionaries with Pick6 multipliers
        """
        all_props = []

        # Get today's events
        events = self.get_nba_events()

        if not events:
            logger.warning("No NBA events found")
            return []

        # Filter to today's games only
        today = datetime.now().strftime('%Y-%m-%d')
        today_events = []

        for event in events:
            commence_time = event.get('commence_time', '')
            if commence_time:
                try:
                    utc_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    est_time = utc_time - timedelta(hours=5)
                    event_date = est_time.strftime('%Y-%m-%d')
                    if event_date == today:
                        today_events.append(event)
                except Exception:
                    pass

        if self.verbose:
            logger.info(f"Fetching Pick6 props for {len(today_events)} games on {today}")

        # Fetch props for each event
        for event in today_events:
            event_id = event.get('id')
            if not event_id:
                continue

            props = self.get_event_player_props(event_id)
            all_props.extend(props)

            if self.verbose:
                logger.info(f"  {event.get('away_team')} @ {event.get('home_team')}: {len(props)} props")

        # Deduplicate
        all_props = self.deduplicate_props(all_props)

        if self.verbose:
            logger.info(f"Total Pick6 props fetched: {len(all_props)}")

            # Stats by multiplier range
            if all_props:
                trap_count = sum(1 for p in all_props if p.get('pick6_multiplier', 1.0) > 5.0)
                easy_count = sum(1 for p in all_props if p.get('pick6_multiplier', 1.0) < 1.0)
                logger.info(f"  Easy (mult < 1.0): {easy_count}")
                logger.info(f"  Traps (mult > 5.0): {trap_count}")

        return all_props

    def fetch_for_date(self, target_date: str) -> List[Dict]:
        """
        Fetch Pick6 props for a specific date.

        Note: For historical dates, this uses the historical API endpoint
        which costs more credits.

        Args:
            target_date: Date string in YYYY-MM-DD format

        Returns:
            List of prop dictionaries
        """
        all_props = []

        # Get events
        events = self.get_nba_events()

        if not events:
            logger.warning("No NBA events found")
            return []

        # Filter to target date
        target_events = []

        for event in events:
            commence_time = event.get('commence_time', '')
            if commence_time:
                try:
                    utc_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    est_time = utc_time - timedelta(hours=5)
                    event_date = est_time.strftime('%Y-%m-%d')
                    if event_date == target_date:
                        target_events.append(event)
                except Exception:
                    pass

        if self.verbose:
            logger.info(f"Fetching Pick6 props for {len(target_events)} games on {target_date}")

        for event in target_events:
            event_id = event.get('id')
            if not event_id:
                continue

            props = self.get_event_player_props(event_id)
            all_props.extend(props)

        return self.deduplicate_props(all_props)

    def save_to_json(self, props: List[Dict], suffix: str = "") -> Path:
        """
        Save props to JSON file.

        Overrides base to save to the-odds-api-data directory.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"pick6_live_{timestamp}"
        if suffix:
            filename = f"{filename}_{suffix}"
        filename = f"{filename}.json"

        # Save to the-odds-api-data directory
        output_dir = Path(__file__).parent.parent.parent.parent / 'the-odds-api-data'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

        output_data = {
            'source': 'pick6',
            'fetch_timestamp': datetime.now().isoformat(),
            'total_props': len(props),
            'props': props
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        if self.verbose:
            logger.info(f"Saved {len(props)} Pick6 props to: {output_file}")

        return output_file


def fetch_pick6_for_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """
    Pipeline-friendly function to fetch Pick6 data.

    Returns:
        Dict with:
        - success: bool
        - props: List of prop dicts
        - props_by_player: Dict mapping (player_name, stat_type) to prop data
        - trap_count: Number of props with mult > 5.0
        - easy_count: Number of props with mult < 1.0
    """
    fetcher = Pick6Fetcher(verbose=verbose)
    props = fetcher.fetch()

    # Build lookup dict for easy joining
    props_by_player = {}
    for prop in props:
        key = (
            prop['player_name'].lower(),
            prop['stat_type'],
            prop.get('game_date', '')
        )
        props_by_player[key] = prop

    # Calculate stats
    trap_count = sum(1 for p in props if p.get('pick6_multiplier', 1.0) > 5.0)
    easy_count = sum(1 for p in props if p.get('pick6_multiplier', 1.0) < 1.0)

    return {
        'success': len(props) > 0,
        'props': props,
        'props_by_player': props_by_player,
        'total': len(props),
        'trap_count': trap_count,
        'easy_count': easy_count,
    }


if __name__ == '__main__':
    # Test the fetcher
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("Pick6 Fetcher Test")
    print("=" * 60)

    fetcher = Pick6Fetcher(verbose=True)
    props = fetcher.fetch()

    if props:
        # Save to file
        output_file = fetcher.save_to_json(props)
        print(f"\nSaved to: {output_file}")

        # Show sample props
        print("\nSample props:")
        for prop in props[:5]:
            mult = prop.get('pick6_multiplier', 1.0)
            mult_signal = "TRAP" if mult > 5.0 else ("EASY" if mult < 1.0 else "STD")
            print(f"  {prop['player_name']}: {prop['stat_type']} O{prop['line']} (mult={mult:.2f} {mult_signal})")
    else:
        print("No props fetched")
