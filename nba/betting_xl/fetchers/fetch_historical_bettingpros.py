#!/usr/bin/env python3
"""
Historical Multi-Book Props Fetcher for NBA XL System
======================================================
Fetches historical player props from BettingPros Premium API with actual results.

This fetcher uses the /v3/props/analysis endpoint which returns:
- Historical prop lines per player per market per book
- ACTUAL RESULTS for each game (critical for training)
- Date range filtering
- Book-specific lines

Strategy:
- Fetch per player per market per book (per-player API limitation)
- Target: 150 players × 4 markets × 7 books × 2 seasons
- Rate limit: 1 req/sec
- Expected runtime: ~4.6 hours for 2 seasons (8,400 API calls)

Usage:
    # Fetch 2023-24 season
    python fetch_historical_bettingpros.py --season 2024 --players-file active_players.txt

    # Fetch 2024-25 season
    python fetch_historical_bettingpros.py --season 2025 --players-file active_players.txt

    # Fetch specific player
    python fetch_historical_bettingpros.py --season 2025 --player "LeBron James"
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from base_fetcher import BaseFetcher

# BettingPros Premium API configuration
API_BASE_URL = "https://api.bettingpros.com/v3/props/analysis"

PREMIUM_HEADERS = {
    'x-api-key': os.getenv('BETTINGPROS_API_KEY'),
    'x-level': 'cHJlbWl1bQ==',  # base64 for "premium"
    'accept': 'application/json'
}

# Market IDs - FIXED 2026-01-03: ASSISTS was 160 (STEALS), correct is 151
MARKETS = {
    'POINTS': 156,
    'REBOUNDS': 157,
    'ASSISTS': 151,  # FIXED: Was 160 (steals)
    'THREES': 162,
}

# Priority sportsbooks (book_id → book_name)
PRIORITY_BOOKS = {
    12: 'draftkings',
    10: 'fanduel',
    19: 'betmgm',
    13: 'caesars',
    24: 'bet365',
    18: 'betrivers',
    14: 'fanatics',
}


class HistoricalBettingProsFetcher(BaseFetcher):
    """Fetches historical multi-book props from BettingPros Premium API"""

    def __init__(self, verbose: bool = True):
        """
        Initialize historical fetcher.

        Args:
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name='bettingpros_historical',
            rate_limit=1.0,  # 1 request per second (conservative)
            max_retries=3,
            timeout=30,
            verbose=verbose
        )

        self.api_base_url = API_BASE_URL
        self.premium_headers = PREMIUM_HEADERS
        self.total_api_calls = 0
        self.total_props_fetched = 0

    def player_name_to_slug(self, player_name: str) -> str:
        """
        Convert player name to BettingPros slug format.

        Examples:
            "LeBron James" → "lebron-james"
            "Nikola Jokić" → "nikola-jokic"
            "Karl-Anthony Towns" → "karl-anthony-towns"

        Args:
            player_name: Full player name

        Returns:
            Slug format (lowercase, hyphenated)
        """
        # Normalize name
        slug = player_name.lower().strip()

        # Remove accents and special characters
        slug = slug.replace('ć', 'c').replace('č', 'c').replace('š', 's')
        slug = slug.replace('ž', 'z').replace('ñ', 'n').replace('é', 'e')
        slug = slug.replace('á', 'a').replace('ö', 'o').replace('ü', 'u')

        # Replace spaces with hyphens
        slug = slug.replace(' ', '-')

        # Remove dots, apostrophes
        slug = slug.replace('.', '').replace("'", '')

        # Remove multiple hyphens
        while '--' in slug:
            slug = slug.replace('--', '-')

        return slug

    def fetch_player_market_book(
        self,
        player_name: str,
        market_id: int,
        market_name: str,
        book_id: int,
        book_name: str,
        season: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for one player, one market, one book, one season.

        Args:
            player_name: Player's full name
            market_id: BettingPros market ID (156=POINTS, etc.)
            market_name: Market name (POINTS, REBOUNDS, etc.)
            book_id: Sportsbook ID (12=DraftKings, etc.)
            book_name: Book name (draftkings, etc.)
            season: Year (2024, 2025, 2026)

        Returns:
            List of historical prop dictionaries with actual results
        """
        # Convert player name to slug
        player_slug = self.player_name_to_slug(player_name)

        # Build request parameters
        params = {
            'sport': 'NBA',
            'player_slug': player_slug,
            'market_id': market_id,
            'season': season,
            'book_id': book_id,
            'limit': 1000  # Max props to return (should cover full season)
        }

        if self.verbose:
            print(f"\n[{player_name}] {market_name} @ {book_name} (season {season})")

        # Make API request
        response = self._make_request(
            url=self.api_base_url,
            method='GET',
            params=params,
            headers=self.premium_headers
        )

        self.total_api_calls += 1

        if not response:
            if self.verbose:
                print(f"  ❌ API request failed")
            return []

        # Parse response
        try:
            data = response.json()
        except Exception as e:
            if self.verbose:
                print(f"  ❌ JSON parse error: {e}")
            return []

        # Extract props from response
        props = self._parse_historical_response(
            data=data,
            player_name=player_name,
            market_name=market_name,
            book_name=book_name,
            season=season
        )

        if self.verbose:
            print(f"  ✅ Fetched {len(props)} historical props")

        self.total_props_fetched += len(props)

        return props

    def _parse_historical_response(
        self,
        data: Dict[str, Any],
        player_name: str,
        market_name: str,
        book_name: str,
        season: int
    ) -> List[Dict[str, Any]]:
        """
        Parse API response into standardized prop format.

        Args:
            data: API response JSON
            player_name: Player name
            market_name: Market name (POINTS, REBOUNDS, etc.)
            book_name: Book name (draftkings, etc.)
            season: Season year

        Returns:
            List of prop dictionaries
        """
        props = []

        # API response structure (expected):
        # {
        #   "props": [
        #     {
        #       "game_date": "2024-10-22",
        #       "game_id": "...",
        #       "opponent_team": "BOS",
        #       "is_home": true,
        #       "over_line": 25.5,
        #       "over_odds": -110,
        #       "under_line": 25.5,
        #       "under_odds": -110,
        #       "actual_value": 28,  # ⭐ KEY FIELD FOR TRAINING
        #       "game_time": "2024-10-22T19:00:00Z"
        #     },
        #     ...
        #   ]
        # }

        # Handle different response structures
        if isinstance(data, dict):
            if 'props' in data:
                raw_props = data['props']
            elif 'data' in data:
                raw_props = data['data']
            else:
                raw_props = [data]
        elif isinstance(data, list):
            raw_props = data
        else:
            return []

        for raw_prop in raw_props:
            try:
                # Extract fields
                game_date = raw_prop.get('game_date') or raw_prop.get('date')
                over_line = raw_prop.get('over_line') or raw_prop.get('line')
                under_line = raw_prop.get('under_line') or over_line
                over_odds = raw_prop.get('over_odds', -110)
                under_odds = raw_prop.get('under_odds', -110)
                actual_value = raw_prop.get('actual_value') or raw_prop.get('actual')
                game_id = raw_prop.get('game_id', '')
                game_time = raw_prop.get('game_time', '')
                opponent_team = raw_prop.get('opponent_team', '') or raw_prop.get('opponent', '')
                is_home = raw_prop.get('is_home')

                # Validate required fields
                if not all([game_date, over_line]):
                    continue

                # Build standardized prop
                prop = {
                    'player_name': player_name,
                    'stat_type': market_name,
                    'book_name': book_name,
                    'game_date': game_date,
                    'game_time': game_time,
                    'over_line': float(over_line),
                    'under_line': float(under_line),
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'actual_value': float(actual_value) if actual_value is not None else None,
                    'game_id': game_id,
                    'opponent_team': opponent_team,
                    'is_home': is_home,
                    'season': season,
                    'fetch_timestamp': datetime.now().isoformat(),
                    'source': 'bettingpros_historical'
                }

                props.append(prop)

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Error parsing prop: {e}")
                continue

        return props

    def fetch_player_all_markets_all_books(
        self,
        player_name: str,
        season: int,
        markets: Optional[List[str]] = None,
        books: Optional[Dict[int, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for one player across all markets and books.

        Args:
            player_name: Player's full name
            season: Season year (2024, 2025, 2026)
            markets: List of market names to fetch (default: all 4)
            books: Dict of book_id → book_name to fetch (default: all 7)

        Returns:
            List of all historical props for this player
        """
        if markets is None:
            markets = list(MARKETS.keys())

        if books is None:
            books = PRIORITY_BOOKS

        all_props = []

        print(f"\n{'='*70}")
        print(f"FETCHING PLAYER: {player_name} (Season {season})")
        print(f"{'='*70}")
        print(f"Markets: {len(markets)}")
        print(f"Books: {len(books)}")
        print(f"Expected API calls: {len(markets) * len(books)}")

        # Loop through markets
        for market_name in markets:
            market_id = MARKETS[market_name]

            # Loop through books
            for book_id, book_name in books.items():
                # Fetch this combination
                props = self.fetch_player_market_book(
                    player_name=player_name,
                    market_id=market_id,
                    market_name=market_name,
                    book_id=book_id,
                    book_name=book_name,
                    season=season
                )

                all_props.extend(props)

        print(f"\n✅ Player complete: {len(all_props)} total props")

        return all_props

    def fetch_all_players(
        self,
        player_names: List[str],
        season: int,
        markets: Optional[List[str]] = None,
        books: Optional[Dict[int, str]] = None,
        save_frequency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for multiple players.

        Args:
            player_names: List of player names
            season: Season year
            markets: Markets to fetch (default: all)
            books: Books to fetch (default: all)
            save_frequency: Save JSON every N players (for recovery)

        Returns:
            List of all historical props
        """
        all_props = []

        print(f"\n{'='*70}")
        print(f"HISTORICAL FETCH: Season {season}")
        print(f"{'='*70}")
        print(f"Players: {len(player_names)}")
        print(f"Markets: {len(markets) if markets else 4}")
        print(f"Books: {len(books) if books else 7}")
        print(f"Expected API calls: {len(player_names) * (len(markets) if markets else 4) * (len(books) if books else 7)}")
        print(f"Estimated time: {len(player_names) * (len(markets) if markets else 4) * (len(books) if books else 7) / 60:.1f} minutes")
        print(f"{'='*70}\n")

        # Loop through players
        for i, player_name in enumerate(player_names, 1):
            print(f"\n[{i}/{len(player_names)}] Processing: {player_name}")

            # Fetch this player
            player_props = self.fetch_player_all_markets_all_books(
                player_name=player_name,
                season=season,
                markets=markets,
                books=books
            )

            all_props.extend(player_props)

            # Save checkpoint every N players
            if i % save_frequency == 0:
                checkpoint_file = self.save_to_json(
                    props=all_props,
                    suffix=f"season_{season}_checkpoint_{i}players"
                )
                print(f"\n💾 Checkpoint saved: {checkpoint_file}")

            # Progress summary
            print(f"\n📊 Progress: {i}/{len(player_names)} players ({i/len(player_names)*100:.1f}%)")
            print(f"   API calls: {self.total_api_calls}")
            print(f"   Props fetched: {self.total_props_fetched}")

        return all_props

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Main fetch method (required by BaseFetcher).

        Not used directly - use fetch_all_players() instead.
        """
        raise NotImplementedError("Use fetch_all_players() method instead")


def load_player_names(filepath: str) -> List[str]:
    """
    Load player names from text file (one per line).

    Args:
        filepath: Path to player names file

    Returns:
        List of player names
    """
    with open(filepath, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Fetch historical multi-book NBA props')
    parser.add_argument('--season', type=int, required=True, help='Season year (2024, 2025, 2026)')
    parser.add_argument('--players-file', type=str, help='Text file with player names (one per line)')
    parser.add_argument('--player', type=str, help='Single player name to fetch')
    parser.add_argument('--markets', type=str, nargs='+', help='Markets to fetch (POINTS REBOUNDS ASSISTS THREES)')
    parser.add_argument('--books', type=str, nargs='+', help='Books to fetch (draftkings fanduel betmgm caesars bet365 betrivers fanatics)')
    parser.add_argument('--save-frequency', type=int, default=10, help='Save checkpoint every N players')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    args = parser.parse_args()

    # Validate inputs
    if not args.players_file and not args.player:
        parser.error("Must specify --players-file or --player")

    # Load player names
    if args.players_file:
        player_names = load_player_names(args.players_file)
    else:
        player_names = [args.player]

    # Parse markets filter
    markets = args.markets if args.markets else None

    # Parse books filter
    books = None
    if args.books:
        # Map book names to IDs
        book_name_to_id = {v: k for k, v in PRIORITY_BOOKS.items()}
        books = {book_name_to_id[name]: name for name in args.books if name in book_name_to_id}

    # Create fetcher
    fetcher = HistoricalBettingProsFetcher(verbose=not args.quiet)

    try:
        # Fetch historical props
        start_time = time.time()

        all_props = fetcher.fetch_all_players(
            player_names=player_names,
            season=args.season,
            markets=markets,
            books=books,
            save_frequency=args.save_frequency
        )

        elapsed = time.time() - start_time

        # Save final output
        output_file = fetcher.save_to_json(
            props=all_props,
            suffix=f"season_{args.season}_COMPLETE"
        )

        # Final summary
        print(f"\n{'='*70}")
        print(f"FETCH COMPLETE")
        print(f"{'='*70}")
        print(f"Season: {args.season}")
        print(f"Players: {len(player_names)}")
        print(f"Total props: {len(all_props)}")
        print(f"API calls: {fetcher.total_api_calls}")
        print(f"Elapsed time: {elapsed/60:.1f} minutes")
        print(f"Output file: {output_file}")
        print(f"{'='*70}\n")

    finally:
        fetcher.close()


if __name__ == '__main__':
    main()
