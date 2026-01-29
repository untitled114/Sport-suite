#!/usr/bin/env python3
"""
Historical Multi-Book Props Fetcher for NBA XL System (Date-Based)
===================================================================
Fetches historical player props from BettingPros Premium API with actual results.

KEY INSIGHT:
The /v3/props endpoint supports date parameter (not player_slug + season).
This is the SAME endpoint used by the live fetcher, just with historical dates.

Strategy:
- Fetch per DATE per MARKET per BOOK (not per player!)
- Iterate through date range (Oct 2023 - Apr 2024 for 2023-24 season)
- Returns props for ALL players that had lines on that date
- Includes actual results in `scoring.actual` field

This is MORE efficient than per-player fetching:
- **OLD approach:** 150 players × 4 markets × 7 books × 2 seasons = 8,400 API calls
- **NEW approach:** ~180 days × 4 markets × 7 books × 2 seasons = 10,080 API calls
  BUT captures ALL players (not just top 150), so more complete dataset!

Usage:
    # Fetch 2023-24 season
    python fetch_historical_by_date.py \\
      --start-date 2023-10-24 \\
      --end-date 2024-04-14

    # Fetch 2024-25 season
    python fetch_historical_by_date.py \\
      --start-date 2024-10-22 \\
      --end-date 2025-04-13

    # Fetch specific month
    python fetch_historical_by_date.py \\
      --start-date 2024-10-01 \\
      --end-date 2024-10-31
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from base_fetcher import BaseFetcher

# BettingPros Premium API configuration
API_BASE_URL = "https://api.bettingpros.com/v3/props"

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


class HistoricalDateFetcher(BaseFetcher):
    """Fetches historical multi-book props by iterating through dates"""

    def __init__(self, verbose: bool = True):
        """
        Initialize historical date fetcher.

        Args:
            verbose: Enable verbose logging
        """
        super().__init__(
            source_name='bettingpros_historical_date',
            rate_limit=1.0,  # 1 request per second (conservative)
            max_retries=3,
            timeout=30,
            verbose=verbose
        )

        self.api_base_url = API_BASE_URL
        self.premium_headers = PREMIUM_HEADERS
        self.total_api_calls = 0
        self.total_props_fetched = 0

    def fetch_date_market_book(
        self,
        date: str,
        market_id: int,
        market_name: str,
        book_id: int,
        book_name: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for one date, one market, one book.

        This returns props for ALL players that had lines on that date.

        Args:
            date: Game date (YYYY-MM-DD)
            market_id: BettingPros market ID (156=POINTS, etc.)
            market_name: Market name (POINTS, REBOUNDS, etc.)
            book_id: Sportsbook ID (12=DraftKings, etc.)
            book_name: Book name (draftkings, etc.)

        Returns:
            List of historical prop dictionaries with actual results
        """
        all_props = []
        page = 1

        while True:
            # Build request parameters
            # NOTE: ev_threshold=false is CRITICAL to get ALL players
            # Without it, API filters out players without projection data (e.g., Jokic, Giddey)
            params = {
                'sport': 'NBA',
                'date': date,
                'market_id': market_id,
                'book_id': book_id,
                'limit': 500,  # Max per page
                'page': page,
                'ev_threshold': 'false',  # Get ALL players, not just those with EV data
                'include_injured': 'true',  # Include injured players
            }

            if self.verbose and page == 1:
                print(f"  [{date}] {market_name} @ {book_name}")

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
                    print(f"    ❌ API request failed")
                break

            # Parse response - handle empty responses gracefully
            # Some books (caesars, bet365, fanatics) return empty responses for historical dates
            try:
                # Check for empty response before parsing
                if not response.content or len(response.content) < 10:
                    # Empty response - this book has no historical data for this date
                    break
                data = response.json()
            except Exception as e:
                # Silently skip - expected for some books on historical dates
                break

            # Extract props from response
            props = data.get('props', [])

            if not props:
                # No more props on this page
                break

            # Parse each prop
            for raw_prop in props:
                try:
                    parsed_prop = self._parse_prop(
                        raw_prop=raw_prop,
                        date=date,
                        market_name=market_name,
                        book_name=book_name
                    )

                    if parsed_prop:
                        all_props.append(parsed_prop)

                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Error parsing prop: {e}")
                    continue

            # Check if there are more pages
            pagination = data.get('_pagination', {})
            total_pages = pagination.get('total_pages', 1)

            if page >= total_pages:
                break

            page += 1

        if self.verbose:
            print(f"    ✅ Fetched {len(all_props)} props")

        self.total_props_fetched += len(all_props)

        return all_props

    def _parse_prop(
        self,
        raw_prop: Dict[str, Any],
        date: str,
        market_name: str,
        book_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse API response prop into standardized format with FULL cheatsheet data.

        API response structure:
        {
          "participant": {"id": "3024", "name": "Devin Booker", ...},
          "over": {"line": 24.5, "odds": -120, "book": 12, "consensus_line": 24.5, ...},
          "under": {"line": 24.5, "odds": -110, "book": 12, ...},
          "projection": {"value": 26.5, "diff": 2.0, "bet_rating": 4, "expected_value": 0.08, ...},
          "performance": {"last_5": {"over": 3, "under": 2}, "season": {...}, ...},
          "extra": {"opposition_rank": {"rank": 15, "value": 23.2, "tied": false}},
          "scoring": {"is_scored": true, "actual": 23, ...},
          "event_id": 25073,
          ...
        }

        Args:
            raw_prop: Raw prop from API
            date: Game date
            market_name: Market name (POINTS, etc.)
            book_name: Book name (draftkings, etc.)

        Returns:
            Standardized prop dictionary or None if invalid
        """
        try:
            # Extract participant fields
            participant = raw_prop.get('participant', {})
            player_name = participant.get('name')
            player_data = participant.get('player', {})
            player_team = player_data.get('team')

            # Extract over/under fields
            over = raw_prop.get('over', {})
            under = raw_prop.get('under', {})
            over_line = over.get('line')
            over_odds = over.get('odds', -110)
            under_line = under.get('line')
            under_odds = under.get('odds', -110)
            consensus_line = over.get('consensus_line')

            # Extract scoring (actual result)
            scoring = raw_prop.get('scoring', {})
            actual_value = scoring.get('actual')

            event_id = raw_prop.get('event_id')

            # Validate required fields
            if not all([player_name, over_line]):
                return None

            # ========== CHEATSHEET DATA: Projection ==========
            projection = raw_prop.get('projection', {})
            proj_value = projection.get('value')
            proj_diff = projection.get('diff')
            bet_rating = projection.get('bet_rating')
            expected_value = projection.get('expected_value')
            probability = projection.get('probability')
            recommended_side = projection.get('recommended_side')

            # ========== CHEATSHEET DATA: Opposition Rank ==========
            extra = raw_prop.get('extra', {})
            opp_rank_data = extra.get('opposition_rank', {})
            opp_rank = opp_rank_data.get('rank')
            opp_value = opp_rank_data.get('value')
            opp_tied = opp_rank_data.get('tied')

            # ========== CHEATSHEET DATA: Performance / Hit Rates ==========
            performance = raw_prop.get('performance', {})

            def calc_hit_rate(perf_data: Dict) -> Optional[float]:
                """Calculate hit rate from performance data"""
                over_count = perf_data.get('over', 0)
                under_count = perf_data.get('under', 0)
                total = over_count + under_count
                if total > 0:
                    return round(over_count / total, 3)
                return None

            # Extract all performance windows
            last_1 = performance.get('last_1', {})
            last_5 = performance.get('last_5', {})
            last_10 = performance.get('last_10', {})
            last_15 = performance.get('last_15', {})
            last_20 = performance.get('last_20', {})
            season_perf = performance.get('season', {})
            prior_season = performance.get('prior_season', {})
            h2h = performance.get('h2h', {})
            streak = performance.get('streak')
            streak_type = performance.get('streak_type')

            # Build comprehensive prop with ALL cheatsheet fields
            prop = {
                # === Basic Prop Info ===
                'player_name': player_name,
                'player_team': player_team,
                'stat_type': market_name,
                'book_name': book_name,
                'game_date': date,
                'game_id': str(event_id) if event_id else '',

                # === Line Data ===
                'line': float(over_line),
                'over_line': float(over_line),
                'under_line': float(under_line) if under_line else float(over_line),
                'over_odds': over_odds,
                'under_odds': under_odds,
                'consensus_line': float(consensus_line) if consensus_line else None,

                # === Actual Result ===
                'actual_value': float(actual_value) if actual_value is not None else None,

                # === Projection (Cheatsheet) ===
                'projection': float(proj_value) if proj_value is not None else None,
                'projection_diff': float(proj_diff) if proj_diff is not None else None,
                'bet_rating': bet_rating,
                'expected_value': expected_value,
                'ev_pct': round(expected_value * 100, 2) if expected_value else None,
                'probability': probability,
                'recommended_side': recommended_side,

                # === Opposition Rank (Cheatsheet) ===
                'opp_rank': opp_rank,
                'opp_value': opp_value,
                'opp_tied': opp_tied,

                # === Hit Rates (Cheatsheet) ===
                'hit_rate_l1': calc_hit_rate(last_1),
                'hit_rate_l5': calc_hit_rate(last_5),
                'hit_rate_l10': calc_hit_rate(last_10),
                'hit_rate_l15': calc_hit_rate(last_15),
                'hit_rate_l20': calc_hit_rate(last_20),
                'hit_rate_season': calc_hit_rate(season_perf),
                'hit_rate_prior_season': calc_hit_rate(prior_season),
                'hit_rate_h2h': calc_hit_rate(h2h),

                # === Raw Performance Counts ===
                'l1_over': last_1.get('over', 0),
                'l1_under': last_1.get('under', 0),
                'l5_over': last_5.get('over', 0),
                'l5_under': last_5.get('under', 0),
                'l10_over': last_10.get('over', 0),
                'l10_under': last_10.get('under', 0),
                'l15_over': last_15.get('over', 0),
                'l15_under': last_15.get('under', 0),
                'l20_over': last_20.get('over', 0),
                'l20_under': last_20.get('under', 0),
                'season_over': season_perf.get('over', 0),
                'season_under': season_perf.get('under', 0),
                'prior_season_over': prior_season.get('over', 0),
                'prior_season_under': prior_season.get('under', 0),
                'h2h_over': h2h.get('over', 0),
                'h2h_under': h2h.get('under', 0),

                # === Streak Info ===
                'streak': streak,
                'streak_type': streak_type,

                # === Metadata ===
                'opponent_team': '',  # Will be enriched later
                'is_home': None,  # Will be enriched later
                'season': int(date.split('-')[0]) if date else None,
                'fetch_timestamp': datetime.now().isoformat(),
                'source': 'bettingpros_historical'
            }

            return prop

        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  Parse error: {e}")
            return None

    def fetch_date_all_markets_all_books(
        self,
        date: str,
        markets: Optional[List[str]] = None,
        books: Optional[Dict[int, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for one date across all markets and books.

        Args:
            date: Game date (YYYY-MM-DD)
            markets: List of market names to fetch (default: all 4)
            books: Dict of book_id → book_name to fetch (default: all 7)

        Returns:
            List of all historical props for this date
        """
        if markets is None:
            markets = list(MARKETS.keys())

        if books is None:
            books = PRIORITY_BOOKS

        all_props = []

        print(f"\n{'='*70}")
        print(f"FETCHING DATE: {date}")
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
                props = self.fetch_date_market_book(
                    date=date,
                    market_id=market_id,
                    market_name=market_name,
                    book_id=book_id,
                    book_name=book_name
                )

                all_props.extend(props)

        print(f"\n✅ Date complete: {len(all_props)} total props")

        return all_props

    def fetch_date_range(
        self,
        start_date: str,
        end_date: str,
        markets: Optional[List[str]] = None,
        books: Optional[Dict[int, str]] = None,
        save_frequency: int = 7  # Save every 7 days (week)
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical props for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            markets: Markets to fetch (default: all)
            books: Books to fetch (default: all)
            save_frequency: Save checkpoint every N days

        Returns:
            List of all historical props
        """
        all_props = []

        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Generate date list
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        print(f"\n{'='*70}")
        print(f"HISTORICAL FETCH: {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"Date range: {len(dates)} days")
        print(f"Markets: {len(markets) if markets else 4}")
        print(f"Books: {len(books) if books else 7}")
        print(f"Expected API calls: {len(dates) * (len(markets) if markets else 4) * (len(books) if books else 7)}")
        print(f"Estimated time: {len(dates) * (len(markets) if markets else 4) * (len(books) if books else 7) / 60:.1f} minutes")
        print(f"{'='*70}\n")

        # Loop through dates
        for i, date in enumerate(dates, 1):
            print(f"\n[{i}/{len(dates)}] Processing date: {date}")

            # Fetch this date
            date_props = self.fetch_date_all_markets_all_books(
                date=date,
                markets=markets,
                books=books
            )

            all_props.extend(date_props)

            # Save checkpoint every N days
            if i % save_frequency == 0:
                checkpoint_file = self.save_to_json(
                    props=all_props,
                    suffix=f"{start_date}_to_{date}_checkpoint"
                )
                print(f"\n💾 Checkpoint saved: {checkpoint_file}")

            # Progress summary
            print(f"\n📊 Progress: {i}/{len(dates)} dates ({i/len(dates)*100:.1f}%)")
            print(f"   API calls: {self.total_api_calls}")
            print(f"   Props fetched: {self.total_props_fetched}")

        return all_props

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Main fetch method (required by BaseFetcher).

        Not used directly - use fetch_date_range() instead.
        """
        raise NotImplementedError("Use fetch_date_range() method instead")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Fetch historical multi-book NBA props by date range')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--markets', type=str, nargs='+', help='Markets to fetch (POINTS REBOUNDS ASSISTS THREES)')
    parser.add_argument('--books', type=str, nargs='+', help='Books to fetch (draftkings fanduel betmgm caesars bet365 betrivers fanatics)')
    parser.add_argument('--save-frequency', type=int, default=7, help='Save checkpoint every N days')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    args = parser.parse_args()

    # Parse markets filter
    markets = args.markets if args.markets else None

    # Parse books filter
    books = None
    if args.books:
        # Map book names to IDs
        book_name_to_id = {v: k for k, v in PRIORITY_BOOKS.items()}
        books = {book_name_to_id[name]: name for name in args.books if name in book_name_to_id}

    # Create fetcher
    fetcher = HistoricalDateFetcher(verbose=not args.quiet)

    try:
        # Fetch historical props
        start_time = time.time()

        all_props = fetcher.fetch_date_range(
            start_date=args.start_date,
            end_date=args.end_date,
            markets=markets,
            books=books,
            save_frequency=args.save_frequency
        )

        elapsed = time.time() - start_time

        # Save final output
        output_file = fetcher.save_to_json(
            props=all_props,
            suffix=f"{args.start_date}_to_{args.end_date}_COMPLETE"
        )

        # Final summary
        print(f"\n{'='*70}")
        print(f"FETCH COMPLETE")
        print(f"{'='*70}")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Total props: {len(all_props)}")
        print(f"API calls: {fetcher.total_api_calls}")
        print(f"Elapsed time: {elapsed/60:.1f} minutes")
        print(f"Output file: {output_file}")
        print(f"{'='*70}\n")

    finally:
        fetcher.close()


if __name__ == '__main__':
    main()
