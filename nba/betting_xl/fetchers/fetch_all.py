#!/usr/bin/env python3
"""
NBA XL System - Multi-Source Prop Fetcher Orchestrator
========================================================
Runs all prop fetchers in sequence and deduplicates results.

This script coordinates fetching from:
1. BettingPros API (7 sportsbooks + consensus)
2. Underdog Cheat Sheet (projections + hit rates)
3. Direct Sportsbook APIs (7 books via ThreadPoolExecutor)

Usage:
    # Fetch all sources (BP + Underdog + Direct)
    python fetch_all.py

    # Fetch for specific date
    python fetch_all.py --date 2025-11-06

    # Skip direct sportsbook fetchers (BP + Underdog only)
    python fetch_all.py --skip-direct

    # Direct sportsbooks only (skip BP + Underdog)
    python fetch_all.py --direct-only

    # Dry run (don't save)
    python fetch_all.py --dry-run

Output:
    Saves to: nba/betting_xl/lines/all_sources_{YYYY-MM-DD}_{HH-MM-SS}.json
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from zoneinfo import ZoneInfo

import requests

from nba.betting_xl.fetchers.fetch_bettingpros import BettingProsFetcher
from nba.betting_xl.fetchers.fetch_cheatsheet import CheatSheetFetcher
from nba.betting_xl.utils.logging_config import add_logging_args, get_logger, setup_logging

EST = ZoneInfo("America/New_York")

# Direct sportsbook fetchers (Phase 2)
# Graceful imports — missing fetchers are excluded at runtime
_DIRECT_FETCHER_IMPORTS: List[Tuple[str, Optional[Type]]] = []

try:
    from nba.betting_xl.fetchers.fetch_draftkings_direct import DraftKingsDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("draftkings", DraftKingsDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("draftkings", None))

try:
    from nba.betting_xl.fetchers.fetch_fanduel_direct import FanDuelDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("fanduel", FanDuelDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("fanduel", None))

try:
    from nba.betting_xl.fetchers.fetch_betmgm_direct import BetMGMDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("betmgm", BetMGMDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("betmgm", None))

try:
    from nba.betting_xl.fetchers.fetch_betrivers_direct import BetRiversDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("betrivers", BetRiversDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("betrivers", None))

try:
    from nba.betting_xl.fetchers.fetch_hardrock_direct import HardRockDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("hardrock", HardRockDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("hardrock", None))

try:
    from nba.betting_xl.fetchers.fetch_underdog_direct import UnderdogDirectFetcher

    _DIRECT_FETCHER_IMPORTS.append(("underdog", UnderdogDirectFetcher))
except ImportError:
    _DIRECT_FETCHER_IMPORTS.append(("underdog", None))

# Logger will be configured in main()
logger = get_logger(__name__)


class FetchOrchestrator:
    """Orchestrates fetching from multiple prop sources"""

    # Direct sportsbook fetchers — only include those that imported successfully
    DIRECT_FETCHERS: List[Tuple[str, Type]] = [
        (name, cls) for name, cls in _DIRECT_FETCHER_IMPORTS if cls is not None
    ]

    # Maximum threads for direct fetcher parallelism
    MAX_DIRECT_THREADS = 6

    def __init__(
        self,
        date: str = None,
        verbose: bool = True,
        dry_run: bool = False,
        skip_direct: bool = False,
        direct_only: bool = False,
    ):
        """
        Initialize orchestrator.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            verbose: Enable verbose logging
            dry_run: Don't save output
            skip_direct: Skip direct sportsbook fetchers (BP + Underdog only)
            direct_only: Skip BP + Underdog (direct sportsbooks only)
        """
        self.date = date or datetime.now(EST).strftime("%Y-%m-%d")
        self.verbose = verbose
        self.dry_run = dry_run
        self.skip_direct = skip_direct
        self.direct_only = direct_only

        # Output directory
        self.output_dir = Path(__file__).parent.parent / "lines"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.props_by_source = {}
        self.errors = []

    def fetch_bettingpros(self) -> List[Dict[str, Any]]:
        """Fetch from BettingPros API"""
        try:
            print("\n" + "=" * 70)
            print("SOURCE 1/3: BETTINGPROS API (7 sportsbooks)")
            print("=" * 70 + "\n")

            with BettingProsFetcher(date=self.date, verbose=self.verbose) as fetcher:
                props = fetcher.fetch()

            return props

        except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"BettingPros fetch failed: {e}")
            self.errors.append(("bettingpros", str(e)))
            return []

    def fetch_underdog_sheet(self) -> List[Dict[str, Any]]:
        """
        Fetch from Underdog Cheat Sheet via BettingPros API.

        Returns full cheat sheet data including:
        - Projections (value, diff)
        - Bet ratings (1-5 stars)
        - Expected Value (EV)
        - Hit rates (L-5, L-15, Season)
        - Opposition rank

        Uses book_id=36 for Underdog-specific lines.
        """
        print("\n" + "=" * 70)
        print("SOURCE 2/3: UNDERDOG CHEAT SHEET")
        print("=" * 70 + "\n")

        try:
            with CheatSheetFetcher(
                date=self.date,
                platform="underdog",
                include_combos=True,  # Include DFS combo markets
                verbose=self.verbose,
            ) as fetcher:
                props = fetcher.fetch()

            print(f"[OK] Underdog cheat sheet: {len(props)} props with analytics\n")
            return props

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Underdog cheat sheet fetch failed: {e}")
            self.errors.append(("underdog_sheet", str(e)))
            return []

    def _fetch_single_direct(
        self, name: str, fetcher_cls: Type
    ) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        """
        Fetch from a single direct sportsbook. Runs inside a thread.

        Args:
            name: Short name for the sportsbook (e.g., "draftkings")
            fetcher_cls: The fetcher class to instantiate

        Returns:
            Tuple of (name, props_list, error_string_or_None)
        """
        try:
            with fetcher_cls(verbose=self.verbose) as fetcher:
                props = fetcher.fetch()
            logger.info(f"[direct] {name}: {len(props)} props")
            return (name, props, None)
        except Exception as e:
            logger.error(f"[direct] {name} failed: {e}")
            return (name, [], str(e))

    def fetch_direct_sportsbooks(self) -> List[Dict[str, Any]]:
        """
        Fetch from all direct sportsbook APIs in parallel.

        Each fetcher runs in its own thread via ThreadPoolExecutor.
        Failures are isolated per-book — a single book failing does not
        affect other books.

        Returns:
            Combined props list from all successful direct fetchers.
        """
        if not self.DIRECT_FETCHERS:
            logger.warning("No direct fetchers available (none imported successfully)")
            return []

        print("\n" + "=" * 70)
        source_label = "SOURCE 3/3" if not self.direct_only else "SOURCE 1/1"
        print(f"{source_label}: DIRECT SPORTSBOOK APIs ({len(self.DIRECT_FETCHERS)} books)")
        print("=" * 70 + "\n")

        available_names = [name for name, _ in self.DIRECT_FETCHERS]
        print(f"  Fetching: {', '.join(available_names)}")
        print(f"  Threads:  {min(self.MAX_DIRECT_THREADS, len(self.DIRECT_FETCHERS))}")
        print()

        combined_props: List[Dict[str, Any]] = []
        direct_results: Dict[str, int] = {}
        start_time = time.time()

        with ThreadPoolExecutor(
            max_workers=min(self.MAX_DIRECT_THREADS, len(self.DIRECT_FETCHERS))
        ) as executor:
            futures = {
                executor.submit(self._fetch_single_direct, name, cls): name
                for name, cls in self.DIRECT_FETCHERS
            }

            for future in as_completed(futures):
                book_name = futures[future]
                try:
                    name, props, error = future.result(timeout=120)
                    direct_results[name] = len(props)
                    if error:
                        self.errors.append((f"direct_{name}", error))
                        status = "[FAIL]"
                    elif len(props) == 0:
                        status = "[WARN]"
                    else:
                        combined_props.extend(props)
                        status = "[ OK ]"
                    print(f"  {status} {name:15s}: {len(props):4d} props")
                except Exception as e:
                    logger.error(f"[direct] {book_name} thread error: {e}")
                    self.errors.append((f"direct_{book_name}", str(e)))
                    direct_results[book_name] = 0
                    print(f"  [FAIL] {book_name:15s}:    0 props ({e})")

        elapsed = time.time() - start_time
        print(f"\n  Total direct props: {len(combined_props)} ({elapsed:.1f}s)")

        # Store per-book results for summary
        self.props_by_source["direct_books"] = direct_results

        return combined_props

    def deduplicate_across_sources(self, all_props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate props across all sources.

        Strategy:
        - Keep all unique (player, stat, book) combinations
        - For consensus lines from multiple sources, keep first seen
        - Preserve book-specific lines (this is what we want!)

        Args:
            all_props: Combined props from all sources

        Returns:
            Deduplicated list
        """
        seen = set()
        deduped = []

        for prop in all_props:
            # Create unique key
            key = (
                prop.get("player_name", "").lower(),
                prop.get("stat_type", "").upper(),
                prop.get("book_name", "").lower(),
                round(
                    float(prop.get("line", 0)), 1
                ),  # Round to 1 decimal to catch minor variations
            )

            if key not in seen:
                seen.add(key)
                deduped.append(prop)

        removed = len(all_props) - len(deduped)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate props across sources")

        return deduped

    def calculate_line_stats(self, props: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate line spread and consensus for each player+stat combination.

        Args:
            props: List of all props

        Returns:
            Dictionary with line statistics
        """
        from collections import defaultdict

        # Group by (player, stat)
        groups = defaultdict(list)
        for prop in props:
            key = (prop.get("player_name", ""), prop.get("stat_type", ""))
            groups[key].append(prop)

        stats = {
            "total_player_stat_combos": len(groups),
            "multi_book_combos": sum(1 for lines in groups.values() if len(lines) > 1),
            "single_book_combos": sum(1 for lines in groups.values() if len(lines) == 1),
            "max_line_spread": 0,
            "avg_line_spread": 0,
            "line_spreads": [],
        }

        spreads = []
        for (player, stat), lines in groups.items():
            if len(lines) > 1:
                line_values = [float(p["line"]) for p in lines]
                spread = max(line_values) - min(line_values)
                spreads.append(spread)
                stats["line_spreads"].append(
                    {
                        "player": player,
                        "stat": stat,
                        "spread": spread,
                        "min_line": min(line_values),
                        "max_line": max(line_values),
                        "num_books": len(lines),
                    }
                )

        if spreads:
            stats["max_line_spread"] = max(spreads)
            stats["avg_line_spread"] = sum(spreads) / len(spreads)

        # Sort by spread (descending)
        stats["line_spreads"].sort(key=lambda x: x["spread"], reverse=True)

        return stats

    def run(self) -> Dict[str, Any]:
        """
        Run all fetchers and combine results.

        Flow:
            Source 1: BettingPros (existing)          — skipped if --direct-only
            Source 2: Underdog cheatsheet (existing)   — skipped if --direct-only
            Source 3: Direct Sportsbooks (NEW)         — skipped if --skip-direct
            Deduplication (all sources combined)

        Returns:
            Dictionary with combined results and metadata
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "NBA XL MULTI-SOURCE PROP FETCHER")
        print("=" * 80)
        print(f"Date: {self.date}")
        mode_parts = []
        if self.dry_run:
            mode_parts.append("DRY RUN")
        if self.skip_direct:
            mode_parts.append("SKIP DIRECT")
        if self.direct_only:
            mode_parts.append("DIRECT ONLY")
        print(f"Mode: {', '.join(mode_parts) if mode_parts else 'LIVE (all sources)'}")
        print(f"Timestamp: {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # Fetch from all sources
        all_props = []
        bettingpros_props = []
        underdog_props = []
        direct_props = []

        if not self.direct_only:
            # Source 1: BettingPros API (7 sportsbooks)
            bettingpros_props = self.fetch_bettingpros()
            self.props_by_source["bettingpros"] = bettingpros_props
            all_props.extend(bettingpros_props)

            # Source 2: Underdog Cheat Sheet
            underdog_props = self.fetch_underdog_sheet()
            self.props_by_source["underdog"] = underdog_props
            all_props.extend(underdog_props)

        if not self.skip_direct:
            # Source 3: Direct Sportsbook APIs (parallel)
            direct_props = self.fetch_direct_sportsbooks()
            all_props.extend(direct_props)

        # Deduplicate across sources
        print("\n" + "=" * 70)
        print("DEDUPLICATION")
        print("=" * 70)
        print(f"Total props before dedup: {len(all_props)}")

        deduped_props = self.deduplicate_across_sources(all_props)
        print(f"Total props after dedup: {len(deduped_props)}")
        print("=" * 70 + "\n")

        # Calculate line statistics
        line_stats = self.calculate_line_stats(deduped_props)

        # Build sources dict
        sources = {}
        if not self.direct_only:
            sources["bettingpros"] = len(bettingpros_props)
            sources["underdog"] = len(underdog_props)
        if not self.skip_direct:
            sources["direct_total"] = len(direct_props)
            # Include per-book breakdown
            direct_books = self.props_by_source.get("direct_books", {})
            for book_name, count in sorted(direct_books.items()):
                sources[f"direct_{book_name}"] = count

        # Build result
        result = {
            "date": self.date,
            "fetch_timestamp": datetime.now(EST).isoformat(),
            "sources": sources,
            "total_props": len(deduped_props),
            "line_stats": line_stats,
            "errors": self.errors,
            "props": deduped_props,
        }

        # Print summary
        self._print_summary(result)

        # Save to file
        if not self.dry_run:
            output_file = self._save_results(result)
            print(f"\n[OK] Saved {len(deduped_props)} props to: {output_file}\n")
        else:
            print("\n[DRY RUN] Results not saved\n")

        return result

    def _print_summary(self, result: Dict[str, Any]):
        """Print summary of fetch results"""
        print("\n" + "=" * 70)
        print("FETCH SUMMARY")
        print("=" * 70)

        print("\nProps by source:")
        for source, count in result["sources"].items():
            # Indent direct per-book lines under the direct_total header
            if source.startswith("direct_") and source != "direct_total":
                status = "[OK]" if count > 0 else "[--]"
                print(f"    {status} {source:20s}: {count:4d} props")
            else:
                status = "[OK]" if count > 0 else "[WARN]"
                print(f"  {status} {source:20s}: {count:4d} props")

        print(f"\nTotal unique props: {result['total_props']}")

        stats = result["line_stats"]
        print(f"\nLine spread analysis:")
        print(f"  Player+stat combinations: {stats['total_player_stat_combos']}")
        print(f"  Multiple books available:  {stats['multi_book_combos']}")
        print(f"  Single book only:          {stats['single_book_combos']}")

        if stats["multi_book_combos"] > 0:
            print(f"  Max line spread:           {stats['max_line_spread']:.2f}")
            print(f"  Avg line spread:           {stats['avg_line_spread']:.2f}")

            print(f"\nTop 5 line spreads (line shopping opportunities):")
            for i, spread_info in enumerate(stats["line_spreads"][:5], 1):
                print(
                    f"  {i}. {spread_info['player']:20s} {spread_info['stat']:10s}: "
                    f"{spread_info['min_line']:.1f} - {spread_info['max_line']:.1f} "
                    f"(spread: {spread_info['spread']:.1f}, {spread_info['num_books']} books)"
                )

        if result["errors"]:
            print(f"\n[WARN]  Errors encountered: {len(result['errors'])}")
            for source, error in result["errors"]:
                print(f"  - {source}: {error}")

        print("=" * 70)

    def _save_results(self, result: Dict[str, Any]) -> Path:
        """Save results to JSON file"""
        timestamp = datetime.now(EST).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"all_sources_{timestamp}.json"
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        return output_file


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NBA props from all sources")
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't save results)")
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip direct sportsbook fetchers (BP + Underdog only)",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip BettingPros and Underdog (direct sportsbooks only)",
    )
    add_logging_args(parser)  # Adds --debug and --quiet flags

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.skip_direct and args.direct_only:
        parser.error("--skip-direct and --direct-only are mutually exclusive")

    # Setup unified logging
    setup_logging("fetch_all", debug=args.debug, quiet=args.quiet)
    logger.info(f"Starting prop fetch for {args.date or 'today'}")

    if args.skip_direct:
        logger.info("Direct sportsbook fetchers SKIPPED (--skip-direct)")
    elif args.direct_only:
        logger.info("BettingPros + Underdog SKIPPED (--direct-only)")

    # Log available direct fetchers
    available = [name for name, cls in FetchOrchestrator.DIRECT_FETCHERS]
    failed = [name for name, cls in _DIRECT_FETCHER_IMPORTS if cls is None]
    if available:
        logger.info(f"Direct fetchers available: {', '.join(available)}")
    if failed:
        logger.warning(f"Direct fetchers failed to import: {', '.join(failed)}")

    # Run orchestrator
    orchestrator = FetchOrchestrator(
        date=args.date,
        verbose=not args.quiet,
        dry_run=args.dry_run,
        skip_direct=args.skip_direct,
        direct_only=args.direct_only,
    )

    try:
        orchestrator.run()
        logger.info("Fetch completed successfully")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\n[WARN]  Interrupted by user\n")
    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
