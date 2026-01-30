#!/usr/bin/env python3
"""
NBA XL System - Multi-Source Prop Fetcher Orchestrator
========================================================
Runs all prop fetchers in sequence and deduplicates results.

This script coordinates fetching from:
1. BettingPros API (7 sportsbooks + consensus)
2. Underdog Cheat Sheet (projections + hit rates)

Usage:
    # Fetch all sources
    python fetch_all.py

    # Fetch for specific date
    python fetch_all.py --date 2025-11-06

    # Dry run (don't save)
    python fetch_all.py --dry-run

Output:
    Saves to: nba/betting_xl/lines/all_sources_{YYYY-MM-DD}_{HH-MM-SS}.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

from nba.betting_xl.fetchers.fetch_bettingpros import BettingProsFetcher
from nba.betting_xl.fetchers.fetch_cheatsheet import CheatSheetFetcher
from nba.betting_xl.utils.logging_config import add_logging_args, get_logger, setup_logging

# Logger will be configured in main()
logger = get_logger(__name__)


class FetchOrchestrator:
    """Orchestrates fetching from multiple prop sources"""

    def __init__(self, date: str = None, verbose: bool = True, dry_run: bool = False):
        """
        Initialize orchestrator.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today.
            verbose: Enable verbose logging
            dry_run: Don't save output
        """
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.verbose = verbose
        self.dry_run = dry_run

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
            print("SOURCE 1/2: BETTINGPROS API (7 sportsbooks)")
            print("=" * 70 + "\n")

            with BettingProsFetcher(date=self.date, verbose=self.verbose) as fetcher:
                props = fetcher.fetch()

            return props

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
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
        print("SOURCE 2/2: UNDERDOG CHEAT SHEET")
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

        Returns:
            Dictionary with combined results and metadata
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "NBA XL MULTI-SOURCE PROP FETCHER")
        print("=" * 80)
        print(f"Date: {self.date}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # Fetch from all sources
        all_props = []

        # Source 1: BettingPros API (7 sportsbooks)
        bettingpros_props = self.fetch_bettingpros()
        self.props_by_source["bettingpros"] = bettingpros_props
        all_props.extend(bettingpros_props)

        # Source 2: Underdog Cheat Sheet
        underdog_props = self.fetch_underdog_sheet()
        self.props_by_source["underdog"] = underdog_props
        all_props.extend(underdog_props)

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

        # Build result
        result = {
            "date": self.date,
            "fetch_timestamp": datetime.now().isoformat(),
            "sources": {
                "bettingpros": len(bettingpros_props),
                "underdog": len(underdog_props),
            },
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
            print("\n🔍 DRY RUN - Results not saved\n")

        return result

    def _print_summary(self, result: Dict[str, Any]):
        """Print summary of fetch results"""
        print("\n" + "=" * 70)
        print("FETCH SUMMARY")
        print("=" * 70)

        print("\nProps by source:")
        for source, count in result["sources"].items():
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    add_logging_args(parser)  # Adds --debug and --quiet flags

    args = parser.parse_args()

    # Setup unified logging
    setup_logging("fetch_all", debug=args.debug, quiet=args.quiet)
    logger.info(f"Starting prop fetch for {args.date or 'today'}")

    # Run orchestrator
    orchestrator = FetchOrchestrator(date=args.date, verbose=not args.quiet, dry_run=args.dry_run)

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
