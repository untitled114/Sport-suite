#!/usr/bin/env python3
"""
NBA XL Picks Formatter
======================
Formats XL predictions for human-readable display.

Part of Phase 5: XL Betting Pipeline (Task 5.4)

Usage:
    python3 format_picks.py predictions/xl_picks_20251107.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def format_picks(picks_file: str):
    """
    Format XL picks for display.

    Output format:
    TONIGHT'S NBA XL PICKS - Nov 7, 2025

    ========================================
    POINTS (8 picks):
    ========================================

    1. Bam Adebayo OVER 18.5 points
       Book: DraftKings
       Edge: +3.3 | Confidence: HIGH
       Model Prediction: 21.8 points
       Why: High-spread goldmine (4.0 pts). Model predicts 21.8 vs softest line 18.5.

    2. ...
    """
    # Load picks
    with open(picks_file, "r") as f:
        data = json.load(f)

    # Header
    date_obj = datetime.fromisoformat(data["generated_at"])
    print("\n" + "=" * 80)
    print(f"TONIGHT'S NBA XL PICKS - {date_obj.strftime('%b %d, %Y')}")
    print("=" * 80)
    print(f"Strategy: {data['strategy']}")
    print(f"Total Picks: {data['total_picks']}")
    tier_counts = data["summary"].get("by_tier", {})
    print(f"High Confidence: {data['summary']['high_confidence']}")
    print(f"Avg Edge: {data['summary']['avg_edge']}")
    if tier_counts:
        tier_line = " | ".join(
            [f"Tier A: {tier_counts.get('tier_a', 0)}", f"Tier B: {tier_counts.get('tier_b', 0)}"]
        )
        print(tier_line)
    print("=" * 80 + "\n")

    # Group by market
    picks = data["picks"]
    by_market = {}
    for pick in picks:
        market = pick["stat_type"]
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(pick)

    # Display each market
    for market in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
        if market not in by_market:
            continue

        market_picks = by_market[market]

        print("=" * 80)
        print(f"{market} ({len(market_picks)} picks):")
        print("=" * 80 + "\n")

        # Sort by edge
        sorted_picks = sorted(market_picks, key=lambda x: x["edge"], reverse=True)

        for i, pick in enumerate(sorted_picks, 1):
            print(f"{i}. {pick['player_name']} {pick['side']} {pick['best_line']} {market.lower()}")
            print(f"   Book: {pick['best_book']}")
            tier = pick.get("filter_tier", "unknown").upper()
            print(f"   Edge: +{pick['edge']:.1f} | Confidence: {pick['confidence']} | Tier: {tier}")
            print(f"   Model Prediction: {pick['prediction']:.1f} {market.lower()}")

            # Show line shopping info
            if pick["line_spread"] > 0:
                print(
                    f"   Line Shopping: {pick['num_books']} books, spread {pick['line_spread']:.1f} pts (softest: {pick['best_line']}, consensus: {pick['consensus_line']:.1f})"
                )

            hit_rates = pick.get("hit_rates", {})
            l10 = hit_rates.get("last_10")
            if l10 and l10.get("total"):
                pct = f"{l10['rate'] * 100:.0f}%" if l10.get("rate") is not None else "n/a"
                print(f"   Hit Rate (L10): {l10['over']}/{l10['total']} ({pct})")
            season = hit_rates.get("season")
            if season and season.get("total"):
                pct = f"{season['rate'] * 100:.0f}%" if season.get("rate") is not None else "n/a"
                print(f"   Season Hit Rate: {season['over']}/{season['total']} ({pct})")

            print(f"   Why: {pick['reasoning']}")
            print()

    # Footer with expected performance
    print("=" * 80)
    print("EXPECTED PERFORMANCE (from validation):")
    print("=" * 80)
    for market, perf in data["expected_performance"].items():
        if market in by_market:
            print(f"{market}: {perf['win_rate']}% WR, {perf['roi']:+.2f}% ROI")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Format XL picks for display")
    parser.add_argument("picks_file", help="Path to xl_picks JSON file")

    args = parser.parse_args()

    if not Path(args.picks_file).exists():
        print(f"[ERROR] File not found: {args.picks_file}")
        sys.exit(1)

    format_picks(args.picks_file)


if __name__ == "__main__":
    main()
