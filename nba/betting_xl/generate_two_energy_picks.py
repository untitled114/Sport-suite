#!/usr/bin/env python3
"""
Two Energy Picks Generator (OPTIMIZED v2)
==========================================
Implements the two-energy betting strategy with volume-optimized filters:

POSITIVE ENERGY (OVER):
- PrizePicks Goblin lines (deflated = easy OVER)
- POINTS: line < 20 AND deflation >= 5.0 → 77.3% WR
- REBOUNDS: deflation >= 2.5 OR line >= 9.0 → 77.9% WR

NEGATIVE ENERGY (UNDER):
- FanDuel REBOUNDS: line > consensus + 0.8 → 76.7% WR
- DraftKings POINTS: line > consensus + 1.5 → 92.3% WR
- DraftKings REBOUNDS: DISABLED (66.7% WR = too much variance)

Usage:
    python3 -m nba.betting_xl.generate_two_energy_picks --date 2026-02-05
    python3 -m nba.betting_xl.generate_two_energy_picks --backtest --start 2026-01-01 --end 2026-02-03
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2

logger = logging.getLogger(__name__)


# Database connection - uses centralized config
def get_db_config():
    """Get database config from centralized config or environment."""
    try:
        from nba.config.database import get_intelligence_db_config

        return get_intelligence_db_config()
    except ImportError:
        # Fallback to environment variables (no hardcoded defaults)
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5539")),
            "database": os.getenv("DB_NAME", "nba_intelligence"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }


# Filter configurations (OPTIMIZED v2)
FILTERS = {
    # POSITIVE ENERGY - OVER on Goblin (deflated lines)
    "goblin_points_over": {
        "enabled": True,
        "book": "prizepicks_goblin",
        "stat_type": "POINTS",
        "side": "OVER",
        "max_line": 20.0,  # Exclude high scorers (20+ lines lose at 34-43%)
        "min_deflate": 5.0,  # 77.3% WR on 185 picks (Jan 1 - Feb 6)
        "description": "Goblin POINTS OVER (line<20, deflation>=5)",
        "expected_wr": 77.3,
    },
    "goblin_rebounds_over": {
        "enabled": True,
        "book": "prizepicks_goblin",
        "stat_type": "REBOUNDS",
        "side": "OVER",
        "min_deflate": 2.5,  # 77.9% WR on 249 picks (Jan 1 - Feb 6)
        "min_line_alt": 9.0,  # OR line >= 9 (91.7% hit rate)
        "description": "Goblin REBOUNDS OVER (deflation>=2.5 OR line>=9)",
        "expected_wr": 77.9,
    },
    # NEGATIVE ENERGY - UNDER on inflated lines
    "fd_rebounds_under": {
        "enabled": True,
        "book": "fanduel",
        "stat_type": "REBOUNDS",
        "side": "UNDER",
        "min_inflate": 0.8,  # line > consensus + 0.8
        "description": "FanDuel REBOUNDS UNDER (inflated +0.8)",
        "expected_wr": 76.7,
    },
    "dk_rebounds_under": {
        "enabled": False,  # DISABLED - 66.7% WR adds variance without value
        "book": "draftkings",
        "stat_type": "REBOUNDS",
        "side": "UNDER",
        "min_inflate": 0.8,
        "description": "DraftKings REBOUNDS UNDER (DISABLED)",
        "expected_wr": 66.7,
    },
    "dk_points_under": {
        "enabled": True,
        "book": "draftkings",
        "stat_type": "POINTS",
        "side": "UNDER",
        "min_inflate": 1.5,  # 92.3% WR on 13 picks (Jan 1 - Feb 6), was 66.2% at 1.0
        "description": "DraftKings POINTS UNDER (inflated +1.5)",
        "expected_wr": 92.3,
    },
}


class TwoEnergyGenerator:
    """Generates picks using the two-energy strategy."""

    def __init__(self, game_date: str = None):
        self.game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        self.conn = None

    def connect(self):
        """Connect to database."""
        self.conn = psycopg2.connect(**get_db_config())

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def get_goblin_picks(self, stat_type: str, config: Dict) -> List[Dict]:
        """Get OVER picks from Goblin lines with optimized filters."""
        cursor = self.conn.cursor()
        # Query supports both data formats:
        # - odds_type = 'goblin' (live PrizePicks data)
        # - book_name = 'prizepicks_goblin' (historical data)
        cursor.execute(
            """
            SELECT
                player_name,
                stat_type,
                over_line,
                book_name,
                consensus_line,
                actual_value,
                opponent_team
            FROM nba_props_xl
            WHERE game_date = %s
              AND (odds_type = 'goblin' OR book_name = 'prizepicks_goblin')
              AND stat_type = %s
              AND over_line IS NOT NULL
            ORDER BY player_name
            """,
            (self.game_date, stat_type),
        )

        picks = []
        max_line = config.get("max_line")
        min_deflate = config.get("min_deflate", 0)
        min_line_alt = config.get("min_line_alt")  # Alternative: line >= X bypasses deflation check

        for row in cursor.fetchall():
            player_name, stat, line, book, consensus, actual, opponent_team = row
            line_val = float(line)
            consensus_val = float(consensus) if consensus else None

            # Calculate deflation (how much below consensus)
            deflation = (consensus_val - line_val) if consensus_val else 0

            # Apply optimized filters
            passes_filter = False

            if stat_type == "POINTS":
                # POINTS: line < max_line AND deflation >= min_deflate
                if max_line and line_val >= max_line:
                    continue  # Skip high scorers
                if deflation >= min_deflate:
                    passes_filter = True
            elif stat_type == "REBOUNDS":
                # REBOUNDS: deflation >= min_deflate OR line >= min_line_alt
                if deflation >= min_deflate:
                    passes_filter = True
                elif min_line_alt and line_val >= min_line_alt:
                    passes_filter = True  # Elite rebounders (line >= 8)
            else:
                # Default: just check deflation
                if deflation >= min_deflate:
                    passes_filter = True

            if not passes_filter:
                continue

            opp = str(opponent_team) if opponent_team else None
            pick = {
                "player_name": player_name,
                "stat_type": stat,
                "line": line_val,
                "side": "OVER",
                "book": book,
                "consensus_line": consensus_val,
                "deflation": round(deflation, 2),
                "filter": f"goblin_{stat.lower()}_over",
                "energy": "POSITIVE",
                "actual_value": float(actual) if actual else None,
                "opponent_team": opp,
                "game_key": opp,
            }
            if actual is not None:
                pick["result"] = "WIN" if actual > line_val else "LOSS"
            picks.append(pick)

        cursor.close()
        return picks

    def get_under_picks(self, book: str, stat_type: str, min_inflate: float) -> List[Dict]:
        """Get UNDER picks from inflated lines."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                player_name,
                stat_type,
                over_line,
                book_name,
                consensus_line,
                actual_value,
                opponent_team
            FROM nba_props_xl
            WHERE game_date = %s
              AND book_name = %s
              AND stat_type = %s
              AND over_line IS NOT NULL
              AND consensus_line IS NOT NULL
              AND over_line > consensus_line + %s
            ORDER BY (over_line - consensus_line) DESC
            """,
            (self.game_date, book, stat_type, min_inflate),
        )

        picks = []
        for row in cursor.fetchall():
            player_name, stat, line, book_name, consensus, actual, opponent_team = row
            inflate = float(line) - float(consensus)
            opp = str(opponent_team) if opponent_team else None
            pick = {
                "player_name": player_name,
                "stat_type": stat,
                "line": float(line),
                "side": "UNDER",
                "book": book_name,
                "consensus_line": float(consensus),
                "line_inflate": round(inflate, 2),
                "filter": f"{book}_{stat.lower()}_under",
                "energy": "NEGATIVE",
                "actual_value": float(actual) if actual else None,
                "opponent_team": opp,
                "game_key": opp,
            }
            if actual is not None:
                pick["result"] = "WIN" if actual < line else "LOSS"
            picks.append(pick)

        cursor.close()
        return picks

    def generate_picks(self) -> Dict:
        """Generate all picks for the game date."""
        all_picks = []

        # POSITIVE ENERGY - Goblin OVER
        for filter_name, config in FILTERS.items():
            if not config["enabled"]:
                continue

            if config["side"] == "OVER" and "goblin" in config["book"]:
                picks = self.get_goblin_picks(config["stat_type"], config)
                for p in picks:
                    p["expected_wr"] = config["expected_wr"]
                    p["filter_name"] = filter_name
                    p["filter_tier"] = "two_energy"
                all_picks.extend(picks)

            elif config["side"] == "UNDER":
                picks = self.get_under_picks(
                    config["book"], config["stat_type"], config.get("min_inflate", 0.5)
                )
                for p in picks:
                    p["expected_wr"] = config["expected_wr"]
                    p["filter_name"] = filter_name
                    p["filter_tier"] = "two_energy"
                all_picks.extend(picks)

        # Dedupe by (player, stat, side) - keep best line
        deduped = {}
        for pick in all_picks:
            key = (pick["player_name"].lower(), pick["stat_type"], pick["side"])
            if key not in deduped:
                deduped[key] = pick
            else:
                # For OVER, keep lower line; for UNDER, keep higher line
                existing = deduped[key]
                if pick["side"] == "OVER" and pick["line"] < existing["line"]:
                    deduped[key] = pick
                elif pick["side"] == "UNDER" and pick["line"] > existing["line"]:
                    deduped[key] = pick

        picks = list(deduped.values())

        # Separate by energy
        positive = [p for p in picks if p["energy"] == "POSITIVE"]
        negative = [p for p in picks if p["energy"] == "NEGATIVE"]

        # Format for validate_predictions.py compatibility
        return {
            "generated_at": datetime.now().isoformat(),
            "game_date": self.game_date,
            "date": self.game_date,  # Alias for compatibility
            "strategy": "Two Energy (Goblin OVER + Inflated UNDER)",
            "tier": "two_energy",
            "total_picks": len(picks),
            "picks": picks,
            "positive_energy_picks": positive,
            "negative_energy_picks": negative,
            "summary": {
                "total": len(picks),
                "total_picks": len(picks),  # Alias
                "positive_energy": len(positive),
                "negative_energy": len(negative),
                "by_stat_type": {},
                "by_filter": {},
            },
        }

    def generate_and_save(self, output_dir: str = "predictions") -> Optional[Path]:
        """Generate picks and save to JSON file."""
        picks = self.generate_picks()

        if picks["total_picks"] == 0:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"two_energy_picks_{self.game_date}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(picks, f, indent=2)

        return filepath

    def generate_historical(self, start_date: str, end_date: str, output_dir: str = "predictions"):
        """Generate picks for a date range and save to files."""
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        generated = 0
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            self.game_date = date_str

            filepath = self.generate_and_save(output_dir)
            if filepath:
                picks = self.generate_picks()
                print(f"{date_str}: {picks['total_picks']} picks saved")
                generated += 1
            else:
                print(f"{date_str}: No picks")

            current += timedelta(days=1)

        print(f"\nGenerated {generated} prediction files in {output_dir}/")

    def backtest(self, start_date: str, end_date: str) -> Dict:
        """Backtest the strategy over a date range."""
        results = {
            "period": f"{start_date} to {end_date}",
            "filters": {},
            "daily_results": [],
            "overall": {},
        }

        # Initialize filter tracking
        for filter_name in FILTERS:
            results["filters"][filter_name] = {
                "total": 0,
                "wins": 0,
                "losses": 0,
            }

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        total_wins = 0
        total_bets = 0

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            self.game_date = date_str

            day_picks = self.generate_picks()
            all_day_picks = day_picks["positive_energy_picks"] + day_picks["negative_energy_picks"]

            day_wins = 0
            day_bets = 0

            for pick in all_day_picks:
                if pick.get("result"):
                    filter_name = pick["filter"]
                    if filter_name in results["filters"]:
                        results["filters"][filter_name]["total"] += 1
                        if pick["result"] == "WIN":
                            results["filters"][filter_name]["wins"] += 1
                            day_wins += 1
                        else:
                            results["filters"][filter_name]["losses"] += 1
                        day_bets += 1

            if day_bets > 0:
                results["daily_results"].append(
                    {
                        "date": date_str,
                        "bets": day_bets,
                        "wins": day_wins,
                        "wr": round(day_wins / day_bets * 100, 1),
                    }
                )
                total_wins += day_wins
                total_bets += day_bets

            current += timedelta(days=1)

        # Calculate overall stats
        results["overall"] = {
            "total_bets": total_bets,
            "wins": total_wins,
            "losses": total_bets - total_wins,
            "win_rate": round(total_wins / total_bets * 100, 1) if total_bets > 0 else 0,
            "roi": (
                round((total_wins * 0.91 - (total_bets - total_wins)) / total_bets * 100, 1)
                if total_bets > 0
                else 0
            ),
        }

        # Calculate per-filter stats
        for _filter_name, stats in results["filters"].items():
            if stats["total"] > 0:
                stats["win_rate"] = round(stats["wins"] / stats["total"] * 100, 1)
            else:
                stats["win_rate"] = 0

        return results


def print_backtest_results(results: Dict):
    """Pretty print backtest results."""
    print("\n" + "=" * 70)
    print("TWO ENERGY STRATEGY BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period: {results['period']}")
    print()

    # Overall
    overall = results["overall"]
    print(f"OVERALL: {overall['wins']}W - {overall['losses']}L ({overall['win_rate']}% WR)")
    print(f"ROI: {overall['roi']:+.1f}%")
    print()

    # By filter
    print("BY FILTER:")
    print("-" * 70)
    print(f"{'Filter':<30} {'Total':>8} {'Wins':>8} {'WR':>8}")
    print("-" * 70)

    for filter_name, stats in sorted(results["filters"].items()):
        if stats["total"] > 0:
            print(
                f"{filter_name:<30} {stats['total']:>8} {stats['wins']:>8} {stats['win_rate']:>7.1f}%"
            )

    print("-" * 70)

    # Daily breakdown
    print("\nDAILY RESULTS:")
    print("-" * 40)
    profitable_days = 0
    for day in results["daily_results"]:
        status = "✓" if day["wr"] >= 52.4 else "✗"
        print(f"{day['date']}: {day['wins']}/{day['bets']} ({day['wr']:.1f}%) {status}")
        if day["wr"] >= 52.4:
            profitable_days += 1

    print("-" * 40)
    print(f"Profitable days: {profitable_days}/{len(results['daily_results'])}")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two Energy Picks Generator")
    parser.add_argument("--date", type=str, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--start", type=str, help="Backtest start date")
    parser.add_argument("--end", type=str, help="Backtest end date")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    generator = TwoEnergyGenerator(game_date=args.date)
    generator.connect()

    try:
        if args.backtest:
            if not args.start or not args.end:
                print("Error: --start and --end required for backtest")
                return

            results = generator.backtest(args.start, args.end)
            print_backtest_results(results)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nSaved to {args.output}")
        else:
            picks = generator.generate_picks()

            print("\n" + "=" * 70)
            print(f"TWO ENERGY PICKS - {picks['game_date']}")
            print("=" * 70)
            print(f"Total: {picks['summary']['total_picks']} picks")
            print(f"  Positive Energy (OVER): {picks['summary']['positive_energy']}")
            print(f"  Negative Energy (UNDER): {picks['summary']['negative_energy']}")
            print()

            if picks["positive_energy_picks"]:
                print("POSITIVE ENERGY (OVER):")
                print("-" * 50)
                for p in picks["positive_energy_picks"]:
                    print(
                        f"  {p['player_name']:<20} {p['stat_type']:<10} O{p['line']:<5} ({p['book']})"
                    )

            print()

            if picks["negative_energy_picks"]:
                print("NEGATIVE ENERGY (UNDER):")
                print("-" * 50)
                for p in picks["negative_energy_picks"]:
                    inflate = p.get("line_inflate", 0)
                    print(
                        f"  {p['player_name']:<20} {p['stat_type']:<10} U{p['line']:<5} ({p['book']}, +{inflate})"
                    )

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(picks, f, indent=2)
                print(f"\nSaved to {args.output}")

    finally:
        generator.close()


if __name__ == "__main__":
    main()
