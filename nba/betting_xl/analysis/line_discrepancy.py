#!/usr/bin/env python3
"""
Line Discrepancy Analyzer
=========================
Cross-source line comparison: direct sportsbook feeds vs BettingPros aggregation.

Identifies four categories of discrepancy:
1. **Latency** — BP lagging behind direct feeds (stale lines)
2. **Rounding/adjustment** — BP rounding or adjusting lines differently
3. **Missing props** — Props present in one source but not the other
4. **Odds/juice** — Different over/under odds for the same line

Usage:
    # Full daily report
    python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --report

    # Filter to a single book
    python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --book draftkings

    # Show only stale lines (>30 min behind)
    python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --stale 30
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

from nba.config.database import get_intelligence_db_config

EST = ZoneInfo("America/New_York")

# Maps direct book_name (with _direct suffix) to BettingPros book_name.
# Direct fetchers write e.g. "draftkings_direct"; BP writes "draftkings".
DIRECT_TO_BP_BOOK_MAP: Dict[str, str] = {
    "draftkings_direct": "draftkings",
    "fanduel_direct": "fanduel",
    "betmgm_direct": "betmgm",
    "caesars_direct": "caesars",
    "betrivers_direct": "betrivers",
    "espnbet_direct": "espnbet",
    "underdog_direct": "underdog",
    "fanatics_direct": "fanatics",
    "hardrock_direct": "hardrock",
    "bet365_direct": "bet365",
}

# Reverse mapping: BP name -> direct name
BP_TO_DIRECT_BOOK_MAP: Dict[str, str] = {v: k for k, v in DIRECT_TO_BP_BOOK_MAP.items()}


def _strip_direct_suffix(book_name: str) -> str:
    """Extract base book name by stripping '_direct' suffix."""
    if book_name.endswith("_direct"):
        return book_name[: -len("_direct")]
    return book_name


class DiscrepancyAnalyzer:
    """Cross-source line comparison: direct sportsbook feeds vs BettingPros."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get or create a database connection to nba_intelligence."""
        if self._conn is None or self._conn.closed:
            config = get_intelligence_db_config()
            self._conn = psycopg2.connect(**config)
        return self._conn

    def close(self):
        """Close the database connection if open."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def compute_discrepancies(
        self, game_date: str, book_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare direct vs BP lines for a given date.

        Queries nba_props_xl for both fetch_source='direct' and 'bettingpros',
        joins on (player_name, stat_type, game_date), compares lines.

        Args:
            game_date: Date string YYYY-MM-DD.
            book_name: Optional base book name (e.g. "draftkings") to filter.

        Returns:
            DataFrame with columns:
            - player_name, stat_type, book_base
            - direct_line, bp_line, line_diff
            - direct_odds_over, bp_odds_over, odds_diff
            - direct_timestamp, bp_timestamp, latency_seconds
        """
        conn = self._get_connection()
        cur = conn.cursor()

        # Query direct props
        direct_query = """
            SELECT
                player_name,
                stat_type,
                book_name,
                over_line,
                over_odds,
                under_odds,
                fetch_timestamp
            FROM nba_props_xl
            WHERE game_date = %s
              AND fetch_source = 'direct'
        """
        params: list = [game_date]

        if book_name:
            direct_book = BP_TO_DIRECT_BOOK_MAP.get(book_name, f"{book_name}_direct")
            direct_query += " AND book_name = %s"
            params.append(direct_book)

        direct_query += " ORDER BY player_name, stat_type, fetch_timestamp DESC"
        cur.execute(direct_query, params)
        direct_rows = cur.fetchall()

        # Query BP props
        bp_query = """
            SELECT
                player_name,
                stat_type,
                book_name,
                over_line,
                over_odds,
                under_odds,
                fetch_timestamp
            FROM nba_props_xl
            WHERE game_date = %s
              AND (fetch_source = 'bettingpros' OR fetch_source IS NULL)
        """
        bp_params: list = [game_date]

        if book_name:
            bp_query += " AND book_name = %s"
            bp_params.append(book_name)

        bp_query += " ORDER BY player_name, stat_type, fetch_timestamp DESC"
        cur.execute(bp_query, bp_params)
        bp_rows = cur.fetchall()

        cur.close()

        # Build lookup dicts keyed by (player_name, stat_type, book_base).
        # Use the most recent row per key (already ordered DESC).
        direct_map: Dict[tuple, dict] = {}
        for row in direct_rows:
            player, stat, bname, line, odds_o, odds_u, ts = row
            base = _strip_direct_suffix(bname)
            key = (player, stat, base)
            if key not in direct_map:
                direct_map[key] = {
                    "direct_line": float(line) if line is not None else None,
                    "direct_odds_over": odds_o,
                    "direct_odds_under": odds_u,
                    "direct_timestamp": ts,
                }

        bp_map: Dict[tuple, dict] = {}
        for row in bp_rows:
            player, stat, bname, line, odds_o, odds_u, ts = row
            base = _strip_direct_suffix(bname)
            key = (player, stat, base)
            if key not in bp_map:
                bp_map[key] = {
                    "bp_line": float(line) if line is not None else None,
                    "bp_odds_over": odds_o,
                    "bp_odds_under": odds_u,
                    "bp_timestamp": ts,
                }

        # Merge on common keys
        all_keys = set(direct_map.keys()) | set(bp_map.keys())
        records: List[dict] = []

        for key in sorted(all_keys):
            player, stat, base = key
            d = direct_map.get(key, {})
            b = bp_map.get(key, {})

            direct_line = d.get("direct_line")
            bp_line = b.get("bp_line")
            line_diff = None
            if direct_line is not None and bp_line is not None:
                line_diff = round(direct_line - bp_line, 2)

            direct_odds = d.get("direct_odds_over")
            bp_odds = b.get("bp_odds_over")
            odds_diff = None
            if direct_odds is not None and bp_odds is not None:
                odds_diff = direct_odds - bp_odds

            direct_ts = d.get("direct_timestamp")
            bp_ts = b.get("bp_timestamp")
            latency_seconds = None
            if direct_ts is not None and bp_ts is not None:
                delta = direct_ts - bp_ts
                latency_seconds = int(delta.total_seconds())

            records.append(
                {
                    "player_name": player,
                    "stat_type": stat,
                    "book_base": base,
                    "direct_line": direct_line,
                    "bp_line": bp_line,
                    "line_diff": line_diff,
                    "direct_odds_over": direct_odds,
                    "bp_odds_over": bp_odds,
                    "odds_diff": odds_diff,
                    "direct_timestamp": direct_ts,
                    "bp_timestamp": bp_ts,
                    "latency_seconds": latency_seconds,
                }
            )

        df = pd.DataFrame(records)
        if self.verbose and not df.empty:
            matched = df["line_diff"].notna().sum()
            total = len(df)
            print(f"[DiscrepancyAnalyzer] {game_date}: {matched} matched pairs, {total} total rows")

        return df

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    def flag_stale_lines(self, game_date: str, staleness_minutes: int = 30) -> List[Dict[str, Any]]:
        """Find props where BP line is stale (>N minutes behind direct).

        Queries nba_line_snapshots to compare most recent direct vs BP snapshot
        for each (player_name, stat_type, book_base) triple.

        Args:
            game_date: Date string YYYY-MM-DD.
            staleness_minutes: Threshold in minutes to flag as stale.

        Returns:
            List of dicts with player, stat, book, staleness info.
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
            WITH latest_direct AS (
                SELECT DISTINCT ON (player_name, stat_type, book_name)
                    player_name,
                    stat_type,
                    book_name,
                    over_line AS direct_line,
                    snapshot_at AS direct_at
                FROM nba_line_snapshots
                WHERE game_date = %s
                  AND fetch_source = 'direct'
                ORDER BY player_name, stat_type, book_name, snapshot_at DESC
            ),
            latest_bp AS (
                SELECT DISTINCT ON (player_name, stat_type, book_name)
                    player_name,
                    stat_type,
                    book_name,
                    over_line AS bp_line,
                    snapshot_at AS bp_at
                FROM nba_line_snapshots
                WHERE game_date = %s
                  AND fetch_source = 'bettingpros'
                ORDER BY player_name, stat_type, book_name, snapshot_at DESC
            )
            SELECT
                d.player_name,
                d.stat_type,
                d.book_name AS direct_book,
                b.book_name AS bp_book,
                d.direct_line,
                b.bp_line,
                d.direct_at,
                b.bp_at,
                EXTRACT(EPOCH FROM (d.direct_at - b.bp_at)) AS lag_seconds
            FROM latest_direct d
            JOIN latest_bp b
              ON d.player_name = b.player_name
             AND d.stat_type = b.stat_type
             AND REPLACE(d.book_name, '_direct', '') = b.book_name
            WHERE d.direct_at - b.bp_at > INTERVAL '%s minutes'
            ORDER BY lag_seconds DESC
        """

        try:
            cur.execute(query, (game_date, game_date, staleness_minutes))
            rows = cur.fetchall()
        except psycopg2.errors.UndefinedTable:
            # nba_line_snapshots may not exist yet
            conn.rollback()
            if self.verbose:
                print("[DiscrepancyAnalyzer] nba_line_snapshots table not found (migration needed)")
            return []
        except psycopg2.Error as e:
            conn.rollback()
            if self.verbose:
                print(f"[DiscrepancyAnalyzer] Error querying snapshots: {e}")
            return []
        finally:
            cur.close()

        stale_lines: List[Dict[str, Any]] = []
        for row in rows:
            (
                player,
                stat,
                direct_book,
                bp_book,
                direct_line,
                bp_line,
                direct_at,
                bp_at,
                lag_sec,
            ) = row
            stale_lines.append(
                {
                    "player_name": player,
                    "stat_type": stat,
                    "book_base": _strip_direct_suffix(direct_book),
                    "direct_line": float(direct_line) if direct_line else None,
                    "bp_line": float(bp_line) if bp_line else None,
                    "line_diff": (
                        round(float(direct_line) - float(bp_line), 2)
                        if direct_line and bp_line
                        else None
                    ),
                    "direct_at": direct_at.isoformat() if direct_at else None,
                    "bp_at": bp_at.isoformat() if bp_at else None,
                    "lag_minutes": round(float(lag_sec) / 60, 1) if lag_sec else None,
                }
            )

        if self.verbose:
            print(
                f"[DiscrepancyAnalyzer] {len(stale_lines)} stale lines "
                f"(>{staleness_minutes} min behind direct)"
            )

        return stale_lines

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def generate_daily_report(self, game_date: str) -> Dict[str, Any]:
        """Generate comprehensive daily discrepancy report.

        Args:
            game_date: Date string YYYY-MM-DD.

        Returns:
            Dict with:
            - total_comparisons: int
            - avg_line_diff: float
            - max_line_diff: float
            - stale_lines_count: int
            - missing_from_bp: list (props in direct but not BP)
            - missing_from_direct: list (props in BP but not direct)
            - per_book_summary: dict[book] -> {avg_diff, max_diff, count}
        """
        df = self.compute_discrepancies(game_date)

        if df.empty:
            return {
                "game_date": game_date,
                "total_comparisons": 0,
                "avg_line_diff": 0.0,
                "max_line_diff": 0.0,
                "avg_abs_line_diff": 0.0,
                "stale_lines_count": 0,
                "missing_from_bp": [],
                "missing_from_direct": [],
                "per_book_summary": {},
            }

        # Matched pairs (both sources have a line)
        matched = df[df["line_diff"].notna()]

        # Missing from BP: direct has line but BP does not
        missing_bp_mask = df["direct_line"].notna() & df["bp_line"].isna()
        missing_from_bp = df[missing_bp_mask][["player_name", "stat_type", "book_base"]].to_dict(
            orient="records"
        )

        # Missing from direct: BP has line but direct does not
        missing_direct_mask = df["bp_line"].notna() & df["direct_line"].isna()
        missing_from_direct = df[missing_direct_mask][
            ["player_name", "stat_type", "book_base"]
        ].to_dict(orient="records")

        # Per-book summary
        per_book: Dict[str, Dict[str, Any]] = {}
        if not matched.empty:
            for book_base, group in matched.groupby("book_base"):
                abs_diffs = group["line_diff"].abs()
                per_book[str(book_base)] = {
                    "count": len(group),
                    "avg_diff": round(float(group["line_diff"].mean()), 3),
                    "avg_abs_diff": round(float(abs_diffs.mean()), 3),
                    "max_diff": round(float(abs_diffs.max()), 2),
                    "zero_diff_pct": round(
                        float((group["line_diff"] == 0).sum() / len(group) * 100), 1
                    ),
                }

        # Stale lines
        stale = self.flag_stale_lines(game_date)

        # Aggregate metrics
        avg_line_diff = round(float(matched["line_diff"].mean()), 3) if not matched.empty else 0.0
        avg_abs_diff = (
            round(float(matched["line_diff"].abs().mean()), 3) if not matched.empty else 0.0
        )
        max_line_diff = (
            round(float(matched["line_diff"].abs().max()), 2) if not matched.empty else 0.0
        )

        report = {
            "game_date": game_date,
            "total_comparisons": len(matched),
            "avg_line_diff": avg_line_diff,
            "avg_abs_line_diff": avg_abs_diff,
            "max_line_diff": max_line_diff,
            "stale_lines_count": len(stale),
            "missing_from_bp": missing_from_bp,
            "missing_from_direct": missing_from_direct,
            "per_book_summary": per_book,
        }

        if self.verbose:
            self._print_report(report)

        return report

    # ------------------------------------------------------------------
    # Direct-only consensus
    # ------------------------------------------------------------------

    def compute_cross_source_consensus(self, game_date: str) -> pd.DataFrame:
        """Build consensus from direct sources only (no BP dependency).

        Groups by (player_name, stat_type), calculates:
        - direct_consensus: mean of direct lines across books
        - direct_spread: max - min of direct lines
        - direct_num_books: count of distinct books
        - softest_direct_book: book with lowest OVER line
        - hardest_direct_book: book with highest OVER line

        Args:
            game_date: Date string YYYY-MM-DD.

        Returns:
            DataFrame with consensus metrics per (player_name, stat_type).
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
            SELECT
                player_name,
                stat_type,
                book_name,
                over_line
            FROM nba_props_xl
            WHERE game_date = %s
              AND fetch_source = 'direct'
              AND over_line IS NOT NULL
            ORDER BY player_name, stat_type
        """

        cur.execute(query, (game_date,))
        rows = cur.fetchall()
        cur.close()

        if not rows:
            if self.verbose:
                print(f"[DiscrepancyAnalyzer] No direct lines found for {game_date}")
            return pd.DataFrame()

        # Build per-prop book lines
        from collections import defaultdict

        prop_lines: Dict[tuple, List[dict]] = defaultdict(list)
        for player, stat, book, line in rows:
            base = _strip_direct_suffix(book)
            prop_lines[(player, stat)].append({"book": base, "line": float(line)})

        records: List[dict] = []
        for (player, stat), books in sorted(prop_lines.items()):
            lines = [b["line"] for b in books]
            min_line = min(lines)
            max_line = max(lines)
            softest = min(books, key=lambda b: b["line"])["book"]
            hardest = max(books, key=lambda b: b["line"])["book"]

            records.append(
                {
                    "player_name": player,
                    "stat_type": stat,
                    "direct_consensus": round(sum(lines) / len(lines), 2),
                    "direct_spread": round(max_line - min_line, 2),
                    "direct_num_books": len(books),
                    "softest_direct_book": softest,
                    "hardest_direct_book": hardest,
                    "direct_min_line": min_line,
                    "direct_max_line": max_line,
                }
            )

        df = pd.DataFrame(records)

        if self.verbose:
            props_with_spread = (df["direct_spread"] > 0).sum() if not df.empty else 0
            total = len(df)
            print(
                f"[DiscrepancyAnalyzer] Direct consensus: {total} props, "
                f"{props_with_spread} with cross-book spread"
            )

        return df

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_report(self, report: Dict[str, Any]) -> None:
        """Pretty-print the daily discrepancy report to stdout."""
        print(f"\n{'=' * 70}")
        print(f"LINE DISCREPANCY REPORT  |  {report['game_date']}")
        print(f"{'=' * 70}")

        print(f"\n  Total matched pairs:    {report['total_comparisons']}")
        print(f"  Avg line diff:          {report['avg_line_diff']:+.3f}")
        print(f"  Avg |line diff|:        {report['avg_abs_line_diff']:.3f}")
        print(f"  Max |line diff|:        {report['max_line_diff']:.2f}")
        print(f"  Stale lines (>30 min):  {report['stale_lines_count']}")
        print(f"  Missing from BP:        {len(report['missing_from_bp'])}")
        print(f"  Missing from Direct:    {len(report['missing_from_direct'])}")

        if report["per_book_summary"]:
            print(
                f"\n  {'Book':<16} {'Count':>6} {'AvgDiff':>9} {'Avg|Diff|':>10} "
                f"{'MaxDiff':>9} {'=0%':>7}"
            )
            print(f"  {'-' * 58}")
            for book, stats in sorted(report["per_book_summary"].items()):
                print(
                    f"  {book:<16} {stats['count']:>6} {stats['avg_diff']:>+9.3f} "
                    f"{stats['avg_abs_diff']:>10.3f} {stats['max_diff']:>9.2f} "
                    f"{stats['zero_diff_pct']:>6.1f}%"
                )

        if report["missing_from_bp"]:
            print(f"\n  Props in Direct but NOT in BP ({len(report['missing_from_bp'])}):")
            for item in report["missing_from_bp"][:10]:
                print(f"    - {item['player_name']} / {item['stat_type']} / {item['book_base']}")
            if len(report["missing_from_bp"]) > 10:
                print(f"    ... and {len(report['missing_from_bp']) - 10} more")

        if report["missing_from_direct"]:
            print(f"\n  Props in BP but NOT in Direct ({len(report['missing_from_direct'])}):")
            for item in report["missing_from_direct"][:10]:
                print(f"    - {item['player_name']} / {item['stat_type']} / {item['book_base']}")
            if len(report["missing_from_direct"]) > 10:
                print(f"    ... and {len(report['missing_from_direct']) - 10} more")

        print(f"\n{'=' * 70}\n")


# ======================================================================
# CLI entry point
# ======================================================================


def main():
    """CLI entry point for standalone discrepancy analysis."""
    parser = argparse.ArgumentParser(
        description="Compare direct sportsbook lines vs BettingPros lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --report\n"
            "  python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --book draftkings\n"
            "  python -m nba.betting_xl.analysis.line_discrepancy --date 2026-03-16 --stale 30\n"
        ),
    )

    today = datetime.now(EST).strftime("%Y-%m-%d")

    parser.add_argument(
        "--date",
        type=str,
        default=today,
        help="Game date to analyze (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Filter to a specific book (base name, e.g. 'draftkings')",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full daily discrepancy report",
    )
    parser.add_argument(
        "--stale",
        type=int,
        default=None,
        metavar="MINUTES",
        help="Show stale lines older than N minutes",
    )
    parser.add_argument(
        "--consensus",
        action="store_true",
        help="Show direct-only consensus (no BP dependency)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of formatted text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    analyzer = DiscrepancyAnalyzer(verbose=not args.quiet)

    try:
        if args.report:
            report = analyzer.generate_daily_report(args.date)
            if args.output_json:
                # Convert non-serializable fields
                report["missing_from_bp"] = report["missing_from_bp"]
                report["missing_from_direct"] = report["missing_from_direct"]
                print(json.dumps(report, indent=2, default=str))

        elif args.stale is not None:
            stale = analyzer.flag_stale_lines(args.date, staleness_minutes=args.stale)
            if args.output_json:
                print(json.dumps(stale, indent=2, default=str))
            elif stale:
                print(f"\nStale BP Lines (>{args.stale} min behind direct):")
                print(
                    f"{'Player':<25} {'Stat':<10} {'Book':<14} "
                    f"{'Direct':>7} {'BP':>7} {'Diff':>6} {'Lag':>8}"
                )
                print("-" * 80)
                for s in stale:
                    d_line = f"{s['direct_line']:.1f}" if s["direct_line"] else "N/A"
                    b_line = f"{s['bp_line']:.1f}" if s["bp_line"] else "N/A"
                    diff = f"{s['line_diff']:+.1f}" if s["line_diff"] else "N/A"
                    lag = f"{s['lag_minutes']:.0f} min" if s["lag_minutes"] else "N/A"
                    print(
                        f"{s['player_name']:<25} {s['stat_type']:<10} {s['book_base']:<14} "
                        f"{d_line:>7} {b_line:>7} {diff:>6} {lag:>8}"
                    )
            else:
                print(f"No stale lines found (threshold: {args.stale} min)")

        elif args.consensus:
            df = analyzer.compute_cross_source_consensus(args.date)
            if df.empty:
                print(f"No direct lines found for {args.date}")
            elif args.output_json:
                print(df.to_json(orient="records", indent=2))
            else:
                print(f"\nDirect-Only Consensus for {args.date}:")
                print(
                    f"{'Player':<25} {'Stat':<10} {'Consensus':>10} {'Spread':>8} "
                    f"{'Books':>6} {'Softest':<14} {'Hardest':<14}"
                )
                print("-" * 90)
                for _, row in df.iterrows():
                    print(
                        f"{row['player_name']:<25} {row['stat_type']:<10} "
                        f"{row['direct_consensus']:>10.1f} {row['direct_spread']:>8.1f} "
                        f"{row['direct_num_books']:>6} {row['softest_direct_book']:<14} "
                        f"{row['hardest_direct_book']:<14}"
                    )

        else:
            # Default: show discrepancy table
            df = analyzer.compute_discrepancies(args.date, book_name=args.book)
            if df.empty:
                print(f"No lines found for {args.date}")
                sys.exit(0)

            if args.output_json:
                print(df.to_json(orient="records", indent=2, default_handler=str))
            else:
                # Show only rows with actual differences
                has_diff = df[df["line_diff"].notna() & (df["line_diff"] != 0)]
                if has_diff.empty:
                    matched = df["line_diff"].notna().sum()
                    print(f"\nAll {matched} matched lines are identical (no discrepancies)")
                else:
                    print(f"\nLine Discrepancies for {args.date}:")
                    print(
                        f"{'Player':<25} {'Stat':<10} {'Book':<14} "
                        f"{'Direct':>7} {'BP':>7} {'Diff':>6} {'Latency':>10}"
                    )
                    print("-" * 82)
                    for _, row in has_diff.iterrows():
                        d_line = f"{row['direct_line']:.1f}" if pd.notna(row["direct_line"]) else ""
                        b_line = f"{row['bp_line']:.1f}" if pd.notna(row["bp_line"]) else ""
                        diff = f"{row['line_diff']:+.1f}" if pd.notna(row["line_diff"]) else ""
                        lat = ""
                        if pd.notna(row.get("latency_seconds")):
                            lat_min = row["latency_seconds"] / 60
                            lat = f"{lat_min:+.0f} min"
                        print(
                            f"{row['player_name']:<25} {row['stat_type']:<10} "
                            f"{row['book_base']:<14} {d_line:>7} {b_line:>7} "
                            f"{diff:>6} {lat:>10}"
                        )
                    print(
                        f"\n  {len(has_diff)} discrepancies out of "
                        f"{df['line_diff'].notna().sum()} matched pairs"
                    )

    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
