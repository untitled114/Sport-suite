#!/usr/bin/env python3
"""
Performance Monitor and Stop-Loss System
========================================
Tracks daily performance, detects stop-loss triggers, and auto-stops betting
when performance falls below acceptable thresholds.

Part of Phase 9: Production Deployment - Operational Safeguards
Created: November 7, 2025
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from betting_xl.config.production_policies import (
    ESCALATION_LEVELS,
    MAX_AVG_EDGE_PER_DAY,
    MAX_CONSECUTIVE_LOSING_DAYS,
    MAX_REASONABLE_EDGE,
    STOP_LOSS_WR_THRESHOLDS,
    VALIDATION_BENCHMARKS,
    WARNING_WR_THRESHOLDS,
    get_escalation_level,
)


class PerformanceMonitor:
    """Monitor performance and apply stop-loss triggers."""

    def __init__(self, picks_log_path: str = None, logger=None):
        """
        Initialize performance monitor.

        Args:
            picks_log_path: Path to picks log file (JSON)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Default picks log path
        if picks_log_path is None:
            picks_log_path = Path(__file__).parent / "logs" / "picks_log.json"

        self.picks_log_path = Path(picks_log_path)
        self.picks_data = self._load_picks_log()

    def _load_picks_log(self) -> Dict:
        """Load picks log from JSON file."""
        if not self.picks_log_path.exists():
            self.logger.debug(f"Picks log not found: {self.picks_log_path}")
            return {"picks": []}

        try:
            with open(self.picks_log_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load picks log: {e}")
            return {"picks": []}

    def _save_picks_log(self):
        """Save picks log to JSON file."""
        try:
            self.picks_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.picks_log_path, "w") as f:
                json.dump(self.picks_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save picks log: {e}")

    def calculate_rolling_performance(self, days: int = 7) -> Dict[str, Dict]:
        """
        Calculate rolling performance metrics.

        Args:
            days: Number of days to look back

        Returns:
            Dict with performance by market and overall
        """
        cutoff_date = date.today() - timedelta(days=days)

        # Filter picks within rolling window
        recent_picks = [
            p
            for p in self.picks_data.get("picks", [])
            if "game_date" in p and date.fromisoformat(p["game_date"]) >= cutoff_date
        ]

        # Calculate metrics by market
        metrics_by_market = {}
        overall_metrics = {
            "total_picks": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "win_rate": 0.0,
            "roi": 0.0,
        }

        # Group by market
        picks_by_market = defaultdict(list)
        for pick in recent_picks:
            market = pick.get("stat_type", "UNKNOWN")
            picks_by_market[market].append(pick)

        # Calculate metrics for each market
        for market, picks in picks_by_market.items():
            metrics = self._calculate_metrics(picks)
            metrics_by_market[market] = metrics

            # Aggregate to overall
            overall_metrics["total_picks"] += metrics["total_picks"]
            overall_metrics["wins"] += metrics["wins"]
            overall_metrics["losses"] += metrics["losses"]
            overall_metrics["pending"] += metrics["pending"]

        # Calculate overall WR and ROI
        if overall_metrics["total_picks"] > 0:
            completed = overall_metrics["wins"] + overall_metrics["losses"]
            if completed > 0:
                overall_metrics["win_rate"] = (overall_metrics["wins"] / completed) * 100
                # ROI calculation: (wins * 0.91 - losses) / completed
                overall_metrics["roi"] = (
                    (overall_metrics["wins"] * 0.91 - overall_metrics["losses"]) / completed
                ) * 100

        return {
            "by_market": metrics_by_market,
            "overall": overall_metrics,
            "days": days,
            "cutoff_date": str(cutoff_date),
        }

    def _calculate_metrics(self, picks: List[Dict]) -> Dict:
        """Calculate metrics for a list of picks."""
        metrics = {
            "total_picks": len(picks),
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_edge": 0.0,
        }

        if not picks:
            return metrics

        edges = []
        for pick in picks:
            result = pick.get("result")
            if result == "WIN":
                metrics["wins"] += 1
            elif result == "LOSS":
                metrics["losses"] += 1
            else:
                metrics["pending"] += 1

            # Track edge
            edge = pick.get("edge", 0)
            if edge:
                edges.append(edge)

        # Calculate WR and ROI
        completed = metrics["wins"] + metrics["losses"]
        if completed > 0:
            metrics["win_rate"] = (metrics["wins"] / completed) * 100
            # ROI: (wins * 0.91 - losses) / completed
            metrics["roi"] = ((metrics["wins"] * 0.91 - metrics["losses"]) / completed) * 100

        # Average edge
        if edges:
            metrics["avg_edge"] = sum(edges) / len(edges)

        return metrics

    def check_stop_loss_triggers(self, performance: Dict) -> Tuple[str, List[str]]:
        """
        Check if stop-loss triggers are hit.

        Args:
            performance: Performance metrics from calculate_rolling_performance()

        Returns:
            Tuple of (status, reasons)
            status: 'SAFE', 'WARNING', 'CAUTION', 'STOP'
            reasons: List of reasons for the status
        """
        reasons = []
        market_statuses = []

        # Check each market against stop-loss thresholds
        for market, metrics in performance["by_market"].items():
            if metrics["total_picks"] < 10:  # Need minimum sample size
                continue

            wr = metrics["win_rate"]
            stop_threshold = STOP_LOSS_WR_THRESHOLDS.get(market, 999.0)
            warn_threshold = WARNING_WR_THRESHOLDS.get(market, 999.0)

            # Check stop-loss
            if wr < stop_threshold:
                reasons.append(f"{market} WR {wr:.1f}% below stop-loss threshold {stop_threshold}%")
                market_statuses.append("STOP")
            elif wr < warn_threshold:
                reasons.append(f"{market} WR {wr:.1f}% below warning threshold {warn_threshold}%")
                market_statuses.append("WARNING")

            # Check vs validation benchmark
            benchmark_wr = VALIDATION_BENCHMARKS.get(market, {}).get("win_rate", 0)
            if benchmark_wr > 0:
                wr_drop = benchmark_wr - wr
                if wr_drop > 15:
                    reasons.append(
                        f"{market} WR dropped {wr_drop:.1f}pp vs validation ({benchmark_wr:.1f}%)"
                    )
                    market_statuses.append("STOP")
                elif wr_drop > 10:
                    reasons.append(
                        f"{market} WR dropped {wr_drop:.1f}pp vs validation ({benchmark_wr:.1f}%)"
                    )
                    market_statuses.append("CAUTION")
                elif wr_drop > 5:
                    reasons.append(
                        f"{market} WR dropped {wr_drop:.1f}pp vs validation ({benchmark_wr:.1f}%)"
                    )
                    market_statuses.append("WARNING")

        # Check consecutive losing days
        consecutive_losses = self._count_consecutive_losing_days()
        if consecutive_losses >= MAX_CONSECUTIVE_LOSING_DAYS:
            reasons.append(
                f"{consecutive_losses} consecutive losing days (max: {MAX_CONSECUTIVE_LOSING_DAYS})"
            )
            market_statuses.append("STOP")
        elif consecutive_losses >= 2:
            reasons.append(f"{consecutive_losses} consecutive losing days")
            market_statuses.append("CAUTION")

        # Check overall ROI
        overall_roi = performance["overall"]["roi"]
        if overall_roi < -5.0 and performance["overall"]["total_picks"] >= 20:
            reasons.append(f"Overall ROI {overall_roi:.2f}% (negative)")
            market_statuses.append("CAUTION")

        # Determine final status
        if "STOP" in market_statuses:
            status = "STOP"
        elif "CAUTION" in market_statuses:
            status = "CAUTION"
        elif "WARNING" in market_statuses:
            status = "WARNING"
        else:
            status = "SAFE"

        return status, reasons

    def _count_consecutive_losing_days(self) -> int:
        """Count consecutive days with negative ROI."""
        # Group picks by date
        picks_by_date = defaultdict(list)
        for pick in self.picks_data.get("picks", []):
            if "game_date" in pick:
                picks_by_date[pick["game_date"]].append(pick)

        # Sort dates in reverse (most recent first)
        sorted_dates = sorted(picks_by_date.keys(), reverse=True)

        consecutive = 0
        for game_date in sorted_dates:
            daily_picks = picks_by_date[game_date]

            # Calculate daily ROI
            wins = sum(1 for p in daily_picks if p.get("result") == "WIN")
            losses = sum(1 for p in daily_picks if p.get("result") == "LOSS")
            completed = wins + losses

            if completed == 0:  # Skip days with no completed picks
                continue

            daily_roi = ((wins * 0.91 - losses) / completed) * 100

            if daily_roi < 0:
                consecutive += 1
            else:
                break  # Stop counting at first positive day

        return consecutive

    def check_edge_anomalies(self, today_picks: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Check for edge anomalies in today's picks.

        Args:
            today_picks: List of picks for today

        Returns:
            Tuple of (has_anomaly, warnings)
        """
        warnings = []
        has_anomaly = False

        if not today_picks:
            return False, []

        # Check max edge
        edges = [p.get("edge", 0) for p in today_picks]
        max_edge = max(edges) if edges else 0
        avg_edge = sum(edges) / len(edges) if edges else 0

        if max_edge > MAX_REASONABLE_EDGE:
            warnings.append(
                f"Edge anomaly: Max edge {max_edge:.1f} exceeds threshold {MAX_REASONABLE_EDGE}"
            )
            has_anomaly = True

        if avg_edge > MAX_AVG_EDGE_PER_DAY:
            warnings.append(
                f"Edge anomaly: Avg edge {avg_edge:.1f} exceeds threshold {MAX_AVG_EDGE_PER_DAY}"
            )
            has_anomaly = True

        return has_anomaly, warnings

    def print_performance_report(self, performance: Dict, status: str, reasons: List[str]):
        """Print performance report."""
        overall = performance["overall"]

        # Skip full report if no picks tracked
        if overall["total_picks"] == 0:
            print("\n[INFO] Performance Monitor: No picks tracked yet (picks_log.json empty)")
            print("[OK] STATUS: SAFE TO CONTINUE (no stop-loss data)\n")
            return

        print("\n" + "=" * 80)
        print("PERFORMANCE MONITORING REPORT")
        print("=" * 80)

        # Overall performance
        print(f"\n[DATA] OVERALL PERFORMANCE (Last {performance['days']} days):")
        print(f"  - Total Picks: {overall['total_picks']}")
        print(
            f"  - Wins: {overall['wins']} | Losses: {overall['losses']} | Pending: {overall['pending']}"
        )
        print(f"  - Win Rate: {overall['win_rate']:.1f}%")
        print(f"  - ROI: {overall['roi']:.2f}%")

        # By market
        print("\n📈 PERFORMANCE BY MARKET:")
        for market, metrics in performance["by_market"].items():
            if metrics["total_picks"] == 0:
                continue

            benchmark_wr = VALIDATION_BENCHMARKS.get(market, {}).get("win_rate", 0)
            wr_diff = metrics["win_rate"] - benchmark_wr if benchmark_wr > 0 else 0

            status_icon = "[OK]" if wr_diff >= 0 else "[ERROR]"
            print(f"\n  {status_icon} {market}:")
            print(
                f"     - Picks: {metrics['total_picks']} | Wins: {metrics['wins']} | Losses: {metrics['losses']}"
            )
            print(
                f"     - Win Rate: {metrics['win_rate']:.1f}% (Benchmark: {benchmark_wr:.1f}%, Diff: {wr_diff:+.1f}pp)"
            )
            print(f"     - ROI: {metrics['roi']:.2f}%")
            print(f"     - Avg Edge: {metrics['avg_edge']:.2f}")

        # Status
        print("\n" + "=" * 80)
        if status == "SAFE":
            print("[OK] STATUS: SAFE TO CONTINUE")
        elif status == "WARNING":
            print("[WARN]  STATUS: WARNING - Continue with caution")
        elif status == "CAUTION":
            print("🛑 STATUS: CAUTION - Review required")
        elif status == "STOP":
            print("🚨 STATUS: STOP - DO NOT GENERATE PICKS")

        if reasons:
            print("\nREASONS:")
            for reason in reasons:
                print(f"  • {reason}")

        print("=" * 80 + "\n")

    def should_generate_picks(self) -> Tuple[bool, str, List[str]]:
        """
        Determine if it's safe to generate picks.

        Returns:
            Tuple of (should_generate, status, reasons)
        """
        # Calculate rolling performance
        performance = self.calculate_rolling_performance(days=7)

        # Check stop-loss triggers
        status, reasons = self.check_stop_loss_triggers(performance)

        # Print report
        self.print_performance_report(performance, status, reasons)

        # Decision
        should_generate = status in ["SAFE", "WARNING"]

        return should_generate, status, reasons


def main():
    """Run performance monitoring check."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    monitor = PerformanceMonitor()
    should_generate, status, reasons = monitor.should_generate_picks()

    if should_generate:
        print("[OK] CLEARED TO GENERATE PICKS\n")
        sys.exit(0)
    else:
        print("🚨 DO NOT GENERATE PICKS - Stop-loss triggered\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
