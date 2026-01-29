#!/usr/bin/env python3
"""
NBA Performance Calculator
===========================
Calculates performance metrics for XL betting system.

Generates detailed performance reports including:
- Overall win rate and ROI
- Performance by market (POINTS, REBOUNDS, etc.)
- Performance by confidence level
- Rolling 7-day and 30-day metrics
- Edge calibration analysis
- Comparison to validation benchmarks

Usage:
    python3 calculate_performance.py
    python3 calculate_performance.py --days 30
    python3 calculate_performance.py --output report_2025-11-07.md
"""
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_INTELLIGENCE = {
    'host': 'localhost',
    'port': 5539,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_intelligence'
}

# Validation benchmarks (from Oct 23 - Nov 4, 2024)
VALIDATION_BENCHMARKS = {
    'overall_line_shopping': {'win_rate': 54.5, 'roi': 4.16},
    'POINTS': {'win_rate': 56.7, 'roi': 8.27},
    'REBOUNDS': {'win_rate': 61.2, 'roi': 16.96},
    'high_spread_goldmine': {'win_rate': 70.6, 'roi': 34.82}
}


class PerformanceCalculator:
    """
    Calculates comprehensive performance metrics from pick_results table.
    """

    def __init__(self, days=30):
        self.days = days
        self.conn = None
        self.picks_df = None

    def connect(self):
        """Connect to intelligence database"""
        self.conn = psycopg2.connect(**DB_INTELLIGENCE)
        logger.info("[OK] Connected to database")

    def load_picks(self):
        """Load picks with results from last N days"""
        query = """
            SELECT
                pick_date,
                game_date,
                player_name,
                stat_type,
                side,
                best_book,
                best_line,
                consensus_line,
                line_spread,
                prediction,
                edge,
                edge_pct,
                confidence,
                actual_result,
                result,
                profit,
                strategy
            FROM pick_results
            WHERE pick_date >= CURRENT_DATE - INTERVAL '%s days'
              AND result IS NOT NULL
              AND result != 'PENDING'
            ORDER BY pick_date DESC;
        """

        self.picks_df = pd.read_sql_query(
            query,
            self.conn,
            params=(self.days,)
        )

        logger.info(f"[DATA] Loaded {len(self.picks_df)} picks from last {self.days} days")

    def calculate_overall_metrics(self):
        """Calculate overall win rate and ROI"""
        if len(self.picks_df) == 0:
            return None

        total_picks = len(self.picks_df)
        wins = len(self.picks_df[self.picks_df['result'] == 'WIN'])
        losses = len(self.picks_df[self.picks_df['result'] == 'LOSS'])
        pushes = len(self.picks_df[self.picks_df['result'] == 'PUSH'])

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        total_profit = self.picks_df['profit'].sum()
        roi = total_profit / (wins + losses) * 100 if (wins + losses) > 0 else 0

        return {
            'total_picks': total_picks,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi
        }

    def calculate_by_market(self):
        """Calculate performance by market (POINTS, REBOUNDS, etc.)"""
        results = {}

        for market in self.picks_df['stat_type'].unique():
            market_df = self.picks_df[self.picks_df['stat_type'] == market]
            wins = len(market_df[market_df['result'] == 'WIN'])
            losses = len(market_df[market_df['result'] == 'LOSS'])

            if wins + losses > 0:
                win_rate = wins / (wins + losses) * 100
                total_profit = market_df['profit'].sum()
                roi = total_profit / (wins + losses) * 100

                results[market] = {
                    'picks': len(market_df),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'roi': roi,
                    'avg_edge': market_df['edge'].mean(),
                    'avg_line_spread': market_df['line_spread'].mean()
                }

        return results

    def calculate_by_confidence(self):
        """Calculate performance by confidence level"""
        results = {}

        for confidence in self.picks_df['confidence'].unique():
            conf_df = self.picks_df[self.picks_df['confidence'] == confidence]
            wins = len(conf_df[conf_df['result'] == 'WIN'])
            losses = len(conf_df[conf_df['result'] == 'LOSS'])

            if wins + losses > 0:
                win_rate = wins / (wins + losses) * 100
                total_profit = conf_df['profit'].sum()
                roi = total_profit / (wins + losses) * 100

                results[confidence] = {
                    'picks': len(conf_df),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'roi': roi,
                    'avg_edge': conf_df['edge'].mean(),
                    'avg_line_spread': conf_df['line_spread'].mean()
                }

        return results

    def calculate_rolling_metrics(self):
        """Calculate 7-day and 30-day rolling metrics"""
        if len(self.picks_df) == 0:
            return None

        # 7-day rolling
        recent_7d = self.picks_df[
            self.picks_df['pick_date'] >= (datetime.now() - timedelta(days=7)).date()
        ]

        if len(recent_7d) > 0:
            wins_7d = len(recent_7d[recent_7d['result'] == 'WIN'])
            losses_7d = len(recent_7d[recent_7d['result'] == 'LOSS'])
            win_rate_7d = wins_7d / (wins_7d + losses_7d) * 100 if (wins_7d + losses_7d) > 0 else 0
            roi_7d = recent_7d['profit'].sum() / (wins_7d + losses_7d) * 100 if (wins_7d + losses_7d) > 0 else 0
        else:
            wins_7d = losses_7d = win_rate_7d = roi_7d = 0

        # 30-day rolling
        recent_30d = self.picks_df[
            self.picks_df['pick_date'] >= (datetime.now() - timedelta(days=30)).date()
        ]

        if len(recent_30d) > 0:
            wins_30d = len(recent_30d[recent_30d['result'] == 'WIN'])
            losses_30d = len(recent_30d[recent_30d['result'] == 'LOSS'])
            win_rate_30d = wins_30d / (wins_30d + losses_30d) * 100 if (wins_30d + losses_30d) > 0 else 0
            roi_30d = recent_30d['profit'].sum() / (wins_30d + losses_30d) * 100 if (wins_30d + losses_30d) > 0 else 0
        else:
            wins_30d = losses_30d = win_rate_30d = roi_30d = 0

        return {
            '7_day': {
                'picks': len(recent_7d),
                'wins': wins_7d,
                'losses': losses_7d,
                'win_rate': win_rate_7d,
                'roi': roi_7d
            },
            '30_day': {
                'picks': len(recent_30d),
                'wins': wins_30d,
                'losses': losses_30d,
                'win_rate': win_rate_30d,
                'roi': roi_30d
            }
        }

    def calculate_high_spread_performance(self):
        """Calculate performance on high-spread bets (≥2.5 points)"""
        high_spread = self.picks_df[self.picks_df['line_spread'] >= 2.5]

        if len(high_spread) == 0:
            return None

        wins = len(high_spread[high_spread['result'] == 'WIN'])
        losses = len(high_spread[high_spread['result'] == 'LOSS'])

        if wins + losses > 0:
            win_rate = wins / (wins + losses) * 100
            total_profit = high_spread['profit'].sum()
            roi = total_profit / (wins + losses) * 100

            return {
                'picks': len(high_spread),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'roi': roi,
                'avg_spread': high_spread['line_spread'].mean()
            }

        return None

    def generate_report(self, output_file=None):
        """Generate comprehensive performance report"""
        logger.info("\n" + "="*80)
        logger.info("NBA XL SYSTEM PERFORMANCE REPORT")
        logger.info("="*80)

        # Overall metrics
        overall = self.calculate_overall_metrics()

        if overall is None or overall['total_picks'] == 0:
            logger.warning("[ERROR] No picks with results found")
            return

        report_lines = []
        report_lines.append("# NBA XL System Performance Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Period: Last {self.days} days\n")

        # Overall performance
        report_lines.append("## Overall Performance\n")
        report_lines.append(f"- **Total Picks**: {overall['total_picks']}")
        report_lines.append(f"- **Wins**: {overall['wins']}")
        report_lines.append(f"- **Losses**: {overall['losses']}")
        report_lines.append(f"- **Pushes**: {overall['pushes']}")
        report_lines.append(f"- **Win Rate**: {overall['win_rate']:.1f}%")
        report_lines.append(f"- **Total Profit**: {overall['total_profit']:+.2f}u")
        report_lines.append(f"- **ROI**: {overall['roi']:+.2f}%\n")

        # Comparison to validation
        val_overall = VALIDATION_BENCHMARKS['overall_line_shopping']
        wr_diff = overall['win_rate'] - val_overall['win_rate']
        roi_diff = overall['roi'] - val_overall['roi']

        report_lines.append("### Comparison to Validation (Oct 23 - Nov 4, 2024)\n")
        report_lines.append(f"- **Validation Win Rate**: {val_overall['win_rate']:.1f}%")
        report_lines.append(f"- **Live Win Rate**: {overall['win_rate']:.1f}% ({wr_diff:+.1f}%)")
        report_lines.append(f"- **Validation ROI**: {val_overall['roi']:+.2f}%")
        report_lines.append(f"- **Live ROI**: {overall['roi']:+.2f}% ({roi_diff:+.2f}%)\n")

        # Performance by market
        by_market = self.calculate_by_market()
        report_lines.append("## Performance by Market\n")

        for market, stats in sorted(by_market.items()):
            report_lines.append(f"### {market}\n")
            report_lines.append(f"- **Picks**: {stats['picks']}")
            report_lines.append(f"- **Win Rate**: {stats['win_rate']:.1f}%")
            report_lines.append(f"- **ROI**: {stats['roi']:+.2f}%")
            report_lines.append(f"- **Avg Edge**: {stats['avg_edge']:.2f}")
            report_lines.append(f"- **Avg Line Spread**: {stats['avg_line_spread']:.2f}\n")

            # Compare to validation
            if market in VALIDATION_BENCHMARKS:
                val = VALIDATION_BENCHMARKS[market]
                wr_diff = stats['win_rate'] - val['win_rate']
                roi_diff = stats['roi'] - val['roi']
                report_lines.append(f"**Validation Comparison**:")
                report_lines.append(f"- Win Rate: {val['win_rate']:.1f}% → {stats['win_rate']:.1f}% ({wr_diff:+.1f}%)")
                report_lines.append(f"- ROI: {val['roi']:+.2f}% → {stats['roi']:+.2f}% ({roi_diff:+.2f}%)\n")

        # Performance by confidence
        by_confidence = self.calculate_by_confidence()
        report_lines.append("## Performance by Confidence\n")

        for confidence, stats in sorted(by_confidence.items()):
            report_lines.append(f"### {confidence}\n")
            report_lines.append(f"- **Picks**: {stats['picks']}")
            report_lines.append(f"- **Win Rate**: {stats['win_rate']:.1f}%")
            report_lines.append(f"- **ROI**: {stats['roi']:+.2f}%")
            report_lines.append(f"- **Avg Edge**: {stats['avg_edge']:.2f}")
            report_lines.append(f"- **Avg Line Spread**: {stats['avg_line_spread']:.2f}\n")

        # High-spread performance
        high_spread = self.calculate_high_spread_performance()
        if high_spread:
            report_lines.append("## High-Spread Goldmine (≥2.5 points)\n")
            report_lines.append(f"- **Picks**: {high_spread['picks']}")
            report_lines.append(f"- **Win Rate**: {high_spread['win_rate']:.1f}%")
            report_lines.append(f"- **ROI**: {high_spread['roi']:+.2f}%")
            report_lines.append(f"- **Avg Spread**: {high_spread['avg_spread']:.2f}\n")

            # Compare to validation
            val = VALIDATION_BENCHMARKS['high_spread_goldmine']
            wr_diff = high_spread['win_rate'] - val['win_rate']
            roi_diff = high_spread['roi'] - val['roi']
            report_lines.append(f"**Validation Comparison**:")
            report_lines.append(f"- Win Rate: {val['win_rate']:.1f}% → {high_spread['win_rate']:.1f}% ({wr_diff:+.1f}%)")
            report_lines.append(f"- ROI: {val['roi']:+.2f}% → {high_spread['roi']:+.2f}% ({roi_diff:+.2f}%)\n")

        # Rolling metrics
        rolling = self.calculate_rolling_metrics()
        if rolling:
            report_lines.append("## Rolling Performance\n")
            report_lines.append("### Last 7 Days\n")
            report_lines.append(f"- **Picks**: {rolling['7_day']['picks']}")
            report_lines.append(f"- **Win Rate**: {rolling['7_day']['win_rate']:.1f}%")
            report_lines.append(f"- **ROI**: {rolling['7_day']['roi']:+.2f}%\n")

            report_lines.append("### Last 30 Days\n")
            report_lines.append(f"- **Picks**: {rolling['30_day']['picks']}")
            report_lines.append(f"- **Win Rate**: {rolling['30_day']['win_rate']:.1f}%")
            report_lines.append(f"- **ROI**: {rolling['30_day']['roi']:+.2f}%\n")

        # Print to console
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(report_text)

            logger.info(f"\n💾 Report saved to: {output_path}")

    def run(self, output_file=None):
        """Main execution"""
        try:
            self.connect()
            self.load_picks()
            self.generate_report(output_file=output_file)
        finally:
            if self.conn:
                self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate NBA XL system performance metrics'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    parser.add_argument(
        '--output',
        help='Output file path for report (markdown)',
        default=None
    )

    args = parser.parse_args()

    # Default output file
    if not args.output:
        output_dir = Path(__file__).parent / 'reports'
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"performance_{datetime.now().strftime('%Y-%m-%d')}.md"

    calculator = PerformanceCalculator(days=args.days)
    calculator.run(output_file=args.output)


if __name__ == '__main__':
    main()
