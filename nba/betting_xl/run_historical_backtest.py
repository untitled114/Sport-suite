#!/usr/bin/env python3
"""
Historical Backtest Harness - Simulates daily pipeline without data leaks.

This script loops through historical dates and generates predictions using
temporal boundaries to prevent future data from leaking into past predictions.

Key Features:
- as_of_date parameter limits calibration data to before the prediction date
- predictions_dir points calibrator to backtest output (builds up real p_over data)
- Runs sequentially so each day's predictions feed into next day's calibration
- All feature queries already use game_date < as_of_date patterns
- SEED PERIOD: Generates predictions for ~35 days before backtest start to warm up calibrator

Usage:
    # Run full backtest with auto seed period (uses March-April 2025 data)
    python3 run_historical_backtest.py --start 2025-11-01 --end 2025-12-28

    # Custom seed period
    python3 run_historical_backtest.py --start 2025-11-01 --end 2025-12-28 \
        --seed-start 2025-03-10 --seed-end 2025-04-13

    # Skip seed period (not recommended)
    python3 run_historical_backtest.py --start 2025-11-01 --end 2025-12-28 --no-seed

    # Dry run (no files saved)
    python3 run_historical_backtest.py --start 2025-11-01 --end 2025-11-07 --dry-run
"""

import sys
import os
import argparse
import json
import psycopg2
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from betting_xl.generate_xl_predictions import XLPredictionsGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_INTELLIGENCE = {
    'host': os.getenv('NBA_INT_DB_HOST', 'localhost'),
    'port': int(os.getenv('NBA_INT_DB_PORT', 5539)),
    'user': os.getenv('NBA_INT_DB_USER', os.getenv('DB_USER', 'nba_user')),
    'password': os.getenv('NBA_INT_DB_PASSWORD', os.getenv('DB_PASSWORD')),
    'database': os.getenv('NBA_INT_DB_NAME', 'nba_intelligence')
}


class BacktestResult:
    """Container for single-day backtest results."""

    def __init__(self, game_date: str):
        self.game_date = game_date
        self.picks_generated = 0
        self.validated = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.by_market: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        self.by_model: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        self.error: Optional[str] = None

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    def __repr__(self):
        return f"<BacktestResult {self.game_date}: {self.wins}W/{self.losses}L ({self.win_rate:.1f}%)>"


class HistoricalBacktest:
    """
    Runs leak-free historical backtest across a date range.

    Process for each date:
    1. Generate predictions with as_of_date = game_date (limits calibration)
    2. Save predictions to backtest/ subfolder (not predictions/)
    3. Validate against actual values from nba_props_xl
    4. Aggregate results

    SEED PERIOD:
    - Before the main backtest, generate predictions for a seed period
    - This warms up the calibrator so it has 30+ samples when backtest starts
    - Default: Oct 22-31, 2025 (start of 2025-26 season, within 35-day lookback of Nov 1)
    """

    # Default seed period: start of 2025-26 season (Oct 22 = opening night)
    # This is within the 35-day lookback window when backtest starts Nov 1
    DEFAULT_SEED_START = date(2025, 10, 22)
    DEFAULT_SEED_END = date(2025, 10, 31)

    def __init__(
        self,
        start_date: date,
        end_date: date,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        skip_validation: bool = False,
        seed_start: Optional[date] = None,
        seed_end: Optional[date] = None,
        no_seed: bool = False,
        pick6_file: Optional[str] = None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.dry_run = dry_run
        self.skip_validation = skip_validation
        self.no_seed = no_seed
        self.pick6_file = pick6_file

        # Seed period for calibrator warmup
        if no_seed:
            self.seed_start = None
            self.seed_end = None
        else:
            self.seed_start = seed_start or self.DEFAULT_SEED_START
            self.seed_end = seed_end or self.DEFAULT_SEED_END

        # Output directory (separate from production predictions/)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / 'backtest'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results (only for main backtest, not seed period)
        self.results: List[BacktestResult] = []
        self.seed_results: List[BacktestResult] = []  # Track seed period separately
        self.conn = None

    def _connect_db(self):
        """Connect to intelligence database."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**DB_INTELLIGENCE)

    def _close_db(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None

    def check_props_exist(self, game_date: date) -> Tuple[int, int]:
        """
        Check if props exist for a date and how many have actuals.

        Returns:
            (total_props, props_with_actuals)
        """
        self._connect_db()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(actual_value) as with_actuals
                FROM nba_props_xl
                WHERE game_date = %s
                  AND stat_type IN ('POINTS', 'REBOUNDS')
            """, (game_date,))
            row = cur.fetchone()
            return (row[0], row[1]) if row else (0, 0)

    def validate_predictions(self, game_date: date, predictions_file: Path) -> BacktestResult:
        """
        Validate predictions against actual values.

        Args:
            game_date: The date to validate
            predictions_file: Path to the predictions JSON file

        Returns:
            BacktestResult with win/loss tallies
        """
        result = BacktestResult(game_date.strftime('%Y-%m-%d'))

        if not predictions_file.exists():
            result.error = "No predictions file"
            return result

        # Load predictions
        with open(predictions_file, 'r') as f:
            data = json.load(f)

        picks = data.get('picks', [])
        result.picks_generated = len(picks)

        if not picks:
            result.error = "No picks in file"
            return result

        # Fetch actuals from database
        self._connect_db()

        for pick in picks:
            player_name = pick['player_name']
            stat_type = pick['stat_type']
            line = pick['best_line']
            model_version = pick.get('model_version', 'xl')

            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT actual_value
                    FROM nba_props_xl
                    WHERE game_date = %s
                      AND player_name = %s
                      AND stat_type = %s
                      AND actual_value IS NOT NULL
                    ORDER BY fetch_timestamp DESC
                    LIMIT 1
                """, (game_date, player_name, stat_type))

                row = cur.fetchone()

            if not row or row[0] is None:
                continue  # No actual available

            actual = float(row[0])
            result.validated += 1

            # All picks are OVER
            if actual > line:
                result.wins += 1
                result.by_market[stat_type]['wins'] += 1
                result.by_model[model_version]['wins'] += 1
            elif actual < line:
                result.losses += 1
                result.by_market[stat_type]['losses'] += 1
                result.by_model[model_version]['losses'] += 1
            else:
                result.pushes += 1
                result.by_market[stat_type]['pushes'] += 1
                result.by_model[model_version]['pushes'] += 1

        return result

    def run_single_date(self, game_date: date) -> BacktestResult:
        """
        Run backtest for a single date.

        Args:
            game_date: Date to generate predictions for

        Returns:
            BacktestResult for this date
        """
        date_str = game_date.strftime('%Y-%m-%d')
        result = BacktestResult(date_str)

        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTESTING: {date_str}")
        logger.info(f"{'='*60}")

        # Check if props exist
        total_props, with_actuals = self.check_props_exist(game_date)
        logger.info(f"Props available: {total_props} total, {with_actuals} with actuals")

        if total_props == 0:
            result.error = "No props in database"
            logger.warning(f"  Skipping {date_str}: No props in database")
            return result

        # Output file - use xl_picks_ prefix so JSONCalibrator can find it
        output_file = self.output_dir / f"xl_picks_{date_str}.json"

        try:
            # Create generator with backtest parameters
            # as_of_date = game_date ensures calibration only uses data before this date
            # predictions_dir = output_dir so calibrator reads from backtest output (real p_over values)
            # This builds up calibration data naturally as backtest progresses day by day
            as_of_date = datetime.combine(game_date, datetime.min.time())

            generator = XLPredictionsGenerator(
                game_date=date_str,
                as_of_date=as_of_date,
                backtest_mode=True,
                predictions_dir=str(self.output_dir),  # Calibrator reads from backtest output
                pick6_file=self.pick6_file,
            )

            # Run the generator
            generator.run(
                output_file=str(output_file),
                dry_run=self.dry_run
            )

            # Load results to count picks
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                result.picks_generated = len(data.get('picks', []))
            else:
                result.picks_generated = 0

            logger.info(f"  Generated {result.picks_generated} picks")

            # Validate if requested
            if not self.skip_validation and with_actuals > 0:
                result = self.validate_predictions(game_date, output_file)
                logger.info(f"  Validation: {result.wins}W / {result.losses}L ({result.win_rate:.1f}%)")

        except Exception as e:
            result.error = str(e)
            logger.error(f"  Error: {e}")

        return result

    def run_seed_period(self):
        """
        Run seed period to warm up the calibrator.

        Generates predictions for historical dates (not counted in results).
        These predictions populate the output directory so the calibrator
        has 30+ samples when the main backtest starts.
        """
        if self.no_seed or not self.seed_start or not self.seed_end:
            logger.info("Seed period skipped (--no-seed)")
            return

        logger.info("\n" + "="*80)
        logger.info("SEED PERIOD - Warming up calibrator")
        logger.info("="*80)
        logger.info(f"Seed dates: {self.seed_start} to {self.seed_end}")
        logger.info("These predictions are NOT counted in backtest results")
        logger.info("="*80)

        seed_days = (self.seed_end - self.seed_start).days + 1
        current = self.seed_start
        processed = 0
        seed_picks_total = 0

        while current <= self.seed_end:
            date_str = current.strftime('%Y-%m-%d')

            # Check if props exist
            total_props, _ = self.check_props_exist(current)

            if total_props == 0:
                logger.debug(f"  {date_str}: No props, skipping")
                current += timedelta(days=1)
                continue

            output_file = self.output_dir / f"xl_picks_{date_str}.json"

            try:
                as_of_date = datetime.combine(current, datetime.min.time())

                generator = XLPredictionsGenerator(
                    game_date=date_str,
                    as_of_date=as_of_date,
                    backtest_mode=True,
                    predictions_dir=str(self.output_dir),
                    pick6_file=self.pick6_file,
                )

                generator.run(output_file=str(output_file), dry_run=self.dry_run)

                # Count picks
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    picks = len(data.get('picks', []))
                    seed_picks_total += picks
                    if picks > 0:
                        logger.info(f"  {date_str}: {picks} picks generated")

            except Exception as e:
                logger.debug(f"  {date_str}: Error - {e}")

            processed += 1
            current += timedelta(days=1)

        logger.info(f"\nSeed period complete: {seed_picks_total} total picks generated")
        logger.info("Calibrator should now have data for adjustments")
        logger.info("="*80 + "\n")

    def run(self) -> List[BacktestResult]:
        """
        Run backtest across the full date range.

        Returns:
            List of BacktestResult for each date
        """
        logger.info("\n" + "="*80)
        logger.info("HISTORICAL BACKTEST")
        logger.info("="*80)
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Seed period: {self.seed_start} to {self.seed_end}" if not self.no_seed else "Seed period: DISABLED")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Skip validation: {self.skip_validation}")
        logger.info("="*80)

        # STEP 1: Run seed period first to warm up calibrator
        self.run_seed_period()

        # STEP 2: Run main backtest
        logger.info("\n" + "="*80)
        logger.info("MAIN BACKTEST")
        logger.info("="*80)

        # Calculate total days
        total_days = (self.end_date - self.start_date).days + 1
        logger.info(f"Total days to process: {total_days}")

        current = self.start_date
        processed = 0

        while current <= self.end_date:
            result = self.run_single_date(current)
            self.results.append(result)

            processed += 1
            if processed % 7 == 0:
                logger.info(f"\n*** Progress: {processed}/{total_days} days ({processed/total_days*100:.0f}%) ***\n")

            current += timedelta(days=1)

        # Close database
        self._close_db()

        # Print summary
        self.print_summary()

        # Save summary
        self.save_summary()

        return self.results

    def print_summary(self):
        """Print aggregated backtest results."""
        print("\n" + "="*80)
        print(f"BACKTEST SUMMARY: {self.start_date} to {self.end_date}")
        print("="*80)

        # Aggregate results
        total_picks = sum(r.picks_generated for r in self.results)
        total_validated = sum(r.validated for r in self.results)
        total_wins = sum(r.wins for r in self.results)
        total_losses = sum(r.losses for r in self.results)
        total_pushes = sum(r.pushes for r in self.results)

        # By market
        market_results: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        for r in self.results:
            for market, stats in r.by_market.items():
                market_results[market]['wins'] += stats['wins']
                market_results[market]['losses'] += stats['losses']
                market_results[market]['pushes'] += stats['pushes']

        # Calculate win rate (excluding pushes)
        total_bets = total_wins + total_losses
        overall_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0.0

        print(f"\nDays processed: {len(self.results)}")
        print(f"Days with errors: {len([r for r in self.results if r.error])}")
        print(f"Total picks generated: {total_picks}")
        print(f"Total validated: {total_validated}")
        print(f"\n--- OVERALL ---")
        print(f"Wins:   {total_wins}")
        print(f"Losses: {total_losses}")
        print(f"Pushes: {total_pushes}")
        print(f"WIN RATE: {overall_wr:.1f}%")

        print(f"\n--- BY MARKET ---")
        for market in ['POINTS', 'REBOUNDS', 'THREES', 'ASSISTS', 'PA', 'PR', 'RA', 'PRA']:
            if market in market_results:
                stats = market_results[market]
                w, l = stats['wins'], stats['losses']
                wr = (w / (w + l) * 100) if (w + l) > 0 else 0.0
                print(f"{market:10}: {w}W / {l}L = {wr:.1f}%")

        # By model version
        model_results: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        for r in self.results:
            for model, stats in r.by_model.items():
                model_results[model]['wins'] += stats['wins']
                model_results[model]['losses'] += stats['losses']
                model_results[model]['pushes'] += stats['pushes']

        if model_results:
            print(f"\n--- BY MODEL ---")
            for model in sorted(model_results.keys()):
                stats = model_results[model]
                w, l = stats['wins'], stats['losses']
                wr = (w / (w + l) * 100) if (w + l) > 0 else 0.0
                print(f"{model:10}: {w}W / {l}L = {wr:.1f}%")

        # Weekly breakdown
        print(f"\n--- WEEKLY BREAKDOWN ---")
        week_results = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for r in self.results:
            game_date = datetime.strptime(r.game_date, '%Y-%m-%d').date()
            # Week number relative to start
            week_num = (game_date - self.start_date).days // 7 + 1
            week_results[week_num]['wins'] += r.wins
            week_results[week_num]['losses'] += r.losses

        for week_num in sorted(week_results.keys()):
            stats = week_results[week_num]
            w, l = stats['wins'], stats['losses']
            wr = (w / (w + l) * 100) if (w + l) > 0 else 0.0
            week_start = self.start_date + timedelta(days=(week_num - 1) * 7)
            week_end = min(week_start + timedelta(days=6), self.end_date)
            print(f"Week {week_num} ({week_start} - {week_end}): {w}W / {l}L = {wr:.1f}%")

        print("="*80)

    def save_summary(self):
        """Save summary to JSON file."""
        summary = {
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'total_days': len(self.results),
            'total_picks': sum(r.picks_generated for r in self.results),
            'total_validated': sum(r.validated for r in self.results),
            'total_wins': sum(r.wins for r in self.results),
            'total_losses': sum(r.losses for r in self.results),
            'total_pushes': sum(r.pushes for r in self.results),
            'daily_results': [
                {
                    'date': r.game_date,
                    'picks': r.picks_generated,
                    'validated': r.validated,
                    'wins': r.wins,
                    'losses': r.losses,
                    'pushes': r.pushes,
                    'win_rate': r.win_rate,
                    'error': r.error
                }
                for r in self.results
            ]
        }

        # Calculate overall win rate
        total_bets = summary['total_wins'] + summary['total_losses']
        summary['overall_win_rate'] = (summary['total_wins'] / total_bets * 100) if total_bets > 0 else 0.0

        summary_file = self.output_dir / f"backtest_summary_{self.start_date}_{self.end_date}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run historical backtest with temporal safety'
    )
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for backtest results'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate predictions without saving'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation against actuals'
    )
    parser.add_argument(
        '--seed-start',
        type=str,
        default=None,
        help='Seed period start date (default: 2025-03-10)'
    )
    parser.add_argument(
        '--seed-end',
        type=str,
        default=None,
        help='Seed period end date (default: 2025-04-13)'
    )
    parser.add_argument(
        '--no-seed',
        action='store_true',
        help='Skip seed period (not recommended - calibrator needs warmup data)'
    )
    parser.add_argument(
        '--pick6-file',
        type=str,
        default=None,
        help='Historical Pick6 JSON file for Odds API backtest (enables Odds API filter path)'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    if start_date > end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)

    # Parse seed dates if provided
    seed_start = datetime.strptime(args.seed_start, '%Y-%m-%d').date() if args.seed_start else None
    seed_end = datetime.strptime(args.seed_end, '%Y-%m-%d').date() if args.seed_end else None

    # Run backtest
    backtest = HistoricalBacktest(
        start_date=start_date,
        end_date=end_date,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        seed_start=seed_start,
        seed_end=seed_end,
        no_seed=args.no_seed,
        pick6_file=args.pick6_file,
    )

    backtest.run()


if __name__ == '__main__':
    main()
