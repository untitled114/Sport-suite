#!/usr/bin/env python3
"""
Validate Predictions Against Actual Results
=============================================
Compare predicted picks against actual game results.

Usage:
    # Single date
    python3 validate_predictions.py --date 2025-11-10

    # Date range
    python3 validate_predictions.py --start-date 2025-11-01 --end-date 2025-11-30

    # Backtest directory
    python3 validate_predictions.py --start-date 2025-11-01 --end-date 2025-11-30 --backtest-dir backtest/
"""

import json
import argparse
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import os

DB_CONFIG = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

def load_predictions(date: str, predictions_dir: str = "predictions") -> Optional[Dict]:
    """Load predictions JSON file (XL picks + Pro picks combined)"""
    # Try both date formats (YYYY-MM-DD and YYYYMMDD)
    date_compact = date.replace('-', '')

    # XL picks files
    xl_files = [
        Path(predictions_dir) / f"xl_picks_{date}.json",
        Path(predictions_dir) / f"xl_picks_{date_compact}.json",
        Path(predictions_dir) / f"backtest_{date_compact}.json",
    ]

    # Pro picks files
    pro_files = [
        Path(predictions_dir) / f"pro_picks_{date}.json",
        Path(predictions_dir) / f"pro_picks_{date_compact}.json",
    ]

    combined_picks = []
    combined_data = None
    markets_enabled = set()

    # Load XL picks
    for filepath in xl_files:
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    combined_data = data
                    combined_picks.extend(data.get('picks', []))
                    markets_enabled.update(data.get('markets_enabled', []))
                    break
            except json.JSONDecodeError:
                continue

    # Load Pro picks
    for filepath in pro_files:
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    pro_data = json.load(f)
                    # Convert pro picks format to match XL format
                    for pick in pro_data.get('picks', []):
                        converted_pick = {
                            'player_name': pick.get('player_name'),
                            'stat_type': pick.get('stat_type'),
                            'side': pick.get('side', 'OVER'),
                            'best_line': pick.get('line'),
                            'prediction': pick.get('projection', pick.get('line', 0)),
                            'filter_tier': pick.get('filter_tier', 'pro'),
                            'source': 'pro_tier'
                        }
                        combined_picks.append(converted_pick)
                    markets_enabled.update([p['stat_type'] for p in pro_data.get('picks', [])])
                    break
            except json.JSONDecodeError:
                continue

    if not combined_picks:
        print(f"[ERROR] Predictions file not found for {date} in {predictions_dir}")
        return None

    # Build combined result
    if combined_data:
        combined_data['picks'] = combined_picks
        combined_data['total_picks'] = len(combined_picks)
        combined_data['markets_enabled'] = list(markets_enabled)
    else:
        combined_data = {
            'date': date,
            'generated_at': datetime.now().isoformat(),
            'picks': combined_picks,
            'total_picks': len(combined_picks),
            'markets_enabled': list(markets_enabled)
        }

    return combined_data

def normalize_name(name: str) -> str:
    """Normalize player name for matching (remove Jr, III, etc.)"""
    import re
    # Remove common suffixes
    suffixes = [' Jr', ' Jr.', ' III', ' II', ' IV', ' Sr', ' Sr.']
    normalized = name.strip()
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    return normalized.lower()


def get_actual_results(date: str) -> Dict[Tuple[str, str], Dict]:
    """Get actual game results from database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    query = """
        SELECT
            p.full_name,
            l.points,
            l.rebounds,
            l.assists,
            l.three_pointers_made,
            l.opponent_abbrev
        FROM player_game_logs l
        JOIN player_profile p ON l.player_id = p.player_id
        WHERE l.game_date = %s
    """

    cursor.execute(query, (date,))
    results = {}

    for row in cursor.fetchall():
        name, points, rebounds, assists, threes, opponent = row
        # Store both original and normalized name keys
        key = (name, opponent)
        normalized_key = (normalize_name(name), opponent)
        stats = {
            'POINTS': points,
            'REBOUNDS': rebounds,
            'ASSISTS': assists,
            'THREES': threes
        }
        results[key] = stats
        results[normalized_key] = stats  # Also store normalized version

    cursor.close()
    conn.close()

    return results

def validate_pick(pick: Dict, actuals: Dict) -> Dict:
    """Validate a single pick against actual results"""
    player = pick['player_name']
    stat_type = pick['stat_type']
    line = pick['best_line']
    side = pick['side']
    prediction = pick['prediction']

    # Try to find actual result (with and without opponent matching)
    actual = None
    matched_key = None

    # First try exact match
    for key, stats in actuals.items():
        if key[0] == player:
            actual = stats.get(stat_type)
            matched_key = key
            break

    # If not found, try normalized name (handles Jr, III, etc.)
    if actual is None:
        normalized_player = normalize_name(player)
        for key, stats in actuals.items():
            if key[0] == normalized_player:
                actual = stats.get(stat_type)
                matched_key = key
                break

    if actual is None:
        return {
            'status': 'NO_DATA',
            'reason': f'No game log found for {player} (DNP?)'
        }

    # Determine if pick won
    if side == 'OVER':
        result = actual > line
        push = actual == line
    else:
        result = actual < line
        push = actual == line

    # Calculate profit (assuming -110 odds)
    if push:
        profit = 0.0
        outcome = 'PUSH'
    elif result:
        profit = 0.909  # Win 0.909 units on 1 unit bet at -110
        outcome = 'WIN'
    else:
        profit = -1.0
        outcome = 'LOSS'

    return {
        'status': 'VALIDATED',
        'outcome': outcome,
        'actual': actual,
        'line': line,
        'prediction': prediction,
        'diff': actual - line,
        'profit': profit,
        'opponent': matched_key[1] if matched_key else 'Unknown'
    }

def print_validation_report(predictions: Dict, actuals: Dict):
    """Print detailed validation report"""
    print("\n" + "="*80)
    print(f"VALIDATION REPORT: {predictions['date']}")
    print("="*80)
    print(f"Generated at: {predictions['generated_at']}")
    print(f"Total picks: {predictions['total_picks']}")
    print(f"Markets: {', '.join(predictions['markets_enabled'])}")
    print()

    results = []
    by_market = {}

    for pick in predictions['picks']:
        validation = validate_pick(pick, actuals)

        market = pick['stat_type']
        if market not in by_market:
            by_market[market] = {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'no_data': 0,
                'profit': 0.0
            }

        by_market[market]['total'] += 1

        if validation['status'] == 'NO_DATA':
            by_market[market]['no_data'] += 1
            print(f"[WARN]  {pick['player_name']:25s} {market:10s} {validation['reason']}")
        else:
            outcome = validation['outcome']
            if outcome == 'WIN':
                by_market[market]['wins'] += 1
                emoji = '[OK]'
            elif outcome == 'LOSS':
                by_market[market]['losses'] += 1
                emoji = '[ERROR]'
            else:
                by_market[market]['pushes'] += 1
                emoji = '[-]'

            by_market[market]['profit'] += validation['profit']

            side_char = 'O' if pick.get('side', 'OVER') == 'OVER' else 'U'
            print(f"{emoji} {pick['player_name']:25s} {market:10s} "
                  f"{side_char}{validation['line']:4.1f} → {validation['actual']:2d} "
                  f"(pred: {validation['prediction']:5.1f}, diff: {validation['diff']:+4.1f})")

            results.append({
                'player': pick['player_name'],
                'market': market,
                'outcome': outcome,
                'profit': validation['profit'],
                **validation
            })

    print("\n" + "="*80)
    print("SUMMARY BY MARKET")
    print("="*80)

    total_wins = 0
    total_losses = 0
    total_pushes = 0
    total_profit = 0.0
    total_bets = 0

    for market in sorted(by_market.keys()):
        stats = by_market[market]
        validated = stats['total'] - stats['no_data']

        if validated > 0:
            win_rate = (stats['wins'] / validated) * 100
            roi = (stats['profit'] / validated) * 100

            print(f"\n{market}:")
            print(f"  Total: {validated} bets")
            print(f"  Wins: {stats['wins']} ({win_rate:.1f}%)")
            print(f"  Losses: {stats['losses']}")
            print(f"  Pushes: {stats['pushes']}")
            print(f"  Profit: {stats['profit']:+.2f} units")
            print(f"  ROI: {roi:+.2f}%")

            total_wins += stats['wins']
            total_losses += stats['losses']
            total_pushes += stats['pushes']
            total_profit += stats['profit']
            total_bets += validated

    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)

    if total_bets > 0:
        overall_win_rate = (total_wins / total_bets) * 100
        overall_roi = (total_profit / total_bets) * 100

        print(f"Total bets: {total_bets}")
        print(f"Wins: {total_wins} ({overall_win_rate:.1f}%)")
        print(f"Losses: {total_losses}")
        print(f"Pushes: {total_pushes}")
        print(f"Total profit: {total_profit:+.2f} units")
        print(f"ROI: {overall_roi:+.2f}%")

        print("\n" + "="*80)
        print("COMPARISON TO VALIDATION BENCHMARKS")
        print("="*80)

        for market in sorted(by_market.keys()):
            stats = by_market[market]
            validated = stats['total'] - stats['no_data']

            if validated > 0:
                win_rate = (stats['wins'] / validated) * 100
                roi = (stats['profit'] / validated) * 100

                expected = predictions.get('expected_performance', {}).get(market, {})
                expected_wr = expected.get('win_rate', 0)
                expected_roi = expected.get('roi', 0)

                wr_diff = win_rate - expected_wr
                roi_diff = roi - expected_roi

                wr_emoji = '[OK]' if wr_diff >= 0 else '[WARN]'
                roi_emoji = '[OK]' if roi_diff >= 0 else '[WARN]'

                print(f"\n{market}:")
                print(f"  Win Rate: {win_rate:.1f}% (expected {expected_wr:.1f}%, {wr_emoji} {wr_diff:+.1f}%)")
                print(f"  ROI: {roi:+.2f}% (expected {expected_roi:+.2f}%, {roi_emoji} {roi_diff:+.2f}%)")

        # Overall line shopping benchmark
        expected_overall = predictions.get('expected_performance', {}).get('overall_line_shopping', {})
        if expected_overall:
            expected_wr = expected_overall.get('win_rate', 0)
            expected_roi = expected_overall.get('roi', 0)

            wr_diff = overall_win_rate - expected_wr
            roi_diff = overall_roi - expected_roi

            wr_emoji = '[OK]' if wr_diff >= 0 else '[WARN]'
            roi_emoji = '[OK]' if roi_diff >= 0 else '[WARN]'

            print(f"\nOVERALL:")
            print(f"  Win Rate: {overall_win_rate:.1f}% (expected {expected_wr:.1f}%, {wr_emoji} {wr_diff:+.1f}%)")
            print(f"  ROI: {overall_roi:+.2f}% (expected {expected_roi:+.2f}%, {roi_emoji} {roi_diff:+.2f}%)")

    print("\n" + "="*80)

def validate_date_range(start_date: str, end_date: str, predictions_dir: str = "predictions") -> Dict:
    """
    Validate predictions across a date range and aggregate results.

    Returns aggregated results dictionary.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Aggregate stats
    total_wins = 0
    total_losses = 0
    total_pushes = 0
    total_profit = 0.0
    by_market: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0, 'profit': 0.0})
    daily_results = []

    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        predictions = load_predictions(date_str, predictions_dir)
        if not predictions:
            daily_results.append({'date': date_str, 'status': 'NO_PREDICTIONS'})
            current += timedelta(days=1)
            continue

        actuals = get_actual_results(date_str)
        if not actuals:
            daily_results.append({'date': date_str, 'status': 'NO_ACTUALS'})
            current += timedelta(days=1)
            continue

        day_wins = 0
        day_losses = 0
        day_pushes = 0
        day_profit = 0.0

        for pick in predictions.get('picks', []):
            validation = validate_pick(pick, actuals)
            market = pick['stat_type']

            if validation['status'] == 'VALIDATED':
                outcome = validation['outcome']
                if outcome == 'WIN':
                    day_wins += 1
                    by_market[market]['wins'] += 1
                elif outcome == 'LOSS':
                    day_losses += 1
                    by_market[market]['losses'] += 1
                else:
                    day_pushes += 1
                    by_market[market]['pushes'] += 1

                day_profit += validation['profit']
                by_market[market]['profit'] += validation['profit']

        total_wins += day_wins
        total_losses += day_losses
        total_pushes += day_pushes
        total_profit += day_profit

        day_total = day_wins + day_losses
        day_wr = (day_wins / day_total * 100) if day_total > 0 else 0

        daily_results.append({
            'date': date_str,
            'wins': day_wins,
            'losses': day_losses,
            'pushes': day_pushes,
            'win_rate': day_wr,
            'profit': day_profit
        })

        print(f"{date_str}: {day_wins}W / {day_losses}L ({day_wr:.1f}%) | Profit: {day_profit:+.2f}u")

        current += timedelta(days=1)

    # Print summary
    print("\n" + "="*80)
    print(f"AGGREGATE RESULTS: {start_date} to {end_date}")
    print("="*80)

    total_bets = total_wins + total_losses
    overall_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    print(f"\nTotal bets: {total_bets}")
    print(f"Wins: {total_wins} ({overall_wr:.1f}%)")
    print(f"Losses: {total_losses}")
    print(f"Pushes: {total_pushes}")
    print(f"Total profit: {total_profit:+.2f} units")
    print(f"ROI: {overall_roi:+.2f}%")

    print("\n--- BY MARKET ---")
    for market in ['POINTS', 'REBOUNDS', 'THREES', 'ASSISTS']:
        if market in by_market:
            stats = by_market[market]
            w, l = stats['wins'], stats['losses']
            wr = (w / (w + l) * 100) if (w + l) > 0 else 0
            roi = (stats['profit'] / (w + l) * 100) if (w + l) > 0 else 0
            print(f"{market:10}: {w}W / {l}L = {wr:.1f}% WR | ROI: {roi:+.1f}%")

    print("="*80)

    return {
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_pushes': total_pushes,
        'total_profit': total_profit,
        'overall_win_rate': overall_wr,
        'overall_roi': overall_roi,
        'by_market': dict(by_market),
        'daily_results': daily_results
    }


def main():
    parser = argparse.ArgumentParser(description='Validate predictions against actual results')
    parser.add_argument('--date', help='Single date to validate (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for range validation (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for range validation (YYYY-MM-DD)')
    parser.add_argument('--predictions-dir', default='predictions',
                        help='Directory containing prediction files (default: predictions/)')
    parser.add_argument('--backtest-dir', help='Alias for --predictions-dir for backtest results')
    args = parser.parse_args()

    # Handle backtest-dir alias
    predictions_dir = args.backtest_dir or args.predictions_dir

    # Validate arguments
    if args.start_date and args.end_date:
        # Date range mode
        validate_date_range(args.start_date, args.end_date, predictions_dir)
        return
    elif args.date:
        # Single date mode
        predictions = load_predictions(args.date, predictions_dir)
        if not predictions:
            return

        print(f"Fetching actual results for {args.date}...")
        actuals = get_actual_results(args.date)
        print(f"[OK] Found {len(actuals)} player game logs\n")

        print_validation_report(predictions, actuals)
    else:
        parser.error("Either --date or both --start-date and --end-date are required")


if __name__ == '__main__':
    main()
