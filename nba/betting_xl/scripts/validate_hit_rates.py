#!/usr/bin/env python3
"""Validate hit-rate gating strategies over historical predictions."""

from __future__ import annotations

import argparse
import json
import psycopg2
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Iterable, List, Callable, Optional, Tuple
import os


DB_PLAYERS = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

STAT_COLUMN_MAP = {
    'POINTS': 'points',
    'REBOUNDS': 'rebounds',
}


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


class HitRateValidator:
    def __init__(self, predictions_dir: Path):
        self.predictions_dir = predictions_dir
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        self.stat_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def close(self):
        if self.conn_players:
            self.conn_players.close()

    def _load_predictions(self, game_date: datetime) -> List[Dict[str, Any]]:
        date_str = game_date.strftime('%Y-%m-%d')
        candidates = [
            self.predictions_dir / f"xl_picks_{date_str}.json",
            self.predictions_dir / f"xl_picks_{date_str.replace('-', '')}.json",
        ]
        pick_file = next((p for p in candidates if p.exists()), None)
        if not pick_file:
            raise FileNotFoundError(f"No predictions file for {date_str} (looked for: {candidates})")

        with open(pick_file, 'r') as f:
            data = json.load(f)

        picks = data.get('picks', [])
        for pick in picks:
            pick['game_date'] = data.get('date') or date_str
            pick['hit_rate_l10'] = self._extract_rate(pick, 'last_10')
            pick['hit_rate_season'] = self._extract_rate(pick, 'season')
        return picks

    @staticmethod
    def _extract_rate(pick: Dict[str, Any], window: str) -> Optional[float]:
        rates = pick.get('hit_rates') or {}
        entry = rates.get(window)
        if entry and entry.get('rate') is not None:
            return float(entry['rate'])
        return None

    def _get_actual_stats(self, player_name: str, game_date: str) -> Optional[Dict[str, Any]]:
        key = (player_name.lower(), game_date)
        if key in self.stat_cache:
            return self.stat_cache[key]

        columns_list = sorted(set(STAT_COLUMN_MAP.values()))
        select_clause = ', '.join([f"pgl.{col}" for col in columns_list])
        query = f"""
            SELECT {select_clause}
            FROM player_game_logs pgl
            JOIN player_profile pp ON pp.player_id = pgl.player_id
            WHERE pp.full_name = %s AND pgl.game_date = %s
            LIMIT 1;
        """

        with self.conn_players.cursor() as cur:
            cur.execute(query, (player_name, game_date))
            row = cur.fetchone()

        if not row:
            self.stat_cache[key] = None
            return None

        stats = {}
        for idx, stats_column in enumerate(columns_list):
            stats_value = row[idx]
            for stat_type, column in STAT_COLUMN_MAP.items():
                if column == stats_column:
                    stats[stat_type] = float(stats_value)
        self.stat_cache[key] = stats
        return stats

    @staticmethod
    def _grade_pick(actual: float, line: float, side: str) -> Tuple[str, float]:
        if actual is None:
            return ('PENDING', 0.0)
        if side == 'OVER':
            if actual > line:
                return ('WIN', 1.0)
            if actual < line:
                return ('LOSS', -1.1)
            return ('PUSH', 0.0)
        if side == 'UNDER':
            if actual < line:
                return ('WIN', 1.0)
            if actual > line:
                return ('LOSS', -1.1)
            return ('PUSH', 0.0)
        return ('PENDING', 0.0)

    def collect_picks(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for day in daterange(start, end):
            try:
                picks = self._load_predictions(day)
            except FileNotFoundError as exc:
                print(f"⚠️  {exc}")
                continue

            for pick in picks:
                stats = self._get_actual_stats(pick['player_name'], pick['game_date'])
                if not stats:
                    continue
                stat_value = stats.get(pick['stat_type'])
                result, profit = self._grade_pick(stat_value, pick['best_line'], pick['side'])
                pick['actual_value'] = stat_value
                pick['result'] = result
                pick['profit'] = profit
                collected.append(pick)

        return collected


def summarize(picks: List[Dict[str, Any]], name: str, selector: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    filtered = [p for p in picks if selector(p)]
    wins = sum(1 for p in filtered if p['result'] == 'WIN')
    losses = sum(1 for p in filtered if p['result'] == 'LOSS')
    pushes = sum(1 for p in filtered if p['result'] == 'PUSH')
    total = len(filtered)
    profit = sum(p['profit'] for p in filtered)
    wr = (wins / total * 100) if total else 0.0
    roi = (profit / total * 100) if total else 0.0
    return {
        'strategy': name,
        'picks': total,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': wr,
        'roi': roi,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate hit-rate gating strategies.')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--predictions-dir', default='predictions', help='Directory with xl_picks_*.json files')
    parser.add_argument('--tierb-min', type=float, default=0.55, help='Min L10 hit rate required for Tier B picks')
    parser.add_argument('--global-min', type=float, default=0.35, help='Drop picks below this L10 hit rate regardless of tier')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    predictions_dir = Path(args.predictions_dir)

    validator = HitRateValidator(predictions_dir)
    try:
        all_picks = validator.collect_picks(start, end)
    finally:
        validator.close()

    if not all_picks:
        print("No picks collected for the specified range.")
        return

    strategies = [
        summarize(all_picks, 'Baseline (current picks)', lambda p: True),
        summarize(
            all_picks,
            f'Tier-B >= {args.tierb_min:.2f}',
            lambda p: p['filter_tier'] != 'tier_b' or (p['hit_rate_l10'] is None or p['hit_rate_l10'] >= args.tierb_min)
        ),
        summarize(
            all_picks,
            f'Global min >= {args.global_min:.2f}',
            lambda p: p['hit_rate_l10'] is None or p['hit_rate_l10'] >= args.global_min
        )
    ]

    print("\nHit-Rate Validation Results")
    print("=" * 70)
    header = "{:<28} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}".format(
        'Strategy', 'Picks', 'Wins', 'Loss', 'Push', 'WR%', 'ROI%'
    )
    print(header)
    print('-' * len(header))
    for stats in strategies:
        print("{:<28} {:>6} {:>6} {:>6} {:>6} {:>8.2f} {:>8.2f}".format(
            stats['strategy'][:28],
            stats['picks'],
            stats['wins'],
            stats['losses'],
            stats['pushes'],
            stats['win_rate'],
            stats['roi'],
        ))


if __name__ == '__main__':
    main()
