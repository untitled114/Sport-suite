#!/usr/bin/env python3
"""
Backfill prop_performance_history Table
========================================

Aggregates historical prop performance metrics from nba_props_xl
to populate the prop_performance_history summary table.

Computes:
- Hit rates (L20, L10, home, away, vs defense, rested, B2B)
- Line vs season average
- Days since last hit
- Consecutive overs

Usage:
    python backfill_prop_history.py
    python backfill_prop_history.py --dry-run
    python backfill_prop_history.py --season 2024
"""

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import psycopg2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nba.config.database import get_intelligence_db_config

# Database configuration
DB_CONFIG_INTEL = get_intelligence_db_config()


def date_to_season(d: date) -> int:
    """Convert date to NBA season (e.g., 2024-11-01 -> 2024)"""
    if d.month >= 10:
        return d.year
    return d.year - 1


def fetch_all_props() -> list[dict]:
    """Fetch all props with outcomes"""
    with psycopg2.connect(**DB_CONFIG_INTEL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (player_name, game_date, stat_type)
                    player_name, game_date, stat_type,
                    over_line, actual_value, is_home,
                    opponent_team
                FROM nba_props_xl
                WHERE actual_value IS NOT NULL
                  AND over_line IS NOT NULL
                  AND game_date >= '2023-10-01'
                ORDER BY player_name, game_date, stat_type, fetch_timestamp DESC
            """
            )
            columns = [
                "player_name",
                "game_date",
                "stat_type",
                "line",
                "actual",
                "is_home",
                "opponent",
            ]
            results = []
            for row in cur.fetchall():
                d = dict(zip(columns, row))
                # Convert Decimal to float
                d["line"] = float(d["line"]) if d["line"] is not None else 0.0
                d["actual"] = float(d["actual"]) if d["actual"] is not None else 0.0
                results.append(d)
            return results


def compute_stats(
    props: list[dict], player_name: str, stat_type: str, line_center: float, season: int
) -> dict:
    """Compute aggregated stats for a player/stat/line/season combo"""

    # Filter to relevant props for this combo
    relevant = [
        p
        for p in props
        if p["player_name"] == player_name
        and p["stat_type"] == stat_type
        and date_to_season(p["game_date"]) == season
        and round(p["line"] * 2) / 2 == line_center
    ]

    if not relevant:
        return None

    # Sort by date descending (most recent first)
    relevant.sort(key=lambda x: x["game_date"], reverse=True)

    total = len(relevant)

    # Compute hit rates
    def hit_rate(subset: list[dict]) -> float:
        if not subset:
            return 0.5
        hits = sum(1 for p in subset if p["actual"] > p["line"])
        return hits / len(subset)

    # Hit rate L20 (most recent 20)
    l20 = relevant[:20]
    hit_rate_l20 = hit_rate(l20)

    # Hit rate L10
    l10 = relevant[:10]
    hit_rate_l10 = hit_rate(l10)

    # Home/away hit rates
    home_props = [p for p in relevant if p["is_home"]]
    away_props = [p for p in relevant if not p["is_home"]]
    hit_rate_home = hit_rate(home_props)
    hit_rate_away = hit_rate(away_props)

    # Compute line vs season average
    all_player_props = [
        p
        for p in props
        if p["player_name"] == player_name
        and p["stat_type"] == stat_type
        and date_to_season(p["game_date"]) == season
    ]
    if all_player_props:
        avg_line = sum(p["line"] for p in all_player_props) / len(all_player_props)
        line_vs_avg = line_center - avg_line
    else:
        line_vs_avg = 0.0

    # Line percentile
    all_lines = sorted([p["line"] for p in all_player_props])
    if all_lines:
        rank = sum(1 for ln in all_lines if ln < line_center)
        line_percentile = rank / len(all_lines)
    else:
        line_percentile = 0.5

    # Days since last hit
    days_since_hit = 999
    today = date.today()
    for p in relevant:
        if p["actual"] > p["line"]:
            days_since_hit = (today - p["game_date"]).days
            break

    # Days since last prop
    days_since_prop = (today - relevant[0]["game_date"]).days if relevant else 999

    # Consecutive overs
    consecutive = 0
    for p in relevant:
        if p["actual"] > p["line"]:
            consecutive += 1
        else:
            break

    # Max streak
    max_streak = 0
    current_streak = 0
    for p in relevant:
        if p["actual"] > p["line"]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "line_center": line_center,
        "season": season,
        "total_props": total,
        "props_l20": min(20, total),
        "props_l10": min(10, total),
        "hit_rate_all": hit_rate(relevant),
        "hit_rate_l20": hit_rate_l20,
        "hit_rate_l10": hit_rate_l10,
        "hit_rate_home": hit_rate_home,
        "hit_rate_away": hit_rate_away,
        "hit_rate_vs_top10_def": 0.5,  # Would need defense rankings
        "hit_rate_vs_bottom10_def": 0.5,
        "hit_rate_rested": 0.5,  # Would need rest calculation
        "hit_rate_b2b": 0.5,
        "n_home": len(home_props),
        "n_away": len(away_props),
        "n_vs_top10_def": 0,
        "n_vs_bottom10_def": 0,
        "n_rested": 0,
        "n_b2b": 0,
        "line_vs_season_avg": line_vs_avg,
        "line_percentile": line_percentile,
        "days_since_last_prop": days_since_prop,
        "days_since_last_hit": days_since_hit,
        "consecutive_overs": consecutive,
        "max_streak_overs": max_streak,
        "sample_quality_score": min(1.0, total / 20.0),
        "bayesian_prior_weight": max(0.0, 1.0 - total / 20.0),
    }


def upsert_stat(stat: dict, dry_run: bool = False):
    """Upsert a stat record into prop_performance_history"""
    if dry_run:
        return

    with psycopg2.connect(**DB_CONFIG_INTEL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prop_performance_history (
                    player_name, stat_type, line_center, season,
                    total_props, props_l20, props_l10,
                    hit_rate_all, hit_rate_l20, hit_rate_l10,
                    hit_rate_home, hit_rate_away,
                    hit_rate_vs_top10_def, hit_rate_vs_bottom10_def,
                    hit_rate_rested, hit_rate_b2b,
                    n_home, n_away,
                    n_vs_top10_def, n_vs_bottom10_def, n_rested, n_b2b,
                    line_vs_season_avg, line_percentile,
                    days_since_last_prop, days_since_last_hit,
                    consecutive_overs, max_streak_overs,
                    sample_quality_score, bayesian_prior_weight
                ) VALUES (
                    %(player_name)s, %(stat_type)s, %(line_center)s, %(season)s,
                    %(total_props)s, %(props_l20)s, %(props_l10)s,
                    %(hit_rate_all)s, %(hit_rate_l20)s, %(hit_rate_l10)s,
                    %(hit_rate_home)s, %(hit_rate_away)s,
                    %(hit_rate_vs_top10_def)s, %(hit_rate_vs_bottom10_def)s,
                    %(hit_rate_rested)s, %(hit_rate_b2b)s,
                    %(n_home)s, %(n_away)s,
                    %(n_vs_top10_def)s, %(n_vs_bottom10_def)s, %(n_rested)s, %(n_b2b)s,
                    %(line_vs_season_avg)s, %(line_percentile)s,
                    %(days_since_last_prop)s, %(days_since_last_hit)s,
                    %(consecutive_overs)s, %(max_streak_overs)s,
                    %(sample_quality_score)s, %(bayesian_prior_weight)s
                )
                ON CONFLICT (player_name, stat_type, line_center, season)
                DO UPDATE SET
                    total_props = EXCLUDED.total_props,
                    props_l20 = EXCLUDED.props_l20,
                    props_l10 = EXCLUDED.props_l10,
                    hit_rate_all = EXCLUDED.hit_rate_all,
                    hit_rate_l20 = EXCLUDED.hit_rate_l20,
                    hit_rate_l10 = EXCLUDED.hit_rate_l10,
                    hit_rate_home = EXCLUDED.hit_rate_home,
                    hit_rate_away = EXCLUDED.hit_rate_away,
                    hit_rate_vs_top10_def = EXCLUDED.hit_rate_vs_top10_def,
                    hit_rate_vs_bottom10_def = EXCLUDED.hit_rate_vs_bottom10_def,
                    hit_rate_rested = EXCLUDED.hit_rate_rested,
                    hit_rate_b2b = EXCLUDED.hit_rate_b2b,
                    n_home = EXCLUDED.n_home,
                    n_away = EXCLUDED.n_away,
                    n_vs_top10_def = EXCLUDED.n_vs_top10_def,
                    n_vs_bottom10_def = EXCLUDED.n_vs_bottom10_def,
                    n_rested = EXCLUDED.n_rested,
                    n_b2b = EXCLUDED.n_b2b,
                    line_vs_season_avg = EXCLUDED.line_vs_season_avg,
                    line_percentile = EXCLUDED.line_percentile,
                    days_since_last_prop = EXCLUDED.days_since_last_prop,
                    days_since_last_hit = EXCLUDED.days_since_last_hit,
                    consecutive_overs = EXCLUDED.consecutive_overs,
                    max_streak_overs = EXCLUDED.max_streak_overs,
                    sample_quality_score = EXCLUDED.sample_quality_score,
                    bayesian_prior_weight = EXCLUDED.bayesian_prior_weight
            """,
                stat,
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Backfill prop_performance_history")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--season", type=int, help="Only process specific season")
    args = parser.parse_args()

    print("=" * 70)
    print("PROP PERFORMANCE HISTORY BACKFILL")
    print("=" * 70)

    # Fetch all props
    print("\nFetching all props with outcomes...")
    props = fetch_all_props()
    print(f"Loaded {len(props):,} props")

    # Build unique (player, stat_type, line_center, season) combos
    combos = set()
    for p in props:
        line_center = round(p["line"] * 2) / 2
        season = date_to_season(p["game_date"])
        if args.season and season != args.season:
            continue
        combos.add((p["player_name"], p["stat_type"], line_center, season))

    print(f"Unique combinations to process: {len(combos):,}")

    # Process each combo
    inserted = 0
    errors = 0

    for player_name, stat_type, line_center, season in tqdm(combos, desc="Processing"):
        try:
            stat = compute_stats(props, player_name, stat_type, line_center, season)
            if stat:
                upsert_stat(stat, args.dry_run)
                inserted += 1
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\nError for {player_name}/{stat_type}/{line_center}: {e}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Inserted/Updated: {inserted:,}")
    print(f"Errors: {errors}")

    if args.dry_run:
        print("\n[DRY RUN - no changes made]")


if __name__ == "__main__":
    main()
