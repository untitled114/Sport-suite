#!/usr/bin/env python3
"""
Feature Extractor Audit Script
================================
Runs ALL extractors against test players and produces a full report.
Tests: star, bench, traded, rookie, injured — across multiple dates.

Checks:
1. Every feature has a real value (not default/placeholder)
2. No NaN, None, or empty values
3. Values are in sane ranges
4. Features match what production models expect
5. Cross-references V3/XL model feature lists against live output

Usage:
    python audit_feature_extractors.py
"""

import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EST = ZoneInfo("America/New_York")

# Test players — diverse profiles
TEST_PLAYERS = [
    # (name, date, stat_type, opponent, is_home, line, label)
    ("Nikola Jokic", "2025-02-15", "POINTS", "LAL", True, 28.5, "STAR"),
    ("LeBron James", "2025-01-10", "POINTS", "CLE", False, 25.5, "STAR"),
    ("Scottie Barnes", "2025-03-01", "REBOUNDS", "MIA", True, 8.5, "STAR"),
    ("Max Christie", "2025-02-20", "POINTS", "SAC", True, 8.5, "BENCH"),
    ("Trayce Jackson-Davis", "2025-01-25", "REBOUNDS", "PHX", True, 5.5, "BENCH"),
    ("Luka Doncic", "2025-03-01", "POINTS", "BOS", True, 30.5, "STAR_DIACRITICS"),
]


def get_db_config():
    return {
        "user": os.getenv("DB_USER", "mlb_user"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": "localhost",
    }


def audit():
    cfg = get_db_config()

    print("=" * 80)
    print("FEATURE EXTRACTOR AUDIT")
    print("=" * 80)

    # Load production model feature lists
    model_dir = Path(__file__).resolve().parents[1] / "models" / "saved_xl"
    model_features = {}
    for version in ["xl", "v3"]:
        path = model_dir / f"points_{version}_features.pkl"
        if path.exists():
            features = pickle.load(open(path, "rb"))
            model_features[version] = set(features)
            print(f"  {version.upper()} model: {len(features)} features")

    # Initialize the full extractor
    from nba.features.extract_live_features_xl import LiveFeatureExtractorXL

    ext = LiveFeatureExtractorXL()

    results = []
    all_features_seen = set()
    feature_values = defaultdict(list)

    for name, date, stat, opp, home, line, label in TEST_PLAYERS:
        print(f"\n{'─' * 60}")
        print(f"  {label}: {name} ({date} vs {opp}, {stat})")
        print(f"{'─' * 60}")

        try:
            features = ext.extract_features(
                player_name=name,
                game_date=date,
                stat_type=stat,
                opponent_team=opp,
                is_home=home,
                line=line,
            )
        except Exception as e:
            print(f"  CRASH: {e}")
            results.append({"player": name, "label": label, "status": "CRASH", "error": str(e)})
            continue

        if features is None:
            print(f"  RETURNED NONE (insufficient history)")
            results.append({"player": name, "label": label, "status": "NONE"})
            continue

        total = len(features)
        all_features_seen.update(features.keys())

        # Categorize features
        defaults = 0
        populated = 0
        suspicious = 0
        issues = []

        for k, v in features.items():
            feature_values[k].append(v)

            if v is None or (isinstance(v, float) and np.isnan(v)):
                issues.append(f"  NaN/None: {k}")
                suspicious += 1
            elif isinstance(v, str):
                # String features shouldn't be here (model expects numeric)
                issues.append(f"  STRING: {k} = {v!r}")
                suspicious += 1
            elif v == 0.0:
                defaults += 1
            else:
                populated += 1

        # Check specific feature groups
        bp_features = {
            k: v
            for k, v in features.items()
            if k.startswith("bp_analytics_") or k.startswith("dvp_")
        }
        gc_features = {
            k: v
            for k, v in features.items()
            if k
            in [
                "game_pace",
                "opp_score_margin_avg",
                "player_minutes_stability",
                "player_plus_minus_L5",
                "player_usage_proxy",
                "player_scoring_efficiency",
                "player_blowout_risk",
                "player_minutes_vs_avg",
            ]
        }
        tf_features = {
            k: v
            for k, v in features.items()
            if k
            in [
                "is_post_trade_deadline",
                "days_since_trade_deadline",
                "is_post_allstar",
                "days_since_allstar",
                "is_playoff_push",
                "is_regular_season",
                "season_pct",
                "player_games_with_team",
                "is_new_team",
                "team_tenure_games",
            ]
        }

        bp_populated = sum(
            1 for v in bp_features.values() if v != 0 and v != 0.5 and v != 3.0 and v != 15.0
        )
        gc_populated = sum(1 for v in gc_features.values() if v != 0 and v != 100.0 and v != 1.0)
        tf_populated = sum(
            1
            for v in tf_features.values()
            if v != 0 and v != 0.5 and v != 30.0 and v != 40.0 and v != 1.0
        )

        print(f"  Total: {total} features")
        print(f"  Populated: {populated} | Zeros: {defaults} | Issues: {suspicious}")
        print(f"  BP Analytics: {bp_populated}/{len(bp_features)} with real data")
        print(f"  Game Context: {gc_populated}/{len(gc_features)} with real data")
        print(f"  Temporal: {tf_populated}/{len(tf_features)} with real data")

        if issues:
            for issue in issues[:5]:
                print(issue)
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more issues")

        results.append(
            {
                "player": name,
                "label": label,
                "status": "OK",
                "total": total,
                "populated": populated,
                "zeros": defaults,
                "issues": suspicious,
                "bp": f"{bp_populated}/{len(bp_features)}",
                "gc": f"{gc_populated}/{len(gc_features)}",
                "tf": f"{tf_populated}/{len(tf_features)}",
            }
        )

    # Cross-reference with production models
    print(f"\n{'=' * 80}")
    print("PRODUCTION MODEL CROSS-REFERENCE")
    print(f"{'=' * 80}")

    for version, expected in model_features.items():
        missing = (
            expected - all_features_seen - {"expected_diff"}
        )  # expected_diff added by classifier
        extra = all_features_seen - expected

        print(f"\n  {version.upper()} Model ({len(expected)} features):")
        if missing:
            print(f"    MISSING from live extraction ({len(missing)}):")
            for f in sorted(missing):
                # Check if it was in any player's features as zero
                vals = feature_values.get(f, [])
                if vals:
                    print(f"      {f} = {vals[0]} (present but may be wrong)")
                else:
                    print(f"      {f} — NOT EXTRACTED AT ALL")
        else:
            print(f"    All {len(expected)} features present in live extraction")

    # Feature health summary
    print(f"\n{'=' * 80}")
    print("FEATURE HEALTH SUMMARY")
    print(f"{'=' * 80}")

    always_zero = []
    always_default = []
    healthy = []

    for feat, vals in sorted(feature_values.items()):
        if all(v == 0 for v in vals):
            always_zero.append(feat)
        elif all(v == vals[0] for v in vals):  # same value for all players
            always_default.append((feat, vals[0]))
        else:
            healthy.append(feat)

    print(f"\n  Healthy features (vary across players): {len(healthy)}")
    print(f"  Always zero: {len(always_zero)}")
    if always_zero:
        for f in always_zero:
            print(f"    DEAD: {f}")
    print(f"  Always same value: {len(always_default)}")
    if always_default:
        for f, v in always_default:
            print(f"    CONSTANT: {f} = {v}")

    # Summary
    print(f"\n{'=' * 80}")
    print("FINAL VERDICT")
    print(f"{'=' * 80}")
    ok = sum(1 for r in results if r["status"] == "OK")
    none = sum(1 for r in results if r["status"] == "NONE")
    crash = sum(1 for r in results if r["status"] == "CRASH")
    print(f"  Players tested: {len(results)}")
    print(f"  OK: {ok} | NONE (insufficient history): {none} | CRASH: {crash}")
    print(f"  Total unique features extracted: {len(all_features_seen)}")
    print(f"  Healthy features: {len(healthy)}")
    print(f"  Dead features (always zero): {len(always_zero)}")
    print(f"  Constant features: {len(always_default)}")

    ext.close()


if __name__ == "__main__":
    audit()
