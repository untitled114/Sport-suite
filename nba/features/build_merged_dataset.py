#!/usr/bin/env python3
"""
Merged Dataset Builder — V3 base + incremental gap + BP analytics
=================================================================

Strategy:
1. Use V3 batched dataset as base (Oct 2023 - Dec 2025, proven quality)
2. Build gap rows (Dec 2, 2025 - Feb 28, 2026) with current extractor
3. Left-join BP historical analytics for all rows
4. Train cutoff: March 1, 2026

Usage:
    python3 nba/features/build_merged_dataset.py --market POINTS
    python3 nba/features/build_merged_dataset.py --market REBOUNDS
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.config.database import get_connection

# Date boundaries
V3_END = date(2025, 12, 1)
GAP_START = date(2025, 12, 2)
TRAIN_END = date(2026, 2, 28)
VAL_START = date(2026, 3, 1)

STAT_MAP = {"POINTS": "actual_points", "REBOUNDS": "actual_rebounds"}

V3_PATHS = {
    "POINTS": "/tmp/v3_original_POINTS.csv",  # nosec B108
    "REBOUNDS": None,  # Will check for batched version
}

V3_BATCHED = {
    "POINTS": "nba/features/datasets/xl_training_POINTS_2023_2025_batched.csv",
    "REBOUNDS": "nba/features/datasets/xl_training_REBOUNDS_2023_2025_batched.csv",
}


def load_v3_base(market):
    """Load the V3 dataset — the proven base."""
    # Try the original V3 first (if available in /tmp), else use batched
    path = V3_PATHS.get(market)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded V3 original: {df.shape} from {path}")
    else:
        path = V3_BATCHED[market]
        df = pd.read_csv(path)
        print(f"  Loaded V3 batched: {df.shape} from {path}")

    # Ensure is_home is int
    if "is_home" in df.columns:
        df["is_home"] = df["is_home"].astype(float).fillna(1).astype(int)

    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Label rate: {df['label'].mean():.4f}")
    return df


def build_gap_rows(market):
    """Build feature rows for the Dec 2 - Feb 28 gap using current extractor."""
    from features.extract_live_features_xl import LiveFeatureExtractorXL

    conn = get_connection("intelligence")
    conn.autocommit = True
    cur = conn.cursor()

    # Fetch props for gap period
    target_col = STAT_MAP[market]
    cur.execute(
        """
        SELECT player_name, game_date, over_line AS line, actual_value,
               opponent_team, is_home, COALESCE(fetch_source, book_name) AS source
        FROM nba_props_xl
        WHERE stat_type = %s
          AND actual_value IS NOT NULL
          AND actual_value > 0
          AND game_date > %s
          AND game_date <= %s
        ORDER BY game_date, player_name
        """,
        (market, GAP_START, TRAIN_END),
    )
    rows = cur.fetchall()
    cols = [
        "player_name",
        "game_date",
        "line",
        "actual_value",
        "opponent_team",
        "is_home",
        "source",
    ]
    props_df = pd.DataFrame(rows, columns=cols)
    cur.close()
    conn.close()

    # Dedup to 1 per (player, date) — keep row with median line
    props_df["line"] = props_df["line"].astype(float)
    deduped = (
        props_df.sort_values("line").groupby(["player_name", "game_date"]).first().reset_index()
    )
    print(f"  Gap props: {len(props_df)} raw → {len(deduped)} deduped")
    print(f"  Gap dates: {deduped['game_date'].min()} to {deduped['game_date'].max()}")

    # Extract features
    extractor = LiveFeatureExtractorXL()
    results = []
    errors = 0
    total = len(deduped)

    for i, (_, prop) in enumerate(deduped.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{total} ({i/total*100:.0f}%) — {errors} errors")

        try:
            features = extractor.extract_features(
                player_name=prop["player_name"],
                game_date=str(prop["game_date"]),
                stat_type=market,
                opponent_team=prop["opponent_team"],
                is_home=bool(prop["is_home"]) if pd.notna(prop["is_home"]) else True,
                line=float(prop["line"]),
            )
            if features is None:
                errors += 1
                continue

            row = {
                "player_name": prop["player_name"],
                "game_date": str(prop["game_date"]),
                "stat_type": market,
                "opponent_team": prop["opponent_team"],
                "is_home": int(prop["is_home"]) if pd.notna(prop["is_home"]) else 1,
                "line": float(prop["line"]),
                "source": prop["source"],
                target_col: float(prop["actual_value"]),
                "label": 1 if float(prop["actual_value"]) > float(prop["line"]) else 0,
            }
            row.update(features)
            results.append(row)
        except Exception as e:
            errors += 1

    extractor.close()
    df = pd.DataFrame(results)
    print(f"  Gap rows built: {len(df)} ({errors} errors)")
    return df


def join_bp_analytics(df, market):
    """Left-join BP historical analytics features."""
    conn = get_connection("intelligence")
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        SELECT player_name, game_date::text,
               bp_projection, bp_projection_diff, bp_probability,
               bp_expected_value, bp_bet_rating, bp_recommended_side,
               bp_opposition_rank, bp_opposition_value,
               bp_hit_rate_l5, bp_hit_rate_l10, bp_hit_rate_l15,
               bp_hit_rate_l20, bp_hit_rate_season
        FROM bp_historical_analytics
        WHERE stat_type = %s
        """,
        (market,),
    )
    rows = cur.fetchall()
    cols = [
        "player_name",
        "game_date",
        "bp_analytics_projection",
        "bp_analytics_projection_diff",
        "bp_analytics_probability",
        "bp_analytics_ev",
        "bp_analytics_bet_rating",
        "bp_analytics_recommended_side",
        "bp_analytics_opp_rank",
        "bp_analytics_opp_value",
        "bp_analytics_hit_rate_L5",
        "bp_analytics_hit_rate_L10",
        "bp_analytics_hit_rate_L15",
        "bp_analytics_hit_rate_L20",
        "bp_analytics_hit_rate_season",
    ]
    bp_df = pd.DataFrame(rows, columns=cols)
    cur.close()
    conn.close()

    # Convert recommended_side to numeric
    bp_df["bp_analytics_recommended_over"] = (
        bp_df["bp_analytics_recommended_side"] == "over"
    ).astype(float)
    bp_df = bp_df.drop(columns=["bp_analytics_recommended_side"])

    # Dedup BP data (keep first per player+date)
    bp_df = bp_df.drop_duplicates(subset=["player_name", "game_date"], keep="first")

    before = len(df)
    merged = df.merge(bp_df, on=["player_name", "game_date"], how="left")
    matched = merged["bp_analytics_projection"].notna().sum()
    print(f"  BP analytics joined: {matched}/{before} matched ({matched/before*100:.1f}%)")

    return merged


def build_merged_dataset(market, output_dir):
    """Main merge pipeline."""
    print(f"\n{'='*60}")
    print(f"BUILDING MERGED {market} DATASET")
    print(f"{'='*60}")

    # Step 1: Load V3 base
    print("\n[1/4] Loading V3 base...")
    v3 = load_v3_base(market)

    # Step 2: Build gap rows
    print("\n[2/4] Building gap rows (Dec 2 - Feb 28)...")
    gap = build_gap_rows(market)

    # Step 3: Concatenate
    print("\n[3/4] Merging...")
    # Align columns — gap may have different columns than V3
    all_cols = sorted(set(v3.columns) | set(gap.columns))
    for c in all_cols:
        if c not in v3.columns:
            v3[c] = np.nan
        if c not in gap.columns:
            gap[c] = np.nan

    merged = pd.concat([v3, gap], ignore_index=True)
    merged = merged.sort_values("game_date").reset_index(drop=True)

    # Dedup (V3 + gap might overlap on Dec 1)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["player_name", "game_date"], keep="first")
    print(f"  Deduped: {before} → {len(merged)}")

    # Assign split
    merged["split"] = merged["game_date"].apply(
        lambda d: "val" if str(d) >= VAL_START.isoformat() else "train"
    )

    # Step 4: Join BP analytics
    print("\n[4/4] Joining BP analytics...")
    merged = join_bp_analytics(merged, market)

    # Ensure is_home is int
    if "is_home" in merged.columns:
        merged["is_home"] = merged["is_home"].astype(float).fillna(1).astype(int)

    # Summary
    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    print(f"\n{'='*60}")
    print(f"RESULT: {market}")
    print(f"{'='*60}")
    print(f"Total rows:  {len(merged)}")
    print(f"Train:       {len(train)} ({train['game_date'].min()} to {train['game_date'].max()})")
    print(
        f"Val:         {len(val)} ({val['game_date'].min()} to {val['game_date'].max()}"
        if len(val) > 0
        else "Val: 0"
    )
    print(f"Columns:     {len(merged.columns)}")
    print(f"Label rate:  {merged['label'].mean():.4f}")
    print(
        f"is_home:     dtype={merged['is_home'].dtype}, home%={merged['is_home'].mean()*100:.1f}%"
    )

    # Save
    if output_dir:
        out_path = Path(output_dir) / f"xl_merged_training_{market}_2023_2026.csv"
        merged.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Build merged V3 + gap + BP dataset")
    parser.add_argument("--market", type=str, default="POINTS", choices=["POINTS", "REBOUNDS"])
    parser.add_argument("--output", type=str, default="nba/features/datasets/")
    args = parser.parse_args()

    os.environ.setdefault("DB_USER", "mlb_user")
    os.environ.setdefault("DB_PASSWORD", "mlb_secure_2025")

    build_merged_dataset(args.market, args.output)


if __name__ == "__main__":
    main()
