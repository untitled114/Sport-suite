#!/usr/bin/env python3
"""
DB-Based Validation — queries nba_prediction_history directly.

Advantages over JSON-based validation:
- No file path issues
- Access to all runs and context_snapshot
- Conviction accuracy, BP signal accuracy, model version comparison
- Deduplicates across runs (grades each unique pick once)

Usage:
    python3 -m nba.betting_xl.validate_from_db
    python3 -m nba.betting_xl.validate_from_db --days 30
    python3 -m nba.betting_xl.validate_from_db --start-date 2026-03-01 --end-date 2026-03-14
    python3 -m nba.betting_xl.validate_from_db --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nba.config.database import get_axiom_db_config

_EST = ZoneInfo("America/New_York")


def _connect_axiom():
    return psycopg2.connect(**get_axiom_db_config())


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "  N/A "
    return f"{num / den * 100:5.1f}%"


def _roi(profit: float, bets: int) -> str:
    if bets == 0:
        return "  N/A "
    return f"{profit / bets * 100:+5.1f}%"


def _profit_str(profit: float) -> str:
    return f"{profit:+.2f}u"


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────


def load_graded_picks(start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Load graded picks from nba_prediction_history, deduplicated.

    For each (run_date, player_name, stat_type), takes the LATEST run_number
    to avoid counting the same pick multiple times.
    """
    conn = _connect_axiom()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY run_date, player_name, stat_type
                               ORDER BY run_number DESC
                           ) AS rn
                    FROM nba_prediction_history
                    WHERE run_date BETWEEN %s AND %s
                      AND is_hit IS NOT NULL
                )
                SELECT * FROM ranked WHERE rn = 1
                ORDER BY run_date, player_name
            """,
                (start_date, end_date),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def load_conviction_data(start_date: str, end_date: str) -> dict[tuple, dict]:
    """Load conviction data keyed by (run_date, player_name, stat_type)."""
    conn = _connect_axiom()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM axiom_conviction
                WHERE run_date BETWEEN %s AND %s
            """,
                (start_date, end_date),
            )
            result = {}
            for r in cur.fetchall():
                key = (str(r["run_date"]), r["player_name"], r["stat_type"])
                result[key] = dict(r)
            return result
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────


def grade_pick(pick: dict) -> dict:
    """Compute outcome and profit for a graded pick."""
    is_hit = pick["is_hit"]
    if is_hit:
        return {"outcome": "WIN", "profit": 0.909}
    else:
        return {"outcome": "LOSS", "profit": -1.0}


# ──────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────


def run_validation(
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full validation from DB."""

    picks = load_graded_picks(start_date, end_date)
    conviction_data = load_conviction_data(start_date, end_date)

    if not picks:
        print(f"\n  No graded picks found for {start_date} to {end_date}")
        print("  (Run write_actuals to grade picks against game logs)\n")
        return {}

    # ── Aggregate buckets ──────────────────────────────────────────
    by_model = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_market = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_model_market = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_tier = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_date = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_conviction_band = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_bp_signal = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_confidence = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_risk = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_run_number = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
    by_models_agreeing = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})

    all_details = []
    total_w, total_l, total_profit = 0, 0, 0.0

    for pick in picks:
        g = grade_pick(pick)
        outcome = g["outcome"]
        profit = g["profit"]
        is_win = outcome == "WIN"

        model = pick.get("model_version", "xl")
        market = pick.get("stat_type", "UNKNOWN")
        tier = pick.get("tier", "unknown")
        run_date = str(pick["run_date"])
        run_number = pick.get("run_number", 1)

        # Parse context_snapshot
        ctx = pick.get("context_snapshot") or {}
        if isinstance(ctx, str):
            ctx = json.loads(ctx)

        confidence = ctx.get("confidence") or "UNKNOWN"
        risk_level = ctx.get("risk_level") or "UNKNOWN"
        models_agreeing = ctx.get("models_agreeing", [model])
        bp_side = ctx.get("bp_recommended_side", "")
        bp_rating = ctx.get("bp_bet_rating")
        bp_rec_stars = ctx.get("bp_rec_stars")
        bp_rec_side = ctx.get("bp_rec_side")

        # Conviction data
        conv_key = (run_date, pick["player_name"], market)
        conv = conviction_data.get(conv_key)
        conviction_score = conv["conviction"] if conv else None
        conviction_label = conv["conviction_label"] if conv else None

        # ── Update aggregates ──────────────────────────────────────
        def _update(bucket, is_win, profit):
            if is_win:
                bucket["w"] += 1
            else:
                bucket["l"] += 1
            bucket["profit"] += profit

        _update(by_model[model], is_win, profit)
        _update(by_market[market], is_win, profit)
        _update(by_model_market[f"{model}|{market}"], is_win, profit)
        _update(by_tier[tier], is_win, profit)
        _update(by_date[run_date], is_win, profit)
        _update(by_confidence[confidence], is_win, profit)
        _update(by_risk[risk_level], is_win, profit)
        _update(by_run_number[run_number], is_win, profit)

        # Models agreeing count
        n_models = len(models_agreeing) if isinstance(models_agreeing, list) else 1
        _update(
            by_models_agreeing[f"{n_models} model{'s' if n_models != 1 else ''}"], is_win, profit
        )

        # Conviction bands (aligned with _get_label thresholds)
        if conviction_score is not None:
            if conviction_score >= 0.85:
                band = "LOCKED (>=0.85)"
            elif conviction_score >= 0.75:
                band = "STRONG (0.75-0.84)"
            elif conviction_score >= 0.50:
                band = "WATCH (0.50-0.74)"
            else:
                band = "SKIP (<0.50)"
            _update(by_conviction_band[band], is_win, profit)
        else:
            _update(by_conviction_band["No conviction data"], is_win, profit)

        # BP signal categories
        bp_label = _classify_bp_signal(ctx)
        _update(by_bp_signal[bp_label], is_win, profit)

        if is_win:
            total_w += 1
        else:
            total_l += 1
        total_profit += profit

        all_details.append(
            {
                "date": run_date,
                "player": pick["player_name"],
                "market": market,
                "model": model,
                "line": float(pick.get("line") or 0),
                "actual": float(pick.get("actual_result") or 0),
                "outcome": outcome,
                "profit": profit,
                "tier": tier,
                "conviction": conviction_score,
                "conviction_label": conviction_label,
                "bp_label": bp_label,
                "confidence": confidence,
            }
        )

    # ── Print Report ───────────────────────────────────────────────
    total = total_w + total_l

    print()
    print("=" * 90)
    print(f"  VALIDATION REPORT (DB): {start_date} to {end_date}")
    print(
        f"  {total} graded picks | {total_w}W-{total_l}L | "
        f"{_pct(total_w, total)} WR | {_roi(total_profit, total)} ROI | {_profit_str(total_profit)}"
    )
    print("=" * 90)

    # ── By Model Version ──────────────────────────────────────────
    _print_section("BY MODEL VERSION", by_model, sort_by_wr=True)

    # ── By Market ─────────────────────────────────────────────────
    _print_section("BY MARKET", by_market, sort_by_wr=True)

    # ── By Model + Market ─────────────────────────────────────────
    _print_section("BY MODEL + MARKET", by_model_market, sort_by_wr=True)

    # ── By Tier ───────────────────────────────────────────────────
    _print_section("BY TIER", by_tier, sort_by_wr=True, min_picks=2)

    # ── By Conviction Band ────────────────────────────────────────
    _print_section("BY CONVICTION BAND", by_conviction_band, sort_by_wr=False)

    # ── By BP Signal ──────────────────────────────────────────────
    _print_section("BY BETTINGPROS SIGNAL", by_bp_signal, sort_by_wr=True)

    # ── By Confidence ─────────────────────────────────────────────
    _print_section("BY MODEL CONFIDENCE", by_confidence, sort_by_wr=False)

    # ── By Risk Level ─────────────────────────────────────────────
    _print_section("BY RISK LEVEL", by_risk, sort_by_wr=False)

    # ── By Models Agreeing ────────────────────────────────────────
    _print_section("BY MODELS AGREEING", by_models_agreeing, sort_by_wr=False)

    # ── By Run Number ─────────────────────────────────────────────
    _print_section("BY RUN NUMBER (earliest appearance)", by_run_number, sort_by_wr=False)

    # ── Daily Breakdown ───────────────────────────────────────────
    print("\n" + "-" * 90)
    print("DAILY BREAKDOWN")
    print("-" * 90)
    for date in sorted(by_date.keys()):
        s = by_date[date]
        t = s["w"] + s["l"]
        print(f"  {date}: {s['w']}W-{s['l']}L ({_pct(s['w'], t)}) | {_profit_str(s['profit'])}")

    # ── Verbose: all picks ────────────────────────────────────────
    if verbose:
        print("\n" + "-" * 90)
        print("ALL GRADED PICKS")
        print("-" * 90)
        for p in sorted(all_details, key=lambda x: (x["date"], x["market"], x["player"])):
            icon = "W" if p["outcome"] == "WIN" else "L"
            conv_str = f"{p['conviction']:.0%}" if p["conviction"] else "---"
            print(
                f"  {icon} {p['date']} {p['player'][:22]:<22} "
                f"{p['market']:<10} {p['model']:<4} "
                f"O{p['line']:<5} -> {p['actual']:<5} "
                f"[{p['tier'][:16]:<16}] conv={conv_str:<5} "
                f"bp={p['bp_label'][:12]}"
            )

    print("\n" + "=" * 90)

    return {
        "total": total,
        "wins": total_w,
        "losses": total_l,
        "win_rate": total_w / total * 100 if total else 0,
        "roi": total_profit / total * 100 if total else 0,
        "profit": total_profit,
        "by_model": dict(by_model),
        "by_market": dict(by_market),
        "by_tier": dict(by_tier),
        "by_conviction_band": dict(by_conviction_band),
        "by_bp_signal": dict(by_bp_signal),
        "details": all_details,
    }


def _classify_bp_signal(ctx: dict) -> str:
    """Classify BettingPros signal for a pick."""
    bp_side = (ctx.get("bp_recommended_side") or "").lower()
    bp_rating = ctx.get("bp_bet_rating")
    bp_rec_stars = ctx.get("bp_rec_stars")
    bp_rec_side = (ctx.get("bp_rec_side") or "").lower()

    # Pick-recommendations signal (stronger)
    if bp_rec_stars:
        if bp_rec_side == "over" and bp_rec_stars >= 4:
            return "BP Aligned 4-5*"
        elif bp_rec_side == "over":
            return "BP Aligned 1-3*"
        elif bp_rec_side == "under" and bp_rec_stars >= 4:
            return "BP Conflict 4-5*"
        elif bp_rec_side == "under":
            return "BP Conflict 1-3*"

    # Cheatsheet signal (weaker)
    if bp_side:
        if bp_side == "over" and bp_rating and bp_rating >= 4:
            return "CS Aligned 4-5"
        elif bp_side == "over":
            return "CS Aligned 1-3"
        elif bp_side == "under" and bp_rating and bp_rating >= 4:
            return "CS Conflict 4-5"
        elif bp_side == "under":
            return "CS Conflict 1-3"

    return "No BP data"


def _print_section(
    title: str,
    data: dict,
    sort_by_wr: bool = True,
    min_picks: int = 1,
):
    """Print a section of the validation report."""
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    print(
        f"  {'Category':<30} {'W':<5} {'L':<5} {'Total':<7} {'Win Rate':<10} {'ROI':<10} {'Profit':<10}"
    )
    print("  " + "-" * 80)

    if sort_by_wr:
        items = sorted(
            data.items(),
            key=lambda x: x[1]["w"] / (x[1]["w"] + x[1]["l"]) if (x[1]["w"] + x[1]["l"]) > 0 else 0,
            reverse=True,
        )
    else:
        items = sorted(data.items())

    for name, s in items:
        t = s["w"] + s["l"]
        if t < min_picks:
            continue
        display = name[:28] + ".." if len(str(name)) > 30 else str(name)
        print(
            f"  {display:<30} {s['w']:<5} {s['l']:<5} {t:<7} "
            f"{_pct(s['w'], t):<10} {_roi(s['profit'], t):<10} {_profit_str(s['profit']):<10}"
        )


# ──────────────────────────────────────────────────────────────────
# Backfill write_actuals
# ──────────────────────────────────────────────────────────────────


def backfill_actuals(start_date: str, end_date: str) -> int:
    """Run write_actuals for every date in range that has ungraded picks."""
    from nba.core.axiom_writer import write_actuals

    conn = _connect_axiom()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT run_date FROM nba_prediction_history
                WHERE run_date BETWEEN %s AND %s AND is_hit IS NULL
                ORDER BY run_date
            """,
                (start_date, end_date),
            )
            dates = [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()

    if not dates:
        print("  No ungraded picks to backfill")
        return 0

    total = 0
    for d in dates:
        updated = write_actuals(d)
        if updated:
            print(f"  {d}: graded {updated} picks")
        total += updated

    return total


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DB-based pick validation")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--backfill", action="store_true", help="Backfill write_actuals before validating"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all individual picks")
    args = parser.parse_args()

    now = datetime.now(_EST)

    if args.start_date and args.end_date:
        start = args.start_date
        end = args.end_date
    else:
        end = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")

    if args.backfill:
        print(f"\n  Backfilling actuals for {start} to {end}...")
        total = backfill_actuals(start, end)
        print(f"  Backfill complete: {total} picks graded\n")

    run_validation(start, end, verbose=args.verbose)


if __name__ == "__main__":
    main()
