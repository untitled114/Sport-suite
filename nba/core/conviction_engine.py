"""
Conviction Engine — computes conviction scores for today's picks.

Runs after each pipeline refresh. Reads nba_prediction_history for today,
groups by (player_name, stat_type), and computes a 0-1 conviction score
that reflects how confident Axiom should be that this pick is actionable.

Design principles:
- Never amplify a weak signal — be conservative with LOCKED labels
- Be specific in narratives — mention actual counts and numbers
- Never rerun the ML model — only observe what the pipeline produced
- All scoring is transparent and explainable

Conviction is NOT a second ML model. It's an audit of pick consistency.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("nba.conviction_engine")

_AXIOM_PORT = 5541
_AXIOM_DB = "cephalon_axiom"
_CONNECT_TIMEOUT = 5

# ─────────────────────────────────────────────────────────────────
# Scoring weights — must sum to 1.0 (base score)
# Additive bonuses applied on top: model agreement, BP rec signal, tier
# ─────────────────────────────────────────────────────────────────
_W_APPEARANCE = 0.40  # Did this pick survive every run? (primary signal)
_W_STABILITY = 0.25  # How consistent was p_over across runs?
_W_TREND = 0.20  # Is the model getting more confident over the day?
_W_LINE = 0.15  # Is the line holding or softening?

# BP pick-recommendations bonus thresholds (third independent signal)
# Applied when bp_pick_recommendations_{date}.json has this pick
_BP_REC_STAR_BONUS = {5: 0.07, 4: 0.05, 3: 0.02}  # per-star-tier bonus
_BP_REC_MARKET_EV_BONUS = 0.04  # extra when best book also has positive EV (>= 0.05)
_BP_REC_CONFLICT_PENALTY = -0.06  # BP says UNDER while we say OVER (4-5 star conflict)

# Minimum appearances before LOCKED is reachable
_MIN_APPEARANCES_FOR_LOCKED = 3

# Line movement threshold — changes smaller than this are noise, not signal
_LINE_MOVEMENT_NOISE = 0.25


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Clip and normalize value to [0, 1]."""
    if max_val <= min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _connect():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=_AXIOM_PORT,
        dbname=_AXIOM_DB,
        user=os.environ.get("DB_USER", "mlb_user"),
        password=os.environ.get("DB_PASSWORD", ""),
        connect_timeout=_CONNECT_TIMEOUT,
    )


# ─────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────


def _score_appearance(appearances: int, entry_run: int, total_runs: int) -> float:
    """Fraction of possible runs (since first appearance) that included this pick.

    A pick seen in all 6 runs scores higher than one that showed up late.
    Entry penalty: appearing first at run 4 of 6 is weaker than run 1 of 6.
    """
    possible_runs = total_runs - entry_run + 1
    if possible_runs <= 0:
        return 0.0
    consistency = appearances / possible_runs

    # Entry bonus: first seen at run 1 gets full score; run 5 gets 60%
    entry_factor = 1.0 - (entry_run - 1) * 0.08  # -8% per delayed run
    entry_factor = max(0.60, entry_factor)

    return min(1.0, consistency * entry_factor)


def _score_stability(p_over_std: float) -> float:
    """Lower std = more stable signal = higher score.

    0.00 std → 1.0 (perfectly consistent)
    0.10 std → 0.33 (model is unsure)
    0.15+ std → 0.0  (model is all over the place)

    Guard: if only 1 run, std is 0 by definition — this is misleading.
    Caller should cap conviction early in the day regardless.
    """
    return _normalize(-p_over_std, -0.15, 0.0)


def _score_trend(p_over_trend: float) -> float:
    """Is the model getting more confident over the day?

    +0.10 trend → 1.0 (model strengthening, very good)
     0.00 trend → 0.5 (model stable, neutral)
    -0.10 trend → 0.0 (model weakening, concerning)
    """
    return _normalize(p_over_trend, -0.10, 0.10)


def _score_line_movement(line_movement: float) -> float:
    """Is the line holding or softening?

    line_movement = line_latest - line_at_entry
    Negative = line dropped (softened, more value) → score above 0.5
    Zero     = line held → score slightly above 0.5 (book hasn't adjusted = overlooked)
    Positive = line rose (sharpened, value eroding) → score below 0.5

    Noise filter: movements < 0.25 pts treated as neutral.
    """
    if abs(line_movement) < _LINE_MOVEMENT_NOISE:
        return 0.55  # Stable line: slight positive (book hasn't caught on)
    return _normalize(-line_movement, -1.5, 1.5)


def _get_label(conviction: float, appearances: int) -> str:
    if appearances < 2:
        return "EARLY"
    if conviction >= 0.80 and appearances >= _MIN_APPEARANCES_FOR_LOCKED:
        return "LOCKED"
    if conviction >= 0.60:
        return "STRONG"
    if conviction >= 0.40:
        return "WATCH"
    return "SKIP"


def _get_status(is_active: bool, line_movement: float, p_over_trend: float) -> str:
    """Classify the pick's lifecycle status."""
    if is_active:
        return "active"
    # Disappeared — why?
    if line_movement > 1.0:
        return "evaporated"  # Line rose significantly — value gone
    if p_over_trend < -0.10:
        return "evaporated"  # Model lost confidence significantly
    return "dropped"  # Disappeared without clear cause (injury? news?)


# ─────────────────────────────────────────────────────────────────
# Narrative generation
# ─────────────────────────────────────────────────────────────────


def _build_narrative(
    player_name: str,
    stat_type: str,
    appearances: int,
    total_runs: int,
    entry_run: int,
    run_pattern: str,
    line_at_entry: Optional[float],
    line_latest: Optional[float],
    line_direction: str,
    p_over_at_entry: Optional[float],
    p_over_latest: Optional[float],
    book_latest: Optional[str],
    status: str,
    conviction_label: str,
    conviction: float,
    context: Optional[dict],
    tier_note: str = "",
) -> str:
    """Build a plain English narrative. Be specific, not promotional.

    This is read by Atlas and Claude on escalation — it must be accurate.
    """
    parts = []

    # Appearance summary
    if appearances == total_runs:
        parts.append(f"Present all {total_runs} runs.")
    elif entry_run > 1:
        parts.append(
            f"Appeared run {entry_run}–{entry_run + appearances - 1} ({appearances}/{total_runs} runs)."
        )
    else:
        parts.append(f"{appearances}/{total_runs} runs (pattern: {run_pattern}).")

    # Line movement
    if line_at_entry is not None and line_latest is not None:
        movement = line_latest - line_at_entry
        if abs(movement) < _LINE_MOVEMENT_NOISE:
            parts.append(f"Line held at {line_latest} @ {book_latest or 'unknown'}.")
        elif movement < 0:
            parts.append(
                f"Line softened {line_at_entry}→{line_latest} (-{abs(movement):.1f}). "
                f"More value now."
            )
        else:
            parts.append(
                f"Line rose {line_at_entry}→{line_latest} (+{movement:.1f}). "
                f"Sharp money — value eroding."
            )

    # Signal strength
    if p_over_at_entry is not None and p_over_latest is not None:
        trend = p_over_latest - p_over_at_entry
        if abs(trend) < 0.02:
            parts.append(f"p_over stable ({p_over_latest:.3f}).")
        elif trend > 0:
            parts.append(
                f"Model strengthening: {p_over_at_entry:.3f}→{p_over_latest:.3f} (+{trend:.3f})."
            )
        else:
            parts.append(
                f"Model weakening: {p_over_at_entry:.3f}→{p_over_latest:.3f} ({trend:.3f})."
            )

    # Status flags
    if status == "evaporated":
        parts.append("EVAPORATED — no longer in pipeline output.")
    elif status == "dropped":
        parts.append("DROPPED — disappeared without clear cause (check injuries/news).")

    # BettingPros intel (from hit-rate enrichment)
    ctx = context or {}
    if ctx.get("bp_streak") is not None:
        streak_dir = ctx.get("bp_streak_type", "")
        parts.append(f"BP streak: {ctx['bp_streak']} {streak_dir}.")

    if ctx.get("bp_bet_rating") is not None:
        rating = ctx["bp_bet_rating"]
        bp_side = ctx.get("bp_recommended_side", "?")
        parts.append(f"BP rating: {rating}/5 ({bp_side}).")

    # BP pick-recommendations signal (independent third signal)
    bp_stars = ctx.get("bp_rec_stars")
    bp_rec_side = ctx.get("bp_rec_side", "")
    bp_rec_ev = ctx.get("bp_rec_ev")
    bp_market_ev = ctx.get("bp_rec_market_ev")
    bp_rec_book = ctx.get("bp_rec_best_book")

    if bp_stars is not None:
        star_str = "★" * bp_stars
        ev_str = f" EV+{bp_rec_ev:.0%}" if bp_rec_ev and bp_rec_ev > 0 else ""
        market_str = (
            f" (market EV {bp_market_ev:+.1%} @ {bp_rec_book})"
            if bp_market_ev and bp_rec_book
            else ""
        )
        bp_proj = ctx.get("bp_rec_projection")
        bp_proj_str = f" (BP projects {bp_proj})" if bp_proj is not None else ""
        if bp_rec_side == "over":
            parts.append(f"BP rec: {star_str} OVER{ev_str}{market_str}.")
        elif bp_rec_side == "under":
            conflict_type = ctx.get("bp_rec_conflict_type", "genuine")
            if conflict_type == "odds_play":
                parts.append(
                    f"BP rec: {star_str} UNDER vs their consensus{bp_proj_str} "
                    f"(odds-value play — not priced vs our line)."
                )
            else:
                parts.append(f"BP rec: {star_str} UNDER{bp_proj_str} — CONFLICTS with our OVER.")

    # Risk context
    risk = ctx.get("risk_level")
    risk_flags = ctx.get("risk_flags") or []
    if risk == "HIGH" or risk_flags:
        flag_str = ", ".join(risk_flags) if risk_flags else ""
        parts.append(f"Risk: {risk or 'UNKNOWN'}{' [' + flag_str + ']' if flag_str else ''}.")

    # Filter performance (only shown when enough historical data exists)
    if tier_note:
        parts.append(f"Filter: {tier_note}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Tier performance (measures whether our filters are actually working)
# ─────────────────────────────────────────────────────────────────

_TIER_MIN_GRADED = 8  # Need at least this many graded picks before trusting the rate
_TIER_UNDERPERFORM_THRESHOLD = 0.47  # Below this WR → soft cap at 0.70
_TIER_STRONG_THRESHOLD = 0.63  # Above this WR → small bonus


def _get_tier_stats(conn, lookback_days: int = 14) -> dict[str, dict]:
    """Query recent graded picks and return win rates by tier.

    Only uses picks with is_hit IS NOT NULL (written by write_actuals()).
    Returns empty dict if no graded data yet — engine degrades gracefully.

    Format: {tier: {"wins": N, "losses": N, "total": N, "win_rate": float}}
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    tier,
                    COUNT(*)                                    AS total,
                    COUNT(*) FILTER (WHERE is_hit = true)       AS wins,
                    COUNT(*) FILTER (WHERE is_hit = false)      AS losses
                FROM nba_prediction_history
                WHERE is_hit IS NOT NULL
                  AND run_date >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY tier
                HAVING COUNT(*) >= %s
                """,
                (lookback_days, _TIER_MIN_GRADED),
            )
            rows = cur.fetchall()
    except Exception as e:
        log.debug(f"_get_tier_stats failed: {e}")
        return {}

    result = {}
    for row in rows:
        tier = row["tier"] or "unknown"
        total = int(row["total"])
        wins = int(row["wins"])
        losses = int(row["losses"])
        result[tier] = {
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": round(wins / total, 3) if total > 0 else 0.0,
        }
    return result


def _tier_score_adjustment(tier: Optional[str], tier_stats: dict) -> tuple[float, str]:
    """Return (score_delta, narrative_fragment) based on historical tier performance.

    Returns (0.0, "") when not enough data — no adjustment until filters prove themselves.
    """
    if not tier or tier not in tier_stats:
        return 0.0, ""

    stats = tier_stats[tier]
    wr = stats["win_rate"]
    total = stats["total"]
    wins = stats["wins"]

    label = f"{tier} L14: {wins}/{total} ({wr:.1%})"

    if wr < _TIER_UNDERPERFORM_THRESHOLD:
        return -0.08, f"{label} — UNDERPERFORMING"
    elif wr > _TIER_STRONG_THRESHOLD:
        return +0.04, f"{label} — performing well"
    else:
        return 0.0, label


# ─────────────────────────────────────────────────────────────────
# BP pick-recommendations signal (third independent signal)
# ─────────────────────────────────────────────────────────────────


def _load_bp_recommendations(run_date: str) -> dict[tuple, dict]:
    """Load BP pick recommendations for today.

    Returns {(player_name_lower, stat_type): record} lookup.
    Returns empty dict silently if file not found — engine degrades gracefully.
    """
    lines_dir = Path(__file__).parent.parent / "betting_xl" / "lines"
    bp_file = lines_dir / f"bp_pick_recommendations_{run_date}.json"

    if not bp_file.exists():
        log.debug(f"BP recommendations file not found: {bp_file}")
        return {}

    try:
        with open(bp_file) as f:
            data = json.load(f)
        lookup: dict[tuple, dict] = {}
        for pick in data.get("picks", []):
            player = (pick.get("player_name") or "").lower().strip()
            stat = (pick.get("stat_type") or "").upper()
            if player and stat:
                lookup[(player, stat)] = pick
        return lookup
    except Exception as e:
        log.debug(f"Failed to load BP recommendations: {e}")
        return {}


def _score_bp_rec(
    bp_rec: Optional[dict], our_best_line: Optional[float] = None
) -> tuple[float, dict]:
    """Compute the BP recommendation bonus/penalty and supplemental context fields.

    Returns (score_delta, extra_ctx) where extra_ctx is merged into context_snapshot
    so the daily card builder and narratives can reference BP signal details.

    Agreement (BP says OVER):
        5★ + market EV ≥ 0.05 → +0.11 (strong independent confirmation)
        5★ only            → +0.07
        4★ + market EV     → +0.09
        4★ only            → +0.05
        3★ + market EV     → +0.06
        3★ only            → +0.02

    Conflict (BP says UNDER, 4-5★):
        Genuine conflict: bp_projection ≥ our_best_line - 1.5  → -0.06
        Odds play:        bp_projection <  our_best_line - 1.5  →  0.0
            BP is pricing vs their consensus (e.g., -108 UNDER has better EV
            than -128 OVER at a 52% prop). Their UNDER is not a directional
            call against our higher demon/goblin line.

    No match / BP absent → 0.0 (no effect, no penalty for absence)
    """
    if not bp_rec:
        return 0.0, {}

    stars = bp_rec.get("bp_bet_rating") or 0
    rec_side = (bp_rec.get("bp_recommended_side") or "").lower()
    model_ev = bp_rec.get("bp_expected_value") or 0.0
    market_ev = bp_rec.get("best_ev") or 0.0
    best_book = bp_rec.get("best_book")
    bp_projection = bp_rec.get("bp_projection")

    extra_ctx = {
        "bp_rec_stars": stars,
        "bp_rec_side": rec_side,
        "bp_rec_ev": round(float(model_ev), 4) if model_ev else None,
        "bp_rec_market_ev": round(float(market_ev), 4) if market_ev else None,
        "bp_rec_best_book": best_book,
        "bp_rec_projection": bp_projection,
    }

    if rec_side == "over":
        star_bonus = _BP_REC_STAR_BONUS.get(stars, 0.0)
        market_bonus = _BP_REC_MARKET_EV_BONUS if market_ev >= 0.05 else 0.0
        delta = min(0.11, star_bonus + market_bonus)
    elif rec_side == "under" and stars >= 4:
        # Only penalize when BP's projection is close to our target line.
        # If bp_projection is well below our line, BP is pricing vs their
        # consensus (odds-value play), not making a directional call against us.
        is_genuine_conflict = (
            our_best_line is None
            or bp_projection is None
            or float(bp_projection) >= our_best_line - 1.5
        )
        if is_genuine_conflict:
            delta = _BP_REC_CONFLICT_PENALTY
            extra_ctx["bp_rec_conflict_type"] = "genuine"
        else:
            delta = 0.0
            extra_ctx["bp_rec_conflict_type"] = "odds_play"
    else:
        delta = 0.0

    return delta, extra_ctx


# ─────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────


def compute_conviction(run_date: str, run_number: int) -> int:
    """Compute conviction scores for all picks seen today up to run_number.

    Reads nba_prediction_history, groups by (player_name, stat_type),
    computes scores, and upserts into axiom_conviction.

    Returns count of conviction rows written.
    """
    try:
        conn = _connect()
    except Exception as e:
        log.warning(f"conviction_engine: cannot connect to axiom DB: {e}")
        return 0

    try:
        # ── Fetch today's prediction history ──────────────────────────
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    player_name, stat_type, model_version,
                    run_number, line, p_over, edge, spread, book,
                    context_snapshot
                FROM nba_prediction_history
                WHERE run_date = %s AND run_number <= %s
                ORDER BY player_name, stat_type, run_number
                """,
                (run_date, run_number),
            )
            rows = cur.fetchall()

        if not rows:
            log.info(f"conviction_engine: no picks found for {run_date} up to run {run_number}")
            conn.close()
            return 0

        # ── Tier performance stats (fetched once, applied per pick) ───
        # Empty dict until enough graded picks accumulate — engine degrades gracefully
        tier_stats = _get_tier_stats(conn)
        if tier_stats:
            log.info(f"conviction_engine: tier stats loaded — {list(tier_stats.keys())}")

        # ── BP pick recommendations (third independent signal) ─────────
        bp_recs = _load_bp_recommendations(run_date)
        if bp_recs:
            log.info(f"conviction_engine: {len(bp_recs)} BP recommendations loaded")

        # ── Group by (player_name, stat_type) ─────────────────────────
        groups: dict[tuple, list] = {}
        for row in rows:
            key = (row["player_name"], row["stat_type"])
            groups.setdefault(key, []).append(dict(row))

        conviction_rows = []

        for (player_name, stat_type), pick_rows in groups.items():
            pick_rows.sort(key=lambda r: r["run_number"])

            run_numbers = [r["run_number"] for r in pick_rows]
            p_overs = [float(r["p_over"]) for r in pick_rows if r["p_over"] is not None]
            lines = [float(r["line"]) for r in pick_rows if r["line"] is not None]

            if not p_overs:
                continue

            appearances = len(pick_rows)
            entry_run = run_numbers[0]
            last_seen_run = run_numbers[-1]
            is_active = last_seen_run == run_number
            run_pattern = ",".join(str(n) for n in run_numbers)

            p_over_entry = p_overs[0]
            p_over_latest = p_overs[-1]
            p_over_trend = p_over_latest - p_over_entry
            import statistics

            p_over_std = statistics.stdev(p_overs) if len(p_overs) > 1 else 0.0

            line_entry = lines[0] if lines else None
            line_latest = lines[-1] if lines else None
            line_movement = (line_latest - line_entry) if (line_entry and line_latest) else 0.0
            line_direction = (
                "rising"
                if line_movement > _LINE_MOVEMENT_NOISE
                else "falling" if line_movement < -_LINE_MOVEMENT_NOISE else "stable"
            )

            book_latest = pick_rows[-1].get("book")

            # Context snapshot from most recent run
            ctx = pick_rows[-1].get("context_snapshot") or {}
            if isinstance(ctx, str):
                import json

                try:
                    ctx = json.loads(ctx)
                except Exception:
                    ctx = {}

            # ── Score the four components ──────────────────────────────
            s_appearance = _score_appearance(appearances, entry_run, run_number)
            s_stability = _score_stability(p_over_std)
            s_trend = _score_trend(p_over_trend)
            s_line = _score_line_movement(line_movement)

            conviction = (
                _W_APPEARANCE * s_appearance
                + _W_STABILITY * s_stability
                + _W_TREND * s_trend
                + _W_LINE * s_line
            )

            # Model agreement bonus: both XL and V3 see this pick
            models_agreeing = ctx.get("models_agreeing") or []
            if len(models_agreeing) > 1:
                conviction = min(1.0, conviction + 0.05)

            # BettingPros hit-rate signal: BP rates the pick from cheatsheet/hit-rate data
            bp_rating = ctx.get("bp_bet_rating")
            bp_side = ctx.get("bp_recommended_side")
            if bp_rating is not None and bp_rating >= 4 and bp_side == "over":
                conviction = min(1.0, conviction + 0.03)

            # BP pick-recommendations signal: independent third opinion with real EV data
            bp_rec_key = (player_name.lower(), stat_type.upper())
            bp_rec = bp_recs.get(bp_rec_key)
            bp_delta, bp_extra_ctx = _score_bp_rec(bp_rec, our_best_line=line_latest)
            if bp_delta != 0.0:
                conviction = max(0.0, min(1.0, conviction + bp_delta))
            # Merge BP rec fields into context so daily card and narrative see them
            if bp_extra_ctx:
                ctx = {**ctx, **bp_extra_ctx}

            # Tier performance adjustment — only when enough history exists
            tier = ctx.get("filter_tier")
            tier_delta, tier_note = _tier_score_adjustment(tier, tier_stats)
            if tier_delta != 0.0:
                conviction = max(0.0, min(1.0, conviction + tier_delta))

            # Early-day cap: with only 1 run, don't let std=0 inflate the score
            if run_number == 1 or appearances < 2:
                conviction = min(conviction, 0.62)

            label = _get_label(conviction, appearances)
            status = _get_status(is_active, line_movement, p_over_trend)

            narrative = _build_narrative(
                player_name=player_name,
                stat_type=stat_type,
                appearances=appearances,
                total_runs=run_number,
                entry_run=entry_run,
                run_pattern=run_pattern,
                line_at_entry=line_entry,
                line_latest=line_latest,
                line_direction=line_direction,
                p_over_at_entry=p_over_entry,
                p_over_latest=p_over_latest,
                book_latest=book_latest,
                status=status,
                conviction_label=label,
                conviction=conviction,
                context=ctx,
                tier_note=tier_note,
            )

            conviction_rows.append(
                (
                    run_date,
                    player_name,
                    stat_type,
                    round(conviction, 4),
                    label,
                    appearances,
                    run_number,
                    entry_run,
                    last_seen_run,
                    is_active,
                    run_pattern,
                    line_entry,
                    line_latest,
                    round(line_movement, 2) if line_movement is not None else None,
                    line_direction,
                    book_latest,
                    p_over_entry,
                    p_over_latest,
                    round(p_over_trend, 4),
                    round(p_over_std, 4),
                    status,
                    psycopg2.extras.Json(ctx) if ctx else None,
                    narrative,
                )
            )

        if not conviction_rows:
            conn.close()
            return 0

        # ── Upsert conviction rows ─────────────────────────────────────
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO axiom_conviction (
                        run_date, player_name, stat_type,
                        conviction, conviction_label,
                        appearances, total_runs,
                        entry_run, last_seen_run, is_active, run_pattern,
                        line_at_entry, line_latest, line_movement, line_direction,
                        book_latest,
                        p_over_at_entry, p_over_latest, p_over_trend, p_over_std,
                        status, context_snapshot, narrative
                    ) VALUES %s
                    ON CONFLICT (run_date, player_name, stat_type) DO UPDATE SET
                        conviction       = EXCLUDED.conviction,
                        conviction_label = EXCLUDED.conviction_label,
                        appearances      = EXCLUDED.appearances,
                        total_runs       = EXCLUDED.total_runs,
                        last_seen_run    = EXCLUDED.last_seen_run,
                        is_active        = EXCLUDED.is_active,
                        run_pattern      = EXCLUDED.run_pattern,
                        line_latest      = EXCLUDED.line_latest,
                        line_movement    = EXCLUDED.line_movement,
                        line_direction   = EXCLUDED.line_direction,
                        book_latest      = EXCLUDED.book_latest,
                        p_over_latest    = EXCLUDED.p_over_latest,
                        p_over_trend     = EXCLUDED.p_over_trend,
                        p_over_std       = EXCLUDED.p_over_std,
                        status           = EXCLUDED.status,
                        context_snapshot = EXCLUDED.context_snapshot,
                        narrative        = EXCLUDED.narrative,
                        computed_at      = NOW()
                    """,
                    conviction_rows,
                )

        conn.close()
        log.info(
            f"conviction_engine: wrote {len(conviction_rows)} conviction rows "
            f"for {run_date} run {run_number}"
        )
        return len(conviction_rows)

    except Exception as e:
        log.error(f"conviction_engine failed: {e}", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return 0
