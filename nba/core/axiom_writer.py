"""
Axiom Writer — pipeline write interface for cephalon_axiom DB (port 5541).

Used by the Airflow pipeline to record prediction history and audit runs.
Separate from axiom_db.py, which is read-only and used by the bot/Atlas.

All functions are fire-and-forget: they log on failure but never raise,
so a DB issue never propagates back to fail the pipeline.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("nba.axiom_writer")

_AXIOM_PORT = 5541
_AXIOM_DB = "cephalon_axiom"
_CONNECT_TIMEOUT = 5
_EST = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────
# Run number
# ─────────────────────────────────────────────────────────────────

# Scheduled runs: (EST hour, run_number)
# Full pipeline: 2 AM EST (run 1)
# Refresh runs:  5 AM, 8 AM, 11 AM, 2 PM, 5 PM EST (runs 2-6)
_SCHEDULED_RUNS = [(2, 1), (5, 2), (8, 3), (11, 4), (14, 5), (17, 6)]

# Runs are 3 hours (180 min) apart — a ±90 min window covers each slot
# with no overlap and no gap, so manual triggers near a window land correctly.
_SLOT_WINDOW_MINUTES = 90


def get_run_number() -> int:
    """Derive run_number (1-6) from current EST time.

    Uses a ±90-minute window around each scheduled EST hour so that manual
    triggers fired close to a scheduled run are assigned that run's slot
    rather than defaulting to 1 and corrupting the history.

    Returns 1 if called entirely outside any scheduled window.
    """
    now = datetime.now(_EST)
    current_minutes = now.hour * 60 + now.minute
    for sched_hour, run_num in _SCHEDULED_RUNS:
        if abs(current_minutes - sched_hour * 60) <= _SLOT_WINDOW_MINUTES:
            return run_num
    return 1


# ─────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────


def _connect():
    """Open a write connection to cephalon_axiom. Caller must close."""
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=_AXIOM_PORT,
        dbname=_AXIOM_DB,
        user=os.environ.get("DB_USER", "mlb_user"),
        password=os.environ.get("DB_PASSWORD", ""),
        connect_timeout=_CONNECT_TIMEOUT,
    )


# ─────────────────────────────────────────────────────────────────
# Pipeline audit
# ─────────────────────────────────────────────────────────────────


def audit_run_start(run_date: str, run_number: int, run_type: str) -> bool:
    """Record that a pipeline run has started.

    Inserts or resets a row in axiom_pipeline_audit with status='running'.
    Safe to call multiple times (ON CONFLICT resets the row).

    Args:
        run_date:   YYYY-MM-DD string
        run_number: 1 (full) or 2-6 (refresh)
        run_type:   'full' or 'refresh'
    """
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO axiom_pipeline_audit
                        (run_date, run_number, run_type, run_timestamp, status)
                    VALUES (%s, %s, %s, NOW(), 'running')
                    ON CONFLICT (run_date, run_number) DO UPDATE
                        SET run_type      = EXCLUDED.run_type,
                            run_timestamp = NOW(),
                            status        = 'running',
                            error_message = NULL,
                            error_traceback = NULL
                    """,
                    (run_date, run_number, run_type),
                )
        conn.close()
        log.info(f"Axiom audit started: run {run_number} ({run_type}) for {run_date}")
        return True
    except Exception as e:
        log.warning(f"axiom audit_run_start failed (non-critical): {e}")
        return False


def audit_run_complete(
    run_date: str,
    run_number: int,
    status: str,
    *,
    props_fetched: Optional[int] = None,
    books_available: Optional[int] = None,
    injuries_updated: Optional[bool] = None,
    games_found: Optional[int] = None,
    duration_seconds: Optional[int] = None,
    picks_generated: Optional[int] = None,
    xl_picks: Optional[int] = None,
    v3_picks: Optional[int] = None,
    error_message: Optional[str] = None,
    error_traceback: Optional[str] = None,
    anomalies: Optional[dict] = None,
) -> bool:
    """Update the audit row with the final result of a pipeline run.

    Args:
        run_date, run_number: identify the row
        status: 'success', 'partial', or 'failed'
        All other kwargs fill the corresponding columns.
    """
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE axiom_pipeline_audit SET
                        status            = %s,
                        props_fetched     = %s,
                        books_available   = %s,
                        injuries_updated  = %s,
                        games_found       = %s,
                        duration_seconds  = %s,
                        picks_generated   = %s,
                        xl_picks          = %s,
                        v3_picks          = %s,
                        error_message     = %s,
                        error_traceback   = %s,
                        anomalies         = %s
                    WHERE run_date = %s AND run_number = %s
                    """,
                    (
                        status,
                        props_fetched,
                        books_available,
                        injuries_updated,
                        games_found,
                        duration_seconds,
                        picks_generated,
                        xl_picks,
                        v3_picks,
                        error_message,
                        error_traceback,
                        json.dumps(anomalies) if anomalies else None,
                        run_date,
                        run_number,
                    ),
                )
        conn.close()
        log.info(f"Axiom audit complete: run {run_number} → {status} ({picks_generated} picks)")
        return True
    except Exception as e:
        log.warning(f"axiom audit_run_complete failed (non-critical): {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Prediction history
# ─────────────────────────────────────────────────────────────────


def _build_context_snapshot(pick: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields from a pick dict that the conviction engine needs.

    These are things the model already computed that we don't want to lose.
    BettingPros streak/trend data will be merged into this snapshot when available.

    Fields left as None/absent are intentional placeholders for future enrichment.
    """
    hr = pick.get("hit_rates") or {}
    l5 = hr.get("last_5") or {}
    l15 = hr.get("last_15") or {}
    season = hr.get("season") or {}
    bp = pick.get("bp_intel") or {}

    ctx: dict[str, Any] = {
        # Model agreement — key for conviction: did both XL and V3 see this?
        "models_agreeing": pick.get("models_agreeing", [pick.get("model_version", "xl")]),
        # Risk filter output — already computed by the risk engine
        "risk_level": pick.get("risk_level"),
        "risk_flags": pick.get("risk_flags", []),
        # Player momentum
        "trend": pick.get("player_context", {}).get("trend"),
        # Model confidence label (HIGH / MEDIUM / STANDARD)
        "confidence": pick.get("confidence"),
        # Line context — needed to track consensus movement separately from best_line
        "consensus_line": pick.get("consensus_line"),
        # Which tier of PrizePicks line this is (goblin / demon / standard book)
        # Tracks across runs: goblin→demon = model got more confident
        "best_book_type": pick.get("best_book"),
        # Kelly sizing — carry this through to the conviction card, don't recompute
        "recommended_stake": pick.get("recommended_stake"),
        "stake_reason": pick.get("stake_reason"),
        # Filter tier (Goldmine, X, Z, META, etc.)
        "filter_tier": pick.get("filter_tier"),
        # Historical hit rates — currently 50% defaults; will be replaced by
        # BettingPros /v3/props/streaks when that enrichment is added
        "hit_rate_L5": l5.get("rate"),
        "hit_rate_L15": l15.get("rate"),
        "hit_rate_season": season.get("rate"),
        # BettingPros intelligence — populated from bp_intel in the pick dict
        # streak: how many games in a row OVER/UNDER this line
        # bp_projection: BP's own model projected value (independent second opinion)
        # bp_probability: BP's p_over (compare to our p_over for agreement check)
        # bp_bet_rating: 1-5 stars from BettingPros
        # opposition_rank: defense rank (1=hardest, 30=easiest)
        "bp_streak": bp.get("streak"),
        "bp_streak_type": bp.get("streak_type"),
        "bp_projection": bp.get("bp_projection"),
        "bp_probability": bp.get("bp_probability"),
        "bp_bet_rating": bp.get("bp_bet_rating"),
        "bp_recommended_side": bp.get("bp_recommended_side"),
        "opposition_rank": bp.get("opposition_rank"),
    }

    return ctx


def write_picks(
    run_date: str,
    run_number: int,
    run_timestamp: str,
    picks: list[dict[str, Any]],
) -> int:
    """Insert picks into nba_prediction_history.

    One row per pick. Skips duplicates silently (ON CONFLICT DO NOTHING).
    Returns count of rows inserted.

    Args:
        run_date:      YYYY-MM-DD
        run_number:    1-6
        run_timestamp: ISO timestamp string of when the run produced these picks
        picks:         list of pick dicts from generate_xl_predictions output
    """
    if not picks:
        return 0

    rows = []
    for pick in picks:
        ctx = _build_context_snapshot(pick)
        rows.append(
            (
                run_date,
                run_number,
                run_timestamp,
                pick.get("player_name"),
                pick.get("stat_type"),
                pick.get("model_version", "xl"),
                pick.get("filter_tier"),
                pick.get("best_line"),
                pick.get("p_over"),
                pick.get("edge"),
                pick.get("line_spread"),
                pick.get("best_book"),
                None,  # game_time — not in pick dict currently
                pick.get("opponent_team"),
                pick.get("is_home"),
                json.dumps(ctx) if ctx else None,
            )
        )

    try:
        import psycopg2.extras

        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO nba_prediction_history
                        (run_date, run_number, run_timestamp,
                         player_name, stat_type, model_version, tier,
                         line, p_over, edge, spread, book,
                         game_time, opponent_team, is_home, context_snapshot)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    rows,
                )
                inserted = cur.rowcount
        conn.close()
        log.info(f"Axiom: wrote {inserted} picks to nba_prediction_history (run {run_number})")
        return inserted
    except Exception as e:
        log.warning(f"axiom write_picks failed (non-critical): {e}")
        return 0


# ─────────────────────────────────────────────────────────────────
# Convenience: count today's props in intel DB
# ─────────────────────────────────────────────────────────────────


def write_actuals(run_date: str) -> int:
    """Match nba_prediction_history picks against player_game_logs and write is_hit.

    Runs after games complete (called by nba_validation_pipeline).
    Updates ALL run_number rows for each (run_date, player_name, stat_type) combo.

    Returns count of picks updated. Returns 0 on error (never raises).
    """
    _STAT_TO_COL = {
        "POINTS": "points",
        "REBOUNDS": "rebounds",
        "ASSISTS": "assists",
        "THREES": "three_pointers_made",
    }

    try:
        import psycopg2

        from nba.config.database import get_players_db_config

        # ── 1. Load picks that need grading ──────────────────────────────
        axiom_conn = _connect()
        try:
            with axiom_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT player_name, stat_type, line
                    FROM nba_prediction_history
                    WHERE run_date = %s AND is_hit IS NULL
                    """,
                    (run_date,),
                )
                picks_to_grade = cur.fetchall()
        finally:
            axiom_conn.close()

        if not picks_to_grade:
            log.info(f"write_actuals: no ungraded picks for {run_date}")
            return 0

        # ── 2. Load actual game stats from player_game_logs ──────────────
        players_conn = psycopg2.connect(**get_players_db_config())
        try:
            with players_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pp.full_name,
                           pgl.points, pgl.rebounds, pgl.assists,
                           pgl.three_pointers_made
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE pgl.game_date = %s
                    """,
                    (run_date,),
                )
                rows = cur.fetchall()
        finally:
            players_conn.close()

        # Build lookup: lower(name) -> {stat: value}
        actuals: dict[str, dict[str, float]] = {}
        for full_name, pts, reb, ast, threes in rows:
            actuals[full_name.lower()] = {
                "POINTS": float(pts or 0),
                "REBOUNDS": float(reb or 0),
                "ASSISTS": float(ast or 0),
                "THREES": float(threes or 0),
            }

        if not actuals:
            log.info(f"write_actuals: no game logs found for {run_date}")
            return 0

        # ── 3. Match and update ───────────────────────────────────────────
        updated = 0
        axiom_conn = _connect()
        try:
            with axiom_conn:
                with axiom_conn.cursor() as cur:
                    for player_name, stat_type, line in picks_to_grade:
                        stat_data = actuals.get(player_name.lower())
                        if stat_data is None:
                            continue
                        actual_val = stat_data.get(stat_type)
                        if actual_val is None:
                            continue

                        is_hit = actual_val > float(line)
                        cur.execute(
                            """
                            UPDATE nba_prediction_history
                               SET actual_result      = %s,
                                   is_hit             = %s,
                                   result_source      = 'game_logs',
                                   result_recorded_at = NOW()
                             WHERE run_date    = %s
                               AND player_name = %s
                               AND stat_type   = %s
                               AND is_hit IS NULL
                            """,
                            (actual_val, is_hit, run_date, player_name, stat_type),
                        )
                        updated += cur.rowcount
        finally:
            axiom_conn.close()

        log.info(f"write_actuals: updated {updated} rows for {run_date}")
        return updated

    except Exception as e:
        log.warning(f"write_actuals failed (non-critical): {e}")
        return 0


def count_todays_props(run_date: str) -> tuple[int, int]:
    """Query nba_intelligence for today's prop count and book count.

    Returns (props_fetched, books_available). Returns (0, 0) on error.
    """
    try:
        import psycopg2

        from nba.config.database import get_intelligence_db_config

        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS props, COUNT(DISTINCT book_name) AS books
                    FROM nba_props_xl
                    WHERE game_date = %s
                    """,
                    (run_date,),
                )
                row = cur.fetchone()
                return (row[0] or 0, row[1] or 0) if row else (0, 0)
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"count_todays_props failed (non-critical): {e}")
        return (0, 0)
