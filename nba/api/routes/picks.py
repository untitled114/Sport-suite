"""
Picks Endpoint
==============
Serves today's (or any date's) generated picks to Lunara for pick sync.
Also accepts pick result callbacks from Lunara (feedback loop).
Authentication: Bearer token via LUNARA_API_KEY env var.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/picks", tags=["Picks"])

_PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "betting_xl" / "predictions"

# NAMESPACE_DNS — agreed with Lunara Claude via claude-to-claude.txt
_PICK_UUID_NAMESPACE = uuid.NAMESPACE_DNS

_LUNARA_API_KEY = os.getenv("LUNARA_API_KEY", "")

# DB config from environment (same pattern as rest of codebase)
_DB_USER = os.getenv("DB_USER", "mlb_user")
_DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def _verify_token(authorization: str | None) -> None:
    if not _LUNARA_API_KEY:
        return
    if authorization != f"Bearer {_LUNARA_API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization header",
        )


def _pick_id(player_name: str, game_date: str, stat_type: str) -> str:
    """Deterministic UUID5 — stable across pipeline runs. Namespace: NAMESPACE_DNS."""
    return str(uuid.uuid5(_PICK_UUID_NAMESPACE, f"{player_name}:{game_date}:{stat_type}"))


# ==============================================================================
# DB Enrichment
# ==============================================================================


def _get_rolling_stats(player_name: str, stat_type: str) -> Optional[dict]:
    """Compute rolling averages from player_game_logs."""
    from nba.config.database import get_players_db_config

    stat_col_map = {
        "POINTS": "points",
        "REBOUNDS": "rebounds",
        "ASSISTS": "assists",
        "THREES": "three_pointers_made",
    }
    col = stat_col_map.get(stat_type.upper(), "points")
    try:
        conn = psycopg2.connect(**get_players_db_config(), connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT pgl.{col}
                FROM player_game_logs pgl
                JOIN player_profile pp ON pgl.player_id = pp.player_id
                WHERE pp.full_name ILIKE %s
                  AND pgl.minutes_played > 0
                ORDER BY pgl.game_date DESC
                LIMIT 20
            """,
                (player_name,),
            )
            rows = [float(r[0] or 0) for r in cur.fetchall()]
        conn.close()
        if not rows:
            return None

        def avg(vals):
            return round(sum(vals) / len(vals), 2) if vals else None

        return {
            "avg_l5": avg(rows[:5]) if len(rows) >= 5 else avg(rows),
            "avg_l10": avg(rows[:10]) if len(rows) >= 10 else avg(rows),
            "avg_l20": avg(rows[:20]) if len(rows) >= 20 else avg(rows),
            "ema_l5": avg(rows[:5]) if len(rows) >= 5 else avg(rows),
        }
    except Exception as e:
        logger.warning(f"rolling_stats lookup failed for {player_name}: {e}")
        return None


def _get_injury_status(player_name: str, game_date: str) -> Optional[str]:
    """Query injuries table."""
    from nba.config.database import get_intelligence_db_config

    try:
        conn = psycopg2.connect(**get_intelligence_db_config(), connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status FROM injuries
                WHERE player_name ILIKE %s
                  AND date = %s
                LIMIT 1
            """,
                (player_name, game_date),
            )
            row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"injury_status lookup failed for {player_name}: {e}")
        return None


# ==============================================================================
# Picks file helpers
# ==============================================================================


def _load_picks_file(date_str: str) -> dict:
    path = _PREDICTIONS_DIR / f"xl_picks_{date_str}.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No picks found for {date_str}"
        )
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read picks file for {date_str}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read picks file"
        ) from e


def _format_picks(data: dict, date_str: str) -> list[dict]:
    """Normalise picks into Lunara API contract format, enriched with DB lookups."""
    formatted = []
    for pick in data.get("picks", []):
        player_name = pick.get("player_name", "")
        stat_type = pick.get("stat_type", "")

        formatted.append(
            {
                "id": _pick_id(player_name, date_str, stat_type),
                "player_name": player_name,
                "team": pick.get("team"),
                "opponent_team": pick.get("opponent_team"),
                "stat_type": stat_type,
                "side": pick.get("side", "OVER"),
                "line": pick.get("consensus_line"),
                "softest_line": pick.get("best_line"),
                "softest_book": pick.get("best_book"),
                "edge": pick.get("edge"),
                "edge_pct": pick.get("edge_pct"),
                "p_over": pick.get("p_over"),
                "tier": pick.get("filter_tier"),
                "model_version": pick.get("model_version"),
                "consensus": pick.get("consensus", False),
                "models_agreeing": pick.get("models_agreeing", []),
                "risk_level": pick.get("risk_level"),
                "risk_flags": pick.get("risk_flags", []),
                "blowout_risk": (
                    pick.get("blowout_risk", {}).get("level") if pick.get("blowout_risk") else None
                ),
                "game_time": pick.get("game_time"),
                "game_date": date_str,
                # Phase 3 enrichment — null if DB unavailable
                "rolling_stats": _get_rolling_stats(player_name, stat_type),
                "injury_status": _get_injury_status(player_name, date_str),
            }
        )
    return formatted


# ==============================================================================
# GET endpoints
# ==============================================================================


@router.get("/today", summary="Get today's picks")
async def get_today_picks(authorization: str | None = Header(default=None)) -> dict:  # noqa: B008
    _verify_token(authorization)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    return await _get_picks_for_date(today)


@router.get("/{date}", summary="Get picks for a date (YYYY-MM-DD)")
async def get_picks_by_date(
    date: str, authorization: str | None = Header(default=None)  # noqa: B008
) -> dict:
    _verify_token(authorization)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Date must be YYYY-MM-DD"
        )
    return await _get_picks_for_date(date)


async def _get_picks_for_date(date_str: str) -> dict:
    data = _load_picks_file(date_str)
    picks = _format_picks(data, date_str)
    return {
        "date": date_str,
        "generated_at": data.get("generated_at"),
        "strategy": data.get("strategy"),
        "total_picks": len(picks),
        "picks": picks,
    }


# ==============================================================================
# PATCH /picks/{id}/result  — feedback loop from Lunara
# ==============================================================================


class PickResultUpdate(BaseModel):
    actual_value: float
    is_hit: bool
    source: str = "lunara"


@router.patch("/{pick_id}/result", summary="Receive pick result from Lunara")
async def update_pick_result(
    pick_id: str,
    body: PickResultUpdate,
    authorization: str | None = Header(default=None),  # noqa: B008
) -> dict:
    """
    Called by Lunara's pick_tracker_poller when a game goes final.
    Stores result in pick_results_{today}.json for populate_actuals.py to consume.
    """
    _verify_token(authorization)

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    results_path = _PREDICTIONS_DIR / f"pick_results_{today}.json"

    # Load existing results for today (or start fresh)
    results = {}
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            results = {}

    # Write/overwrite this pick's result
    results[pick_id] = {
        "actual_value": body.actual_value,
        "is_hit": body.is_hit,
        "source": body.source,
        "updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write pick result for {pick_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist result"
        ) from e

    logger.info(f"Pick result stored: {pick_id} → is_hit={body.is_hit} actual={body.actual_value}")
    return {"ok": True, "pick_id": pick_id}
