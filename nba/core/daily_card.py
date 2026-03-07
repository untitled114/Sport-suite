"""
Daily Card Builder — Cephalon Axiom's autonomous conviction card.

Reads axiom_conviction for LOCKED + STRONG picks, formats a Discord embed,
sends it to the owner's DM once per day, and records the post in axiom_posts.

Called by:
  - nba_daily_card_dag.py at T-1hr before first game (dynamic, polls 5-10:30 PM EST)
  - /nba-card Discord command (manual trigger, force=True)
"""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("nba.daily_card")

_EST = ZoneInfo("America/New_York")
_AXIOM_PORT = 5541
_AXIOM_DB = "cephalon_axiom"
_CONNECT_TIMEOUT = 5
_DISCORD_API = "https://discord.com/api/v10"

_LABEL_EMOJI = {"LOCKED": "\U0001f512", "STRONG": "\U0001f4aa", "WATCH": "\U0001f440"}

_STAT_ABBREV = {
    "POINTS": "PTS",
    "REBOUNDS": "REB",
    "ASSISTS": "AST",
    "THREES": "3PM",
    "STEALS": "STL",
    "BLOCKS": "BLK",
}

_BOOK_DISPLAY = {
    "prizepicks_goblin": "PrizePicks (goblin)",
    "prizepicks_demon": "PrizePicks (demon)",
    "underdog": "Underdog",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "betrivers": "BetRivers",
    "espnbet": "ESPNBet",
}


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------


def _connect():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=_AXIOM_PORT,
        dbname=_AXIOM_DB,
        user=os.environ.get("DB_USER", "mlb_user"),
        password=os.environ.get("DB_PASSWORD", ""),
        connect_timeout=_CONNECT_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_first_tip_time(run_date: str) -> str:
    """Get first game tip time from ESPN API. Returns '7:00 PM ET' as fallback."""
    try:
        date_compact = run_date.replace("-", "")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            f"?dates={date_compact}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310
            data = json.loads(resp.read())

        times = []
        for event in data.get("events", []):
            date_str = event.get("date", "")
            if date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                times.append(dt.astimezone(_EST))

        if times:
            return min(times).strftime("%-I:%M %p ET")
    except Exception as e:
        log.debug(f"ESPN tip time fetch failed: {e}")
    return "7:00 PM ET"


def _load_conviction_picks(run_date: str) -> tuple[list[dict], int]:
    """Load LOCKED + STRONG active picks from axiom_conviction.

    Returns (picks_list, max_run_number).
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT player_name, stat_type, conviction, conviction_label,
                       appearances, total_runs, line_at_entry, line_latest,
                       line_direction, p_over_at_entry, p_over_latest,
                       p_over_trend, book_latest, context_snapshot
                FROM axiom_conviction
                WHERE run_date = %s
                  AND conviction_label IN ('LOCKED', 'STRONG')
                  AND status = 'active'
                ORDER BY
                    CASE conviction_label WHEN 'LOCKED' THEN 0 ELSE 1 END,
                    conviction DESC
                """,
                (run_date,),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

            cur.execute(
                """
                SELECT COALESCE(MAX(run_number), 1)
                FROM nba_prediction_history
                WHERE run_date = %s
                """,
                (run_date,),
            )
            max_run = cur.fetchone()[0]
    finally:
        conn.close()

    return [dict(zip(cols, r)) for r in rows], max_run


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_line(line_at_entry, line_latest, line_direction: str) -> str:
    entry = float(line_at_entry)
    latest = float(line_latest)
    if line_direction == "falling":
        return f"line {entry}->{latest} (dropped)"
    elif line_direction == "rising":
        return f"line {entry}->{latest} (up)"
    return f"line stable @ {latest}"


def _fmt_p_over(p_entry, p_latest, p_trend) -> str:
    pe, pl = float(p_entry), float(p_latest)
    trend = float(p_trend) if p_trend is not None else 0.0
    if abs(trend) < 0.005:
        return f"p_over stable ({pl:.3f})"
    elif trend > 0:
        return f"p_over {pe:.3f}->{pl:.3f} (up)"
    return f"p_over {pe:.3f}->{pl:.3f} (down)"


def _bp_stars(ctx: dict) -> str:
    rating = ctx.get("bp_bet_rating")
    if not rating:
        return ""
    side = ctx.get("bp_recommended_side", "")
    stars = "\u2b50" * int(rating)
    return f"BP {stars}" + (f" ({side})" if side else "")


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------


def build_embed(run_date: str, tip_time: str, picks: list[dict], max_run: int) -> dict:
    """Build a Discord embed payload dict (raw JSON, not discord.py Embed)."""
    dt = datetime.strptime(run_date, "%Y-%m-%d")
    date_display = dt.strftime("%b %-d, %Y")
    now_est = datetime.now(_EST).isoformat()

    if not picks:
        return {
            "title": "NBA Picks",
            "description": (
                f"**{date_display}**  |  Games tip at {tip_time}\n\n"
                "Nothing meets conviction threshold tonight.\n"
                "Monitoring all picks."
            ),
            "color": 0x36393F,
            "footer": {"text": f"0 picks  |  Run {max_run} of 6"},
            "timestamp": now_est,
        }

    fields = []
    models_seen = set()

    for pick in picks:
        ctx = pick.get("context_snapshot") or {}
        label = pick["conviction_label"]
        emoji = _LABEL_EMOJI.get(label, "")
        book = _BOOK_DISPLAY.get(pick.get("book_latest", ""), pick.get("book_latest", ""))
        conviction = float(pick["conviction"])
        stat = _STAT_ABBREV.get(pick["stat_type"], pick["stat_type"][:3])

        line_info = _fmt_line(
            pick["line_at_entry"], pick["line_latest"], pick["line_direction"] or "stable"
        )
        p_info = _fmt_p_over(pick["p_over_at_entry"], pick["p_over_latest"], pick["p_over_trend"])

        models = ctx.get("models_agreeing", [])
        model_str = "+".join(m.upper() for m in models) if models else "XL"
        models_seen.update(m.upper() for m in models)

        bp = _bp_stars(ctx)
        bp_part = f"  |  {bp}" if bp else ""

        name = (
            f"{emoji} {label}  {pick['player_name']} "
            f"OVER {float(pick['line_latest'])} {stat}  @{book}"
        )
        value = (
            f"{pick['appearances']}/{pick['total_runs']} runs  |  {p_info}  |  {line_info}\n"
            f"**conviction: {conviction:.3f}**  |  [{model_str}]{bp_part}"
        )
        fields.append({"name": name, "value": value, "inline": False})

    models_str = "+".join(sorted(models_seen)) if models_seen else "XL"
    count = len(picks)
    pick_word = "pick" if count == 1 else "picks"

    return {
        "title": "NBA Picks",
        "description": f"**{date_display}**  |  Games tip at {tip_time}",
        "color": 0xFFD700,
        "fields": fields,
        "footer": {"text": f"{count} {pick_word}  |  Run {max_run} of 6  |  {models_str}"},
        "timestamp": now_est,
    }


# ---------------------------------------------------------------------------
# Discord posting (bot token DM, same pattern as discord_notify.py)
# ---------------------------------------------------------------------------


def _get_bot_token() -> Optional[str]:
    token = os.environ.get("AXIOM_BOT_TOKEN")
    if not token:
        env_path = os.path.join(os.environ.get("HOME", "/home/sportsuite"), "sport-suite", ".env")
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AXIOM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except OSError:
            pass
    return token


def _post_dm_embed(embed: dict) -> Optional[str]:
    """Send embed to owner DM. Returns message_id on success, None on failure."""
    owner_id = os.environ.get("DISCORD_OWNER_ID", "")
    token = _get_bot_token()
    if not token:
        log.error("AXIOM_BOT_TOKEN not set — cannot post daily card")
        return None

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "User-Agent": "CephalonAxiom/1.0",
    }

    try:
        # Open DM channel
        req = urllib.request.Request(
            f"{_DISCORD_API}/users/@me/channels",
            data=json.dumps({"recipient_id": owner_id}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            channel_id = json.loads(resp.read())["id"]

        # Post embed
        req = urllib.request.Request(
            f"{_DISCORD_API}/channels/{channel_id}/messages",
            data=json.dumps({"embeds": [embed]}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            return json.loads(resp.read()).get("id")

    except urllib.error.HTTPError as e:
        log.error(f"Discord DM HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        log.error(f"Discord DM failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Dedup + recording
# ---------------------------------------------------------------------------


def _already_sent_today(run_date: str) -> bool:
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM axiom_posts "
                    "WHERE run_date = %s AND post_type = 'daily_card'",
                    (run_date,),
                )
                return cur.fetchone()[0] > 0
        finally:
            conn.close()
    except Exception:
        return False


def _record_post(run_date: str, picks: list, message_id: Optional[str], trigger: str) -> None:
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO axiom_posts
                        (run_date, post_type, trigger, picks_sent, channel_id, message_id, sent_at)
                    VALUES (%s, 'daily_card', %s, %s, NULL, %s, NOW())
                    """,
                    (
                        run_date,
                        trigger,
                        json.dumps(
                            [
                                {
                                    "player_name": p["player_name"],
                                    "stat_type": p["stat_type"],
                                    "conviction": float(p["conviction"]),
                                    "label": p["conviction_label"],
                                }
                                for p in picks
                            ]
                        ),
                        message_id,
                    ),
                )
        conn.close()
    except Exception as e:
        log.warning(f"_record_post failed (non-critical): {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def send_daily_card(
    run_date: str,
    *,
    force: bool = False,
    trigger: str = "airflow",
) -> dict:
    """Build and send the daily conviction card to the owner via Discord DM.

    Args:
        run_date: YYYY-MM-DD (EST date)
        force:    Skip dedup check (for manual /nba-card triggers)
        trigger:  'airflow' | 'discord' | 'manual'

    Returns dict with keys: sent (bool), picks_count (int),
                             message_id (str|None), reason (str)

    Never raises — all errors are logged and returned in the result dict.
    """
    if not force and _already_sent_today(run_date):
        log.info(f"Daily card already sent for {run_date} — skipping")
        return {"sent": False, "reason": "already_sent", "picks_count": 0, "message_id": None}

    try:
        picks, max_run = _load_conviction_picks(run_date)
    except Exception as e:
        log.error(f"Failed to load conviction picks: {e}")
        return {"sent": False, "reason": f"db_error: {e}", "picks_count": 0, "message_id": None}

    tip_time = get_first_tip_time(run_date)
    embed = build_embed(run_date, tip_time, picks, max_run)

    message_id = _post_dm_embed(embed)
    if message_id is None:
        return {
            "sent": False,
            "reason": "discord_error",
            "picks_count": len(picks),
            "message_id": None,
        }

    _record_post(run_date, picks, message_id, trigger)
    log.info(
        f"Daily card sent: {len(picks)} picks for {run_date} " f"(trigger={trigger}, run={max_run})"
    )
    return {"sent": True, "picks_count": len(picks), "message_id": message_id, "reason": "ok"}


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now(_EST).strftime("%Y-%m-%d")
    force_arg = "--force" in sys.argv

    print(f"Building card for {date_arg} (force={force_arg})")
    result = send_daily_card(date_arg, force=force_arg, trigger="manual")
    print(f"Result: {result}")
