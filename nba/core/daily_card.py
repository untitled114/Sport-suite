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

from nba.config.database import get_axiom_db_config

log = logging.getLogger("nba.daily_card")

_EST = ZoneInfo("America/New_York")
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

    config = get_axiom_db_config()
    config["connect_timeout"] = _CONNECT_TIMEOUT
    return psycopg2.connect(**config)


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
                FROM axiom_pipeline_audit
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
        return f"Line dropped {entry} \u2192 {latest}"
    elif line_direction == "rising":
        return f"Line rose {entry} \u2192 {latest}"
    return f"Line holding at {latest}"


def _fmt_p_over(p_entry, p_latest, p_trend) -> str:
    pe, pl = float(p_entry), float(p_latest)
    pe_pct, pl_pct = round(pe * 100), round(pl * 100)
    trend = float(p_trend) if p_trend is not None else 0.0
    if abs(trend) < 0.005:
        return f"Probability steady at {pl_pct}%"
    elif trend > 0:
        return f"Probability {pe_pct}% \u2192 {pl_pct}% (rising)"
    return f"Probability {pe_pct}% \u2192 {pl_pct}% (fading)"


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


def _conviction_color(picks: list[dict]) -> int:
    """Embed sidebar color based on average conviction."""
    if not picks:
        return 0x36393F  # dark gray
    avg = sum(float(p["conviction"]) for p in picks) / len(picks)
    if avg >= 0.90:
        return 0xFFD700  # gold — elite slate
    if avg >= 0.85:
        return 0x2ECC71  # green — strong slate
    return 0x3498DB  # blue — standard


def _bp_signal(ctx: dict) -> str:
    """Format BP signal as alignment/conflict — never show raw 'under'."""
    rating = ctx.get("bp_bet_rating")
    if not rating:
        return ""
    rating = int(rating)
    side = (ctx.get("bp_recommended_side") or "").lower()

    # We always take OVER — BP either confirms or conflicts
    if side == "over":
        if rating >= 5:
            return f"BP {rating}/5 — sharp alignment"
        return f"BP {rating}/5 confirms"
    elif side == "under":
        return f"BP {rating}/5 leans other way (contrarian)"
    return f"BP {rating}/5"


def _axiom_note(pick: dict, rank: int, total: int) -> str:
    """Generate Axiom's analytical note for a pick."""
    ctx = pick.get("context_snapshot") or {}
    conv = float(pick["conviction"])
    appearances = pick["appearances"]
    total_runs = pick["total_runs"]
    trend = float(pick["p_over_trend"]) if pick.get("p_over_trend") is not None else 0.0
    line_dir = pick.get("line_direction", "stable")
    models = ctx.get("models_agreeing", [])
    bp_rating = int(ctx.get("bp_bet_rating", 0) or 0)
    bp_side = (ctx.get("bp_recommended_side") or "").lower()

    notes = []

    # Conviction-based
    if rank == 1 and conv >= 0.93:
        notes.append("Top play of the slate")
    elif conv >= 0.95:
        notes.append("Elite conviction")

    # Model consensus
    if len(models) >= 2:
        notes.append("both models agree")

    # Consistency across runs
    if appearances == total_runs and total_runs >= 3:
        notes.append(f"picked every run ({appearances}/{total_runs})")
    elif appearances >= 3:
        notes.append(f"{appearances}/{total_runs} runs")

    # BP alignment
    if bp_rating >= 5 and bp_side == "over":
        notes.append("BP sharp money confirms")
    elif bp_rating >= 4 and bp_side == "over":
        notes.append("BP aligns")
    elif bp_side == "under" and bp_rating >= 3:
        notes.append("contrarian vs BP")

    # Trend
    if trend > 0.01:
        notes.append("probability rising")
    elif trend < -0.01:
        notes.append("slight fade across runs")

    # Line movement
    if line_dir == "falling":
        notes.append("line moved our way")
    elif line_dir == "rising":
        notes.append("line moved against")

    if not notes:
        notes.append(f"{appearances}/{total_runs} runs")

    return " — ".join(notes[:3])


def build_embed(run_date: str, tip_time: str, picks: list[dict], max_run: int) -> dict:
    """Build a unified Discord embed — one field per pick with Axiom analysis."""
    dt = datetime.strptime(run_date, "%Y-%m-%d")
    date_display = dt.strftime("%b %-d, %Y")
    now_est = datetime.now(_EST).isoformat()

    if not picks:
        return {
            "title": "NBA Picks",
            "description": (
                f"**{date_display}** | Games tip at {tip_time}\n\n"
                "Nothing meets conviction threshold tonight.\n"
                "Monitoring all picks across runs."
            ),
            "color": 0x36393F,
            "footer": {"text": f"Run {max_run} of 7 | 0 picks"},
            "timestamp": now_est,
        }

    # Gather stats
    models_seen = set()
    locked = sum(1 for p in picks if p["conviction_label"] == "LOCKED")
    strong = sum(1 for p in picks if p["conviction_label"] == "STRONG")
    for p in picks:
        ctx = p.get("context_snapshot") or {}
        for m in ctx.get("models_agreeing", []):
            models_seen.add(m.upper())
    models_str = "+".join(sorted(models_seen)) if models_seen else "XL"

    # Header
    label_parts = []
    if locked:
        label_parts.append(f"{locked} locked")
    if strong:
        label_parts.append(f"{strong} strong")

    description = (
        f"**{date_display}** | Games tip at {tip_time}\n"
        f"Run {max_run} of 7 | {', '.join(label_parts)} | {models_str}"
    )

    # Group by market, then build fields
    market_order = ["POINTS", "REBOUNDS", "ASSISTS", "THREES", "STEALS", "BLOCKS"]
    markets: dict[str, list[dict]] = {}
    for p in picks:
        markets.setdefault(p["stat_type"], []).append(p)

    fields = []
    pick_rank = 0

    for market in market_order:
        market_picks = markets.get(market)
        if not market_picks:
            continue

        abbrev = _STAT_ABBREV.get(market, market[:3])

        # Market separator
        fields.append(
            {
                "name": f"\u2500\u2500\u2500 {abbrev} ({len(market_picks)}) \u2500\u2500\u2500",
                "value": "\u200b",  # required non-empty
                "inline": False,
            }
        )

        for p in market_picks:
            pick_rank += 1
            ctx = p.get("context_snapshot") or {}
            conv = float(p["conviction"])
            conv_pct = round(conv * 100, 1)
            label = p["conviction_label"]
            line_val = float(p["line_latest"])
            book = _BOOK_DISPLAY.get(p.get("book_latest", ""), p.get("book_latest", ""))
            models = ctx.get("models_agreeing", [])
            model_tag = "+".join(m.upper() for m in models) if models else "XL"

            p_info = _fmt_p_over(p["p_over_at_entry"], p["p_over_latest"], p["p_over_trend"])
            line_info = _fmt_line(
                p["line_at_entry"], p["line_latest"], p.get("line_direction") or "stable"
            )
            bp = _bp_signal(ctx)
            axiom = _axiom_note(p, pick_rank, len(picks))

            # --- Build rich context lines ---
            tag = "LOCKED" if label == "LOCKED" else "STRONG"
            field_name = f"{p['player_name']} \u2014 OVER {line_val:g} | {conv_pct}% [{tag}]"

            # Line 1: where to bet + model + consistency
            value_lines = [
                f"{book} | {model_tag} | {p['appearances']} of {p['total_runs']} runs",
            ]

            # Line 2: player context — recent averages, matchup, home/away
            player_ctx = ctx.get("player_context") or {}
            opp = ctx.get("opponent_team", "")
            is_home = ctx.get("is_home")
            projection = ctx.get("prediction")
            edge = ctx.get("edge")
            avg_l5 = player_ctx.get("avg_L5")
            avg_l10 = player_ctx.get("avg_L10")
            h2h_avg = player_ctx.get("h2h_avg")
            h2h_games = player_ctx.get("h2h_games", 0)
            trend_label = player_ctx.get("trend", "")
            opp_rank = ctx.get("opposition_rank") or ctx.get("opp_rank")

            perf_parts = []
            if avg_l5 is not None and avg_l10 is not None:
                perf_parts.append(f"Avg L5: **{avg_l5}** | L10: **{avg_l10}**")
            if h2h_avg is not None and h2h_games and int(h2h_games) >= 2:
                perf_parts.append(f"vs {opp}: {h2h_avg} ({h2h_games}g)")
            elif opp:
                loc = "home" if is_home else "away" if is_home is not None else ""
                perf_parts.append(f"{'vs' if is_home else '@'} {opp}" if loc else f"vs {opp}")
            if trend_label and trend_label not in ("STABLE", ""):
                perf_parts.append(trend_label)
            if perf_parts:
                value_lines.append(" | ".join(perf_parts))

            # Line 3: projection + edge (if available) or probability + line
            if projection is not None and edge is not None:
                value_lines.append(
                    f"Projects **{float(projection):.1f}** (edge {float(edge):+.1f}) | {line_info}"
                )
            else:
                value_lines.append(f"{p_info} | {line_info}")

            # Line 4: hit rates + opp defense rank
            hr_parts = []
            hit_l5 = ctx.get("hit_rate_L5")
            hit_szn = ctx.get("hit_rate_season")
            if hit_l5 is not None:
                hr_parts.append(f"Hit rate L5: {round(float(hit_l5) * 100)}%")
            if hit_szn is not None:
                hr_parts.append(f"Season: {round(float(hit_szn) * 100)}%")
            if opp_rank:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(int(opp_rank) % 10, "th")
                if 11 <= int(opp_rank) <= 13:
                    suffix = "th"
                hr_parts.append(f"Opp defense: {opp_rank}{suffix}")
            if hr_parts:
                value_lines.append(" | ".join(hr_parts))

            # Line 5: BP signal
            if bp:
                value_lines.append(bp)

            # Line 6: Axiom analysis
            value_lines.append(f"\u25b8 *{axiom}*")

            fields.append(
                {
                    "name": field_name,
                    "value": "\n".join(value_lines),
                    "inline": False,
                }
            )

    count = len(picks)

    return {
        "title": "NBA Picks",
        "description": description,
        "color": _conviction_color(picks),
        "fields": fields,
        "footer": {"text": f"{count} picks | Run {max_run} of 7 | {models_str}"},
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
