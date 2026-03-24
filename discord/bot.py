#!/usr/bin/env python3
"""
Cephalon Axiom - Sport-suite NBA Discord Bot
DM-only standalone bot for NBA betting picks.

Background intelligence tasks:
- Morning brief (7:00 AM EST) — yesterday's results + today's outlook
- Pipeline monitor (30s poll) — run completion alerts
- Alert engine (post-pipeline) — line movements, evaporations, pipeline health
- Injury monitor (5 min) — cross-ref injuries with active picks
- Pre-game monitor (5 min) — T-60 mini-brief per game
- Live game monitor (2 min) — real-time pick tracking during games
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp

import discord
from discord import app_commands
from discord.ext import commands

# Add cephalon module path (works both locally and on server)
_CEPHALON_PATHS = [
    "/home/untitled/Sport-suite",  # local dev
    "/home/cephalons",  # server shared location
]
for _p in _CEPHALON_PATHS:
    if os.path.isdir(os.path.join(_p, "cephalon")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

import nba_commands
from cephalon import BotIdentity, CephalonBrain
from cephalon.axiom_tools import AXIOM_TOOLS, handle_tool
from cephalon.context import axiom_context
from cephalon.persistence import _init_dedup_table, claim_message
from cephalon.personalities import AXIOM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("axiom")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize AI brain
ADMIN_IDS = {int(uid) for uid in os.environ.get("AXIOM_ADMIN_IDS", "").split(",") if uid.strip()}
brain = CephalonBrain(
    BotIdentity(
        name="Axiom",
        system_prompt=AXIOM,
        context_fn=axiom_context,
        admin_ids=ADMIN_IDS,
        persistent=True,
        tools=AXIOM_TOOLS,
        tool_handler=handle_tool,
    )
)

nba_commands.register(bot)

_start_time = datetime.now(timezone.utc)
_EST = ZoneInfo("America/New_York")
_notified_run_ids: set[str] = set()

# Dedup sets for proactive intelligence (reset daily at midnight)
_sent_alerts: set[str] = set()
_sent_live_alerts: set[str] = set()
_sent_pregame: set[str] = set()
_morning_brief_sent: set[str] = set()
_last_dedup_reset: str = ""

_init_dedup_table()


def _get_today() -> str:
    """Get today's date in EST as YYYY-MM-DD."""
    return datetime.now(_EST).strftime("%Y-%m-%d")


def _reset_daily_dedup():
    """Reset in-memory dedup sets when the date changes."""
    global _last_dedup_reset
    today = _get_today()
    if _last_dedup_reset != today:
        _sent_alerts.clear()
        _sent_live_alerts.clear()
        _sent_pregame.clear()
        _morning_brief_sent.clear()
        _last_dedup_reset = today


async def _dm_admins(message: str):
    """Send a DM to all admin users."""
    for admin_id in ADMIN_IDS:
        try:
            user = await bot.fetch_user(admin_id)
            for chunk in _split_message(message):
                await user.send(chunk)
        except Exception as e:
            log.warning(f"DM failed for {admin_id}: {e}")


async def _dm_admins_embed(embed: discord.Embed):
    """Send a Discord embed DM to all admin users."""
    for admin_id in ADMIN_IDS:
        try:
            user = await bot.fetch_user(admin_id)
            await user.send(embed=embed)
        except Exception as e:
            log.warning(f"Embed DM failed for {admin_id}: {e}")


def _get_user_settings(user_id: int) -> dict:
    """Load user settings from axiom_memory table."""
    try:
        from cephalon.axiom_db import execute_query

        rows = execute_query(
            "axiom",
            "SELECT value FROM axiom_memory WHERE key = %s",
            (f"settings:{user_id}",),
        )
        if rows:
            val = rows[0]["value"]
            if isinstance(val, str):
                return json.loads(val)
            return val if isinstance(val, dict) else {}
    except Exception:
        pass
    return {}


def _is_feature_enabled(user_id: int, feature: str) -> bool:
    """Check if a proactive feature is enabled for a user.

    Defaults to True (opt-out model).
    """
    settings = _get_user_settings(user_id)
    return settings.get(feature, True)


def _check_post_for_dedup(run_date: str, post_type: str) -> bool:
    """Check axiom_posts for existing post today. Returns True if already sent."""
    try:
        from cephalon.axiom_db import execute_query

        rows = execute_query(
            "axiom",
            "SELECT id FROM axiom_posts WHERE run_date = %s AND post_type = %s LIMIT 1",
            (run_date, post_type),
        )
        return bool(rows)
    except Exception:
        return False


def _record_post(run_date: str, post_type: str, data: dict = None):
    """Record a post in axiom_posts for dedup."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", "5500")),
            dbname=os.environ.get("DB_NAME", "sportsuite"),
            user=os.environ.get("DB_USER", "mlb_user"),
            password=os.environ.get("DB_PASSWORD", ""),
            options="-c search_path=axiom,public",
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO axiom_posts (run_date, post_type, trigger, picks_sent, sent_at)
            VALUES (%s, %s, 'bot', %s, NOW())
            """,
            (run_date, post_type, json.dumps(data) if data else None),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.debug(f"Failed to record post: {e}")


# ═══════════════════════════════════════════════════════════════
#  EXISTING BACKGROUND TASKS
# ═══════════════════════════════════════════════════════════════


async def _send_heartbeat():
    """Send heartbeat to Cephalon Atlas every 60 seconds."""
    await bot.wait_until_ready()
    secret = os.environ.get("ATLAS_HEARTBEAT_SECRET", "")
    while not bot.is_closed():
        try:
            uptime = int((datetime.now(timezone.utc) - _start_time).total_seconds())
            async with aiohttp.ClientSession() as session:
                await session.post(
                    "http://localhost:8100/heartbeat",
                    json={"bot": "axiom", "status": "ok", "uptime": uptime, "secret": secret},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass
        await asyncio.sleep(60)


async def _pipeline_completion_monitor():
    """Poll axiom_pipeline_audit and DM admins when a pipeline run completes.

    After notifying, runs the alert engine for line movements, evaporations, etc.
    """
    await bot.wait_until_ready()
    await asyncio.sleep(90)

    while not bot.is_closed():
        await asyncio.sleep(30)
        _reset_daily_dedup()
        try:
            from cephalon.axiom_db import execute_query

            rows = execute_query(
                "axiom",
                """
                SELECT run_timestamp, run_number, run_type, status,
                       picks_generated, xl_picks, v3_picks,
                       duration_seconds, error_message, run_date
                FROM axiom_pipeline_audit
                WHERE run_timestamp >= %s
                  AND status IN ('success', 'failed', 'partial')
                ORDER BY run_timestamp DESC
                LIMIT 10
                """,
                (_start_time.isoformat(),),
            )
            for row in rows:
                run_id = str(row["run_timestamp"])
                if run_id in _notified_run_ids:
                    continue
                _notified_run_ids.add(run_id)

                status = row["status"]
                icon = "✅" if status == "success" else ("⚠️" if status == "partial" else "❌")
                picks = row["picks_generated"]
                v5 = row["xl_picks"]  # xl_picks column now holds V5 count
                dur = row["duration_seconds"]
                err = row["error_message"]
                num = row["run_number"]
                rtype = row["run_type"] or "run"
                run_date = str(row.get("run_date") or _get_today())

                picks_str = f"{picks} picks" if picks is not None else "?"
                if v5 is not None and v5 > 0:
                    picks_str += f" (V5:{v5})"
                dur_str = f" in {int(dur)}s" if dur else ""
                err_str = f"\n⚠️ {err[:120]}" if err else ""
                ts = datetime.now(_EST).strftime("%H:%M EST")
                msg = (
                    f"{icon} **Run {num}** ({rtype}) — {status}{dur_str}\n"
                    f"{picks_str}{err_str}\n"
                    f"-# {ts}"
                )
                await _dm_admins(msg)

                # === POST-PIPELINE ALERT ENGINE ===
                try:
                    from cephalon.alerts import AlertEngine

                    alert_engine = AlertEngine()

                    # Check for alerts
                    alerts = alert_engine.check_post_pipeline(run_date, num)
                    for alert in alerts:
                        if alert.dedup_key not in _sent_alerts:
                            _sent_alerts.add(alert.dedup_key)
                            await _dm_admins(alert.format_dm())

                    # Self-healing check
                    action = alert_engine.check_self_heal(run_date, num)
                    if action == "refresh" and status != "failed":
                        log.info("Self-heal: auto-triggering refresh due to low data")
                        await _dm_admins(
                            "🔧 **Self-healing**: Low data detected, "
                            "triggering automatic refresh..."
                        )
                        # Import and run quick refresh
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                f"cd /home/sportsuite/sport-suite && "
                                f"source venv/bin/activate && "
                                f"source .env && export DB_USER DB_PASSWORD "
                                f"BETTINGPROS_API_KEY TERM=xterm TZ=America/New_York && "
                                f"python3 nba/betting_xl/quick_refresh.py",
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                executable="/bin/bash",
                            )
                            await asyncio.wait_for(proc.communicate(), timeout=300)
                        except Exception as heal_err:
                            log.warning(f"Self-heal refresh failed: {heal_err}")

                except Exception as alert_err:
                    log.debug(f"Post-pipeline alert check failed: {alert_err}")

        except Exception as e:
            log.debug(f"Pipeline monitor error: {e}")


# ═══════════════════════════════════════════════════════════════
#  PROACTIVE INTELLIGENCE TASKS
# ═══════════════════════════════════════════════════════════════


async def _morning_brief_loop():
    """Send morning intelligence brief at 7:00 AM EST daily.

    Combines yesterday's pick autopsy + rolling performance + today's outlook.
    Uses Axiom's AI brain to narrate in character.
    """
    await bot.wait_until_ready()

    while not bot.is_closed():
        now_est = datetime.now(_EST)
        # Calculate seconds until next 7:00 AM EST
        target = now_est.replace(hour=7, minute=0, second=0, microsecond=0)
        if now_est >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now_est).total_seconds()
        log.info(f"Morning brief: next in {wait_seconds/3600:.1f}h")
        await asyncio.sleep(wait_seconds)

        if bot.is_closed():
            break

        _reset_daily_dedup()
        today = _get_today()

        # Dedup: only once per day
        if today in _morning_brief_sent:
            continue
        if _check_post_for_dedup(today, "morning_brief"):
            _morning_brief_sent.add(today)
            continue

        log.info("Generating morning brief...")
        try:
            from cephalon.context import _fetch_espn_schedule, _get_injury_section
            from cephalon.intelligence import MorningBriefAnalyzer

            analyzer = MorningBriefAnalyzer()
            brief = analyzer.analyze(today)
            brief_text = analyzer.format_for_brain(brief)

            # Add today's schedule and injuries
            schedule = await _fetch_espn_schedule(today)
            injuries = _get_injury_section()

            extra_context = brief_text
            if schedule:
                extra_context += f"\n\n=== TODAY'S SCHEDULE ===\n{schedule}"
            if injuries:
                extra_context += f"\n\n=== INJURY REPORT ===\n{injuries}"

            # Let Axiom's brain narrate the brief
            if brain.available:
                owner_id = next(iter(ADMIN_IDS), 0)
                narrated = await brain.ask_with_context(
                    user_id=owner_id,
                    question=(
                        "Generate the morning intelligence brief for the Operator. "
                        "Cover: yesterday's results with miss analysis, rolling performance trends, "
                        "hot/cold tiers, today's schedule and injury impacts. "
                        "Be analytical and direct. Use data from the brief below. "
                        "Keep under 1800 characters for Discord."
                    ),
                    extra_context=extra_context,
                )
                message = f"**Morning Brief — {today}**\n\n{narrated}"
            else:
                # Fallback: template-based brief without AI
                message = f"**Morning Brief — {today}**\n\n{brief_text[:1800]}"

            await _dm_admins(message)
            _morning_brief_sent.add(today)
            _record_post(
                today,
                "morning_brief",
                {
                    "wins": brief.yesterday_wins,
                    "losses": brief.yesterday_losses,
                },
            )
            log.info("Morning brief sent.")

        except Exception as e:
            log.error(f"Morning brief failed: {e}", exc_info=True)


async def _injury_monitor_loop():
    """Check for injury impacts on active picks every 5 minutes.

    Only active during game-day hours (8 AM - 11 PM EST).
    """
    await bot.wait_until_ready()
    await asyncio.sleep(120)  # Initial delay

    while not bot.is_closed():
        await asyncio.sleep(300)  # 5 minutes
        _reset_daily_dedup()

        now_est = datetime.now(_EST)
        if not (8 <= now_est.hour <= 23):
            continue

        try:
            from cephalon.alerts import AlertEngine

            alert_engine = AlertEngine()
            today = _get_today()

            alerts = alert_engine.check_injuries(today)
            for alert in alerts:
                if alert.dedup_key not in _sent_alerts:
                    _sent_alerts.add(alert.dedup_key)
                    await _dm_admins(alert.format_dm())

        except Exception as e:
            log.debug(f"Injury monitor error: {e}")


async def _pregame_monitor_loop():
    """Send pre-game mini-brief ~60 minutes before each game with active picks.

    Polls every 5 minutes, sends once per game.
    """
    await bot.wait_until_ready()
    await asyncio.sleep(180)  # Initial delay

    while not bot.is_closed():
        await asyncio.sleep(300)  # 5 minutes
        _reset_daily_dedup()

        now_est = datetime.now(_EST)
        # Only during game-day hours
        if not (12 <= now_est.hour <= 23):
            continue

        try:
            today = _get_today()

            # Fetch ESPN schedule
            from cephalon.axiom_db import execute_query
            from cephalon.context import _fetch_espn_schedule

            date_compact = today.replace("-", "")
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_compact}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

            events = data.get("events", [])

            for event in events:
                game_id = event.get("id", "")
                state = event.get("status", {}).get("type", {}).get("state", "pre")
                if state != "pre":
                    continue  # Only pre-game

                # Parse game time
                date_str = event.get("date", "")
                if not date_str:
                    continue
                try:
                    game_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    game_time_est = game_time.astimezone(_EST)
                except Exception:
                    continue

                # Check if 55-65 minutes until tip
                minutes_until = (game_time_est - now_est).total_seconds() / 60
                if not (55 <= minutes_until <= 65):
                    continue

                # Dedup: one pregame brief per game
                pregame_key = f"pregame:{game_id}"
                if pregame_key in _sent_pregame:
                    continue
                if _check_post_for_dedup(today, pregame_key):
                    _sent_pregame.add(pregame_key)
                    continue

                # Get teams in this game
                teams = set()
                game_name = event.get("shortName", "")
                for comp in event.get("competitions", [{}]):
                    for team_entry in comp.get("competitors", []):
                        abbr = team_entry.get("team", {}).get("abbreviation", "")
                        _map = {"WSH": "WAS", "NO": "NOP", "SA": "SAS", "GS": "GSW"}
                        abbr = _map.get(abbr, abbr)
                        if abbr:
                            teams.add(abbr)

                # Get active picks for this game
                active_picks = execute_query(
                    "axiom",
                    """
                    SELECT player_name, stat_type, conviction_label, conviction,
                           line_latest, book_latest, p_over_latest,
                           appearances, total_runs, narrative, context_snapshot
                    FROM axiom_conviction
                    WHERE run_date = %s
                      AND conviction_label IN ('LOCKED', 'STRONG')
                      AND status = 'active'
                    """,
                    (today,),
                )

                # Filter to picks in this game (match by opponent_team)
                game_picks = []
                for p in active_picks:
                    ctx = p.get("context_snapshot") or {}
                    if isinstance(ctx, str):
                        try:
                            ctx = json.loads(ctx)
                        except Exception:
                            ctx = {}
                    opp = ctx.get("opponent_team", "")
                    _map = {"WSH": "WAS", "NO": "NOP", "SA": "SAS", "GS": "GSW"}
                    opp = _map.get(opp, opp)
                    if opp in teams:
                        p["_ctx"] = ctx
                        game_picks.append(p)

                if not game_picks:
                    continue

                # Build pre-game brief
                tip = game_time_est.strftime("%-I:%M %p ET")
                pick_lines = []
                for gp in game_picks:
                    ctx = gp.get("_ctx", {})
                    pc = ctx.get("player_context", {})
                    pick_lines.append(
                        f"  {gp['player_name']} {gp['stat_type']} OVER {float(gp['line_latest']):.1f} "
                        f"[{gp['conviction_label']}] — "
                        f"L5: {pc.get('avg_L5', '?')}, trend: {pc.get('trend', '?')}, "
                        f"{gp['appearances']}/{gp['total_runs']} runs"
                    )

                extra_context = (
                    f"PRE-GAME BRIEF for {game_name} (tip {tip}):\n"
                    f"Active picks:\n" + "\n".join(pick_lines)
                )

                # Use AI brain for narration if available
                if brain.available:
                    owner_id = next(iter(ADMIN_IDS), 0)
                    narrated = await brain.ask_with_context(
                        user_id=owner_id,
                        question=(
                            f"Generate a pre-game mini-brief for {game_name} "
                            f"tipping at {tip}. Cover the active picks listed below, "
                            f"key matchup context, and any injury concerns. "
                            f"Keep under 1200 characters."
                        ),
                        extra_context=extra_context,
                    )
                    message = f"**Pre-Game: {game_name}** (tip {tip})\n\n{narrated}"
                else:
                    message = f"**Pre-Game: {game_name}** (tip {tip})\n\n" + "\n".join(pick_lines)

                await _dm_admins(message)
                _sent_pregame.add(pregame_key)
                _record_post(
                    today,
                    pregame_key,
                    {
                        "game": game_name,
                        "picks": len(game_picks),
                    },
                )
                log.info(f"Pre-game brief sent for {game_name}")

        except Exception as e:
            log.debug(f"Pregame monitor error: {e}")


async def _live_game_monitor_loop():
    """Track live games and picks in real-time every 2 minutes.

    Only active during game window (5 PM - 1 AM EST).
    Sends alerts for pick hits, busts, halftime pace, and blowout risk.
    """
    await bot.wait_until_ready()
    await asyncio.sleep(240)  # Initial delay

    while not bot.is_closed():
        await asyncio.sleep(120)  # 2 minutes
        _reset_daily_dedup()

        now_est = datetime.now(_EST)
        # Only during game window: 5 PM - 1 AM (next day)
        if not (17 <= now_est.hour or now_est.hour < 1):
            continue

        try:
            from cephalon.live_tracker import LiveGameTracker

            today = _get_today()
            tracker = LiveGameTracker()
            alerts = await tracker.check_live_games(today)

            for alert in alerts:
                if alert.dedup_key not in _sent_live_alerts:
                    _sent_live_alerts.add(alert.dedup_key)

                    # Format with Axiom personality
                    icon = {
                        "pick_hit": "✅",
                        "pick_bust": "❌",
                        "halftime_pace": "📊",
                        "blowout_risk": "⚠️",
                    }.get(alert.alert_type, "📌")

                    msg = f"{icon} **{alert.player_name} {alert.stat_type}**\n{alert.message}"
                    await _dm_admins(msg)

        except Exception as e:
            log.debug(f"Live game monitor error: {e}")


# ═══════════════════════════════════════════════════════════════
#  BOT EVENTS
# ═══════════════════════════════════════════════════════════════


@bot.event
async def on_ready():
    log.info(f"Cephalon Axiom online: {bot.user} (ID: {bot.user.id})")
    log.info(f"AI brain: {'ONLINE' if brain.available else 'OFFLINE'}")
    try:
        synced = await bot.tree.sync()
        log.info(f"Synced {len(synced)} global commands")
    except Exception as e:
        log.error(f"Command sync failed: {e}")
    nba_commands.start_scheduled_tasks(bot)

    # Core tasks
    bot.loop.create_task(_send_heartbeat())
    bot.loop.create_task(_pipeline_completion_monitor())

    # Proactive intelligence tasks
    bot.loop.create_task(_morning_brief_loop())
    bot.loop.create_task(_injury_monitor_loop())
    bot.loop.create_task(_pregame_monitor_loop())
    bot.loop.create_task(_live_game_monitor_loop())
    log.info("All intelligence tasks started.")


@bot.event
async def on_message(message):
    # Ignore own messages
    if message.author == bot.user:
        return

    # Only respond to DMs
    if not isinstance(message.channel, discord.DMChannel):
        await bot.process_commands(message)
        return

    # Deduplicate: SQLite-backed claim so cross-restart / two-process races are safe
    if not claim_message(message.id):
        log.debug(f"Skipping already-claimed message id={message.id}")
        return

    # DM conversation via AI
    async with message.channel.typing():
        reply = await brain.respond(message.author.id, message.content)

    # Split long replies for Discord's 2000 char limit
    for chunk in _split_message(reply):
        await message.channel.send(chunk)

    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask Cephalon Axiom a question about NBA picks")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    reply = await brain.ask_with_context(interaction.user.id, question)
    for chunk in _split_message(reply):
        await interaction.followup.send(chunk)


@bot.tree.command(name="clear-history", description="Reset your conversation with Axiom")
async def clear_history(interaction: discord.Interaction):
    brain.clear_history(interaction.user.id)
    await interaction.response.send_message(
        "Conversation history cleared. A fresh start, Operator.", ephemeral=True
    )


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks that fit Discord's message limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


def main():
    token = os.environ.get("AXIOM_BOT_TOKEN")
    if not token:
        log.error("AXIOM_BOT_TOKEN not set")
        sys.exit(1)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
