#!/usr/bin/env python3
"""
Cephalon Axiom - Sport-suite NBA Discord Bot
DM-only standalone bot for NBA betting picks.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
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

_init_dedup_table()


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
    """Poll axiom_pipeline_audit and DM admins when a pipeline run completes."""
    await bot.wait_until_ready()
    await asyncio.sleep(90)
    while not bot.is_closed():
        await asyncio.sleep(30)
        try:
            from cephalon.axiom_db import execute_query

            rows = execute_query(
                "axiom",
                """
                SELECT run_timestamp, run_number, run_type, status,
                       picks_generated, xl_picks, v3_picks,
                       duration_seconds, error_message
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
                xl, v3 = row["xl_picks"], row["v3_picks"]
                dur = row["duration_seconds"]
                err = row["error_message"]
                num = row["run_number"]
                rtype = row["run_type"] or "run"

                picks_str = f"{picks} picks" if picks is not None else "?"
                if xl is not None and v3 is not None:
                    picks_str += f" (XL:{xl} V3:{v3})"
                dur_str = f" in {int(dur)}s" if dur else ""
                err_str = f"\n⚠️ {err[:120]}" if err else ""
                ts = datetime.now(_EST).strftime("%H:%M EST")
                msg = (
                    f"{icon} **Run {num}** ({rtype}) — {status}{dur_str}\n"
                    f"{picks_str}{err_str}\n"
                    f"-# {ts}"
                )
                for admin_id in ADMIN_IDS:
                    try:
                        user = await bot.fetch_user(admin_id)
                        await user.send(msg)
                    except Exception as dm_err:
                        log.warning(f"Pipeline DM failed for {admin_id}: {dm_err}")
        except Exception as e:
            log.debug(f"Pipeline monitor error: {e}")


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
    bot.loop.create_task(_send_heartbeat())
    bot.loop.create_task(_pipeline_completion_monitor())


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
