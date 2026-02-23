#!/usr/bin/env python3
"""
Cephalon Axiom - Sport-suite NBA Discord Bot
DM-only standalone bot for NBA betting picks.
"""

import logging
import os
import sys

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
        tools=AXIOM_TOOLS,
        tool_handler=handle_tool,
    )
)

nba_commands.register(bot)


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


@bot.event
async def on_message(message):
    # Ignore own messages
    if message.author == bot.user:
        return

    # Only respond to DMs
    if not isinstance(message.channel, discord.DMChannel):
        await bot.process_commands(message)
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
