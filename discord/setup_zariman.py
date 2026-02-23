#!/usr/bin/env python3
"""
Setup The Zariman server as the Cephalon Fleet hub.
Purges old channels and rebuilds with Orokin-themed structure.

Usage: python3 setup_zariman.py
"""

import asyncio
import os
import sys

import discord

GUILD_NAME = "The Zariman"

# ═══════════════════════════════════════════════════════
# Server structure — Orokin aesthetic
# ═══════════════════════════════════════════════════════
STRUCTURE = [
    (
        "✧ CEPHALON FLEET ✧",
        [
            ("🔱┃fleet-nexus", "Central command — status updates & heartbeats"),
            ("📡┃fleet-telemetry", "Diagnostics — errors & alerts from all Cephalons"),
        ],
    ),
    (
        "◈ AXIOM ─ NBA",
        [
            ("🎯┃axiom-picks", "Daily predictions — auto-posted at 9:15 AM EST"),
            ("📊┃axiom-pipeline", "Pipeline output & refresh logs"),
            ("💬┃axiom-discussion", "Strategy talk & manual commands"),
        ],
    ),
    (
        "◈ LUMEN ─ PALWORLD",
        [
            ("🌐┃lumen-status", "Server status & player activity"),
            ("📜┃lumen-chronicles", "Server events & admin logs"),
        ],
    ),
    (
        "◈ SOLACE ─ TRADING",
        [
            ("⚡┃solace-signals", "Trade entries, exits & alerts"),
            ("📈┃solace-ledger", "Strategy performance & analytics"),
        ],
    ),
    (
        "◈ ATLAS ─ UTILITY",
        [
            ("🔮┃atlas-workshop", "General utility — Ordis at your service"),
        ],
    ),
    (
        "─── THE VOID ───",
        [
            ("🕳️┃void-archive", "Historical data & retired configurations"),
        ],
    ),
]


async def setup():
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")

        guild = discord.utils.get(client.guilds, name=GUILD_NAME)
        if not guild:
            print(f"[ERROR] Not in a guild named '{GUILD_NAME}'")
            print(f"  Guilds: {[g.name for g in client.guilds]}")
            await client.close()
            return

        print(f"Found guild: {guild.name} (ID: {guild.id})")
        print()

        # ── Phase 1: Purge old channels & categories ──
        print("═══ Purging old structure ═══")
        new_channel_names = {ch for _, channels in STRUCTURE for ch, _ in channels}
        new_category_names = {cat for cat, _ in STRUCTURE}

        for channel in guild.text_channels:
            if channel.name not in new_channel_names:
                try:
                    await channel.delete(reason="Zariman restructure")
                    print(f"  [DEL] #{channel.name}")
                except discord.Forbidden:
                    print(f"  [SKIP] #{channel.name} (no permission)")

        for category in guild.categories:
            if category.name not in new_category_names:
                try:
                    await category.delete(reason="Zariman restructure")
                    print(f"  [DEL] Category '{category.name}'")
                except discord.Forbidden:
                    print(f"  [SKIP] Category '{category.name}' (no permission)")

        print()

        # ── Phase 2: Build Orokin structure ──
        print("═══ Building Orokin structure ═══")
        existing_categories = {c.name: c for c in guild.categories}
        existing_channels = {c.name: c for c in guild.text_channels}

        for category_name, channels in STRUCTURE:
            if category_name in existing_categories:
                category = existing_categories[category_name]
                print(f"  [SKIP] {category_name}")
            else:
                category = await guild.create_category(category_name)
                print(f"  [✧] Created {category_name}")

            for channel_name, topic in channels:
                if channel_name in existing_channels:
                    print(f"    [SKIP] {channel_name}")
                else:
                    await guild.create_text_channel(channel_name, category=category, topic=topic)
                    print(f"    [✧] Created {channel_name}")

        print()
        print("═══════════════════════════════════")
        print("  The Zariman stands eternal.")
        print("═══════════════════════════════════")
        await client.close()

    token = os.environ.get("AXIOM_BOT_TOKEN")
    if not token:
        print("Set AXIOM_BOT_TOKEN environment variable")
        sys.exit(1)

    await client.start(token)


if __name__ == "__main__":
    asyncio.run(setup())
