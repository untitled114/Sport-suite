#!/usr/bin/env python3
"""
NBA Betting Picks Commands for Cephalon Axiom
"""

import asyncio
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from discord.ext import tasks

# ==================== CONFIG ====================

NBA_PROJECT = "/home/sportsuite/sport-suite"
PREDICTIONS_DIR = f"{NBA_PROJECT}/nba/betting_xl/predictions"
VENV_ACTIVATE = f"source {NBA_PROJECT}/venv/bin/activate"
ENV_SETUP = f"source {NBA_PROJECT}/.env && export DB_USER DB_PASSWORD BETTINGPROS_API_KEY ODDS_API_KEY TERM=dumb"

NBA_OWNER_ID = 759254862423916564
NBA_ADMIN_IDS = []

# Colors
COLOR_GOLD = 0xFFD700
COLOR_GREEN = 0x2ECC71
COLOR_BLUE = 0x3498DB
COLOR_PURPLE = 0x9B59B6
COLOR_RED = 0xE74C3C
COLOR_ORANGE = 0xE67E22


# ==================== HELPERS ====================


def _is_admin(user_id: int) -> bool:
    return user_id == NBA_OWNER_ID or user_id in NBA_ADMIN_IDS


def _get_today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _get_picks_file(date_str: str = None) -> Path:
    return Path(f"{PREDICTIONS_DIR}/xl_picks_{date_str or _get_today_str()}.json")


def _load_picks(date_str: str = None) -> Optional[dict]:
    """Load picks from both XL and PRO files, merging them."""
    date = date_str or _get_today_str()
    xl_file = Path(f"{PREDICTIONS_DIR}/xl_picks_{date}.json")
    pro_file = Path(f"{PREDICTIONS_DIR}/pro_picks_{date}.json")

    xl_data = None
    pro_picks = []

    # Load XL picks
    if xl_file.exists():
        try:
            with open(xl_file) as f:
                xl_data = json.load(f)
        except Exception:
            pass

    # Load PRO picks
    if pro_file.exists():
        try:
            with open(pro_file) as f:
                pro_data = json.load(f)
                pro_picks = pro_data.get("picks", []) if isinstance(pro_data, dict) else pro_data
                # Normalize PRO picks to match XL format
                for p in pro_picks:
                    p["source"] = "PRO"
                    p["model_version"] = "pro"
                    # Map PRO fields to XL fields
                    if "line" in p and "best_line" not in p:
                        p["best_line"] = p["line"]
                    if "projection" in p and "prediction" not in p:
                        p["prediction"] = p["projection"]
                    if "platform" in p and "best_book" not in p:
                        p["best_book"] = p["platform"]
                    if "ev_pct" in p and "edge_pct" not in p:
                        p["edge_pct"] = p["ev_pct"]
                    if "projection_diff" in p and "edge" not in p:
                        p["edge"] = p["projection_diff"]
                    if "probability" in p and "p_over" not in p:
                        p["p_over"] = p["probability"]
                    if "consensus_line" not in p:
                        p["consensus_line"] = p.get("line", 0)
                    if "num_books" not in p:
                        p["num_books"] = 1
                    if "opponent_team" not in p:
                        p["opponent_team"] = "OPP"
                    if "filter_tier" not in p:
                        p["filter_tier"] = p.get("filter_name", "PRO")
        except Exception:
            pass

    if not xl_data and not pro_picks:
        return None

    # Merge picks
    if xl_data:
        xl_picks = xl_data.get("picks", [])
        for p in xl_picks:
            if "source" not in p:
                p["source"] = "XL"
        all_picks = xl_picks + pro_picks
        xl_data["picks"] = all_picks
        xl_data["total_picks"] = len(all_picks)
        return xl_data
    else:
        return {"picks": pro_picks, "total_picks": len(pro_picks)}


async def _run_command(command: str, timeout: int = 600) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=NBA_PROJECT,
            executable="/bin/bash",
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode == 0, stdout.decode() if proc.returncode == 0 else stderr.decode()
    except asyncio.TimeoutError:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def _normalize_tier(tier: str) -> str:
    """Normalize tier names to match display config."""
    tier_map = {
        "goldmine": "GOLDMINE",
        "tier_a": "GOLDMINE",
        "x": "X",
        "z": "Z",
        "meta": "META",
        "star_tier": "star_tier",
    }
    return tier_map.get(tier.lower(), tier.upper())


def _format_pick_card(pick: dict) -> discord.Embed:
    """Format a detailed pick card with player context and hit rates."""

    # Basic info
    player = pick.get("player_name", "Unknown")
    market = pick.get("stat_type", pick.get("market", "POINTS"))
    side = pick.get("side", "OVER")
    opponent = pick.get("opponent_team", "???")
    is_home = pick.get("is_home", True)
    location = "vs" if is_home else "@"

    # Lines & books
    best_line = pick.get("best_line", 0)
    best_book = pick.get("best_book", "unknown").upper()
    consensus = pick.get("consensus_line", best_line)
    num_books = pick.get("num_books", 1)
    line_spread = pick.get("line_spread", 0)

    # Prediction & edge
    prediction = pick.get("prediction", 0)
    edge = pick.get("edge", 0)
    edge_pct = pick.get("edge_pct", 0)
    p_over = pick.get("p_over", 0.5)
    prob_pct = p_over * 100

    # Tier & confidence
    filter_tier = _normalize_tier(pick.get("filter_tier", ""))
    confidence = pick.get("confidence", "MEDIUM")
    model = pick.get("model_version", "xl").upper()

    # Stake sizing
    stake = pick.get("recommended_stake", 1.0)
    risk_level = pick.get("risk_level", "")

    # Model agreement
    models_agreeing = pick.get("models_agreeing", [])
    p_over_by_model = pick.get("p_over_by_model", {})
    both_agree = len(models_agreeing) >= 2 and "xl" in models_agreeing and "v3" in models_agreeing

    # Player context
    player_context = pick.get("player_context", {})

    # Tier config
    tier_info = {
        "GOLDMINE": ("GOLDMINE", COLOR_GOLD, "~70% WR"),
        "X": ("TIER X", COLOR_RED, "~65% WR"),
        "Z": ("TIER Z", COLOR_ORANGE, "~60% WR"),
        "META": ("META", COLOR_PURPLE, "~70% WR"),
        "star_tier": ("STAR", COLOR_BLUE, "~80% WR"),
    }
    tier_label, color, tier_wr = tier_info.get(filter_tier, ("STANDARD", COLOR_GREEN, "~55% WR"))

    # Build embed
    embed = discord.Embed(color=color)

    # Title: Player REBOUNDS OVER 8.5
    title = f"{player} {market} {side} {best_line}"
    embed.title = title

    # Description: Main info
    desc_lines = [
        f"**{location} {opponent}** | {tier_label} | {model}",
        "",
        f"**Prediction:** {prediction:.1f} | **Edge:** {edge_pct:+.1f}%",
        f"**Book:** {best_book} | **Prob:** {prob_pct:.0f}%",
    ]

    # Line info
    if abs(consensus - best_line) >= 0.5:
        desc_lines.append(
            f"**Consensus:** {consensus:.1f} ({best_line - consensus:+.1f}) | {num_books} books"
        )
    else:
        desc_lines.append(f"**Line:** {best_line} | {num_books} books")

    # Model agreement
    if both_agree:
        xl_p = p_over_by_model.get("xl", 0)
        v3_p = p_over_by_model.get("v3", 0)
        desc_lines.append(f"**Both Models Agree** (XL: {xl_p*100:.0f}% | V3: {v3_p*100:.0f}%)")

    # Stake
    stake_str = f"**Stake:** {stake}u"
    if risk_level and risk_level not in ("LOW", "MEDIUM", ""):
        stake_str += f" ({risk_level})"
    desc_lines.append(stake_str)

    embed.description = "\n".join(desc_lines)

    # Player Context field
    if player_context:
        avg_L5 = player_context.get("avg_L5", 0)
        avg_L10 = player_context.get("avg_L10", 0)
        avg_L20 = player_context.get("avg_L20", player_context.get("avg_season", 0))
        h2h_avg = player_context.get("h2h_avg", 0)
        h2h_games = player_context.get("h2h_games", 0)
        trend = player_context.get("trend", "STABLE")
        minutes = player_context.get("minutes_L5", 0)

        trend_emoji = {"HOT": "^", "RISING": "^", "COLD": "v", "FALLING": "v"}.get(trend, "-")

        context_lines = [
            f"L5: **{avg_L5}** | L10: **{avg_L10}** | L20: **{avg_L20}**",
        ]

        # Add H2H if significant (different from recent avg)
        if h2h_avg > 0 and h2h_games >= 2 and abs(h2h_avg - avg_L5) > 1:
            context_lines.append(f"vs OPP: **{h2h_avg}** ({h2h_games} games)")

        context_lines.append(f"MIN: {minutes:.0f} | Trend: {trend} {trend_emoji}")
        embed.add_field(name="Player Stats", value="\n".join(context_lines), inline=False)

    # Hit rates field
    hit_rates = pick.get("hit_rates", {})
    if hit_rates:
        hr_parts = []
        for period in ["last_5", "last_10", "last_15", "season"]:
            hr = hit_rates.get(period, {})
            rate = hr.get("rate", 0)
            total = hr.get("total", 0)
            if rate > 0 and total > 0:
                label = period.replace("last_", "L").replace("season", "SZN").upper()
                est = "*" if hr.get("estimated") else ""
                hr_parts.append(f"{label}: {rate*100:.0f}%{est}")
        if hr_parts:
            embed.add_field(name="Hit Rates", value=" | ".join(hr_parts), inline=False)

    # Line distribution (compact)
    line_dist = pick.get("line_distribution", [])
    if line_dist and len(line_dist) > 1:
        dist_lines = []
        for ld in line_dist[:4]:
            line_val = ld.get("line", 0)
            books = ld.get("books", [])
            books_str = ", ".join([b[:3].upper() for b in books[:4]])
            if len(books) > 4:
                books_str += f"+{len(books)-4}"
            dist_lines.append(f"{line_val}: {books_str}")

        embed.add_field(name="Lines", value=" | ".join(dist_lines), inline=False)

    return embed


def _format_summary_embed(data: dict) -> discord.Embed:
    """Format picks summary"""
    date_str = data.get("date", _get_today_str())
    picks = data.get("picks", [])
    total = len(picks)

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_display = date_obj.strftime("%A, %B %d")
    except ValueError:
        date_display = date_str

    embed = discord.Embed(
        title="NBA Picks", description=f"**{date_display}** | {total} picks", color=COLOR_BLUE
    )

    if total == 0:
        embed.add_field(name="Status", value="No picks passed filters", inline=False)
        return embed

    # Group by tier (normalize tier names)
    tiers = {}
    for p in picks:
        t = _normalize_tier(p.get("filter_tier", p.get("confidence", "OTHER")))
        tiers[t] = tiers.get(t, 0) + 1

    # Group by market
    markets = {}
    for p in picks:
        m = p.get("stat_type", p.get("market", "OTHER"))
        markets[m] = markets.get(m, 0) + 1

    # Count model agreement
    both_agree_count = sum(
        1
        for p in picks
        if len(p.get("models_agreeing", [])) >= 2
        and "xl" in p.get("models_agreeing", [])
        and "v3" in p.get("models_agreeing", [])
    )

    # Tier breakdown
    tier_order = ["GOLDMINE", "X", "Z", "META", "star_tier"]
    tier_lines = []
    for tier in tier_order:
        if tier in tiers:
            tier_lines.append(f"{tier}: **{tiers[tier]}**")
    # Add any others
    for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
        if tier not in tier_order:
            tier_lines.append(f"{tier}: **{count}**")

    embed.add_field(name="By Tier", value="\n".join(tier_lines) or "-", inline=True)

    # Market breakdown with agreement count
    market_lines = [f"{m}: **{c}**" for m, c in sorted(markets.items(), key=lambda x: -x[1])]
    if both_agree_count > 0:
        market_lines.append(f"Consensus: **{both_agree_count}**")
    embed.add_field(name="By Market", value="\n".join(market_lines) or "-", inline=True)

    # Top pick preview
    if picks:
        top = picks[0]
        player = top.get("player_name", "?")
        market = top.get("stat_type", top.get("market", "?"))
        line = top.get("best_line", 0)
        side = top.get("side", "OVER")
        edge_pct = top.get("edge_pct", 0)
        embed.add_field(
            name="Top Pick",
            value=f"**{player}** {side} {line} {market} ({edge_pct:+.1f}%)",
            inline=False,
        )

    embed.set_footer(text="/nba-detail for full cards")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


# ==================== COMMANDS ====================


def register(bot):

    @bot.tree.command(name="nba", description="Show today's NBA betting picks")
    async def nba(interaction: discord.Interaction):
        await interaction.response.defer()
        data = _load_picks()
        if not data or not data.get("picks"):
            embed = discord.Embed(
                title="NBA Picks",
                description="No picks available.\nUse `/nba-run` to generate.",
                color=COLOR_ORANGE,
            )
            await interaction.followup.send(embed=embed)
            return
        await interaction.followup.send(embed=_format_summary_embed(data))

    @bot.tree.command(name="nba-detail", description="Show detailed NBA pick cards")
    @app_commands.describe(count="Number of picks (1-25, default all)")
    async def nba_detail(interaction: discord.Interaction, count: int = 0):
        await interaction.response.defer()
        data = _load_picks()
        if not data or not data.get("picks"):
            embed = discord.Embed(
                title="NBA Picks", description="No picks available.", color=COLOR_ORANGE
            )
            await interaction.followup.send(embed=embed)
            return

        picks = data.get("picks", [])

        # Default to all picks, max 25
        if count <= 0:
            count = len(picks)
        count = max(1, min(count, 25))

        await interaction.followup.send(embed=_format_summary_embed(data))
        for pick in picks[:count]:
            await interaction.channel.send(embed=_format_pick_card(pick))
        if len(picks) > count:
            await interaction.channel.send(f"*+{len(picks) - count} more picks*")

    @bot.tree.command(name="nba-refresh", description="Quick refresh lines and picks (~1 min)")
    async def nba_refresh(interaction: discord.Interaction):
        if not _is_admin(interaction.user.id):
            await interaction.response.send_message("Admin only", ephemeral=True)
            return
        await interaction.response.defer()
        embed = discord.Embed(title="Refreshing", description="Updating lines...", color=COLOR_BLUE)
        await interaction.followup.send(embed=embed)

        cmd = f"cd {NBA_PROJECT} && {VENV_ACTIVATE} && {ENV_SETUP} && ./nba/nba-predictions.sh refresh"
        success, output = await _run_command(cmd, timeout=300)

        if success:
            data = _load_picks()
            n = len(data.get("picks", [])) if data else 0
            embed = discord.Embed(
                title="Refresh Complete", description=f"**{n}** picks", color=COLOR_GREEN
            )
            await interaction.channel.send(embed=embed)
            if data and n > 0:
                await interaction.channel.send(embed=_format_summary_embed(data))
        else:
            embed = discord.Embed(
                title="Failed", description=f"```{output[:300]}```", color=COLOR_RED
            )
            await interaction.channel.send(embed=embed)

    @bot.tree.command(name="nba-run", description="Run full prediction pipeline (~5 min)")
    async def nba_run(interaction: discord.Interaction):
        if not _is_admin(interaction.user.id):
            await interaction.response.send_message("Admin only", ephemeral=True)
            return
        await interaction.response.defer()
        embed = discord.Embed(
            title="Running Pipeline", description="~5 minutes...", color=COLOR_BLUE
        )
        await interaction.followup.send(embed=embed)

        cmd = f"cd {NBA_PROJECT} && {VENV_ACTIVATE} && {ENV_SETUP} && ./nba/nba-predictions.sh"
        success, output = await _run_command(cmd, timeout=600)

        if success:
            data = _load_picks()
            n = len(data.get("picks", [])) if data else 0
            embed = discord.Embed(title="Complete", description=f"**{n}** picks", color=COLOR_GREEN)
            await interaction.channel.send(embed=embed)
            if data and n > 0:
                await interaction.channel.send(embed=_format_summary_embed(data))
        else:
            embed = discord.Embed(
                title="Failed", description=f"```{output[:300]}```", color=COLOR_RED
            )
            await interaction.channel.send(embed=embed)

    @bot.tree.command(name="nba-status", description="Check NBA pipeline status")
    async def nba_status(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        picks_file = _get_picks_file()
        data = _load_picks()

        lines = [f"**{_get_today_str()}**\n"]
        if picks_file.exists() and data:
            n = len(data.get("picks", []))
            mtime = datetime.fromtimestamp(picks_file.stat().st_mtime).strftime("%I:%M %p")
            lines.append(f"**{n}** picks (generated {mtime})")
        else:
            lines.append("No picks yet")

        cmd = "systemctl is-active airflow-scheduler"
        ok, out = await _run_command(cmd, timeout=10)
        lines.append(f"Airflow: {'OK' if ok and 'active' in out else 'DOWN'}")

        embed = discord.Embed(title="Status", description="\n".join(lines), color=COLOR_BLUE)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @bot.tree.command(
        name="nba-reload", description="[ADMIN] Hot-reload NBA commands without bot restart"
    )
    async def nba_reload(interaction: discord.Interaction):
        if not _is_admin(interaction.user.id):
            await interaction.response.send_message("Admin only", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        try:
            import importlib

            import nba_commands as nba_mod

            # Reload the module
            importlib.reload(nba_mod)

            await interaction.followup.send(
                "NBA commands reloaded. New code active.", ephemeral=True
            )
        except Exception as e:
            await interaction.followup.send(f"Reload failed: {e}", ephemeral=True)

    @bot.tree.command(name="nba-validate", description="Show validation results for recent picks")
    @app_commands.describe(days="Number of days to validate (1-7, default 1)")
    async def nba_validate(interaction: discord.Interaction, days: int = 1):
        """Show validation results for recent picks."""
        if not _is_admin(interaction.user.id):
            await interaction.response.send_message("Admin only", ephemeral=True)
            return

        days = max(1, min(7, days))

        await interaction.response.send_message(
            embed=discord.Embed(
                title="Validating Picks",
                description=f"Checking last {days} day(s)...",
                color=COLOR_BLUE,
            )
        )

        cmd = f"source .env && export DB_USER DB_PASSWORD && ./nba/nba-predictions.sh validate"
        if days > 1:
            cmd = f"source .env && export DB_USER DB_PASSWORD && python3 nba/betting_xl/validate_predictions.py --days {days}"

        success, output = await _run_command(cmd, timeout=120)

        if not success:
            await interaction.edit_original_response(
                embed=discord.Embed(
                    title="Validation Failed", description=f"```{output[:1500]}```", color=COLOR_RED
                )
            )
            return

        lines = output.split("\n")

        embed = discord.Embed(title="Validation Results", color=COLOR_GREEN)

        system_results = []
        for line in lines:
            if ("XL" in line or "PRO" in line) and "%" in line:
                parts = line.split()
                if len(parts) >= 4:
                    system_results.append(line.strip())

        if system_results:
            results_text = "```\n"
            for r in system_results[:10]:
                results_text += r + "\n"
            results_text += "```"
            embed.add_field(name="Results by System", value=results_text, inline=False)

        daily_lines = []
        in_daily = False
        for line in lines:
            if "DAILY BREAKDOWN" in line:
                in_daily = True
                continue
            if in_daily and line.strip() and "===" not in line:
                if "W-" in line or (":" in line and ("XL" in line or "PRO" in line)):
                    daily_lines.append(line.strip())

        if daily_lines:
            daily_text = "```\n" + "\n".join(daily_lines[:8]) + "\n```"
            embed.add_field(name="Daily Breakdown", value=daily_text[:1024], inline=False)

        embed.set_footer(text=f"Validated {days} day(s) | {_get_today_str()}")
        await interaction.edit_original_response(embed=embed)

    print(
        "[NBA] Registered: /nba, /nba-detail, /nba-refresh, /nba-run, /nba-status, /nba-reload, /nba-validate"
    )


# ==================== SCHEDULED ====================


def start_scheduled_tasks(bot):
    @tasks.loop(time=time(hour=14, minute=15, tzinfo=timezone.utc))
    async def auto_post():
        """Auto-post all picks to owner at 9:15 AM EST."""
        if not NBA_OWNER_ID:
            return
        owner = bot.get_user(NBA_OWNER_ID) or await bot.fetch_user(NBA_OWNER_ID)
        if not owner:
            return
        await asyncio.sleep(60)
        data = _load_picks()
        if not data or not data.get("picks"):
            await owner.send(
                embed=discord.Embed(title="NBA", description="No picks today", color=COLOR_ORANGE)
            )
            return

        picks = data.get("picks", [])
        await owner.send(embed=_format_summary_embed(data))

        # Send ALL picks (up to 25 to avoid rate limits)
        for pick in picks[:25]:
            await owner.send(embed=_format_pick_card(pick))
            await asyncio.sleep(0.5)  # Small delay to avoid rate limits

        if len(picks) > 25:
            await owner.send(f"*+{len(picks)-25} more picks (use /nba-detail in server)*")

    @auto_post.before_loop
    async def before():
        await bot.wait_until_ready()

    auto_post.start()
    print("[NBA] Auto-post started (9:15 AM EST - sends ALL picks)")
