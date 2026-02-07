#!/usr/bin/env python3
"""
NBA Betting Picks Commands for Cephalon Axiom
"""

import asyncio
import json
import logging
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
ENV_SETUP = f"source {NBA_PROJECT}/.env && export DB_USER DB_PASSWORD BETTINGPROS_API_KEY ODDS_API_KEY THEODDSAPI_KEY TERM=xterm TZ=America/New_York"

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
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


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
    try:
        with open(xl_file) as f:
            xl_data = json.load(f)
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load XL picks from {xl_file}: {e}")

    # Load PRO picks
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
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load PRO picks from {pro_file}: {e}")

    # Load Two Energy picks
    te_file = Path(f"{PREDICTIONS_DIR}/two_energy_picks_{date}.json")
    te_picks = []
    try:
        with open(te_file) as f:
            te_data = json.load(f)
            te_picks = te_data.get("picks", []) if isinstance(te_data, dict) else te_data
            for p in te_picks:
                p["source"] = "TWO_ENERGY"
                p["model_version"] = "two_energy"
                if "line" in p and "best_line" not in p:
                    p["best_line"] = p["line"]
                if "book" in p and "best_book" not in p:
                    p["best_book"] = p["book"]
                if "filter_tier" not in p:
                    p["filter_tier"] = p.get("energy", "TWO_ENERGY")
                if "consensus_line" not in p:
                    p["consensus_line"] = p.get("line", 0)
                if "prediction" not in p:
                    p["prediction"] = p.get("line", 0)
                if "edge" not in p:
                    p["edge"] = p.get("deflation", p.get("line_inflate", 0))
                if "edge_pct" not in p:
                    line_val = p.get("line", 1)
                    p["edge_pct"] = (p["edge"] / line_val * 100) if line_val else 0
                if "p_over" not in p:
                    wr = p.get("expected_wr", 75)
                    p["p_over"] = wr / 100.0
                if "num_books" not in p:
                    p["num_books"] = 1
                if "opponent_team" not in p:
                    p["opponent_team"] = p.get("game_key", "OPP")
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load Two Energy picks from {te_file}: {e}")

    if not xl_data and not pro_picks and not te_picks:
        return None

    # Merge picks
    if xl_data:
        xl_picks = xl_data.get("picks", [])
        for p in xl_picks:
            if "source" not in p:
                p["source"] = "XL"
        all_picks = xl_picks + pro_picks + te_picks
        xl_data["picks"] = all_picks
        xl_data["total_picks"] = len(all_picks)
        return xl_data
    else:
        all_picks = pro_picks + te_picks
        return {"picks": all_picks, "total_picks": len(all_picks)}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re

    # Remove all ANSI escape sequences
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\[[\?0-9;]*[a-zA-Z]")
    return ansi_pattern.sub("", text)


async def _run_command(command: str, timeout: int = 600) -> tuple[bool, str]:
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=NBA_PROJECT,
            executable="/bin/bash",
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (
            stdout.decode()
            if proc.returncode == 0
            else (stderr.decode() or stdout.decode()[-2000:])
        )
        # Strip ANSI escape codes for clean Discord display
        output = _strip_ansi(output)
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            await proc.wait()
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

    # Blowout risk warning
    blowout = pick.get("blowout_risk")
    if blowout:
        level = blowout["level"]
        spread = blowout["abs_spread"]
        if level == "EXTREME":
            desc_lines.append(f"**!! BLOWOUT RISK** (spread {spread:.0f}) - bench risk")
        elif level == "HIGH":
            desc_lines.append(f"**! BLOWOUT RISK** (spread {spread:.0f})")
        else:
            desc_lines.append(f"**~ Blowout Watch** (spread {spread:.0f})")

    # Same-game correlation warning
    same_game = pick.get("same_game_players", [])
    if same_game:
        names = ", ".join(same_game[:3])
        desc_lines.append(f"**Same game as:** {names}")

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

    # Top 3 picks preview
    if picks:
        top_lines = []
        for p in picks[:3]:
            player = p.get("player_name", "?")
            market = p.get("stat_type", p.get("market", "?"))
            line = p.get("best_line", 0)
            side = p.get("side", "OVER")
            edge_pct = p.get("edge_pct", 0)
            top_lines.append(f"**{player}** {side} {line} {market} ({edge_pct:+.1f}%)")
        embed.add_field(
            name="Top Picks",
            value="\n".join(top_lines),
            inline=False,
        )

    # Blowout Watch section
    blowout_picks = [p for p in picks if p.get("blowout_risk")]
    if blowout_picks:
        bl_lines = []
        for p in sorted(blowout_picks, key=lambda x: x["blowout_risk"]["abs_spread"], reverse=True)[
            :5
        ]:
            level = p["blowout_risk"]["level"]
            spread = p["blowout_risk"]["abs_spread"]
            bl_lines.append(f"{p['player_name']}: {level} (spread {spread:.0f})")
        embed.add_field(name="Blowout Watch", value="\n".join(bl_lines), inline=False)

    # Same-Game Groups section (don't parlay together)
    game_groups = {}
    for p in picks:
        gk = p.get("game_key")
        if gk:
            game_groups.setdefault(gk, []).append(p["player_name"])
    multi = {k: v for k, v in game_groups.items() if len(v) > 1}
    if multi:
        gg_lines = []
        for gk, players in list(multi.items())[:4]:
            gg_lines.append(f"**{gk}**: {', '.join(players[:4])}")
        embed.add_field(
            name="Same-Game Groups (don't parlay together)",
            value="\n".join(gg_lines),
            inline=False,
        )

    embed.set_footer(text="/nba-detail for full cards")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


class PickPaginator(discord.ui.View):
    """Paginated view for browsing pick cards with Prev/Next buttons."""

    def __init__(
        self, picks: list, summary_embed: discord.Embed, per_page: int = 5, timeout: int = 600
    ):
        super().__init__(timeout=timeout)
        self.picks = picks
        self.summary_embed = summary_embed
        self.per_page = per_page
        self.page = 0
        self.total_pages = max(1, (len(picks) + per_page - 1) // per_page)
        self._update_buttons()

    def _update_buttons(self):
        self.prev_btn.disabled = self.page <= 0
        self.next_btn.disabled = self.page >= self.total_pages - 1
        self.page_indicator.label = f"{self.page + 1}/{self.total_pages}"

    def get_embeds(self) -> list:
        start = self.page * self.per_page
        end = min(start + self.per_page, len(self.picks))
        embeds = []
        if self.page == 0:
            embeds.append(self.summary_embed)
        else:
            header = discord.Embed(
                description=f"Page {self.page + 1}/{self.total_pages} | Picks {start + 1}-{end} of {len(self.picks)}",
                color=COLOR_BLUE,
            )
            embeds.append(header)
        for pick in self.picks[start:end]:
            embeds.append(_format_pick_card(pick))
        return embeds

    @discord.ui.button(label="<", style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page = max(0, self.page - 1)
        self._update_buttons()
        await interaction.response.edit_message(embeds=self.get_embeds(), view=self)

    @discord.ui.button(label="1/1", style=discord.ButtonStyle.secondary, disabled=True)
    async def page_indicator(self, interaction: discord.Interaction, button: discord.ui.Button):
        pass  # Non-interactive page label

    @discord.ui.button(label=">", style=discord.ButtonStyle.primary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page = min(self.total_pages - 1, self.page + 1)
        self._update_buttons()
        await interaction.response.edit_message(embeds=self.get_embeds(), view=self)

    async def on_timeout(self):
        self.prev_btn.disabled = True
        self.next_btn.disabled = True
        self.page_indicator.label = "Expired"


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
    @app_commands.describe(count="Number of picks (default all)")
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
        if count > 0:
            picks = picks[:count]

        summary = _format_summary_embed(data)

        if len(picks) <= 5:
            # Few enough to show all at once (no pagination needed)
            embeds = [summary] + [_format_pick_card(p) for p in picks]
            await interaction.followup.send(embeds=embeds)
        else:
            # Paginate: 5 picks per page with Prev/Next buttons
            view = PickPaginator(picks, summary)
            await interaction.followup.send(embeds=view.get_embeds(), view=view)

    @bot.tree.command(name="nba-refresh", description="Quick refresh lines and picks (~1 min)")
    async def nba_refresh(interaction: discord.Interaction):
        if not _is_admin(interaction.user.id):
            await interaction.response.send_message("Admin only", ephemeral=True)
            return
        await interaction.response.defer()
        embed = discord.Embed(title="Refreshing", description="Updating lines...", color=COLOR_BLUE)
        await interaction.followup.send(embed=embed)

        cmd = f"cd {NBA_PROJECT} && {VENV_ACTIVATE} && {ENV_SETUP} && python3 nba/betting_xl/quick_refresh.py"
        success, output = await _run_command(cmd, timeout=600)

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
        summary = _format_summary_embed(data)

        if len(picks) <= 5:
            embeds = [summary] + [_format_pick_card(p) for p in picks]
            await owner.send(embeds=embeds)
        else:
            view = PickPaginator(picks, summary, timeout=1800)
            await owner.send(embeds=view.get_embeds(), view=view)

    @auto_post.before_loop
    async def before():
        await bot.wait_until_ready()

    auto_post.start()
    print("[NBA] Auto-post started (9:15 AM EST - sends ALL picks)")
