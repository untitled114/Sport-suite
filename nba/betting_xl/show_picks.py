#!/usr/bin/env python3
"""
Show Today's Picks - Complete betting details for informed decisions
Usage: python3 show_picks.py [--date YYYY-MM-DD] [--table] [--detail]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ANSI colors
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
MUTED = "\033[90m"
WHITE = "\033[97m"

# Projection color (used for predicted values)
PROJ_COLOR = WHITE  # Change to CYAN, MAGENTA, or GREEN if preferred

# Book name abbreviations for table view
BOOK_ABBREV = {
    "draftkings": "DK",
    "fanduel": "FD",
    "betmgm": "MGM",
    "caesars": "CZR",
    "betrivers": "BR",
    "espnbet": "ESPN",
    "fanatics": "FAN",
    "underdog": "UNDRDG",
    "prizepicks": "PP",
    "bet365": "B365",
}


def abbrev_book(book):
    """Abbreviate book name for table display."""
    return BOOK_ABBREV.get(book.lower(), book.upper()[:6])


PREDICTIONS_DIR = Path(__file__).parent / "predictions"


def load_json(filepath):
    """Load JSON file if it exists."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def print_header(text):
    """Print section header."""
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"  {BOLD}{WHITE}{text}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")


def print_divider():
    """Print light divider between picks."""
    print(f"  {MUTED}{'─'*66}{RESET}")


def format_confidence(confidence, p_over):
    """Format confidence with color."""
    p_pct = int(p_over * 100)
    if confidence == "HIGH":
        return f"{GREEN}{BOLD}HIGH{RESET} ({p_pct}%)"
    elif confidence == "MEDIUM":
        return f"{CYAN}{BOLD}MEDIUM{RESET} ({p_pct}%)"
    else:
        return f"{CYAN}STANDARD{RESET} ({p_pct}%)"


def format_edge(edge_pct, edge):
    """Format edge with color (green positive, red negative)."""
    color = GREEN if edge_pct >= 0 else RED
    sign = "+" if edge_pct >= 0 else ""
    return f"{color}{BOLD}{sign}{edge_pct:.1f}%{RESET} {MUTED}({edge:+.1f}){RESET}"


def format_line_distribution(line_dist):
    """Format line distribution as a compact string."""
    if not line_dist:
        return ""
    parts = []
    for item in line_dist:
        books = ", ".join(b.upper()[:3] for b in item.get("books", []))
        parts.append(f"{item['line']}:{item['count']} ({books})")
    return " | ".join(parts)


def find_picks_file(prefix, date_str):
    """Find picks file trying both date formats."""
    # Try with dashes first (YYYY-MM-DD)
    filepath = PREDICTIONS_DIR / f"{prefix}_{date_str}.json"
    if filepath.exists():
        return filepath

    # Try without dashes (YYYYMMDD)
    date_nodash = date_str.replace("-", "")
    filepath = PREDICTIONS_DIR / f"{prefix}_{date_nodash}.json"
    if filepath.exists():
        return filepath

    return None


def format_tier_short(tier):
    """Short tier name for table view (no ANSI, plain text)."""
    t = tier.upper()
    if "STAR" in t:
        return "Star"
    elif "GOLDMINE" in t:
        return "Goldmine"
    elif "STANDARD" in t:
        return "Standard"
    elif "JAN_CONFIDENT" in t:
        return "JanConf"
    elif t in ("X", "Y") or "HIGHCONF" in t or "META" in t:
        return "Goldmine"
    elif t in ("Z", "E", "A") or "EDGE" in t or "TIER_A" in t:
        return "Standard"
    return tier[:8]


def format_tier(tier):
    """Format model tier with color."""
    # All tiers use XL model (102 features) - names reflect filter criteria
    # Current tiers: Star, Goldmine, Standard
    tier_base = tier.upper()

    if "STAR" in tier_base:
        return f"{GREEN}{BOLD}Star{RESET} {MUTED}(~80% WR){RESET}"
    elif "GOLDMINE" in tier_base:
        return f"{GREEN}{BOLD}Goldmine{RESET} {MUTED}(~70% WR){RESET}"
    elif "STANDARD" in tier_base:
        return f"{CYAN}{BOLD}Standard{RESET} {MUTED}(~56% WR){RESET}"
    elif "JAN_CONFIDENT" in tier_base:
        return f"{GREEN}{BOLD}JAN_CONF{RESET} {MUTED}(~87% WR){RESET}"
    # Legacy tier names for backwards compatibility
    elif tier_base in ("X", "Y") or "XL_HIGHCONF" in tier_base or "META" in tier_base:
        return f"{GREEN}{BOLD}Goldmine{RESET} {MUTED}(legacy){RESET}"
    elif tier_base in ("Z", "E", "A") or "XL_EDGE" in tier_base or "TIER_A" in tier_base:
        return f"{CYAN}{BOLD}Standard{RESET} {MUTED}(legacy){RESET}"
    else:
        return f"{MUTED}{tier}{RESET}"


def show_xl_picks(date_str, compact=False):
    """Display XL model picks with full betting details, grouped by stat type."""
    filepath = find_picks_file("xl_picks", date_str)
    if not filepath:
        print(f"\n  {MUTED}No XL picks file for {date_str}.{RESET}")
        return

    data = load_json(filepath)

    if not data or not data.get("picks"):
        print(f"\n  {MUTED}No XL picks for {date_str}.{RESET}")
        return

    picks = data["picks"]
    # Count by model version
    xl_count = sum(1 for p in picks if p.get("model_version", "xl") == "xl")
    v3_count = sum(1 for p in picks if p.get("model_version") == "v3")
    model_breakdown = f"XL: {xl_count}, V3: {v3_count}" if v3_count > 0 else f"{len(picks)} total"

    print_header(f"XL + V3 MODEL PICKS ({len(picks)})")
    print(f"  {MUTED}Models:{RESET} {model_breakdown}")
    print(f"  {MUTED}Strategy:{RESET} {data.get('strategy', 'N/A')}")
    print(f"  {MUTED}Markets:{RESET} {', '.join(data.get('markets_enabled', []))}")

    # Group by stat type
    by_stat = {}
    for pick in picks:
        stat = pick.get("stat_type", "OTHER")
        if stat not in by_stat:
            by_stat[stat] = []
        by_stat[stat].append(pick)

    # Define stat type order
    stat_order = ["POINTS", "REBOUNDS", "ASSISTS", "THREES", "PA", "PR", "RA", "PRA"]

    for stat in stat_order:
        if stat not in by_stat:
            continue

        stat_picks = by_stat[stat]
        # Sort by edge within stat type
        stat_picks_sorted = sorted(stat_picks, key=lambda p: p.get("edge_pct", 0), reverse=True)

        print(f"\n  {BOLD}{CYAN}── {stat} ({len(stat_picks)}) ──{RESET}")

        for i, pick in enumerate(stat_picks_sorted):
            if i > 0:
                print_divider()

            player = pick["player_name"]
            side = pick["side"]
            best_line = pick["best_line"]
            best_book = pick["best_book"].upper()
            edge_pct = pick.get("edge_pct", 0)
            edge = pick.get("edge", 0)
            pred = pick["prediction"]
            p_over = pick.get("p_over", 0.5)
            confidence = pick.get("confidence", "STANDARD")
            opp = pick.get("opponent_team", "?")
            is_home = pick.get("is_home", True)
            consensus = pick.get("consensus_line", best_line)
            line_spread = pick.get("line_spread", 0)
            num_books = pick.get("num_books", 1)
            line_dist = pick.get("line_distribution", [])
            top_3 = pick.get("top_3_lines", [])
            tier = pick.get("filter_tier", "?")

            matchup = f"vs {opp}" if is_home else f"@ {opp}"
            home_away = "HOME" if is_home else "AWAY"

            # Goldmine indicator for high spread
            spread_tag = ""
            if line_spread >= 2.0:
                spread_tag = f" {GREEN}{BOLD}[GOLDMINE]{RESET}"
            elif line_spread >= 1.0:
                spread_tag = f" {MUTED}[spread: {line_spread}]{RESET}"

            print()
            # Player name + matchup + home/away
            print(f"  {BOLD}{WHITE}{player}{RESET}  {MUTED}{matchup} ({home_away}){RESET}")

            # Main bet line + tier + model version
            model_ver = pick.get("model_version", "xl").upper()
            print(
                f"  {MUTED}│{RESET} {side} {BOLD}{best_line}{RESET} @ {BOLD}{best_book}{RESET}{spread_tag}"
            )
            print(
                f"  {MUTED}│{RESET}  Tier: {format_tier(tier)}  {MUTED}│{RESET}  Model: {CYAN}{model_ver}{RESET}"
            )

            # Projection and Edge
            print(
                f"  {MUTED}│{RESET}  Prediction: {BOLD}{PROJ_COLOR}{pred:.1f}{RESET}  {MUTED}│{RESET}  Edge: {format_edge(edge_pct, edge)}"
            )

            # Consensus and Books
            consensus_diff = best_line - consensus
            diff_color = GREEN if consensus_diff < 0 else (RED if consensus_diff > 0 else MUTED)
            print(
                f"  {MUTED}│{RESET}  Consensus: {consensus:.1f} {diff_color}({consensus_diff:+.1f}){RESET}  {MUTED}│{RESET}  Books: {num_books} offering"
            )

            # Confidence and P(over)
            print(f"  {MUTED}│{RESET}  Confidence: {format_confidence(confidence, p_over)}")

            # Model agreement indicator
            models_agreeing = pick.get("models_agreeing", [])
            p_over_by_model = pick.get("p_over_by_model", {})
            both_agree = (
                len(models_agreeing) >= 2 and "xl" in models_agreeing and "v3" in models_agreeing
            )
            if both_agree:
                xl_p = p_over_by_model.get("xl", 0)
                v3_p = p_over_by_model.get("v3", 0)
                print(
                    f"  {MUTED}│{RESET}  {GREEN}{BOLD}✓ BOTH MODELS AGREE{RESET}  "
                    f"{MUTED}XL: {xl_p*100:.0f}% | V3: {v3_p*100:.0f}%{RESET}"
                )

            # Stake sizing (always show)
            stake = pick.get("recommended_stake", 1.0)
            stake_reason = pick.get("stake_reason", "")
            risk_level = pick.get("risk_level", "")
            if stake > 1.0:
                stake_color = GREEN
            elif stake < 1.0:
                stake_color = CYAN if stake >= 0.5 else RED
            else:
                stake_color = WHITE
            risk_str = (
                f" {MUTED}({risk_level} risk){RESET}"
                if risk_level and risk_level not in ("LOW", "")
                else ""
            )
            print(
                f"  {MUTED}│{RESET}  {BOLD}Stake:{RESET} {stake_color}{BOLD}{stake}u{RESET}{risk_str}"
            )
            if stake_reason and stake != 1.0:
                print(f"  {MUTED}│{RESET}  {MUTED}Reason: {stake_reason}{RESET}")

            if not compact:
                # Line distribution
                if line_dist:
                    dist_str = format_line_distribution(line_dist)
                    if len(dist_str) > 60:
                        # Multi-line for long distributions
                        print(f"  {MUTED}│{RESET}  Lines:")
                        for item in line_dist:
                            books_list = ", ".join(item.get("books", []))
                            item_edge = item.get("edge_pct", 0)
                            edge_color = GREEN if item_edge >= 0 else RED
                            print(
                                f"  {MUTED}│{RESET}    {item['line']}: {books_list} {edge_color}({item_edge:+.1f}%){RESET}"
                            )
                    else:
                        print(f"  {MUTED}│{RESET}  Lines: {dist_str}")

                # Alternative books (from top_3 excluding best)
                alt_books = [
                    t["book"].upper() for t in top_3[1:3] if t["book"].lower() != best_book.lower()
                ]
                if alt_books:
                    print(f"  {MUTED}│{RESET}  Also at: {', '.join(alt_books)}")

            print()


def show_pro_picks(date_str, compact=False):
    """Display pro/cheatsheet picks with detailed stats."""
    filepath = find_picks_file("pro_picks", date_str)
    if not filepath:
        print(f"\n  {MUTED}No pro picks file for {date_str}.{RESET}")
        return

    data = load_json(filepath)

    if not data or not data.get("picks"):
        print(f"\n  {MUTED}No pro picks for {date_str}.{RESET}")
        return

    picks = data["picks"]
    print_header(f"PRO PICKS ({len(picks)})")

    # Group by stat type
    by_stat = {}
    for pick in picks:
        stat = pick.get("stat_type", "OTHER")
        if stat not in by_stat:
            by_stat[stat] = []
        by_stat[stat].append(pick)

    for stat, stat_picks in by_stat.items():
        print(f"\n  {BOLD}{CYAN}{stat}{RESET} ({len(stat_picks)} picks)")
        print_divider()

        for pick in stat_picks:
            player = pick.get("player_name", "Unknown")
            side = pick.get("side", "OVER")
            line = pick.get("line", "?")
            projection = pick.get("projection", 0)
            diff = pick.get("projection_diff", 0)
            l5_rate = pick.get("hit_rate_l5", 0)
            l15_rate = pick.get("hit_rate_l15", 0)
            season_rate = pick.get("hit_rate_season", 0)
            opp_rank = pick.get("opp_rank", "?")
            expected_wr = pick.get("expected_wr", "?")
            filter_name = pick.get("filter_name", "")

            edge_color = GREEN if diff > 0 else RED

            # Format filter name for display (use cyan for visibility)
            filter_display = f"  {CYAN}[{filter_name}]{RESET}" if filter_name else ""

            print(f"  {BOLD}{WHITE}{player}{RESET}{filter_display}")
            print(f"  {MUTED}│{RESET} {stat} {side} {BOLD}{line}{RESET} @ Underdog")
            print(
                f"  {MUTED}│{RESET}  Projection: {BOLD}{CYAN}{projection:.1f}{RESET}  {MUTED}│{RESET}  Edge: {edge_color}{BOLD}{diff:+.1f}{RESET}"
            )

            if not compact:
                # Hit rates
                rates = []
                if l5_rate:
                    rates.append(f"L5: {int(l5_rate*100) if l5_rate < 1 else int(l5_rate)}%")
                if l15_rate:
                    rates.append(f"L15: {int(l15_rate*100) if l15_rate < 1 else int(l15_rate)}%")
                if season_rate:
                    rates.append(
                        f"Season: {int(season_rate*100) if season_rate < 1 else int(season_rate)}%"
                    )
                if rates:
                    print(f"  {MUTED}│{RESET}  Hit rates: {' | '.join(rates)}")

                print(
                    f"  {MUTED}│{RESET}  vs #{opp_rank} defense  {MUTED}│{RESET}  Expected: {GREEN}{BOLD}{expected_wr}%{RESET} WR"
                )

            print()


def load_all_picks(date_str):
    """Load all picks from all systems, returning (all_picks, sources dict)."""
    sources = {}
    for prefix in ("xl_picks", "pro_picks"):
        path = find_picks_file(prefix, date_str)
        if path:
            data = load_json(path)
            if data and data.get("picks"):
                sources[prefix] = data
    return sources


def count_all_picks(date_str):
    """Count total picks across all systems without printing."""
    sources = load_all_picks(date_str)
    return sum(len(d.get("picks", [])) for d in sources.values())


def show_summary(date_str):
    """Display rich daily summary matching Discord bot format."""
    sources = load_all_picks(date_str)
    if not sources:
        return

    # Merge all picks into one list with source tags
    all_picks = []
    for prefix, data in sources.items():
        for p in data.get("picks", []):
            p["_source"] = prefix
            all_picks.append(p)

    total = len(all_picks)

    # Date display
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_display = date_obj.strftime("%A, %B %d")
    except ValueError:
        date_display = date_str

    print_header(f"SUMMARY  {MUTED}{date_display} | {total} picks{RESET}")

    # --- By Tier ---
    tiers = {}
    for p in all_picks:
        src = p["_source"]
        if src == "xl_picks":
            tier = p.get("filter_tier", "OTHER")
            t = tier.upper()
            if "GOLDMINE" in t or t in ("X", "Y") or "HIGHCONF" in t or "META" in t:
                label = "GOLDMINE"
            elif "STAR" in t:
                label = "STAR"
            elif "STANDARD" in t or t in ("Z", "E", "A") or "EDGE" in t:
                label = "STANDARD"
            else:
                label = tier
        elif src == "pro_picks":
            label = "PRO"
        else:
            label = "OTHER"
        tiers[label] = tiers.get(label, 0) + 1

    # --- By Market ---
    markets = {}
    for p in all_picks:
        m = p.get("stat_type", "OTHER")
        markets[m] = markets.get(m, 0) + 1

    # --- Consensus (both XL + V3 agree) ---
    consensus_count = sum(
        1
        for p in all_picks
        if len(p.get("models_agreeing", [])) >= 2
        and "xl" in p.get("models_agreeing", [])
        and "v3" in p.get("models_agreeing", [])
    )

    # Print tier and market side by side
    tier_lines = []
    tier_order = ["GOLDMINE", "STAR", "STANDARD", "PRO"]
    for t in tier_order:
        if t in tiers:
            tier_lines.append(f"{t}: {BOLD}{tiers[t]}{RESET}")
    for t, c in sorted(tiers.items(), key=lambda x: -x[1]):
        if t not in tier_order:
            tier_lines.append(f"{t}: {BOLD}{c}{RESET}")

    market_lines = []
    for m, c in sorted(markets.items(), key=lambda x: -x[1]):
        market_lines.append(f"{m}: {BOLD}{c}{RESET}")
    if consensus_count > 0:
        market_lines.append(f"Consensus: {GREEN}{BOLD}{consensus_count}{RESET}")

    print(f"\n  {BOLD}{CYAN}By Tier{RESET}                    {BOLD}{CYAN}By Market{RESET}")
    max_lines = max(len(tier_lines), len(market_lines))
    for i in range(max_lines):
        left = f"  {tier_lines[i]}" if i < len(tier_lines) else ""
        right = market_lines[i] if i < len(market_lines) else ""
        # Strip ANSI for padding calculation
        left_plain = re.sub(r"\033\[[0-9;]*m", "", left)
        padding = max(1, 30 - len(left_plain))
        print(f"{left}{' ' * padding}{right}")

    # --- Top Picks (highest edge from XL picks, sorted) ---
    xl_picks = [p for p in all_picks if p["_source"] == "xl_picks"]
    if xl_picks:
        top = sorted(xl_picks, key=lambda p: p.get("edge_pct", 0), reverse=True)[:3]
        print(f"\n  {BOLD}{CYAN}Top Picks{RESET}")
        for p in top:
            player = p.get("player_name", "?")
            side = p.get("side", "OVER")
            line = p.get("best_line", 0)
            stat = p.get("stat_type", "?")
            edge_pct = p.get("edge_pct", 0)
            print(f"  {WHITE}{player}{RESET} {side} {line} {stat} ({GREEN}+{edge_pct:.1f}%{RESET})")

    # --- Blowout Watch ---
    blowout_picks = [p for p in all_picks if p.get("blowout_risk")]
    if blowout_picks:
        bl_sorted = sorted(
            blowout_picks, key=lambda x: x["blowout_risk"]["abs_spread"], reverse=True
        )
        # Deduplicate by player name (same player can appear in multiple picks)
        seen = set()
        bl_unique = []
        for p in bl_sorted:
            if p["player_name"] not in seen:
                seen.add(p["player_name"])
                bl_unique.append(p)
        print(f"\n  {BOLD}{RED}Blowout Watch{RESET}")
        for p in bl_unique[:5]:
            level = p["blowout_risk"]["level"]
            spread = p["blowout_risk"]["abs_spread"]
            level_color = RED if level == "EXTREME" else YELLOW
            print(
                f"  {WHITE}{p['player_name']}{RESET}: {level_color}{level}{RESET} (spread {spread:.0f})"
            )

    # --- Same-Game Groups ---
    game_groups = {}
    for p in all_picks:
        gk = p.get("game_key")
        if gk:
            game_groups.setdefault(gk, set()).add(p["player_name"])
    multi = {k: v for k, v in game_groups.items() if len(v) > 1}
    if multi:
        print(f"\n  {BOLD}{CYAN}Same-Game Groups{RESET} {MUTED}(don't parlay together){RESET}")
        for gk, players in sorted(multi.items(), key=lambda x: -len(x[1]))[:6]:
            player_list = ", ".join(sorted(players)[:5])
            if len(players) > 5:
                player_list += f" +{len(players)-5}"
            print(f"  {BOLD}{gk}{RESET}: {player_list}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Show today's betting picks with full details")
    parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD)", default=None)
    parser.add_argument("--xl-only", action="store_true", help="Show only XL picks")
    parser.add_argument("--pro-only", action="store_true", help="Show only pro picks")
    parser.add_argument("--compact", "-c", action="store_true", help="Compact output (less detail)")
    parser.add_argument(
        "--sort",
        choices=["edge", "player", "confidence"],
        default="edge",
        help="Sort picks by (default: edge)",
    )
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"\n  {BOLD}NBA PICKS FOR {date_str}{RESET}")
    print(f"  {MUTED}Generated: {datetime.now().strftime('%I:%M %p')}{RESET}")

    if args.xl_only:
        show_xl_picks(date_str, compact=args.compact)
    elif args.pro_only:
        show_pro_picks(date_str, compact=args.compact)
    else:
        show_summary(date_str)
        show_xl_picks(date_str, compact=args.compact)
        show_pro_picks(date_str, compact=args.compact)

    print(f"{BOLD}{'═'*70}{RESET}")
    print()


if __name__ == "__main__":
    main()
