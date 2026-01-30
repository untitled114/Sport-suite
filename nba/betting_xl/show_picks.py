#!/usr/bin/env python3
"""
Show Today's Picks - Complete betting details for informed decisions
Usage: python3 show_picks.py [--date YYYY-MM-DD] [--compact]
"""

import argparse
import json
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
MUTED = "\033[90m"
WHITE = "\033[97m"

# Projection color (used for predicted values)
PROJ_COLOR = WHITE  # Change to CYAN, MAGENTA, or GREEN if preferred

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


def format_tier(tier):
    """Format model tier with color."""
    # All tiers use XL model (102 features) - names reflect filter criteria
    tier_base = tier.upper()
    # Current tier names
    if tier_base == "X":
        return f"{GREEN}{BOLD}X{RESET} {MUTED}(p>=0.85){RESET}"
    elif tier_base == "Z":
        return f"{CYAN}{BOLD}Z{RESET} {MUTED}(p>=0.70){RESET}"
    elif "META" in tier_base:
        return f"{GREEN}{BOLD}META{RESET} {MUTED}(~70% WR){RESET}"
    elif tier_base == "A" or "TIER_A" in tier_base:
        return f"{CYAN}{BOLD}A{RESET} {MUTED}(fallback){RESET}"
    elif "STAR" in tier_base:
        return f"{GREEN}{BOLD}STAR{RESET} {MUTED}(~80% WR){RESET}"
    elif "JAN_CONFIDENT" in tier_base:
        return f"{GREEN}{BOLD}JAN_CONF{RESET} {MUTED}(~87% WR){RESET}"
    elif "GOLDMINE" in tier_base:
        return f"{GREEN}{BOLD}GOLDMINE{RESET} {MUTED}(~70% WR){RESET}"
    # Legacy tier names for backwards compatibility
    elif "XL_HIGHCONF" in tier_base or "V3" in tier_base:
        return f"{GREEN}{BOLD}X{RESET} {MUTED}(legacy){RESET}"
    elif "XL_EDGE" in tier_base:
        return f"{CYAN}{BOLD}Z{RESET} {MUTED}(legacy){RESET}"
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
    print_header(f"XL MODEL PICKS ({len(picks)})")
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

            # Main bet line + tier
            print(
                f"  {MUTED}│{RESET} {side} {BOLD}{best_line}{RESET} @ {BOLD}{best_book}{RESET}{spread_tag}"
            )
            print(f"  {MUTED}│{RESET}  Tier: {format_tier(tier)}")

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

            edge_color = GREEN if diff > 0 else RED

            print(f"  {BOLD}{WHITE}{player}{RESET}")
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


def show_odds_api_picks(date_str, compact=False):
    """Display Odds API picks (Pick6 + BettingPros), grouped by stat type."""
    filepath = find_picks_file("odds_api_picks", date_str)
    if not filepath:
        print(f"\n  {MUTED}No Odds API picks file for {date_str}.{RESET}")
        return

    data = load_json(filepath)

    if not data or not data.get("picks"):
        print(f"\n  {MUTED}No Odds API picks for {date_str}.{RESET}")
        return

    picks = data["picks"]
    print_header(f"ODDS API PICKS ({len(picks)})")
    print(f"  {MUTED}Strategy:{RESET} Pick6 multipliers + BettingPros cheatsheet")

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
        # Sort by multiplier (lower = better) within stat type
        stat_picks_sorted = sorted(stat_picks, key=lambda p: p.get("pick6_multiplier", 1.0))

        print(f"\n  {BOLD}{CYAN}── {stat} ({len(stat_picks)}) ──{RESET}")

        for i, pick in enumerate(stat_picks_sorted):
            if i > 0:
                print_divider()

            player = pick.get("player_name", "Unknown")
            side = pick.get("side", "OVER")
            line = pick.get("line", "?")
            multiplier = pick.get("pick6_multiplier", 1.0)
            projection = pick.get("projection", 0)
            proj_diff = pick.get("projection_diff", 0)
            ev_pct = pick.get("ev_pct", 0)
            l5 = pick.get("hit_rate_l5", 0)
            l15 = pick.get("hit_rate_l15", 0)
            season = pick.get("hit_rate_season", 0)
            opp_rank = pick.get("opp_rank", "?")
            expected_wr = pick.get("expected_wr", "?")
            reasoning = pick.get("reasoning", "")
            platform = pick.get("platform", "underdog").upper()

            # Multiplier color (lower = easier = green)
            mult_color = GREEN if multiplier < 0.9 else (CYAN if multiplier < 1.2 else RED)

            # Edge color
            edge_color = GREEN if proj_diff > 0 else RED

            print()
            print(f"  {BOLD}{WHITE}{player}{RESET}")
            print(f"  {MUTED}│{RESET} {side} {BOLD}{line}{RESET} @ {BOLD}{platform}{RESET}")
            print(
                f"  {MUTED}│{RESET}  Pick6 Mult: {mult_color}{BOLD}{multiplier:.2f}x{RESET}  {MUTED}│{RESET}  EV: {GREEN}{BOLD}{ev_pct:.1f}%{RESET}"
            )
            print(
                f"  {MUTED}│{RESET}  Projection: {BOLD}{CYAN}{projection:.1f}{RESET}  {MUTED}│{RESET}  Edge: {edge_color}{BOLD}{proj_diff:+.1f}{RESET}"
            )

            # Hit rates
            l5_pct = int(l5 * 100) if l5 <= 1 else int(l5)
            l15_pct = int(l15 * 100) if l15 <= 1 else int(l15)
            szn_pct = int(season * 100) if season <= 1 else int(season)
            print(
                f"  {MUTED}│{RESET}  L5: {BOLD}{l5_pct}%{RESET}  {MUTED}│{RESET}  L15: {BOLD}{l15_pct}%{RESET}  {MUTED}│{RESET}  Season: {BOLD}{szn_pct}%{RESET}"
            )

            # Opponent rank
            print(
                f"  {MUTED}│{RESET}  vs #{opp_rank} defense  {MUTED}│{RESET}  Expected: {GREEN}{BOLD}{expected_wr}%{RESET} WR"
            )

            if not compact and reasoning:
                # Wrap reasoning if too long
                if len(reasoning) > 65:
                    print(f"  {MUTED}│{RESET}  {MUTED}Reason: {reasoning[:65]}...{RESET}")
                else:
                    print(f"  {MUTED}│{RESET}  {MUTED}Reason: {reasoning}{RESET}")

            print()


def show_summary(date_str):
    """Display daily summary with expected performance."""
    # Check for XL picks
    xl_path = find_picks_file("xl_picks", date_str)
    xl_data = load_json(xl_path) if xl_path else None

    # Check for pro picks
    pro_path = find_picks_file("pro_picks", date_str)
    pro_data = load_json(pro_path) if pro_path else None

    # Check for odds API picks
    odds_path = find_picks_file("odds_api_picks", date_str)
    odds_data = load_json(odds_path) if odds_path else None

    xl_count = len(xl_data.get("picks", [])) if xl_data else 0
    pro_count = len(pro_data.get("picks", [])) if pro_data else 0
    odds_count = len(odds_data.get("picks", [])) if odds_data else 0

    if xl_count == 0 and pro_count == 0 and odds_count == 0:
        return

    print_header("SUMMARY")

    print(f"  {MUTED}Source{RESET}           {MUTED}Picks{RESET}   {MUTED}Expected WR{RESET}")
    print(f"  {MUTED}{'─'*45}{RESET}")

    total = 0

    if xl_count > 0:
        expected = xl_data.get("expected_performance", {})
        points_wr = expected.get("POINTS", {}).get("win_rate", 56.7)
        rebounds_wr = expected.get("REBOUNDS", {}).get("win_rate", 61.2)

        # Count by market and tier
        by_market = {}
        by_tier = {}
        for p in xl_data.get("picks", []):
            m = p.get("stat_type", "OTHER")
            t = p.get("filter_tier", "?")
            by_market[m] = by_market.get(m, 0) + 1
            by_tier[t] = by_tier.get(t, 0) + 1

        pts_count = by_market.get("POINTS", 0)
        reb_count = by_market.get("REBOUNDS", 0)

        if pts_count:
            print(f"  XL POINTS        {pts_count:>3}     ~{points_wr:.0f}%")
            total += pts_count
        if reb_count:
            print(f"  XL REBOUNDS      {reb_count:>3}     ~{rebounds_wr:.0f}%")
            total += reb_count

        # Tier breakdown (all use XL model - tiers are filter criteria)
        # Count current + legacy tier names
        x_count = by_tier.get("X", 0) + by_tier.get("XL_HIGHCONF", 0) + by_tier.get("V3", 0)
        z_count = by_tier.get("Z", 0) + by_tier.get("XL_EDGE", 0)
        meta_count = by_tier.get("META", 0)
        if x_count:
            print(f"    {MUTED}└ X{RESET}            {x_count:>3}     {GREEN}p>=0.85{RESET}")
        if z_count:
            print(f"    {MUTED}└ Z{RESET}            {z_count:>3}     {CYAN}p>=0.70{RESET}")
        if meta_count:
            print(f"    {MUTED}└ META{RESET}         {meta_count:>3}     {GREEN}~70%{RESET}")

        # Goldmine stats
        goldmine = [p for p in xl_data.get("picks", []) if p.get("line_spread", 0) >= 2.0]
        if goldmine:
            print(
                f"  {GREEN}GOLDMINE{RESET}         {len(goldmine):>3}     {GREEN}{BOLD}~70%{RESET} (spread ≥2.0)"
            )

    if pro_count > 0:
        print(f"  PRO PICKS        {pro_count:>3}     75-88%")
        total += pro_count

    if odds_count > 0:
        print(f"  ODDS API         {odds_count:>3}     ~100% (tight filters)")
        total += odds_count

    print(f"  {MUTED}{'─'*45}{RESET}")
    print(f"  {BOLD}TOTAL{RESET}            {BOLD}{total:>3}{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show today's betting picks with full details")
    parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD)", default=None)
    parser.add_argument("--xl-only", action="store_true", help="Show only XL picks")
    parser.add_argument("--pro-only", action="store_true", help="Show only pro picks")
    parser.add_argument("--odds-only", action="store_true", help="Show only Odds API picks")
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
    elif args.odds_only:
        show_odds_api_picks(date_str, compact=args.compact)
    else:
        show_xl_picks(date_str, compact=args.compact)
        show_pro_picks(date_str, compact=args.compact)
        show_odds_api_picks(date_str, compact=args.compact)
        show_summary(date_str)

    print(f"{BOLD}{'═'*70}{RESET}")
    print()


if __name__ == "__main__":
    main()
