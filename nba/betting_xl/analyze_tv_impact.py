#!/usr/bin/env python3
"""
Analyze NBA pick performance split by nationally televised vs local-only games.

Uses ESPN API to determine broadcast status, then grades picks from JSON files
against player_game_logs to compute win rates by TV status.
"""

import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg2
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREDICTIONS_DIR = Path(__file__).parent / "predictions"
TV_CACHE_FILE = Path(__file__).parent / "tv_broadcast_cache.json"

DB_CONFIG = {
    "host": "localhost",
    "port": 5536,
    "dbname": "nba_players",
    "user": os.environ.get("DB_USER", "mlb_user"),
    "password": os.environ.get("DB_PASSWORD", "mlb_secure_2025"),
}

ESPN_TEAM_MAP = {
    "WSH": "WAS",
    "GS": "GSW",
    "NO": "NOP",
    "SA": "SAS",
    "NY": "NYK",
    "UTAH": "UTA",
    "PHX": "PHX",
}

STAT_COLUMN = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "three_pointers_made",
}


# ---------------------------------------------------------------------------
# ESPN API: get national TV broadcasts
# ---------------------------------------------------------------------------
def fetch_broadcasts_for_date(game_date: str) -> dict:
    """Fetch broadcast info from ESPN for a given date (YYYY-MM-DD).

    Returns: {(home_team, away_team): [network_names]} for nationally televised games.
    """
    dt = game_date.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={dt}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    national_games = {}
    all_games = {}

    for event in data.get("events", []):
        short = event.get("shortName", "")  # "OKC @ NYK"
        parts = short.split(" @ ")
        if len(parts) != 2:
            continue

        away_raw, home_raw = parts[0].strip(), parts[1].strip()
        away = ESPN_TEAM_MAP.get(away_raw, away_raw)
        home = ESPN_TEAM_MAP.get(home_raw, home_raw)

        national_nets = []
        for comp in event.get("competitions", []):
            for gb in comp.get("geoBroadcasts", []):
                if (
                    gb.get("market", {}).get("type") == "National"
                    and gb.get("type", {}).get("shortName") == "TV"
                ):
                    national_nets.append(gb["media"]["shortName"])

        key = f"{home}_{away}"
        all_games[key] = True
        if national_nets:
            national_games[key] = national_nets

    return {"national": national_games, "all": list(all_games.keys())}


def load_or_fetch_tv_cache(dates: list) -> dict:
    """Load cached TV data or fetch from ESPN API for missing dates."""
    cache = {}
    if TV_CACHE_FILE.exists():
        with open(TV_CACHE_FILE) as f:
            cache = json.load(f)

    missing = [d for d in dates if d not in cache]
    if missing:
        print(f"Fetching TV broadcast data for {len(missing)} dates from ESPN API...")
        for i, dt in enumerate(sorted(missing)):
            try:
                cache[dt] = fetch_broadcasts_for_date(dt)
                if (i + 1) % 20 == 0:
                    print(f"  ...fetched {i+1}/{len(missing)}")
            except Exception as e:
                print(f"  Warning: failed for {dt}: {e}")
                cache[dt] = {"national": {}, "all": []}
            time.sleep(0.3)  # be polite

        with open(TV_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Cached to {TV_CACHE_FILE}")

    return cache


# ---------------------------------------------------------------------------
# Load picks from JSON files
# ---------------------------------------------------------------------------
def load_all_picks() -> list:
    """Load all pick files and return unified list of picks with metadata."""
    picks = []

    for pattern, system in [
        ("xl_picks_*.json", "XL"),
        ("pro_picks_*.json", "PRO"),
        ("two_energy_picks_*.json", "TWO_ENERGY"),
    ]:
        files = sorted(glob.glob(str(PREDICTIONS_DIR / pattern)))
        for fpath in files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            game_date = data.get("date") or data.get("game_date")
            if not game_date:
                # Try to parse from filename
                fname = os.path.basename(fpath)
                for fmt in [
                    f"{pattern.split('*')[0]}%Y%m%d.json",
                    f"{pattern.split('*')[0]}%Y-%m-%d.json",
                ]:
                    try:
                        dt = datetime.strptime(fname, fmt)
                        game_date = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            if not game_date:
                continue

            # Normalize date format
            if len(game_date) == 8 and "-" not in game_date:
                game_date = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:]}"

            for p in data.get("picks", []):
                stat_type = p.get("stat_type", "")
                if stat_type not in STAT_COLUMN:
                    continue

                pick = {
                    "player_name": p.get("player_name"),
                    "stat_type": stat_type,
                    "side": p.get("side", "OVER"),
                    "line": p.get("best_line") or p.get("line"),
                    "opponent_team": p.get("opponent_team"),
                    "is_home": p.get("is_home"),
                    "game_date": game_date,
                    "system": system,
                    "model_version": p.get("model_version", ""),
                    "filter_tier": p.get("filter_tier", ""),
                    "confidence": p.get("confidence", ""),
                }
                if pick["player_name"] and pick["line"] is not None:
                    picks.append(pick)

    return picks


# ---------------------------------------------------------------------------
# Grade picks against player_game_logs
# ---------------------------------------------------------------------------
def grade_picks(picks: list) -> list:
    """Look up actual values from player_game_logs and grade each pick."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Build lookup: (player_name, game_date) -> {stat_col: value, team_abbrev: str}
    dates = list(set(p["game_date"] for p in picks))
    if not dates:
        return []

    cur.execute(
        """
        SELECT pp.full_name, pgl.game_date, pgl.team_abbrev, pgl.opponent_abbrev,
               pgl.points, pgl.rebounds, pgl.assists, pgl.three_pointers_made,
               pgl.minutes_played
        FROM player_game_logs pgl
        JOIN player_profile pp ON pp.player_id = pgl.player_id
        WHERE pgl.game_date = ANY(%s::date[])
    """,
        (dates,),
    )

    results = {}
    for row in cur.fetchall():
        name, gdate, team, opp, pts, reb, ast, threes, mins = row
        key = (name, str(gdate))
        results[key] = {
            "points": pts,
            "rebounds": reb,
            "assists": ast,
            "three_pointers_made": threes,
            "minutes_played": mins,
            "team_abbrev": team,
            "opponent_abbrev": opp,
        }

    cur.close()
    conn.close()

    graded = []
    for p in picks:
        key = (p["player_name"], p["game_date"])
        if key not in results:
            continue

        r = results[key]
        col = STAT_COLUMN[p["stat_type"]]
        actual = r.get(col)
        if actual is None:
            continue

        line = float(p["line"])
        won = (actual > line) if p["side"] == "OVER" else (actual < line)
        # Push = loss (standard)
        if actual == line:
            won = False

        p["actual"] = actual
        p["won"] = won
        p["team_abbrev"] = r["team_abbrev"]
        p["minutes_played"] = r.get("minutes_played")
        graded.append(p)

    return graded


# ---------------------------------------------------------------------------
# Tag picks with national TV status
# ---------------------------------------------------------------------------
def tag_national_tv(picks: list, tv_cache: dict) -> list:
    """Tag each pick with is_national_tv based on ESPN broadcast data."""
    for p in picks:
        gd = p["game_date"]
        team = p.get("team_abbrev", "")
        opp = p.get("opponent_team", "")

        tv_data = tv_cache.get(gd, {})
        national = tv_data.get("national", {})

        # Check both orderings: home_away or away_home
        is_ntv = False
        networks = []
        for game_key, nets in national.items():
            teams_in_game = game_key.split("_")
            if team in teams_in_game or opp in teams_in_game:
                is_ntv = True
                networks = nets
                break

        p["is_national_tv"] = is_ntv
        p["tv_networks"] = networks

    return picks


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_stats(picks: list, label: str) -> dict:
    """Compute win rate and other stats for a group of picks."""
    if not picks:
        return {"label": label, "n": 0, "wins": 0, "losses": 0, "wr": 0, "roi": 0}

    wins = sum(1 for p in picks if p["won"])
    losses = len(picks) - wins
    wr = wins / len(picks) * 100
    # ROI at -110 odds: win = +0.909u, loss = -1.0u
    profit = wins * 0.909 - losses * 1.0
    roi = profit / len(picks) * 100

    return {
        "label": label,
        "n": len(picks),
        "wins": wins,
        "losses": losses,
        "wr": round(wr, 1),
        "roi": round(roi, 2),
        "profit": round(profit, 2),
    }


def chi_squared_test(group_a: list, group_b: list) -> dict:
    """Simple chi-squared test for independence between two groups."""
    a_win = sum(1 for p in group_a if p["won"])
    a_loss = len(group_a) - a_win
    b_win = sum(1 for p in group_b if p["won"])
    b_loss = len(group_b) - b_win

    total = a_win + a_loss + b_win + b_loss
    if total == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False}

    # Expected values
    total_win = a_win + b_win
    total_loss = a_loss + b_loss
    n_a = a_win + a_loss
    n_b = b_win + b_loss

    if n_a == 0 or n_b == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False}

    e_a_win = n_a * total_win / total
    e_a_loss = n_a * total_loss / total
    e_b_win = n_b * total_win / total
    e_b_loss = n_b * total_loss / total

    chi2 = 0
    for obs, exp in [
        (a_win, e_a_win),
        (a_loss, e_a_loss),
        (b_win, e_b_win),
        (b_loss, e_b_loss),
    ]:
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    # p-value approximation (1 df) using survival function
    # For chi2 with 1 df: p ≈ erfc(sqrt(chi2/2))
    import math

    p_value = math.erfc(math.sqrt(chi2 / 2))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_comparison(ntv_stats: dict, local_stats: dict, test: dict):
    """Print a formatted comparison table."""
    print(f"\n  {'':30s} {'National TV':>14s} {'Local Only':>14s} {'Delta':>10s}")
    print(f"  {'':30s} {'─'*14:>14s} {'─'*14:>14s} {'─'*10:>10s}")
    print(f"  {'Sample Size':30s} {ntv_stats['n']:>14d} {local_stats['n']:>14d}")
    print(f"  {'Wins':30s} {ntv_stats['wins']:>14d} {local_stats['wins']:>14d}")
    print(f"  {'Losses':30s} {ntv_stats['losses']:>14d} {local_stats['losses']:>14d}")

    delta_wr = ntv_stats["wr"] - local_stats["wr"]
    wr_arrow = "+" if delta_wr > 0 else ""
    print(
        f"  {'Win Rate':30s} {ntv_stats['wr']:>13.1f}% {local_stats['wr']:>13.1f}% {wr_arrow}{delta_wr:>8.1f}%"
    )

    delta_roi = ntv_stats["roi"] - local_stats["roi"]
    roi_arrow = "+" if delta_roi > 0 else ""
    print(
        f"  {'ROI':30s} {ntv_stats['roi']:>13.2f}% {local_stats['roi']:>13.2f}% {roi_arrow}{delta_roi:>7.2f}%"
    )

    sig_str = "YES (p<0.05)" if test["significant"] else "NO"
    print(
        f"\n  Chi-squared: {test['chi2']:.4f}  |  p-value: {test['p_value']:.4f}  |  Significant: {sig_str}"
    )


def run_analysis():
    print("=" * 70)
    print("  NBA PICK PERFORMANCE: NATIONAL TV vs LOCAL BROADCAST")
    print("=" * 70)

    # Step 1: Load picks
    print("\n[1/4] Loading picks from JSON files...")
    picks = load_all_picks()
    print(f"  Loaded {len(picks)} raw picks across all systems")

    systems = defaultdict(int)
    for p in picks:
        systems[p["system"]] += 1
    for sname, cnt in sorted(systems.items()):
        print(f"    {sname}: {cnt} picks")

    # Step 2: Grade picks
    print("\n[2/4] Grading picks against player_game_logs...")
    graded = grade_picks(picks)
    print(f"  Graded {len(graded)} picks (had actual results)")

    if not graded:
        print("\n  ERROR: No graded picks found. Check database connection.")
        return

    # Step 3: Fetch TV data
    print("\n[3/4] Fetching broadcast data from ESPN API...")
    dates = sorted(set(p["game_date"] for p in graded))
    print(f"  Need broadcast data for {len(dates)} unique dates ({dates[0]} to {dates[-1]})")
    tv_cache = load_or_fetch_tv_cache(dates)

    # Step 4: Tag and analyze
    print("\n[4/4] Analyzing...")
    graded = tag_national_tv(graded, tv_cache)

    ntv_picks = [p for p in graded if p["is_national_tv"]]
    local_picks = [p for p in graded if not p["is_national_tv"]]

    # ── Overall ──
    print_section("OVERALL (All Systems Combined)")
    overall_ntv = compute_stats(ntv_picks, "National TV")
    overall_local = compute_stats(local_picks, "Local Only")
    overall_test = chi_squared_test(ntv_picks, local_picks)
    print_comparison(overall_ntv, overall_local, overall_test)

    total_stats = compute_stats(graded, "Total")
    print(
        f"\n  Total graded: {total_stats['n']} | Overall WR: {total_stats['wr']}% | Overall ROI: {total_stats['roi']}%"
    )
    ntv_pct = len(ntv_picks) / len(graded) * 100 if graded else 0
    print(
        f"  National TV games: {len(ntv_picks)} ({ntv_pct:.1f}%) | Local games: {len(local_picks)} ({100-ntv_pct:.1f}%)"
    )

    # ── By System ──
    print_section("BY SYSTEM")
    for sys_name in ["XL", "PRO", "TWO_ENERGY"]:
        sys_picks = [p for p in graded if p["system"] == sys_name]
        if len(sys_picks) < 10:
            continue

        sys_ntv = [p for p in sys_picks if p["is_national_tv"]]
        sys_local = [p for p in sys_picks if not p["is_national_tv"]]

        print(f"\n  ── {sys_name} System ──")
        s_ntv = compute_stats(sys_ntv, "National TV")
        s_local = compute_stats(sys_local, "Local Only")
        s_test = chi_squared_test(sys_ntv, sys_local)
        print_comparison(s_ntv, s_local, s_test)

    # ── By Market ──
    print_section("BY MARKET")
    for market in ["POINTS", "REBOUNDS"]:
        mkt_picks = [p for p in graded if p["stat_type"] == market]
        if len(mkt_picks) < 10:
            continue

        mkt_ntv = [p for p in mkt_picks if p["is_national_tv"]]
        mkt_local = [p for p in mkt_picks if not p["is_national_tv"]]

        print(f"\n  ── {market} ──")
        m_ntv = compute_stats(mkt_ntv, "National TV")
        m_local = compute_stats(mkt_local, "Local Only")
        m_test = chi_squared_test(mkt_ntv, mkt_local)
        print_comparison(m_ntv, m_local, m_test)

    # ── By Side (OVER/UNDER) ──
    print_section("BY SIDE")
    for side in ["OVER", "UNDER"]:
        side_picks = [p for p in graded if p["side"] == side]
        if len(side_picks) < 10:
            continue

        side_ntv = [p for p in side_picks if p["is_national_tv"]]
        side_local = [p for p in side_picks if not p["is_national_tv"]]

        print(f"\n  ── {side} ──")
        sd_ntv = compute_stats(side_ntv, "National TV")
        sd_local = compute_stats(side_local, "Local Only")
        sd_test = chi_squared_test(side_ntv, side_local)
        print_comparison(sd_ntv, sd_local, sd_test)

    # ── Network breakdown ──
    print_section("WIN RATE BY NETWORK")
    network_picks = defaultdict(list)
    for p in ntv_picks:
        for net in p.get("tv_networks", []):
            network_picks[net].append(p)

    print(f"\n  {'Network':15s} {'Picks':>8s} {'Wins':>8s} {'WR%':>8s} {'ROI%':>8s}")
    print(f"  {'─'*15:15s} {'─'*8:>8s} {'─'*8:>8s} {'─'*8:>8s} {'─'*8:>8s}")
    for net, net_picks in sorted(network_picks.items(), key=lambda x: -len(x[1])):
        s = compute_stats(net_picks, net)
        print(f"  {net:15s} {s['n']:>8d} {s['wins']:>8d} {s['wr']:>7.1f}% {s['roi']:>7.2f}%")

    # ── Day of week breakdown ──
    print_section("BY DAY OF WEEK (TV vs Local)")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(
        f"\n  {'Day':5s} {'NTV Games':>10s} {'NTV WR%':>10s} {'Local':>10s} {'Local WR%':>10s} {'Delta':>8s}"
    )
    print(f"  {'─'*5:5s} {'─'*10:>10s} {'─'*10:>10s} {'─'*10:>10s} {'─'*10:>10s} {'─'*8:>8s}")
    for dow_idx, dow in enumerate(dow_names):
        day_ntv = [
            p
            for p in ntv_picks
            if datetime.strptime(p["game_date"], "%Y-%m-%d").weekday() == dow_idx
        ]
        day_local = [
            p
            for p in local_picks
            if datetime.strptime(p["game_date"], "%Y-%m-%d").weekday() == dow_idx
        ]
        s_ntv = compute_stats(day_ntv, dow)
        s_local = compute_stats(day_local, dow)
        delta = s_ntv["wr"] - s_local["wr"] if s_ntv["n"] > 0 and s_local["n"] > 0 else 0
        delta_str = (
            f"{'+' if delta > 0 else ''}{delta:.1f}%"
            if (s_ntv["n"] > 0 and s_local["n"] > 0)
            else "N/A"
        )
        print(
            f"  {dow:5s} {s_ntv['n']:>10d} {s_ntv['wr']:>9.1f}% {s_local['n']:>10d} {s_local['wr']:>9.1f}% {delta_str:>8s}"
        )

    # ── Confidence split ──
    print_section("BY CONFIDENCE (XL Only)")
    xl_picks = [p for p in graded if p["system"] == "XL"]
    for conf in ["HIGH", "MEDIUM", "STANDARD"]:
        conf_picks = [p for p in xl_picks if p["confidence"] == conf]
        if len(conf_picks) < 5:
            continue
        conf_ntv = [p for p in conf_picks if p["is_national_tv"]]
        conf_local = [p for p in conf_picks if not p["is_national_tv"]]
        print(f"\n  ── {conf} Confidence ──")
        c_ntv = compute_stats(conf_ntv, "National TV")
        c_local = compute_stats(conf_local, "Local Only")
        c_test = chi_squared_test(conf_ntv, conf_local)
        print_comparison(c_ntv, c_local, c_test)

    # ── Summary / Recommendation ──
    print_section("FEATURE EVALUATION SUMMARY")
    delta_wr = overall_ntv["wr"] - overall_local["wr"]
    print(
        f"""
  Win Rate Delta (NTV - Local): {'+' if delta_wr > 0 else ''}{delta_wr:.1f}%
  Statistical Significance:     {'YES (p < 0.05)' if overall_test['significant'] else 'NO (p >= 0.05)'}
  Chi-squared:                  {overall_test['chi2']:.4f}
  P-value:                      {overall_test['p_value']:.4f}
  Sample (NTV / Local):         {overall_ntv['n']} / {overall_local['n']}
"""
    )

    if overall_test["significant"] and abs(delta_wr) >= 3.0:
        print("  RECOMMENDATION: WORTH ADDING as a feature.")
        print(f"  The {abs(delta_wr):.1f}% win rate difference is statistically significant")
        print("  and large enough to be actionable.")
    elif overall_test["significant"]:
        print("  RECOMMENDATION: MARGINAL - statistically significant but small effect.")
        print(
            f"  The {abs(delta_wr):.1f}% difference is real but may not justify model complexity."
        )
        print("  Consider adding as a low-weight feature or filter.")
    else:
        print("  RECOMMENDATION: NOT WORTH ADDING as a feature.")
        print(f"  The {abs(delta_wr):.1f}% difference is NOT statistically significant.")
        print("  National TV status does not meaningfully predict pick outcomes")
        print("  in our current dataset.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_analysis()
