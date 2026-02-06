#!/usr/bin/env python3
"""
Analyze cheatsheet filters to find optimal combinations for current regime.
"""

import argparse

import psycopg2

from nba.config.database import get_intelligence_db_config, get_players_db_config


def analyze_losses():
    """Analyze the losing picks to find patterns."""
    intel_conn = psycopg2.connect(**get_intelligence_db_config())
    players_conn = psycopg2.connect(**get_players_db_config())

    intel_cur = intel_conn.cursor()
    players_cur = players_conn.cursor()

    # Get cheatsheet data
    intel_cur.execute(
        """
    SELECT
        player_name, game_date, stat_type, line, projection, projection_diff,
        bet_rating, ev_pct, opp_rank, hit_rate_l5, hit_rate_l15, hit_rate_season
    FROM cheatsheet_data
    WHERE game_date >= CURRENT_DATE - INTERVAL '21 days'
      AND stat_type IN ('POINTS', 'REBOUNDS', 'ASSISTS')
      AND line IS NOT NULL
    """
    )
    cheatsheet_rows = intel_cur.fetchall()

    # Get actuals
    players_cur.execute(
        """
    SELECT p.full_name, l.game_date, l.points, l.rebounds, l.assists
    FROM player_game_logs l
    JOIN player_profile p ON l.player_id = p.player_id
    WHERE l.game_date >= CURRENT_DATE - INTERVAL '21 days'
    """
    )

    actuals = {}
    for row in players_cur.fetchall():
        name, date, pts, reb, ast = row
        key = (name.lower().strip(), str(date))
        actuals[key] = {"POINTS": pts or 0, "REBOUNDS": reb or 0, "ASSISTS": ast or 0}

    # Build results
    results = []
    for row in cheatsheet_rows:
        (
            player,
            date,
            stat,
            line,
            proj,
            proj_diff,
            rating,
            ev_pct,
            opp_rank,
            hr_l5,
            hr_l15,
            hr_szn,
        ) = row
        key = (player.lower().strip(), str(date))
        if key not in actuals:
            continue
        actual = actuals[key].get(stat, 0)
        hit = 1 if actual > float(line) else 0
        results.append(
            {
                "player": player,
                "date": str(date),
                "stat": stat,
                "line": float(line),
                "actual": actual,
                "hit": hit,
                "proj_diff": float(proj_diff) if proj_diff else None,
                "rating": rating,
                "ev_pct": float(ev_pct) if ev_pct else None,
                "opp_rank": opp_rank,
                "hr_l5": float(hr_l5) if hr_l5 else None,
                "hr_l15": float(hr_l15) if hr_l15 else None,
                "hr_szn": float(hr_szn) if hr_szn else None,
            }
        )

    def analyze_filter(stat_name, filter_desc, filter_fn):
        stat_results = [r for r in results if r["stat"] == stat_name and filter_fn(r)]
        losses = [r for r in stat_results if r["hit"] == 0]
        wins = [r for r in stat_results if r["hit"] == 1]

        print()
        print("=" * 80)
        print(f"{stat_name} LOSSES ({filter_desc}) - {len(losses)} losses, {len(wins)} wins")
        print("=" * 80)

        if losses:
            print(
                f"\n{'Player':<22} {'Date':<12} {'Line':>5} {'Act':>4} {'Diff':>5} {'L5':>4} {'L15':>4} {'Szn':>4} {'EV':>4} {'Opp':>3} {'R':>2}"
            )
            print("-" * 80)
            for r in sorted(losses, key=lambda x: x["date"]):
                diff = r["proj_diff"] if r["proj_diff"] else 0
                l5 = int(r["hr_l5"] * 100) if r["hr_l5"] else 0
                l15 = int(r["hr_l15"] * 100) if r["hr_l15"] else 0
                szn = int(r["hr_szn"] * 100) if r["hr_szn"] else 0
                ev = int(r["ev_pct"]) if r["ev_pct"] else 0
                opp = r["opp_rank"] or 0
                rat = r["rating"] or 0
                print(
                    f"{r['player'][:21]:<22} {r['date']:<12} {r['line']:>5.1f} {r['actual']:>4} {diff:>5.1f} {l5:>3}% {l15:>3}% {szn:>3}% {ev:>3}% {opp:>3} {rat:>2}"
                )

        print("\nAdditional filter options:")
        additional_filters = [
            ("+ hr_l5 >= 60%", lambda r: r["hr_l5"] and r["hr_l5"] >= 0.60),
            ("+ hr_l5 >= 80%", lambda r: r["hr_l5"] and r["hr_l5"] >= 0.80),
            ("+ hr_l15 >= 60%", lambda r: r["hr_l15"] and r["hr_l15"] >= 0.60),
            ("+ hr_l15 >= 70%", lambda r: r["hr_l15"] and r["hr_l15"] >= 0.70),
            ("+ hr_szn >= 50%", lambda r: r["hr_szn"] and r["hr_szn"] >= 0.50),
            ("+ hr_szn >= 60%", lambda r: r["hr_szn"] and r["hr_szn"] >= 0.60),
            ("+ ev_pct >= 10", lambda r: r["ev_pct"] and r["ev_pct"] >= 10),
            ("+ ev_pct >= 20", lambda r: r["ev_pct"] and r["ev_pct"] >= 20),
            ("+ rating >= 3", lambda r: r["rating"] and r["rating"] >= 3),
            ("+ rating >= 4", lambda r: r["rating"] and r["rating"] >= 4),
            ("+ opp >= 11", lambda r: r["opp_rank"] and r["opp_rank"] >= 11),
            ("+ opp >= 21", lambda r: r["opp_rank"] and r["opp_rank"] >= 21),
            ("+ opp >= 25", lambda r: r["opp_rank"] and r["opp_rank"] >= 25),
            ("+ diff >= 1.0", lambda r: r["proj_diff"] and r["proj_diff"] >= 1.0),
            ("+ diff >= 1.5", lambda r: r["proj_diff"] and r["proj_diff"] >= 1.5),
            ("+ diff >= 2.0", lambda r: r["proj_diff"] and r["proj_diff"] >= 2.0),
            ("+ diff >= 2.5", lambda r: r["proj_diff"] and r["proj_diff"] >= 2.5),
        ]

        for name, check in additional_filters:
            filt_wins = len([r for r in wins if check(r)])
            filt_losses = len([r for r in losses if check(r)])
            total = filt_wins + filt_losses
            if total >= 3:
                wr = filt_wins / total * 100
                elim = len(losses) - filt_losses
                kept = filt_wins
                if elim > 0 or wr > (len(wins) / (len(wins) + len(losses)) * 100 + 5):
                    marker = " ***" if wr >= 80 else ""
                    print(
                        f"  {name:<18}: {filt_wins}W-{filt_losses}L ({wr:>5.1f}%) | -{elim}L, keeps {kept}/{len(wins)}W{marker}"
                    )

    # Analyze each filter
    analyze_filter(
        "POINTS",
        "hr_szn>=70 + diff>=1.5",
        lambda r: r["hr_szn"] and r["hr_szn"] >= 0.70 and r["proj_diff"] and r["proj_diff"] >= 1.5,
    )

    analyze_filter(
        "REBOUNDS",
        "hr_l15>=60 + opp>=21 + diff>=1.0",
        lambda r: r["hr_l15"]
        and r["hr_l15"] >= 0.60
        and r["opp_rank"]
        and r["opp_rank"] >= 21
        and r["proj_diff"]
        and r["proj_diff"] >= 1.0,
    )

    analyze_filter("ASSISTS", "hr_szn>=70", lambda r: r["hr_szn"] and r["hr_szn"] >= 0.70)

    intel_conn.close()
    players_conn.close()


def main():
    # Connect to both databases
    intel_conn = psycopg2.connect(**get_intelligence_db_config())
    players_conn = psycopg2.connect(**get_players_db_config())

    intel_cur = intel_conn.cursor()
    players_cur = players_conn.cursor()

    # Get cheatsheet data from last 21 days
    intel_cur.execute(
        """
    SELECT
        player_name,
        game_date,
        stat_type,
        line,
        projection,
        projection_diff,
        bet_rating,
        ev_pct,
        opp_rank,
        hit_rate_l5,
        hit_rate_l15,
        hit_rate_season
    FROM cheatsheet_data
    WHERE game_date >= CURRENT_DATE - INTERVAL '21 days'
      AND stat_type IN ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')
      AND line IS NOT NULL
    ORDER BY game_date, player_name
    """
    )

    cheatsheet_rows = intel_cur.fetchall()
    print(f"Found {len(cheatsheet_rows)} cheatsheet props from last 21 days")

    # Get actual results
    players_cur.execute(
        """
    SELECT
        p.full_name,
        l.game_date,
        l.points,
        l.rebounds,
        l.assists,
        l.three_pointers_made
    FROM player_game_logs l
    JOIN player_profile p ON l.player_id = p.player_id
    WHERE l.game_date >= CURRENT_DATE - INTERVAL '21 days'
    """
    )

    # Build lookup dict
    actuals = {}
    for row in players_cur.fetchall():
        name, date, pts, reb, ast, threes = row
        key = (name.lower().strip(), str(date))
        actuals[key] = {
            "POINTS": pts or 0,
            "REBOUNDS": reb or 0,
            "ASSISTS": ast or 0,
            "THREES": threes or 0,
        }

    print(f"Found {len(actuals)} game logs for matching")

    # Analyze each prop
    results = []
    for row in cheatsheet_rows:
        (
            player,
            date,
            stat,
            line,
            proj,
            proj_diff,
            rating,
            ev_pct,
            opp_rank,
            hr_l5,
            hr_l15,
            hr_szn,
        ) = row

        key = (player.lower().strip(), str(date))
        if key not in actuals:
            continue

        actual = actuals[key].get(stat, 0)
        hit = 1 if actual > float(line) else 0

        results.append(
            {
                "player": player,
                "date": str(date),
                "stat": stat,
                "line": float(line),
                "actual": actual,
                "hit": hit,
                "proj_diff": float(proj_diff) if proj_diff else None,
                "rating": rating,
                "ev_pct": float(ev_pct) if ev_pct else None,
                "opp_rank": opp_rank,
                "hr_l5": float(hr_l5) if hr_l5 else None,
                "hr_l15": float(hr_l15) if hr_l15 else None,
                "hr_szn": float(hr_szn) if hr_szn else None,
            }
        )

    print(f"Matched {len(results)} props with actual results")
    print()

    # Analyze by stat type
    for stat in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
        stat_results = [r for r in results if r["stat"] == stat]
        if not stat_results:
            continue

        wins = sum(r["hit"] for r in stat_results)
        total = len(stat_results)
        wr = wins / total * 100 if total > 0 else 0

        print()
        print("=" * 70)
        print(f"{stat}: {wins}W/{total-wins}L ({wr:.1f}% baseline)")
        print("=" * 70)

        # Test various filter combinations
        filters_to_test = [
            ("hr_szn >= 0.70", lambda r: r["hr_szn"] and r["hr_szn"] >= 0.70),
            ("hr_szn >= 0.60", lambda r: r["hr_szn"] and r["hr_szn"] >= 0.60),
            ("hr_l5 >= 0.80", lambda r: r["hr_l5"] and r["hr_l5"] >= 0.80),
            ("hr_l5 >= 0.60", lambda r: r["hr_l5"] and r["hr_l5"] >= 0.60),
            ("hr_l15 >= 0.70", lambda r: r["hr_l15"] and r["hr_l15"] >= 0.70),
            ("hr_l15 >= 0.60", lambda r: r["hr_l15"] and r["hr_l15"] >= 0.60),
            ("ev_pct >= 20", lambda r: r["ev_pct"] and r["ev_pct"] >= 20),
            ("ev_pct >= 10", lambda r: r["ev_pct"] and r["ev_pct"] >= 10),
            ("rating >= 4", lambda r: r["rating"] and r["rating"] >= 4),
            ("rating >= 3", lambda r: r["rating"] and r["rating"] >= 3),
            ("opp_rank >= 21", lambda r: r["opp_rank"] and r["opp_rank"] >= 21),
            ("opp_rank >= 11", lambda r: r["opp_rank"] and r["opp_rank"] >= 11),
            ("proj_diff >= 2.5", lambda r: r["proj_diff"] and r["proj_diff"] >= 2.5),
            ("proj_diff >= 2.0", lambda r: r["proj_diff"] and r["proj_diff"] >= 2.0),
            ("proj_diff >= 1.5", lambda r: r["proj_diff"] and r["proj_diff"] >= 1.5),
            ("proj_diff >= 1.0", lambda r: r["proj_diff"] and r["proj_diff"] >= 1.0),
            # Combos
            (
                "hr_szn>=70 + ev>=20",
                lambda r: r["hr_szn"] and r["hr_szn"] >= 0.70 and r["ev_pct"] and r["ev_pct"] >= 20,
            ),
            (
                "hr_szn>=70 + diff>=1.0",
                lambda r: r["hr_szn"]
                and r["hr_szn"] >= 0.70
                and r["proj_diff"]
                and r["proj_diff"] >= 1.0,
            ),
            (
                "hr_szn>=70 + diff>=1.5",
                lambda r: r["hr_szn"]
                and r["hr_szn"] >= 0.70
                and r["proj_diff"]
                and r["proj_diff"] >= 1.5,
            ),
            (
                "hr_szn>=60 + opp>=11 + diff>=1.5",
                lambda r: r["hr_szn"]
                and r["hr_szn"] >= 0.60
                and r["opp_rank"]
                and r["opp_rank"] >= 11
                and r["proj_diff"]
                and r["proj_diff"] >= 1.5,
            ),
            (
                "hr_l5>=60 + hr_szn>=60 + diff>=1.0",
                lambda r: r["hr_l5"]
                and r["hr_l5"] >= 0.60
                and r["hr_szn"]
                and r["hr_szn"] >= 0.60
                and r["proj_diff"]
                and r["proj_diff"] >= 1.0,
            ),
            (
                "rating>=4 + ev>=10",
                lambda r: r["rating"] and r["rating"] >= 4 and r["ev_pct"] and r["ev_pct"] >= 10,
            ),
            (
                "rating>=4 + diff>=1.5",
                lambda r: r["rating"]
                and r["rating"] >= 4
                and r["proj_diff"]
                and r["proj_diff"] >= 1.5,
            ),
            (
                "opp>=21 + diff>=1.5",
                lambda r: r["opp_rank"]
                and r["opp_rank"] >= 21
                and r["proj_diff"]
                and r["proj_diff"] >= 1.5,
            ),
            (
                "opp>=21 + diff>=2.0",
                lambda r: r["opp_rank"]
                and r["opp_rank"] >= 21
                and r["proj_diff"]
                and r["proj_diff"] >= 2.0,
            ),
            (
                "hr_l15>=70 + diff>=1.0",
                lambda r: r["hr_l15"]
                and r["hr_l15"] >= 0.70
                and r["proj_diff"]
                and r["proj_diff"] >= 1.0,
            ),
            (
                "hr_szn>=60 + diff>=2.0",
                lambda r: r["hr_szn"]
                and r["hr_szn"] >= 0.60
                and r["proj_diff"]
                and r["proj_diff"] >= 2.0,
            ),
            (
                "ev>=20 + diff>=1.5",
                lambda r: r["ev_pct"]
                and r["ev_pct"] >= 20
                and r["proj_diff"]
                and r["proj_diff"] >= 1.5,
            ),
            (
                "rating>=4 + opp>=21",
                lambda r: r["rating"]
                and r["rating"] >= 4
                and r["opp_rank"]
                and r["opp_rank"] >= 21,
            ),
            (
                "hr_l5>=80 + diff>=1.0",
                lambda r: r["hr_l5"]
                and r["hr_l5"] >= 0.80
                and r["proj_diff"]
                and r["proj_diff"] >= 1.0,
            ),
            (
                "hr_l5>=80 + opp>=11",
                lambda r: r["hr_l5"]
                and r["hr_l5"] >= 0.80
                and r["opp_rank"]
                and r["opp_rank"] >= 11,
            ),
            (
                "ev>=15 + diff>=2.0",
                lambda r: r["ev_pct"]
                and r["ev_pct"] >= 15
                and r["proj_diff"]
                and r["proj_diff"] >= 2.0,
            ),
            (
                "hr_szn>=70 + rating>=3",
                lambda r: r["hr_szn"] and r["hr_szn"] >= 0.70 and r["rating"] and r["rating"] >= 3,
            ),
            (
                "hr_l15>=60 + opp>=21 + diff>=1.0",
                lambda r: r["hr_l15"]
                and r["hr_l15"] >= 0.60
                and r["opp_rank"]
                and r["opp_rank"] >= 21
                and r["proj_diff"]
                and r["proj_diff"] >= 1.0,
            ),
        ]

        print(f"{'Filter':<45} {'W':>3} {'L':>3} {'WR%':>6} {'n':>3}")
        print("-" * 65)

        # Sort by win rate, but only show filters with 3+ samples
        filter_results = []
        for name, filt in filters_to_test:
            filtered = [r for r in stat_results if filt(r)]
            if len(filtered) < 3:
                continue
            w = sum(r["hit"] for r in filtered)
            losses = len(filtered) - w
            wr = w / len(filtered) * 100
            filter_results.append((name, w, losses, wr, len(filtered)))

        # Sort by WR descending
        filter_results.sort(key=lambda x: (-x[3], -x[4]))

        for name, w, losses, wr, n in filter_results:
            marker = " ***" if wr >= 65 and n >= 5 else ""
            print(f"{name:<45} {w:>3} {losses:>3} {wr:>5.1f}% {n:>3}{marker}")

    intel_conn.close()
    players_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cheatsheet filters")
    parser.add_argument("--losses", action="store_true", help="Analyze losses for top filters")
    args = parser.parse_args()

    if args.losses:
        analyze_losses()
    else:
        main()
