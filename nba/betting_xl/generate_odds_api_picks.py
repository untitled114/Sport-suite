#!/usr/bin/env python3
"""
Odds API Picks Generator
=========================
Standalone script that generates picks by combining Pick6 multipliers
(from The Odds API) with BettingPros cheatsheet features (hit rates,
bet rating, opponent rank).

This mirrors the generate_cheatsheet_picks.py pattern:
  fetch_pick6_live.py -> load to DB -> generate_odds_api_picks.py -> JSON -> validate

Validated filters (Jan 15-24, 2026):
- TIER 1: mult<1.5 + R>=3 + opp>=6 + L5=100% + L15>=60%
- TIER 2: mult<1.0 + opp>=20 + L5>=80% + L15>=60%
- TRAP:   mult > 5.0 = 0% WR (NEVER bet)

Usage:
    python3 generate_odds_api_picks.py --date 2026-01-25
    python3 generate_odds_api_picks.py --pick6-file /path/to/historical.json
    python3 generate_odds_api_picks.py --dry-run
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

from nba.utils.name_normalizer import NameNormalizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Database configs
DB_INTELLIGENCE = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv(
        "NBA_INT_DB_USER", os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
    ),
    "password": os.getenv(
        "NBA_INT_DB_PASSWORD", os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    ),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}

DB_PLAYERS = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv(
        "NBA_PLAYERS_DB_USER", os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
    ),
    "password": os.getenv(
        "NBA_PLAYERS_DB_PASSWORD", os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    ),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}


# =============================================================================
# INJURY RETURN FILTER - Skip players returning from extended absence
# =============================================================================
MAX_DAYS_SINCE_LAST_GAME = 5


# =============================================================================
# ODDS API FILTER CONFIGURATIONS
# =============================================================================
# Conservative starting filters based on clean data analysis.
# Iterate using validate_odds_api_picks.py.

ODDS_API_FILTERS = {
    # POINTS: Low multiplier + floor-level cheatsheet gates
    # Backtested: 100% WR (10W-0L) Jan 10-24, 0.7 picks/day
    "points_low_mult": {
        "enabled": True,
        "stat_type": "POINTS",
        "max_multiplier": 0.9,
        "min_bet_rating": None,
        "min_opp_rank": 6,
        "max_opp_rank": None,
        "min_hit_rate_l5": 0.40,
        "min_hit_rate_l15": 0.40,
        "min_hit_rate_season": None,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 100.0,
        "description": "POINTS + mult<0.9 + opp>=6 + L5>=40% + L15>=40%",
    },
    # ASSISTS: Moderate multiplier + season consistency
    # Backtested: 100% WR (10W-0L) Jan 10-24, 0.7 picks/day
    "assists_season": {
        "enabled": True,
        "stat_type": "ASSISTS",
        "max_multiplier": 1.5,
        "min_bet_rating": None,
        "min_opp_rank": 6,
        "max_opp_rank": None,
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": None,
        "min_hit_rate_season": 0.60,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 100.0,
        "description": "ASSISTS + mult<1.5 + opp>=6 + szn>=60%",
    },
    # REBOUNDS: Very low multiplier + L15 guard
    # Backtested: 100% WR (5W-0L) Jan 10-24, 0.4 picks/day
    "rebounds_low_mult": {
        "enabled": True,
        "stat_type": "REBOUNDS",
        "max_multiplier": 0.8,
        "min_bet_rating": None,
        "min_opp_rank": None,
        "max_opp_rank": None,
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": 0.60,
        "min_hit_rate_season": None,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 100.0,
        "description": "REBOUNDS + mult<0.8 + L15>=60%",
    },
    # TRAP FILTER - always applied regardless of individual filters
    # Validated: 0% WR on 71 props with mult > 5.0
    "TRAP": {
        "min_multiplier": 5.0,
    },
}


class OddsApiPicksGenerator:
    """
    Generate picks by combining Pick6 multipliers with BettingPros cheatsheet data.

    Flow: load Pick6 data -> load cheatsheet data -> merge on (player, stat_type) -> apply filters -> output
    """

    def __init__(self, game_date: str = None, pick6_file: str = None):
        """
        Args:
            game_date: Target date (YYYY-MM-DD)
            pick6_file: Path to historical Pick6 JSON for backtest mode.
                        If None, fetches live from API.
        """
        self.game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        self.pick6_file = pick6_file
        self.normalizer = NameNormalizer()
        self.conn_intel = None
        self.conn_players = None
        self.pick6_data = {}  # (normalized_name, stat_type) -> pick6 prop
        self.cheatsheet_data = {}  # (normalized_name, stat_type) -> cheatsheet record
        self.merged_data = []
        self.picks = []

    def connect(self):
        """Connect to databases."""
        self.conn_intel = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        logger.info("[OK] Connected to databases")

    def close(self):
        """Close database connections."""
        if self.conn_intel:
            self.conn_intel.close()
        if self.conn_players:
            self.conn_players.close()

    def load_pick6_data(self):
        """
        Load Pick6 multipliers - from file (backtest) or live API.

        Populates self.pick6_data keyed by (normalized_player_name, stat_type).
        """
        if self.pick6_file:
            self._load_pick6_from_file(self.pick6_file)
        else:
            self._load_pick6_live()

    def _load_pick6_live(self):
        """Fetch live Pick6 data from The Odds API."""
        try:
            from betting_xl.fetchers.fetch_pick6_live import fetch_pick6_for_pipeline

            result = fetch_pick6_for_pipeline(verbose=True)

            if not result["success"]:
                logger.warning("[WARN] No Pick6 data available from API")
                return

            for prop in result["props"]:
                normalized = self.normalizer.normalize_name(prop["player_name"]).lower()
                key = (normalized, prop["stat_type"])
                # Keep prop with game_date matching target, or any if not specified
                prop_date = prop.get("game_date", "")
                if prop_date and prop_date != self.game_date:
                    continue
                self.pick6_data[key] = prop

            logger.info(
                f"[DATA] Loaded {len(self.pick6_data)} Pick6 props for {self.game_date} "
                f"(easy: {result['easy_count']}, traps: {result['trap_count']})"
            )
        except Exception as e:
            logger.error(f"Failed to load live Pick6 data: {e}")

    def _load_pick6_from_file(self, filepath: str):
        """Load historical Pick6 data from JSON file."""
        logger.info(f"Loading Pick6 data from file: {filepath}")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Support both raw list and wrapped format
            if isinstance(data, list):
                all_props = data
            else:
                all_props = data.get("props", data.get("data", []))

            # Filter to game_date
            day_props = [p for p in all_props if p.get("game_date") == self.game_date]

            if not day_props:
                logger.info(f"  No Pick6 data for {self.game_date} in file")
                return

            easy_count = 0
            trap_count = 0
            for prop in day_props:
                normalized = self.normalizer.normalize_name(prop["player_name"]).lower()
                key = (normalized, prop["stat_type"])
                self.pick6_data[key] = prop
                mult = prop.get("pick6_multiplier", 1.0)
                if mult < 1.0:
                    easy_count += 1
                elif mult > 5.0:
                    trap_count += 1

            logger.info(
                f"[DATA] Loaded {len(day_props)} Pick6 props from file "
                f"(easy: {easy_count}, traps: {trap_count})"
            )
        except Exception as e:
            logger.error(f"Failed to load Pick6 file: {e}")

    def load_cheatsheet_data(self):
        """
        Query cheatsheet_data table for game_date.

        Gets hit rates, bet rating, opp rank, projection, etc.
        """
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT
                player_name, stat_type, platform,
                line, projection, projection_diff,
                bet_rating, ev_pct, probability,
                hit_rate_l5, hit_rate_l15, hit_rate_season,
                opp_rank, opp_value,
                recommended_side, use_for_betting
            FROM cheatsheet_data
            WHERE game_date = %s
              AND platform IN ('underdog', 'all')
              AND recommended_side = 'over'
              AND use_for_betting = true
            ORDER BY stat_type, player_name
        """,
            (self.game_date,),
        )

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()

        for row in rows:
            data = dict(zip(columns, row))
            normalized = self.normalizer.normalize_name(data["player_name"]).lower()
            key = (normalized, data["stat_type"])
            self.cheatsheet_data[key] = data

        logger.info(
            f"[DATA] Found {len(self.cheatsheet_data)} cheatsheet props for {self.game_date}"
        )

    def merge_data(self) -> List[Dict]:
        """
        Join Pick6 + cheatsheet on (normalized_player_name, stat_type).

        Only includes rows that exist in BOTH sources.
        """
        merged = []

        for key, pick6_prop in self.pick6_data.items():
            cheatsheet = self.cheatsheet_data.get(key)
            if not cheatsheet:
                continue

            merged_row = {
                # Player info
                "player_name": cheatsheet["player_name"],
                "stat_type": cheatsheet["stat_type"],
                "platform": cheatsheet.get("platform", "underdog"),
                # Pick6 data
                "pick6_multiplier": float(pick6_prop.get("pick6_multiplier", 1.0)),
                "pick6_line": float(pick6_prop.get("line", 0)),
                # Cheatsheet data
                "line": float(cheatsheet.get("line") or 0),
                "projection": float(cheatsheet.get("projection") or 0),
                "projection_diff": float(cheatsheet.get("projection_diff") or 0),
                "bet_rating": cheatsheet.get("bet_rating"),
                "ev_pct": float(cheatsheet.get("ev_pct") or 0),
                "probability": float(cheatsheet.get("probability") or 0),
                "hit_rate_l5": float(cheatsheet.get("hit_rate_l5") or 0),
                "hit_rate_l15": float(cheatsheet.get("hit_rate_l15") or 0),
                "hit_rate_season": float(cheatsheet.get("hit_rate_season") or 0),
                "opp_rank": cheatsheet.get("opp_rank"),
                "opp_value": cheatsheet.get("opp_value"),
            }
            merged.append(merged_row)

        self.merged_data = merged
        logger.info(
            f"[MERGE] {len(merged)} props matched "
            f"(Pick6: {len(self.pick6_data)}, Cheatsheet: {len(self.cheatsheet_data)})"
        )
        return merged

    def get_days_since_last_game(self, player_name: str) -> Optional[int]:
        """Get days since player's last game before the target game date."""
        cursor = self.conn_players.cursor()
        cursor.execute(
            """
            SELECT MAX(g.game_date)
            FROM player_game_logs g
            JOIN player_profile p ON g.player_id = p.player_id
            WHERE LOWER(p.full_name) = LOWER(%s) AND g.game_date < %s
        """,
            (player_name, self.game_date),
        )

        result = cursor.fetchone()
        cursor.close()

        if result and result[0]:
            game_date = datetime.strptime(self.game_date, "%Y-%m-%d").date()
            last_game = result[0]
            return (game_date - last_game).days
        return None

    def apply_filters(self, merged_props: List[Dict]) -> List[Dict]:
        """
        Apply ODDS_API_FILTERS configs to merged props.

        Each prop is checked against all enabled filters. First match wins.
        TRAP filter is always applied first (hard filter).
        """
        all_picks = []
        seen_keys = set()

        for filter_name, filter_config in ODDS_API_FILTERS.items():
            # Skip the TRAP meta-filter (applied inline below)
            if filter_name == "TRAP":
                continue

            if not filter_config.get("enabled", True):
                continue

            for prop in merged_props:
                # Dedup key
                key = (prop["player_name"], prop["stat_type"])
                if key in seen_keys:
                    continue

                # TRAP hard filter - always skip mult > 5.0
                trap_threshold = ODDS_API_FILTERS.get("TRAP", {}).get("min_multiplier", 5.0)
                if prop["pick6_multiplier"] >= trap_threshold:
                    continue

                # Stat type filter
                required_stat = filter_config.get("stat_type")
                if required_stat and prop["stat_type"] != required_stat:
                    continue

                # Multiplier filter
                max_mult = filter_config.get("max_multiplier")
                if max_mult is not None and prop["pick6_multiplier"] >= max_mult:
                    continue

                # Bet rating filter
                min_rating = filter_config.get("min_bet_rating")
                if min_rating is not None:
                    prop_rating = prop.get("bet_rating")
                    if prop_rating is None or int(prop_rating) < min_rating:
                        continue

                # Opponent rank filter (min)
                min_opp = filter_config.get("min_opp_rank")
                if min_opp is not None:
                    prop_opp = prop.get("opp_rank")
                    if prop_opp is None or int(prop_opp) < min_opp:
                        continue

                # Opponent rank filter (max)
                max_opp = filter_config.get("max_opp_rank")
                if max_opp is not None:
                    prop_opp = prop.get("opp_rank")
                    if prop_opp is None or int(prop_opp) > max_opp:
                        continue

                # L5 hit rate filter
                min_l5 = filter_config.get("min_hit_rate_l5")
                if min_l5 is not None:
                    if prop.get("hit_rate_l5", 0) < min_l5:
                        continue

                # L15 hit rate filter
                min_l15 = filter_config.get("min_hit_rate_l15")
                if min_l15 is not None:
                    if prop.get("hit_rate_l15", 0) < min_l15:
                        continue

                # Season hit rate filter
                min_season = filter_config.get("min_hit_rate_season")
                if min_season is not None:
                    if prop.get("hit_rate_season", 0) < min_season:
                        continue

                # EV% filter
                min_ev = filter_config.get("min_ev_pct")
                if min_ev is not None:
                    if prop.get("ev_pct", 0) < min_ev:
                        continue

                # Projection diff filter
                min_diff = filter_config.get("min_projection_diff")
                if min_diff is not None:
                    if prop.get("projection_diff", 0) < min_diff:
                        continue

                # Injury return filter
                days_out = self.get_days_since_last_game(prop["player_name"])
                if days_out is not None and days_out > MAX_DAYS_SINCE_LAST_GAME:
                    logger.info(
                        f"[SKIP] {prop['player_name']} - returning from {days_out} days out"
                    )
                    continue

                # Passed all filters
                seen_keys.add(key)
                pick = self._build_pick(prop, filter_name, filter_config)
                all_picks.append(pick)

            count = sum(1 for p in all_picks if p["filter_name"] == filter_name)
            if count > 0:
                logger.info(
                    f"[OK] {filter_name}: {count} picks "
                    f"(expected {filter_config.get('expected_wr', 0):.1f}% WR)"
                )

        self.picks = all_picks
        return all_picks

    def _build_pick(self, prop: Dict, filter_name: str, filter_config: Dict) -> Dict:
        """Build a pick record from merged prop data."""
        mult = prop["pick6_multiplier"]
        expected_wr = filter_config.get("expected_wr", 60.0)

        return {
            "player_name": prop["player_name"],
            "stat_type": prop["stat_type"],
            "side": "OVER",
            "line": prop["line"],
            "pick6_multiplier": mult,
            "projection": prop.get("projection", 0),
            "projection_diff": prop.get("projection_diff", 0),
            "bet_rating": prop.get("bet_rating"),
            "ev_pct": prop.get("ev_pct", 0),
            "hit_rate_l5": prop.get("hit_rate_l5", 0),
            "hit_rate_l15": prop.get("hit_rate_l15", 0),
            "hit_rate_season": prop.get("hit_rate_season", 0),
            "opp_rank": prop.get("opp_rank"),
            "opp_value": prop.get("opp_value"),
            "platform": prop.get("platform", "underdog"),
            "filter_name": filter_name,
            "filter_tier": "odds_api",
            "expected_wr": expected_wr,
            "confidence": (
                "ELITE" if expected_wr >= 90 else ("HIGH" if expected_wr >= 75 else "MEDIUM")
            ),
            "source": "odds_api_pick6",
            "reasoning": self._generate_reasoning(prop, filter_config),
        }

    def _generate_reasoning(self, prop: Dict, filter_config: Dict) -> str:
        """Generate human-readable reasoning for the pick."""
        reasons = [filter_config.get("description", "")]

        mult = prop["pick6_multiplier"]
        if mult < 1.0:
            reasons.append(f"Pick6 mult={mult:.2f} (easy)")
        elif mult < 1.5:
            reasons.append(f"Pick6 mult={mult:.2f} (favorable)")
        else:
            reasons.append(f"Pick6 mult={mult:.2f}")

        if prop.get("hit_rate_l5"):
            reasons.append(f"L5={prop['hit_rate_l5']*100:.0f}%")

        if prop.get("opp_rank"):
            reasons.append(f"vs #{prop['opp_rank']} defense")

        if prop.get("projection_diff") and prop["projection_diff"] > 0:
            reasons.append(f"Proj +{prop['projection_diff']:.1f} vs line")

        return ". ".join(reasons) + "."

    def generate(self) -> List[Dict]:
        """Main flow: load -> merge -> filter -> return picks."""
        self.load_pick6_data()

        if not self.pick6_data:
            logger.warning("[WARN] No Pick6 data loaded - cannot generate Odds API picks")
            return []

        self.load_cheatsheet_data()

        if not self.cheatsheet_data:
            logger.warning("[WARN] No cheatsheet data loaded - cannot merge with Pick6")
            return []

        merged = self.merge_data()

        if not merged:
            logger.warning("[WARN] No merged props (Pick6 + cheatsheet) found")
            return []

        picks = self.apply_filters(merged)
        return picks

    def save_picks(self, output_file: str, dry_run: bool = False) -> None:
        """Save picks to JSON file."""
        output = {
            "generated_at": datetime.now().isoformat(),
            "date": self.game_date,
            "strategy": "Odds API Picks (Pick6 + BettingPros)",
            "tier": "odds_api",
            "total_picks": len(self.picks),
            "picks": self.picks,
            "summary": {
                "total": len(self.picks),
                "by_stat_type": {},
                "by_filter": {},
                "high_confidence": len(
                    [p for p in self.picks if p["confidence"] in ("ELITE", "HIGH")]
                ),
            },
            "data_sources": {
                "pick6_props": len(self.pick6_data),
                "cheatsheet_props": len(self.cheatsheet_data),
                "merged_props": len(self.merged_data),
            },
            "filter_configs": {
                name: {
                    "enabled": cfg.get("enabled", True),
                    "expected_wr": cfg.get("expected_wr"),
                    "description": cfg.get("description", ""),
                }
                for name, cfg in ODDS_API_FILTERS.items()
                if name != "TRAP"
            },
        }

        # Count by stat type
        for pick in self.picks:
            st = pick["stat_type"]
            output["summary"]["by_stat_type"][st] = output["summary"]["by_stat_type"].get(st, 0) + 1

        # Count by filter
        for pick in self.picks:
            fn = pick["filter_name"]
            output["summary"]["by_filter"][fn] = output["summary"]["by_filter"].get(fn, 0) + 1

        if dry_run:
            logger.info("\n[DRY RUN] Would save picks:")
            self._print_summary(output)
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"\n[OK] Saved {len(self.picks)} Odds API picks to: {output_path}")
        self._print_summary(output)

    def _print_summary(self, output: Dict) -> None:
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print(f"ODDS API PICKS - {output['date']}")
        print("=" * 80)
        print(f"Strategy: {output['strategy']}")
        print(f"Total Picks: {output['total_picks']}")

        print("\nData Sources:")
        ds = output.get("data_sources", {})
        print(f"  - Pick6 props: {ds.get('pick6_props', 0)}")
        print(f"  - Cheatsheet props: {ds.get('cheatsheet_props', 0)}")
        print(f"  - Merged (matched): {ds.get('merged_props', 0)}")

        print("\nBy Stat Type:")
        for st, count in output["summary"]["by_stat_type"].items():
            print(f"  - {st}: {count}")

        print("\nBy Filter:")
        for fn, count in output["summary"]["by_filter"].items():
            cfg = ODDS_API_FILTERS.get(fn, {})
            wr = cfg.get("expected_wr", 0)
            print(f"  - {fn}: {count} picks (expected {wr:.1f}% WR)")

        print("=" * 80)

        if output["picks"]:
            print("\nPICKS:")
            for i, pick in enumerate(output["picks"], 1):
                mult = pick.get("pick6_multiplier", 1.0)
                l5 = pick.get("hit_rate_l5", 0)
                opp = pick.get("opp_rank", "?")
                rating = pick.get("bet_rating", "?")
                print(f"\n{i}. {pick['player_name']} {pick['stat_type']} OVER {pick['line']}")
                print(f"   Mult: {mult:.2f} | L5: {l5*100:.0f}% | Opp: #{opp} | Rating: {rating}")
                print(
                    f"   Projection: {pick.get('projection', 0):.1f} (+{pick.get('projection_diff', 0):.1f})"
                )
                print(f"   Filter: {pick['filter_name']} ({pick['expected_wr']:.1f}% WR)")

        print("=" * 80 + "\n")

    def run(self, output_file: str, dry_run: bool = False) -> None:
        """Main execution."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("ODDS API PICKS GENERATOR")
            logger.info("=" * 80)
            logger.info(f"Date: {self.game_date}")
            logger.info(f"Source: {'File' if self.pick6_file else 'Live API'}")
            logger.info("=" * 80 + "\n")

            self.connect()
            self.generate()

            if not self.picks:
                logger.warning("\n[WARN] No Odds API picks found")
                logger.info("   Possible reasons:")
                logger.info("   - No Pick6 data available for today")
                logger.info("   - No cheatsheet data loaded for today")
                logger.info("   - No props pass the filter criteria")
                logger.info("   - Run morning workflow first")
                return

            self.save_picks(output_file, dry_run=dry_run)
            logger.info("\n[OK] Odds API picks generation complete!")

        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Odds API picks (Pick6 multipliers + BettingPros cheatsheet)"
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument(
        "--pick6-file", default=None, help="Historical Pick6 JSON file (for backtest mode)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate without saving")

    args = parser.parse_args()

    # Default output path
    if not args.output:
        predictions_dir = Path(__file__).parent / "predictions"
        args.output = predictions_dir / f"odds_api_picks_{args.date.replace('-', '')}.json"

    generator = OddsApiPicksGenerator(
        game_date=args.date,
        pick6_file=args.pick6_file,
    )
    generator.run(output_file=str(args.output), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
