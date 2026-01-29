#!/usr/bin/env python3
"""
BettingPros Cheatsheet Picks Generator
=======================================
Generates picks from BettingPros cheat sheet recommendations
using validated high-WR filters.

RECALIBRATED Jan 15, 2026 (regime shift detected Jan 5-14):
- Old filters (L5 80%+ L15 70%+) collapsed from ~80% to ~50% WR
- New filters require EV% and/or projection_diff

Current Performance (Jan 5-14, 2026):
- POINTS: L15 60%+ Season 60%+ Opp 11+ Diff 2+ = 77.8% WR (1.8/day)
- ASSISTS: L5 80%+ EV 10%+ = 70.8% WR (2.4/day)
- REBOUNDS: L15 60%+ Season 60%+ EV 10%+ = 84.2% WR (1.9/day)

This is SEPARATE from the XL model predictions - uses BettingPros
projections and hit rate data directly.

Usage:
    python3 generate_cheatsheet_picks.py --date 2026-01-15
    python3 generate_cheatsheet_picks.py --output predictions/pro_picks_20260115.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

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
# Players returning from injury have stale hit rates that don't reflect current form
# BettingPros doesn't account for this, so we filter them out
MAX_DAYS_SINCE_LAST_GAME = 5  # Skip players who haven't played in 5+ days


# =============================================================================
# LINE FILTERS - Minimum line thresholds to filter phantom/pulled lines
# =============================================================================
LINE_FILTERS = {
    "POINTS": {"min_line": None},  # No filter needed (XL handles this)
    "REBOUNDS": {"min_line": None},  # No filter needed (XL handles this)
    "ASSISTS": {"min_line": 2.5},  # Filter ghost lines (0.5-2.0) like Jarrett Allen 1.5
    "THREES": {"min_line": 1.5},  # Filter 0.5 lines
    # Combo stats
    "PA": {"min_line": 15.0},  # Points + Assists
    "PR": {"min_line": 15.0},  # Points + Rebounds
    "RA": {"min_line": 6.0},  # Rebounds + Assists
    "PRA": {"min_line": 25.0},  # Points + Rebounds + Assists
}


# =============================================================================
# VALIDATED FILTER CONFIGURATIONS
# =============================================================================
# RECALIBRATED Jan 15, 2026 based on Jan 5-14, 2026 data (regime shift detected)
# Old hit-rate-only filters collapsed from ~80% to ~50% WR
# New filters add EV% and projection_diff requirements

PRO_FILTERS = {
    # =========================================================================
    # POINTS FILTERS - Recalibrated Jan 15, 2026
    # Key insight: Need Season + Opp + Diff, not just L5/L15 hit rates
    # =========================================================================
    # POINTS: Season 70%+ EV 20%+ = 85.7% WR (6W/1L, 1.0/day) - HYBRID Jan 16
    "points_season_ev": {
        "enabled": True,
        "stat_type": "POINTS",
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": None,
        "min_hit_rate_season": 0.70,
        "min_opp_rank": None,
        "max_opp_rank": None,
        "min_bet_rating": None,
        "min_ev_pct": 20.0,
        "min_projection_diff": None,
        "expected_wr": 85.7,
        "expected_volume": 1.0,
        "tier_label": "pro",
        "description": "POINTS Season 70%+ EV 20%+ (85.7% WR)",
    },
    # POINTS: Season_60 + Opp11+ + EV20+ + Diff2+ = 76.5% WR (13W/4L, 1.7/day)
    # DISABLED: Using only points_season_ev (tighter, higher WR)
    "points_ev_diff": {
        "enabled": False,
        "stat_type": "POINTS",
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": None,
        "min_hit_rate_season": 0.60,
        "min_opp_rank": 11,
        "max_opp_rank": 30,
        "min_bet_rating": None,
        "min_ev_pct": 20.0,
        "min_projection_diff": 2.0,
        "expected_wr": 76.5,
        "expected_volume": 1.7,
        "tier_label": "pro",
        "description": "POINTS Season 60%+ Opp 11+ EV 20%+ Diff 2+ (76.5% WR)",
    },
    # =========================================================================
    # ASSISTS FILTERS - Recalibrated Jan 15, 2026
    # Key insight: EV 10%+ is now required for consistent performance
    # =========================================================================
    # ASSISTS: L5/L15 60%+ Opp 21-30 Rating 3+ = 71.4% WR (10W/4L, 2.0/day) - HYBRID Jan 16
    "assists_L5_L15_opp_rating": {
        "enabled": True,
        "stat_type": "ASSISTS",
        "min_hit_rate_l5": 0.60,
        "min_hit_rate_l15": 0.60,
        "min_hit_rate_season": None,
        "min_opp_rank": 21,
        "max_opp_rank": 30,
        "min_bet_rating": 3,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 71.4,
        "expected_volume": 2.0,
        "tier_label": "pro",
        "description": "ASSISTS L5/L15 60%+ Opp 21-30 Rating 3+ (71.4% WR)",
    },
    # ASSISTS: L5_80 + Opp11+ + EV10+ = 73.7% WR (14W/5L, 1.9/day)
    # DISABLED: Using only assists_L5_L15_opp_rating (tighter Opp 21+)
    "assists_L5_opp_ev": {
        "enabled": False,
        "stat_type": "ASSISTS",
        "min_hit_rate_l5": 0.80,
        "min_hit_rate_l15": None,
        "min_hit_rate_season": None,
        "min_opp_rank": 11,
        "max_opp_rank": 30,
        "min_bet_rating": None,
        "min_ev_pct": 10.0,
        "min_projection_diff": None,
        "expected_wr": 73.7,
        "expected_volume": 1.9,
        "tier_label": "pro",
        "description": "ASSISTS L5 80%+ Opp 11+ EV 10%+ (73.7% WR)",
    },
    # ASSISTS: L5_60 + L15_60 + Opp11+ + Rating4+ = 76.5% WR (13W/4L, 1.7/day)
    # DISABLED: Using only assists_L5_L15_opp_rating (tighter Opp 21+)
    "assists_rating_opp": {
        "enabled": False,
        "stat_type": "ASSISTS",
        "min_hit_rate_l5": 0.60,
        "min_hit_rate_l15": 0.60,
        "min_hit_rate_season": None,
        "min_opp_rank": 11,
        "max_opp_rank": 30,
        "min_bet_rating": 4,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 76.5,
        "expected_volume": 1.7,
        "tier_label": "pro",
        "description": "ASSISTS L5 60%+ L15 60%+ Opp 11+ Rating 4+ (76.5% WR)",
    },
    # =========================================================================
    # REBOUNDS FILTERS - Recalibrated Jan 15, 2026
    # Key insight: EV 10%+ combined with season hit rate is key
    # =========================================================================
    # REBOUNDS: L5/L15/Season 60%+ Opp 11-30 Rating 3+ = 77.8% WR (7W/2L, 1.3/day) - HYBRID Jan 16
    "rebounds_full_filter": {
        "enabled": True,
        "stat_type": "REBOUNDS",
        "min_hit_rate_l5": 0.60,
        "min_hit_rate_l15": 0.60,
        "min_hit_rate_season": 0.60,
        "min_opp_rank": 11,
        "max_opp_rank": 30,
        "min_bet_rating": 3,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 77.8,
        "expected_volume": 1.3,
        "tier_label": "pro",
        "description": "REBOUNDS L5/L15/Season 60%+ Opp 11-30 Rating 3+ (77.8% WR)",
    },
    # =========================================================================
    # COMBO STATS - Recalibrated Jan 15, 2026
    # Based on Jan 5-14, 2026 data (regime shift from old L5 80%+ L15 70%+)
    # =========================================================================
    # PA (Points + Assists): L15 70%+ Season 60%+ Line 20+ = 83.3% WR (5W/1L) - TIGHTENED Jan 17
    # Key insight: Low-line picks (<20) and low-season HR (<60%) are losers
    # Filtered: VJ Edgecombe (Season 53%), Amen Thompson (Season 58%), rookies
    "pa_L15_opp": {
        "enabled": True,
        "stat_type": "PA",
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": 0.70,
        "min_hit_rate_season": 0.60,  # Added Jan 17 - filters low-season players
        "min_opp_rank": 11,
        "max_opp_rank": 30,
        "min_bet_rating": None,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "min_line": 20.0,  # Added Jan 17 - filters rookies/low-volume
        "expected_wr": 83.3,
        "expected_volume": 0.6,
        "tier_label": "combo",
        "description": "PA L15 70%+ Season 60%+ Line 20+ (83.3% WR)",
    },
    # PR (Points + Rebounds): L15 70%+ Opp 16-30 = 75% WR (6W/2L, 1.6/day) - USER OPTIMIZED Jan 16
    "pr_L15_opp": {
        "enabled": True,
        "stat_type": "PR",
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": 0.70,
        "min_hit_rate_season": None,
        "min_opp_rank": 16,
        "max_opp_rank": 30,
        "min_bet_rating": None,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 75.0,
        "expected_volume": 1.6,
        "tier_label": "combo",
        "description": "PR L15 70%+ Opp 16-30 (75% WR)",
    },
    # RA (Rebounds + Assists): Diff 2+ Opp 21-30 = 100% WR (6W/0L, 1.2/day) - USER OPTIMIZED Jan 16
    "ra_opp_diff": {
        "enabled": True,
        "stat_type": "RA",
        "min_hit_rate_l5": None,
        "min_hit_rate_l15": None,
        "min_hit_rate_season": None,
        "min_opp_rank": 21,
        "max_opp_rank": 30,
        "min_bet_rating": None,
        "min_ev_pct": None,
        "min_projection_diff": 2.0,
        "expected_wr": 100.0,
        "expected_volume": 1.2,
        "tier_label": "combo",
        "description": "RA Diff 2+ Opp 21-30 (100% WR)",
    },
    # =========================================================================
    # THREES - DISABLED (29.7% WR on overs - terrible performance)
    # =========================================================================
    # THREES: DISABLED - 29.7% WR means "under" wins 70.3% of the time
    # The BettingPros OVER recommendations for THREES are consistently wrong
    "threes_L5_L15": {
        "enabled": False,  # DISABLED - 29.7% WR on overs
        "stat_type": "THREES",
        "min_hit_rate_l5": 0.80,
        "min_hit_rate_l15": 0.70,
        "min_hit_rate_season": None,
        "min_opp_rank": None,
        "max_opp_rank": None,
        "min_bet_rating": None,
        "min_ev_pct": None,
        "min_projection_diff": None,
        "expected_wr": 29.7,  # TERRIBLE - Do not use
        "expected_volume": 0,
        "tier_label": "disabled",
        "description": "THREES DISABLED - 29.7% WR (overs losing)",
    },
}


class CheatsheetPicksGenerator:
    """
    Generate picks from BettingPros cheat sheet data.

    Uses validated filters to identify high-probability picks
    from the cheatsheet_data table.
    """

    def __init__(self, game_date: str = None, platform: str = "underdog"):
        self.game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        self.platform = platform
        self.conn_intel = None
        self.conn_players = None
        self.picks = []

    def connect(self):
        """Connect to databases."""
        self.conn_intel = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        logger.info("[OK] Connected to databases")

    def get_days_since_last_game(self, player_name: str) -> Optional[int]:
        """
        Get days since player's last game before the target game date.

        Returns None if player not found, otherwise days since last game.
        Players returning from injury (5+ days out) should be filtered.
        """
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
            from datetime import datetime

            game_date = datetime.strptime(self.game_date, "%Y-%m-%d").date()
            last_game = result[0]
            return (game_date - last_game).days
        return None

    def close(self):
        """Close database connections."""
        if self.conn_intel:
            self.conn_intel.close()
        if self.conn_players:
            self.conn_players.close()

    def query_cheatsheet_data(self) -> List[Dict]:
        """
        Query cheat sheet data for the game date.

        Fetches from both the specified platform AND 'all' platform
        (combo stats like PA, PR, RA are stored in 'all' platform).

        Returns list of dicts with all cheat sheet fields.
        """
        cursor = self.conn_intel.cursor()

        # Query from both specified platform and 'all' platform
        # Combo stats (PA, PR, RA) are stored in 'all' platform
        query = """
            SELECT
                player_name, game_date, stat_type, platform,
                line, projection, projection_diff,
                bet_rating, ev_pct, probability,
                hit_rate_l5, hit_rate_l15, hit_rate_season,
                opp_rank, opp_value,
                recommended_side, use_for_betting
            FROM cheatsheet_data
            WHERE game_date = %s
              AND platform IN (%s, 'all')
              AND recommended_side = 'over'
              AND use_for_betting = true
            ORDER BY stat_type, player_name
        """

        cursor.execute(query, (self.game_date, self.platform))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()

        data = [dict(zip(columns, row)) for row in rows]
        logger.info(f"[DATA] Found {len(data)} cheat sheet props for {self.game_date}")

        return data

    def apply_filter(self, props: List[Dict], filter_config: Dict) -> List[Dict]:
        """
        Apply a filter configuration to props.

        Returns list of props that pass the filter.
        """
        if not filter_config.get("enabled", True):
            return []

        filtered = []

        for prop in props:
            # Stat type filter
            required_stat = filter_config.get("stat_type")
            if required_stat and prop["stat_type"] != required_stat:
                continue

            # Line minimum filter (filters phantom/pulled lines)
            stat_type = prop.get("stat_type", "")
            line_config = LINE_FILTERS.get(stat_type, {})
            min_line = line_config.get("min_line")
            if min_line is not None:
                prop_line = float(prop.get("line") or 0)
                if prop_line < min_line:
                    logger.debug(
                        f"Filtered {prop.get('player_name')} {stat_type} - line {prop_line} < {min_line}"
                    )
                    continue

            # Filter-specific min_line (stricter than global LINE_FILTERS)
            filter_min_line = filter_config.get("min_line")
            if filter_min_line is not None:
                prop_line = float(prop.get("line") or 0)
                if prop_line < filter_min_line:
                    logger.debug(
                        f"Filtered {prop.get('player_name')} {stat_type} - line {prop_line} < {filter_min_line} (filter-specific)"
                    )
                    continue

            # L5 hit rate filter
            min_l5 = filter_config.get("min_hit_rate_l5")
            if min_l5 is not None:
                prop_l5 = float(prop.get("hit_rate_l5") or 0)
                if prop_l5 < min_l5:
                    continue

            # L15 hit rate filter
            min_l15 = filter_config.get("min_hit_rate_l15")
            if min_l15 is not None:
                prop_l15 = float(prop.get("hit_rate_l15") or 0)
                if prop_l15 < min_l15:
                    continue

            # Season hit rate filter
            min_season = filter_config.get("min_hit_rate_season")
            if min_season is not None:
                prop_season = float(prop.get("hit_rate_season") or 0)
                if prop_season < min_season:
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

            # Bet rating filter
            min_rating = filter_config.get("min_bet_rating")
            if min_rating is not None:
                prop_rating = prop.get("bet_rating")
                if prop_rating is None or int(prop_rating) < min_rating:
                    continue

            # EV% filter
            min_ev = filter_config.get("min_ev_pct")
            if min_ev is not None:
                prop_ev = float(prop.get("ev_pct") or 0)
                if prop_ev < min_ev:
                    continue

            # Projection diff filter
            min_diff = filter_config.get("min_projection_diff")
            if min_diff is not None:
                prop_diff = float(prop.get("projection_diff") or 0)
                if prop_diff < min_diff:
                    continue

            # Injury return filter - skip players who haven't played recently
            # Players returning from injury have stale hit rates
            days_out = self.get_days_since_last_game(prop.get("player_name", ""))
            if days_out is not None and days_out > MAX_DAYS_SINCE_LAST_GAME:
                logger.info(
                    f"[SKIP] {prop.get('player_name')} - returning from {days_out} days out"
                )
                continue

            # Passed all filters
            filtered.append(prop)

        return filtered

    def generate_picks(self) -> List[Dict]:
        """
        Generate picks using all enabled filters.

        Returns list of picks with tier labels and metadata.
        """
        # Query cheat sheet data
        props = self.query_cheatsheet_data()

        if not props:
            logger.warning(f"[WARN] No cheat sheet data for {self.game_date}")
            return []

        all_picks = []
        seen_keys = set()  # Deduplication

        for filter_name, filter_config in PRO_FILTERS.items():
            if not filter_config.get("enabled", True):
                continue

            filtered = self.apply_filter(props, filter_config)

            for prop in filtered:
                # Dedup key
                key = (prop["player_name"], prop["stat_type"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Build pick record
                pick = {
                    "player_name": prop["player_name"],
                    "stat_type": prop["stat_type"],
                    "side": "OVER",
                    "line": float(prop["line"]),
                    "projection": float(prop.get("projection") or 0),
                    "projection_diff": float(prop.get("projection_diff") or 0),
                    "bet_rating": prop.get("bet_rating"),
                    "ev_pct": float(prop.get("ev_pct") or 0),
                    "probability": float(prop.get("probability") or 0),
                    "hit_rate_l5": float(prop.get("hit_rate_l5") or 0),
                    "hit_rate_l15": float(prop.get("hit_rate_l15") or 0),
                    "hit_rate_season": float(prop.get("hit_rate_season") or 0),
                    "opp_rank": prop.get("opp_rank"),
                    "opp_value": prop.get("opp_value"),
                    "platform": prop["platform"],
                    "filter_name": filter_name,
                    "filter_tier": filter_config["tier_label"],
                    "expected_wr": filter_config["expected_wr"],
                    "confidence": "HIGH" if filter_config["expected_wr"] >= 80 else "MEDIUM",
                    "source": "bettingpros_cheatsheet",
                    "reasoning": self._generate_reasoning(prop, filter_config),
                }

                all_picks.append(pick)

            if filtered:
                logger.info(
                    f"[OK] {filter_name}: {len(filtered)} picks (expected {filter_config['expected_wr']:.1f}% WR)"
                )

        self.picks = all_picks
        return all_picks

    def _generate_reasoning(self, prop: Dict, filter_config: Dict) -> str:
        """Generate human-readable reasoning for the pick."""
        reasons = []

        # Filter description
        reasons.append(filter_config["description"])

        # Specific metrics
        if prop.get("hit_rate_l5"):
            reasons.append(f"L5 hit rate: {float(prop['hit_rate_l5'])*100:.0f}%")

        if prop.get("opp_rank"):
            reasons.append(f"vs #{prop['opp_rank']} defense")

        if prop.get("projection_diff") and float(prop["projection_diff"]) > 0:
            reasons.append(f"Projection: {prop['projection']} (+{prop['projection_diff']} vs line)")

        return ". ".join(reasons) + "."

    def save_picks(self, output_file: str, dry_run: bool = False) -> None:
        """Save picks to JSON file."""
        output = {
            "generated_at": datetime.now().isoformat(),
            "date": self.game_date,
            "strategy": "Pro Tier Filters",
            "tier": "pro",
            "platform": self.platform,
            "total_picks": len(self.picks),
            "picks": self.picks,
            "summary": {
                "total": len(self.picks),
                "by_stat_type": {},
                "by_filter": {},
                "high_confidence": len([p for p in self.picks if p["confidence"] == "HIGH"]),
            },
            "filter_configs": {
                name: {
                    "enabled": cfg["enabled"],
                    "expected_wr": cfg["expected_wr"],
                    "description": cfg["description"],
                }
                for name, cfg in PRO_FILTERS.items()
            },
            "expected_performance": {
                "assists_primary": {"win_rate": 93.8, "avg_picks_per_day": 3.0},
                "l15_elite": {"win_rate": 77.4, "avg_picks_per_day": 1.0},
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

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"\n[OK] Saved {len(self.picks)} Pro picks to: {output_path}")
        self._print_summary(output)

    def _print_summary(self, output: Dict) -> None:
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print(f"PRO PICKS - {output['date']}")
        print("=" * 80)
        print(f"Tier: PRO (76-87% WR validated)")
        print(f"Platform: {output['platform']}")
        print(f"Total Picks: {output['total_picks']}")

        print("\nBy Stat Type:")
        for st, count in output["summary"]["by_stat_type"].items():
            print(f"  - {st}: {count}")

        print("\nBy Filter:")
        for fn, count in output["summary"]["by_filter"].items():
            cfg = PRO_FILTERS.get(fn, {})
            wr = cfg.get("expected_wr", 0)
            print(f"  - {fn}: {count} picks (expected {wr:.1f}% WR)")

        print("=" * 80)

        if output["picks"]:
            print("\nTOP PICKS:")
            for i, pick in enumerate(output["picks"][:5], 1):
                print(f"\n{i}. {pick['player_name']} {pick['stat_type']} OVER {pick['line']}")
                print(f"   Projection: {pick['projection']} | Diff: +{pick['projection_diff']:.1f}")
                print(
                    f"   L5 Hit Rate: {pick['hit_rate_l5']*100:.0f}% | Opp Rank: #{pick['opp_rank']}"
                )
                print(f"   Filter: {pick['filter_name']} ({pick['expected_wr']:.1f}% WR)")

        print("=" * 80 + "\n")

    def run(self, output_file: str, dry_run: bool = False) -> None:
        """Main execution."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("PRO PICKS GENERATOR")
            logger.info("=" * 80)
            logger.info(f"Date: {self.game_date}")
            logger.info(f"Platform: {self.platform}")
            logger.info(f"Tier: PRO (76-87% WR validated)")
            logger.info("=" * 80 + "\n")

            self.connect()
            self.generate_picks()

            if len(self.picks) == 0:
                logger.warning("\n[WARN] No Pro picks found")
                logger.info("   Possible reasons:")
                logger.info("   - No cheat sheet data loaded for today")
                logger.info("   - No props pass the Pro filter criteria")
                logger.info("   - Run: python3 fetchers/fetch_cheatsheet.py first")
                return

            self.save_picks(output_file, dry_run=dry_run)
            logger.info("\n[OK] Pro picks generation complete!")

        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Generate picks from BettingPros cheat sheet")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument(
        "--platform",
        default="underdog",
        choices=["underdog"],  # PrizePicks removed - no book_id, cannot filter
        help="Platform to use (default: underdog)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate without saving")

    args = parser.parse_args()

    # Default output path
    if not args.output:
        predictions_dir = Path(__file__).parent / "predictions"
        args.output = predictions_dir / f"pro_picks_{args.date.replace('-', '')}.json"

    # Run generator
    generator = CheatsheetPicksGenerator(game_date=args.date, platform=args.platform)
    generator.run(output_file=str(args.output), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
