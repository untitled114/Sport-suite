#!/usr/bin/env python3
"""
NBA Line Shopping Optimizer
============================
Finds softest lines across multiple books and calculates edge.

Part of Phase 5: XL Betting Pipeline (Task 5.2)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import psycopg2

# Centralized configuration (available for future migration)
# TODO: Migrate hardcoded values below to use config.thresholds
from nba.config.thresholds import (
    POINTS_CONFIG,
    REBOUNDS_CONFIG,
    STAKE_SIZING_CONFIG,
    STAR_POINTS_CONFIG,
    STAR_REBOUNDS_CONFIG,
    TRAP_BOOKS,
    calculate_stake,
    get_market_config,
)

logger = logging.getLogger(__name__)

# Star players - profitable performers only (updated Jan 26, 2026)
# REMOVED (Jan 26): Curry (31.7%), Sengun (44.3%), Jalen Johnson (43.6%),
#   Jokic (injured), Booker (injured), Brunson (injured), KAT (slumping),
#   Ingram (41.5%), SGA (44.6%), Cade (assists-only player)
# BLACKLISTED (poor WR): Luka (19%), Lauri (7%), Paolo (33%), Giannis (35%), Randle (20-39%), Miles (0-48%)
STAR_PLAYERS = {
    # Keepers
    "Donovan Mitchell",
    "Tyrese Maxey",
    "Anthony Edwards",
    "Jaylen Brown",
    "De'Aaron Fox",
    "Franz Wagner",
    "Pascal Siakam",
    "Kevin Durant",
    # Gold (75%+ WR, Dec-Jan 2026)
    "Joel Embiid",
    "Michael Porter Jr.",
    # Standard (70%+ WR)
    "Jamal Murray",
    "OG Anunoby",
    # Minimum (65%+ WR)
    "LeBron James",
    "Deni Avdija",
    "Jaren Jackson Jr.",
    "Brandon Miller",
}

# =============================================================================
# POINTS-SPECIFIC STAR PLAYERS (Jan 30, 2026)
# =============================================================================
# Analysis of Jan 15-30 data - only stars with 60%+ OVER WR on POINTS:
#   Combined: 28W-5L = 84.8% WR
# EXCLUDED (cold on POINTS): Tyrese Maxey (25%), Giannis (0%), Lauri (0%),
#   SGA (37.5%), Jaylen Brown (37.5%), Curry (14%), Kawhi (33%)
# =============================================================================
POINTS_STAR_PLAYERS = {
    # 60%+ WR on POINTS (Jan 15-30, 2026)
    "Joel Embiid",  # 100% (6W-0L), +7.3 margin
    "Brandon Miller",  # 89% (8W-1L), +4.8 margin
    "Pascal Siakam",  # 83% (5W-1L), +2.2 margin
    "Luka Doncic",  # 71% (5W-2L), +2.1 margin
    "Anthony Edwards",  # 67% (4W-2L), +3.5 margin
}

# =============================================================================
# REBOUNDS-SPECIFIC STAR PLAYERS (Jan 30, 2026)
# =============================================================================
# Top rebounders with 70%+ OVER WR - Combined: 14W-4L = 77.8% WR
# EXCLUDED: Zubac, Bam, Sengun, Chet (67% or lower, smaller margins)
# =============================================================================
REBOUNDS_STAR_PLAYERS = {
    "Donovan Clingan",  # 86% (6W-1L), +2.9 margin
    "Mitchell Robinson",  # 75% (3W-1L), +2.5 margin
    "Karl-Anthony Towns",  # 71% (5W-2L), +3.1 margin
    "Jalen Johnson",  # 71% (5W-2L), +2.9 margin
    "Victor Wembanyama",  # 71% (5W-2L), +2.4 margin
}

# =============================================================================
# BOOK-AWARE FILTERING (Jan 2, 2026)
# =============================================================================
# Investigation found that traditional sportsbooks ADAPTED in December:
# - November: DK/BetMGM/BetRivers softest = reliable signal (slow to adjust)
# - December: DK/BetMGM/BetRivers softest = TRAP signal (baiting rec bettors)
# - Underdog went from trap to profitable when it's softest
#
# Solution: Apply additional scrutiny when traditional books are softest
# =============================================================================

# Books that showed trap behavior when softest in December
TRAP_BOOKS_WHEN_SOFTEST = {
    "DraftKings": {"min_spread_required": 3.5, "min_p_over_boost": 0.05},
    "BetMGM": {"min_spread_required": 3.5, "min_p_over_boost": 0.05},
    "BetRivers": {"min_spread_required": 3.0, "min_p_over_boost": 0.03},
    "Caesars": {"min_spread_required": 3.0, "min_p_over_boost": 0.03},
}

# Books that showed reliability when softest in December
# DFS platforms (Underdog, PrizePicks) typically have softer lines
RELIABLE_BOOKS_WHEN_SOFTEST = {
    "Underdog",
    "Underdog Fantasy",
    "ESPNBet",
    "prizepicks",  # Added Jan 31, 2026 - DFS platform with soft lines
    "PrizePicks",
}

# =============================================================================
# BLACKLISTED BOOKS (Jan 2, 2026)
# =============================================================================
# Based on agent audit of Nov-Dec 2025 backtest:
# - FanDuel POINTS: 16.7% WR (1W-5L) vs 61.1% for all other books
# - When FanDuel is softest for POINTS, use next softest book instead
# =============================================================================
# ★ REGIME SHIFT FIX: Added BetRivers to blacklist (0W/7L = 0% WR in January)
BLACKLISTED_BOOKS = {
    "POINTS": {"FanDuel", "fanduel", "BetRivers", "betrivers"},  # FD 16.7% WR, BR 0% WR in Jan
    "REBOUNDS": set(),  # No blacklist for rebounds
    "ASSISTS": set(),
    "THREES": set(),
}

# =============================================================================
# SOFT BOOKS FILTER (Jan 2, 2026) - POINTS ONLY
# =============================================================================
# When enabled, only accept POINTS props where specified "soft" books are softest.
# REBOUNDS keeps normal filtering (was 65-67% WR - don't fix what's not broken)
#
# December 2025 validation:
#   - POINTS with Underdog softest + spread>=2.0: 62% WR (vs 41% with traps)
#   - Testing: Adding BetRivers + ESPNBet to allowed soft books
#
# Usage: Set SOFT_BOOKS_FILTER = True or pass underdog_only=True to optimize_line()
# =============================================================================
UNDERDOG_ONLY_MODE = False  # Jan 26, 2026: Disabled for backtest comparison

# DFS platforms with typically softer lines (Underdog + PrizePicks variants)
DFS_SOFT_BOOKS = {
    "underdog",
    "Underdog",
    "Underdog Fantasy",
    "prizepicks",
    "PrizePicks",
    "prizepicks_goblin",
    "prizepicks_demon",
    "prizepicks_alt",  # PP line variants
}

UNDERDOG_CONFIG = {
    "POINTS": {
        "enabled": True,  # Apply soft-book filter to POINTS
        "min_spread": 0.5,
        "underdog_names": DFS_SOFT_BOOKS,  # Expanded to include PrizePicks - Jan 31, 2026
    },
    "REBOUNDS": {
        "enabled": False,  # KEEP NORMAL FILTERING - was 65-67% WR
        "min_spread": 1.0,
        "underdog_names": DFS_SOFT_BOOKS,
    },
    "ASSISTS": {
        "enabled": False,
        "min_spread": 1.5,
        "underdog_names": DFS_SOFT_BOOKS,
    },
    "THREES": {
        "enabled": False,
        "min_spread": 1.5,
        "underdog_names": DFS_SOFT_BOOKS,
    },
}

# Database config
DB_DEFAULT_USER = os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
DB_DEFAULT_PASSWORD = os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))

DB_INTELLIGENCE = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv("NBA_INT_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_INT_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}

# =============================================================================
# PRODUCTION FILTER TIERS (Jan 30, 2026)
# =============================================================================
# Simplified tier system based on what actually works:
#   - Star: Manually calibrated star players (weekly updates)
#   - Goldmine: High spread (≥2.5 pts) = market inefficiency = 70.6% WR
#   - Standard: Basic thresholds (p_over ≥ 0.70, edge > 0)
#
# All tiers use XL model (102 features) - tier names reflect FILTER criteria
# =============================================================================
TIER_CONFIG = {
    "POINTS": {
        "enabled": True,
        "min_probability": 0.65,
        "min_line": 12.0,
        "max_line": 35.0,
        "max_edge_points": 6.0,
        "max_edge_low_variance": 3.0,
        "avoid_books": {"betrivers", "BetRivers"},
        "tiers": {
            # =================================================================
            # GOLDMINE: High spread = market inefficiency (80% WR validated)
            # When books disagree by 2.5+ pts, someone is wrong
            # =================================================================
            "Goldmine": {
                "min_spread": 2.5,
                "min_p_over": 0.65,
                "min_edge_points": 1.0,
                "require_positive_edge": True,
                "require_both": True,
                "expected_wr": 0.80,
            },
            # =================================================================
            # STANDARD: Moderate spread + higher confidence (100% WR validated)
            # =================================================================
            "Standard": {
                "min_spread": 1.5,
                "min_p_over": 0.75,
                "min_edge_points": 2.0,
                "require_positive_edge": True,
                "require_both": True,
                "expected_wr": 0.70,
            },
        },
    },
    "REBOUNDS": {
        "enabled": True,
        "min_probability": 0.55,
        "min_line": 5.5,
        "max_edge_low_variance": 2.0,
        "tiers": {
            # =================================================================
            # GOLDMINE: High spread for rebounds
            # =================================================================
            "Goldmine": {
                "min_spread": 2.0,
                "min_p_over": 0.60,
                "min_edge_points": 1.0,
                "require_positive_edge": True,
                "require_both": True,
                "expected_wr": 0.80,
            },
            # =================================================================
            # STANDARD: Moderate spread + higher confidence for rebounds
            # =================================================================
            "Standard": {
                "min_spread": 1.5,
                "min_p_over": 0.75,
                "min_edge_points": 1.0,
                "require_positive_edge": True,
                "require_both": True,
                "expected_wr": 0.70,
            },
        },
    },
    "ASSISTS": {"enabled": False},
    "THREES": {"enabled": False},
}

# =============================================================================
# V3 MODEL TIERS (Jan 16, 2026) - DEEP BACKTEST RECALIBRATION
# =============================================================================
# Full backtest: December 1, 2025 - January 16, 2026 (156 picks with outcomes)
#
# OVERALL: 118W/38L = 75.6% WR
#   December: 93W/22L = 80.9% WR
#   January:  25W/16L = 61.0% WR (regime shift confirmed)
#
# KEY DISCOVERY: p_over >= 0.95 is DANGER ZONE (42.9% WR - 3W/4L)
# The model becomes overconfident and WRONG at extreme probabilities
#
# TOP PERFORMING FILTERS (verified across both Dec AND Jan):
# 1. UNDER + line >= 25:            90.9% WR (10W/1L) - Dec 100%, Jan 75%
# 2. UNDER + edge >= 6:             85.7% WR (18W/3L) - Dec 94%, Jan 60%
# 3. draftkings + OVER + edge >= 4: 84.2% WR (16W/3L) - Dec 91%, Jan 75%
# 4. (dk|fd) + OVER + p 0.75-0.90:  82.6% WR (19W/4L) - Dec 85%, Jan 80%
# 5. OVER + line 10-18 + edge >= 4: 82.1% WR (23W/5L) - Dec 75%, Jan 92%
# 6. OVER + p 0.75-0.85 + !betmgm:  81.8% WR (27W/6L) - Dec 77%, Jan 100%
#
# ROBUST ACROSS BOTH MONTHS (70%+ in Dec AND Jan):
#   - OVER + p_over 0.75-0.85 + NOT (betmgm|betrivers): Avg 88.5%
#   - OVER + line 10-18 + edge >= 4: Avg 83.3%
#   - (dk|fd) + OVER + p_over 0.75-0.90: Avg 82.3%
#   - draftkings + p_over 0.75-0.90: Avg 82.9%
# =============================================================================
V3_TIER_CONFIG = {
    "POINTS": {
        "enabled": True,
        "model_version": "v3",
        # ★ JANUARY-OPTIMIZED: Blacklist betmgm+betrivers for ALL picks
        "avoid_books_over": {"betrivers", "BetRivers", "betmgm", "BetMGM"},
        "avoid_books_under": {"betrivers", "BetRivers"},
        "tiers": {
            # =================================================================
            # JANUARY-OPTIMIZED TIERS (Jan 16, 2026)
            # Target: 80%+ WR in January (matching December's performance)
            # =================================================================
            # ★ TIER 1: JAN_PRIME_OVER - DISABLED (Jan 28, 2026)
            # Actual Jan performance: 4W/6L = 40% WR, -23.6% ROI
            # Backtest was overfitted - real performance is losing money
            # 'JAN_PRIME_OVER': {
            #     'direction': 'OVER',
            #     'min_p_over': 0.70,
            #     'max_p_over': 0.90,
            #     'min_line': 10.0,
            #     'max_line': 18.0,
            #     'min_edge': 4.0,
            #     'min_spread': 0,
            #     'expected_wr': 0.917,
            # },
            # ★ TIER 2: JAN_CONFIDENT_OVER (January 87.5% WR - 7W/1L)
            # p_over 0.75-0.85 sweet spot (avoids overconfidence trap)
            "JAN_CONFIDENT_OVER": {
                "direction": "OVER",
                "min_p_over": 0.75,
                "max_p_over": 0.85,  # Critical cap - Jan 100% in this range w/ book filter
                "min_line": 0,
                "max_line": 25.0,
                "min_edge": 4.0,
                "min_spread": 0,
                "expected_wr": 0.875,  # 87.5% from Jan backtest (7W/1L)
            },
            # ★ TIER 3: JAN_LINE_OVER - DISABLED (Jan 28, 2026)
            # Actual Jan performance: 7W/5L = 58.3% WR, +11.4% ROI
            # Marginal - not worth the noise, disabling per user request
            # 'JAN_LINE_OVER': {
            #     'direction': 'OVER',
            #     'min_p_over': 0.65,
            #     'max_p_over': 0.95,
            #     'min_line': 10.0,
            #     'max_line': 18.0,
            #     'min_edge': 3.0,
            #     'min_spread': 0,
            #     'expected_wr': 0.824,
            # },
            # UNDER TIERS - Disabled for January (50% WR overall)
            # Only enable high-confidence UNDERs with strict filters
            # V3_HIGHLINE_UNDER - Keep but require NOT betrivers
            # 'V3_HIGHLINE_UNDER': {
            #     'direction': 'UNDER',
            #     'min_p_under': 0.80,
            #     'min_line': 25.0,
            #     'max_line': 999,
            #     'min_spread': 0,
            #     'min_edge': 6.0,
            #     'expected_wr': 0.75,
            # },
        },
    },
    "REBOUNDS": {"enabled": False},
    "ASSISTS": {"enabled": False},
    "THREES": {"enabled": False},
}

# NOTE: Odds API tier config moved to standalone generate_odds_api_picks.py (Jan 26, 2026)

# =============================================================================
# STAR TIER FILTER (Jan 16, 2026) - RECALIBRATED
# =============================================================================
# Backtested Dec 15, 2025 - Jan 15, 2026 (30 days)
#
# STAR POINTS: 65.8% WR (25W/13L) - OK but V3_OPTIMAL_OVER is better
# STAR REBOUNDS: 66.7% WR (24W/12L) - Improved with max_p_over + max_line caps
#
# REBOUNDS OPTIMAL FILTER DISCOVERY:
#   All REBOUNDS:           64.3% WR (27W/15L)
#   max_p_over <= 0.80:     71.0% WR (22W/9L)
#   max_line <= 8:          73.3% WR (11W/4L)
#   BOTH combined:          82.6% WR (19W/4L) ★
#
# Key insight: p_over 0.80+ zone is overconfident (45% WR)
# =============================================================================
# ★ REGIME SHIFT FIX: Apply moderate edge filter to STAR tier for POINTS
# STAR tier was dragging down overall WR: 25% WR with edge 0.3-0.5
# Edge >= 3.0 balances WR vs volume (edge >= 5 was too aggressive)
# STAR_TIER_CONFIG - DISABLED Feb 3, 2026
# 7-day validation (Jan 27 - Feb 2): 1W-6L (14.3% WR, -72.7% ROI)
# Star players underperforming vs lines - disable until regime stabilizes
STAR_TIER_CONFIG = {
    "POINTS": {
        "enabled": False,  # DISABLED Feb 3 - 14.3% WR in last 7 days
        "min_p_over": 0.60,
        "max_p_over": 0.85,
        "min_spread": 0.5,
        "min_line": 15.0,
        "max_line": 35.0,
        "min_edge": 0.5,
        "avoid_books_softest": {"betrivers", "BetRivers"},
    },
    "REBOUNDS": {
        "enabled": False,  # DISABLED Feb 3 - 14.3% WR in last 7 days
        "min_p_over": 0.60,
        "max_p_over": 0.75,
        "min_spread": 0.5,
        "min_line": 4.5,
        "max_line": 12.0,
        "min_edge": 0.25,
    },
    "ASSISTS": {"enabled": False},
    "THREES": {"enabled": False},
}

# Legacy config for backwards compatibility
PRODUCTION_CONFIG = {
    "POINTS": {
        "enabled": True,
        "use_hybrid_filter": True,
        "min_probability": 0.58,
        "min_edge_points": 1.5,
        "max_edge_points": 5.0,
        "min_line": 12.0,
        # max_line REMOVED - was filtering out star players
        "min_spread": 2.5,
        "high_confidence_p_over": 0.65,
        "tier_b": None,
        "max_edge_low_variance": 3.0,
    },
    "REBOUNDS": {
        "enabled": True,
        "use_hybrid_filter": True,
        "min_probability": 0.58,
        "min_edge_points": 1.0,
        "min_line": 3.0,  # Filter bad ESPNBet data
        "min_spread": 2.5,
        "high_confidence_p_over": 0.65,
        "max_edge_low_variance": 2.0,
        "tier_b": None,
    },
    "ASSISTS": {"enabled": False},
    "THREES": {"enabled": False},
}


class LineOptimizer:
    """
    Optimizes line selection across multiple sportsbooks.
    Finds softest lines and calculates edge for each book.
    """

    def __init__(self) -> None:
        """
        Initialize LineOptimizer.

        Database connection is lazy - call connect() or use methods that auto-connect.
        """
        self.conn: Optional[Any] = None  # psycopg2 connection

    def connect(self) -> None:
        """
        Connect to nba_intelligence database (port 5539).

        Connection is lazy and only established on first call.
        Safe to call multiple times - subsequent calls are no-ops.

        Raises:
            psycopg2.OperationalError: If database connection fails
        """
        if not self.conn:
            self.conn = psycopg2.connect(**DB_INTELLIGENCE)

    def get_all_book_lines(
        self,
        player_name: str,
        game_date: str,
        stat_type: str,
        opponent_team: str = None,
        is_home: bool = None,
    ) -> Optional[pd.DataFrame]:
        """
        Query all available book lines for a player/game/stat.

        Args:
            player_name: Player's full name
            game_date: Game date (YYYY-MM-DD)
            stat_type: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
            opponent_team: Opponent team abbrev (optional, filters to specific matchup)
            is_home: Home/away status (optional, filters to specific matchup)

        Returns:
            DataFrame with columns: book_name, over_line, opponent_team, is_home
            Or None if no props found
        """
        self.connect()

        # Build query with optional opponent/is_home filters
        query = """
        WITH latest_props AS (
            SELECT
                book_name,
                over_line,
                opponent_team,
                is_home,
                fetch_timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY book_name
                    ORDER BY fetch_timestamp DESC
                ) as rn
            FROM nba_props_xl
            WHERE player_name = %s
                AND game_date = %s
                AND stat_type = %s
                AND is_active = true
                AND over_line IS NOT NULL
        """

        params = [player_name, game_date, stat_type]

        # Add matchup filters if provided (CRITICAL: matches main query grouping)
        if opponent_team is not None:
            query += "                AND opponent_team = %s\n"
            params.append(opponent_team)

        if is_home is not None:
            query += "                AND is_home = %s\n"
            # Convert numpy.bool_ to Python bool to avoid psycopg2 adaptation error
            params.append(bool(is_home))

        query += """        )
        SELECT
            book_name,
            over_line,
            opponent_team,
            is_home
        FROM latest_props
        WHERE rn = 1
        ORDER BY over_line ASC;
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df if len(df) > 0 else None
        except psycopg2.Error as e:
            logger.error(f"Database error querying book lines: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Data error querying book lines: {e}")
            return None

    def optimize_line(
        self,
        player_name: str,
        game_date: str,
        stat_type: str,
        prediction: float,
        p_over: float,
        opponent_team: str = None,
        is_home: bool = None,
        underdog_only: bool = None,
        avg_minutes: float = None,
        volatility_features: Dict[str, float] = None,
    ) -> Optional[Dict]:
        """
        Find best book/line to bet based on model prediction and line shopping.

        Strategy (from validation):
        - Always bet OVER for POINTS/REBOUNDS (validated strategy)
        - Select SOFTEST line (min over_line across all books)
        - Require edge ≥2.5% OR line_spread ≥2.5 (high-spread goldmine)

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: Market type
            prediction: Model's predicted stat value
            p_over: Model's P(OVER) probability
            opponent_team: Opponent team (filters to specific matchup)
            is_home: Home/away status (filters to specific matchup)
            underdog_only: If True, only accept when Underdog is softest
                          If None, uses global UNDERDOG_ONLY_MODE setting

        Returns:
            Dict with:
            - best_book: Book name with softest line
            - best_line: Softest over line
            - edge: prediction - best_line
            - consensus_line: Average of all books
            - line_spread: max_line - min_line
            - num_books: Number of books offering
            Or None if no actionable bet
        """
        # Check if market is enabled
        if stat_type not in PRODUCTION_CONFIG or not PRODUCTION_CONFIG[stat_type]["enabled"]:
            return None

        # Get all book lines (filtered by matchup to match main query grouping)
        lines_df = self.get_all_book_lines(
            player_name, game_date, stat_type, opponent_team, is_home
        )

        if lines_df is None or len(lines_df) == 0:
            return None

        # Calculate metrics
        softest_line = lines_df["over_line"].min()
        hardest_line = lines_df["over_line"].max()
        consensus_line = lines_df["over_line"].mean()
        line_spread = hardest_line - softest_line
        num_books = len(lines_df)

        # Get top 3 softest lines (sorted ascending)
        top_3_lines = lines_df.nsmallest(min(3, len(lines_df)), "over_line")

        # =============================================================================
        # UNDERDOG-ONLY MODE CHECK (Jan 2, 2026) - POINTS ONLY
        # When enabled, only accept POINTS props where Underdog is softest
        # REBOUNDS uses normal filtering (was 65-67% WR)
        # =============================================================================
        use_underdog_only = underdog_only if underdog_only is not None else UNDERDOG_ONLY_MODE
        underdog_cfg = UNDERDOG_CONFIG.get(stat_type, {})
        market_underdog_enabled = underdog_cfg.get("enabled", False)

        # Only apply underdog filter if: global flag is on AND this market has it enabled
        if use_underdog_only and market_underdog_enabled:
            underdog_names = underdog_cfg.get("underdog_names", {"underdog", "Underdog"})
            min_spread_for_underdog = underdog_cfg.get("min_spread", 1.5)

            # Check if softest book is Underdog
            softest_book_name = lines_df.sort_values("over_line").iloc[0]["book_name"]
            is_underdog_softest = softest_book_name in underdog_names

            if not is_underdog_softest:
                logger.debug(
                    f"Underdog-only REJECT: {player_name} {stat_type} - "
                    f"Softest is {softest_book_name}, not Underdog"
                )
                return None

            # Check minimum spread requirement
            if line_spread < min_spread_for_underdog:
                logger.debug(
                    f"Underdog-only REJECT: {player_name} {stat_type} - "
                    f"Spread {line_spread:.2f} < {min_spread_for_underdog:.1f} required"
                )
                return None

            logger.debug(
                f"Underdog-only PASS: {player_name} {stat_type} - "
                f"Underdog softest, spread={line_spread:.2f}"
            )

        # =============================================================================
        # BLACKLIST CHECK (Jan 2, 2026)
        # Skip blacklisted books and use next softest
        # =============================================================================
        blacklist = BLACKLISTED_BOOKS.get(stat_type, set())

        # Also exclude pseudo-books that aren't real sportsbooks
        pseudo_books = {"consensus", "Consensus", "average", "Average"}
        blacklist = blacklist | pseudo_books

        # Find first non-blacklisted book from softest to hardest
        best_book = None
        best_line = None
        for _idx, row in lines_df.sort_values("over_line").iterrows():
            if row["book_name"] not in blacklist:
                best_book = row["book_name"]
                best_line = row["over_line"]
                break

        # If all books are blacklisted, skip this prop
        if best_book is None:
            logger.debug(f"All books blacklisted for {player_name} {stat_type}")
            return None

        # Log if we skipped a blacklisted book
        softest_book = lines_df.sort_values("over_line").iloc[0]["book_name"]
        if softest_book in blacklist:
            logger.info(
                f"Blacklist skip: {player_name} {stat_type} - "
                f"Skipped {softest_book}, using {best_book} instead"
            )

        # Calculate edge using softest line
        edge = prediction - best_line

        # Get game context
        opponent_team = lines_df["opponent_team"].iloc[0]
        is_home = lines_df["is_home"].iloc[0]

        # Apply production filters
        config = PRODUCTION_CONFIG[stat_type]

        # =============================================================================
        # BOOK-AWARE FILTERING (Jan 2, 2026)
        # When traditional books are softest, require higher confidence/spread
        # =============================================================================
        book_penalty_applied = False
        if best_book in TRAP_BOOKS_WHEN_SOFTEST:
            trap_config = TRAP_BOOKS_WHEN_SOFTEST[best_book]
            min_spread_required = trap_config["min_spread_required"]
            min_p_over_boost = trap_config["min_p_over_boost"]

            # If spread is too low when a trap book is softest, reject
            if line_spread < min_spread_required:
                logger.debug(
                    f"Book-aware filter REJECT: {player_name} {stat_type} - "
                    f"{best_book} is softest but spread {line_spread:.2f} < {min_spread_required:.2f} required"
                )
                return None

            # Also require higher p_over when trap book is softest
            # (effectively raising the bar for these picks)
            book_penalty_applied = True
            logger.debug(
                f"Book-aware filter PENALTY: {player_name} {stat_type} - "
                f"{best_book} is softest, requiring +{min_p_over_boost:.0%} p_over boost"
            )

        # Clamp edge for low-variance situations (few books, tight spread)
        clamp_edge = config.get("max_edge_low_variance")
        if clamp_edge and num_books <= 3 and line_spread < 1.0 and edge > clamp_edge:
            logger.debug(
                f"Clamping edge for low-variance prop: {player_name} {stat_type} "
                f"edge {edge:.2f} -> {clamp_edge:.2f} (books={num_books}, spread={line_spread:.2f})"
            )
            edge = clamp_edge

        # =============================================================================
        # TIERED FILTER SYSTEM (Backtested Nov 18 - Dec 28, 2025)
        # Tier A: 67.1% WR - High confidence (p_over >= 0.65 allows low-edge picks)
        # Tier B: 63.2% WR - Stable proven (original filter, requires edge > 0)
        # Tier C: 59.5% WR - Volume picks (min_spread=2.0, requires edge > 0)
        # Star Tier: 63.8% WR - Relaxed thresholds for profitable star players
        # =============================================================================
        passes_filter = False
        filter_tier = None

        # =============================================================================
        # STAR TIER CHECK (Jan 16, 2026) - Check FIRST before regular tiers
        # Updated to include max_p_over cap (avoid overconfident trap zone)
        # Use market-specific star lists for POINTS/REBOUNDS
        # =============================================================================
        if stat_type == "POINTS":
            is_star_player = player_name in POINTS_STAR_PLAYERS
        elif stat_type == "REBOUNDS":
            is_star_player = player_name in REBOUNDS_STAR_PLAYERS
        else:
            is_star_player = player_name in STAR_PLAYERS
        star_config = STAR_TIER_CONFIG.get(stat_type, {})

        if is_star_player and star_config.get("enabled", False):
            star_min_p_over = star_config.get("min_p_over", 0.55)
            star_max_p_over = star_config.get("max_p_over", 1.0)  # ★ NEW: max_p_over cap
            star_min_spread = star_config.get("min_spread", 1.0)
            star_min_line = star_config.get("min_line", 0)  # ★ NEW: min_line support
            star_max_line = star_config.get("max_line", 999)
            star_min_edge = star_config.get("min_edge", 0.25)
            star_avoid_books = star_config.get("avoid_books_softest", set())

            # Star tier gates (with max_p_over cap)
            star_prob_gate = p_over >= star_min_p_over and p_over <= star_max_p_over  # ★ UPDATED
            star_spread_gate = line_spread >= star_min_spread
            star_line_gate = best_line >= star_min_line and best_line <= star_max_line  # ★ UPDATED
            star_edge_gate = edge >= star_min_edge
            star_book_gate = best_book.lower() not in {b.lower() for b in star_avoid_books}

            if (
                star_prob_gate
                and star_spread_gate
                and star_line_gate
                and star_edge_gate
                and star_book_gate
            ):
                passes_filter = True
                filter_tier = "star_tier"
                logger.info(
                    f"STAR TIER pass: {player_name} {stat_type} - "
                    f"p_over {p_over:.3f} / edge {edge:.2f} / spread {line_spread:.1f} / line {best_line:.1f}"
                )
            else:
                # Log why star tier failed
                fail_reasons = []
                if not star_prob_gate:
                    if p_over < star_min_p_over:
                        fail_reasons.append(f"p_over {p_over:.3f} < {star_min_p_over}")
                    else:
                        fail_reasons.append(
                            f"p_over {p_over:.3f} > {star_max_p_over} (overconfident)"
                        )
                if not star_spread_gate:
                    fail_reasons.append(f"spread {line_spread:.1f} < {star_min_spread}")
                if not star_line_gate:
                    if best_line < star_min_line:
                        fail_reasons.append(f"line {best_line:.1f} < {star_min_line}")
                    else:
                        fail_reasons.append(f"line {best_line:.1f} > {star_max_line}")
                if not star_edge_gate:
                    fail_reasons.append(f"edge {edge:.2f} < {star_min_edge}")
                if not star_book_gate:
                    fail_reasons.append(f"book {best_book} in avoid list")
                logger.debug(
                    f"Star tier FAIL: {player_name} {stat_type} - {', '.join(fail_reasons)}"
                )

        # Get tier config for this market (fallback if star tier didn't pass)
        tier_config = TIER_CONFIG.get(stat_type, {})
        tiers = tier_config.get("tiers", {})

        # Common gates (apply to all tiers)
        min_prob = tier_config.get("min_probability", config.get("min_probability", 0.58))
        probability_gate = p_over >= min_prob

        min_line_val = tier_config.get("min_line", config.get("min_line", 0))
        max_line_val = tier_config.get("max_line", config.get("max_line", 999))
        line_gate = (best_line >= min_line_val) and (best_line < max_line_val)

        max_edge_val = tier_config.get("max_edge_points", config.get("max_edge_points", 999))
        edge_cap_gate = edge < max_edge_val

        # Calculate edge_pct for META tier (Jan 29, 2026)
        # edge_pct = (consensus_line - softest_line) / softest_line * 100
        edge_pct = ((consensus_line - best_line) / best_line * 100) if best_line > 0 else 0

        # Check tiers: Goldmine (high spread) first, then Standard
        # Skip if star tier already passed
        for tier_name in [
            "Goldmine",  # High spread = market inefficiency (80% WR)
            "Standard",  # Moderate spread + higher confidence (70% WR)
        ]:
            if passes_filter:  # Star tier already passed
                break
            tier_cfg = tiers.get(tier_name)
            if not tier_cfg:
                continue

            tier_min_spread = tier_cfg.get("min_spread", 0.0)
            tier_min_edge = tier_cfg.get("min_edge_points", 1.5)
            tier_min_edge_pct = tier_cfg.get("min_edge_pct", 0)
            tier_min_p_over = tier_cfg.get("min_p_over", 0.58)
            tier_max_p_over = tier_cfg.get("max_p_over", 1.0)
            tier_require_pos_edge = tier_cfg.get("require_positive_edge", True)
            tier_require_both = tier_cfg.get("require_both", False)

            # Apply book-aware penalty to p_over threshold when trap book is softest
            if book_penalty_applied and best_book in TRAP_BOOKS_WHEN_SOFTEST:
                p_over_boost = TRAP_BOOKS_WHEN_SOFTEST[best_book]["min_p_over_boost"]
                tier_min_p_over += p_over_boost

            # Probability gate
            tier_prob_gate = p_over >= tier_min_p_over and p_over <= tier_max_p_over

            # Spread/edge gate
            spread_gate = line_spread >= tier_min_spread
            edge_gate = edge >= tier_min_edge
            edge_pct_gate = edge_pct >= tier_min_edge_pct

            # Value gate: depends on require_both setting
            if tier_require_both:
                if tier_min_edge_pct > 0:
                    # Y tier: spread AND edge_pct required
                    value_gate = spread_gate and edge_pct_gate
                else:
                    value_gate = spread_gate and edge_gate
            else:
                # Simple edge check
                value_gate = edge_gate

            if tier_require_pos_edge:
                value_gate = value_gate and (edge > 0)

            # All gates must pass for this tier
            if tier_prob_gate and value_gate and line_gate and edge_cap_gate:
                passes_filter = True
                filter_tier = tier_name
                logger.debug(
                    f"Tier-{tier_name} pass: {player_name} {stat_type} - "
                    f"p_over {p_over:.3f} / edge {edge:.2f} / edge_pct {edge_pct:.1f}% / spread {line_spread:.1f}"
                )
                break  # Use highest tier that passes

        if not passes_filter:
            if not probability_gate:
                logger.debug(
                    f"Filtered (prob gate): {player_name} {stat_type} - "
                    f"p_over {p_over:.3f} < {min_prob:.3f}"
                )
            elif not line_gate:
                logger.debug(
                    f"Filtered (line gate): {player_name} {stat_type} - "
                    f"line {best_line:.1f} not in [{min_line_val:.1f}, {max_line_val:.1f})"
                )
            elif not edge_cap_gate:
                logger.debug(
                    f"Filtered (edge cap): {player_name} {stat_type} - "
                    f"edge {edge:.2f} >= {max_edge_val:.2f}"
                )
            else:
                logger.debug(
                    f"Filtered (all tiers): {player_name} {stat_type} - "
                    f"p_over {p_over:.3f} / edge {edge:.2f} / edge_pct {edge_pct:.1f}% / spread {line_spread:.1f}"
                )

        # Return None if filtered
        if not passes_filter:
            return None

        # Warning for low line spread (books agree = less edge)
        if line_spread < 0.5:
            logger.warning(
                f"[WARN]  LOW SPREAD: {player_name} {stat_type} - "
                f"spread {line_spread:.2f} pts (books agree, lower edge confidence)"
            )

        # Determine confidence based on hybrid filter metrics
        if p_over >= 0.70 and edge >= 3.0:
            confidence = "HIGH"  # Strong model confidence + large edge
        elif p_over >= 0.60 and edge >= 2.0:
            confidence = "MEDIUM"  # Good confidence + decent edge
        elif line_spread >= 2.5:
            confidence = "MEDIUM"  # High line spread (book disagreement)
        elif edge >= 3.0:
            confidence = "MEDIUM"
        else:
            confidence = "STANDARD"

        if filter_tier == "tier_b" and confidence == "MEDIUM":
            confidence = "STANDARD"  # Tier B picks carry slightly less conviction

        # Build top 3 lines with individual edges
        top_lines = []
        for _idx, row in top_3_lines.iterrows():
            line_edge = prediction - row["over_line"]
            top_lines.append(
                {
                    "book": row["book_name"],
                    "line": float(row["over_line"]),
                    "edge": float(line_edge),
                    "edge_pct": (line_edge / row["over_line"] * 100) if row["over_line"] > 0 else 0,
                }
            )

        # Calculate consensus offset (how far softest line is from consensus)
        consensus_offset = best_line - consensus_line

        # Group books by line value for line range display
        from collections import defaultdict

        line_groups = defaultdict(list)
        for _idx, row in lines_df.iterrows():
            line_groups[float(row["over_line"])].append(row["book_name"])

        # Sort lines (softest to hardest)
        sorted_lines = sorted(line_groups.keys())

        # Build line distribution for display
        line_distribution = []
        for line_value in sorted_lines:
            books = line_groups[line_value]
            line_edge = prediction - line_value
            line_distribution.append(
                {
                    "line": line_value,
                    "books": books,
                    "count": len(books),
                    "edge": float(line_edge),
                    "edge_pct": (line_edge / line_value * 100) if line_value > 0 else 0,
                }
            )

        # Determine if softest book is considered reliable or trap
        is_trap_book = best_book in TRAP_BOOKS_WHEN_SOFTEST
        is_reliable_book = best_book in RELIABLE_BOOKS_WHEN_SOFTEST

        # Check if underdog-only mode was used
        use_underdog_only = underdog_only if underdog_only is not None else UNDERDOG_ONLY_MODE

        # =============================================================================
        # VOLATILITY-AWARE STAKE SIZING (Feb 3, 2026)
        # Uses player volatility features to adjust recommended stake
        # Low volatility + high confidence = PRESS (increase stake)
        # High volatility = FADE (decrease stake, even with high confidence)
        # =============================================================================
        stake_info = None
        if volatility_features and STAKE_SIZING_CONFIG.enabled:
            # Get usage_volatility_score (combined CV of stat + minutes)
            # Fallback to computing from std/mean if not available
            vol_score = volatility_features.get("usage_volatility_score", 0.0)

            # If usage_volatility_score not available, compute from std features
            if vol_score == 0.0:
                stat_key = stat_type.lower()
                std_key = f"{stat_key}_std_L5"
                mean_key = f"ema_{stat_key}_L5"
                if std_key in volatility_features and mean_key in volatility_features:
                    std_val = volatility_features.get(std_key, 0.0)
                    mean_val = volatility_features.get(mean_key, 1.0)
                    if mean_val > 0:
                        vol_score = std_val / mean_val  # Coefficient of variation

            stake_info = calculate_stake(
                p_over=p_over,
                edge=edge,
                volatility_score=vol_score,
                market=stat_type,
            )
            logger.debug(
                f"Stake sizing: {player_name} {stat_type} - "
                f"vol={vol_score:.3f} → {stake_info['stake_units']}u ({stake_info['stake_label']})"
            )

        return {
            "best_book": best_book,
            "best_line": float(best_line),
            "edge": float(edge),
            "line_edge_pct": float(edge_pct),  # NEW: edge_pct for META tier
            "consensus_line": float(consensus_line),
            "consensus_offset": float(consensus_offset),
            "line_spread": float(line_spread),
            "num_books": int(num_books),
            "confidence": confidence,
            "opponent_team": opponent_team,
            "is_home": bool(is_home),
            "p_over": float(p_over),
            "filter_tier": filter_tier or "unknown",
            "top_3_lines": top_lines,  # Top 3 softest lines with individual edges
            "line_distribution": line_distribution,  # All lines grouped by value
            "book_aware_filter": {
                "is_trap_book": is_trap_book,
                "is_reliable_book": is_reliable_book,
                "penalty_applied": book_penalty_applied,
            },
            "underdog_only_mode": use_underdog_only,
            "stake_sizing": stake_info,  # NEW: Volatility-aware stake recommendation
        }

    def optimize_line_v3(
        self,
        player_name: str,
        game_date: str,
        stat_type: str,
        prediction: float,
        p_over: float,
        opponent_team: str = None,
        is_home: bool = None,
        volatility_features: Dict[str, float] = None,
    ) -> Optional[Dict]:
        """
        V3 line optimizer with UNDER betting support.

        Key difference from optimize_line():
        - Supports both OVER and UNDER directions based on p_over
        - Uses V3_TIER_CONFIG tiers with direction-specific filters
        - OVER: selects softest line (minimum), edge = prediction - line
        - UNDER: selects hardest line (maximum), edge = line - prediction

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: Market type (POINTS, REBOUNDS, etc.)
            prediction: Model's predicted stat value
            p_over: Model's P(OVER) probability
            opponent_team: Opponent team (filters to specific matchup)
            is_home: Home/away status (filters to specific matchup)

        Returns:
            Dict with:
            - direction: 'OVER' or 'UNDER'
            - best_book: Book name with optimal line for direction
            - best_line: Optimal line (softest for OVER, hardest for UNDER)
            - edge: Calculated edge for the direction
            - tier: V3 tier that passed (V3_ELITE_OVER, etc.)
            Or None if no actionable bet
        """
        # Check if V3 is enabled for this market
        v3_config = V3_TIER_CONFIG.get(stat_type, {})
        if not v3_config.get("enabled", False):
            return None

        tiers = v3_config.get("tiers", {})
        if not tiers:
            return None

        # Get all book lines
        lines_df = self.get_all_book_lines(
            player_name, game_date, stat_type, opponent_team, is_home
        )

        if lines_df is None or len(lines_df) == 0:
            return None

        # Calculate metrics
        softest_line = lines_df["over_line"].min()
        hardest_line = lines_df["over_line"].max()
        consensus_line = lines_df["over_line"].mean()
        line_spread = hardest_line - softest_line
        num_books = len(lines_df)

        # Exclude pseudo-books from best book selection
        pseudo_books = {"consensus", "Consensus", "average", "Average"}
        valid_lines_df = lines_df[~lines_df["book_name"].isin(pseudo_books)]

        if len(valid_lines_df) == 0:
            return None

        # ★ REGIME SHIFT FIX: Exclude blacklisted books for OVER picks
        # BetRivers: 0W/7L = 0% WR in January for OVER picks
        avoid_books_over = v3_config.get("avoid_books_over", set())
        if avoid_books_over:
            # Create filtered df for OVER direction (case-insensitive)
            avoid_books_lower = {b.lower() for b in avoid_books_over}
            valid_lines_df_over = valid_lines_df[
                ~valid_lines_df["book_name"].str.lower().isin(avoid_books_lower)
            ]
            # Recalculate softest_line for OVER using filtered books
            if len(valid_lines_df_over) > 0:
                softest_line = valid_lines_df_over["over_line"].min()
        else:
            valid_lines_df_over = valid_lines_df

        if len(valid_lines_df_over) == 0:
            logger.debug(
                f"V3 filter REJECT: {player_name} {stat_type} - all OVER books blacklisted"
            )
            return None

        # ★ NEW: Also exclude blacklisted books for UNDER picks (betrivers)
        avoid_books_under = v3_config.get("avoid_books_under", set())
        if avoid_books_under:
            avoid_books_under_lower = {b.lower() for b in avoid_books_under}
            valid_lines_df_under = valid_lines_df[
                ~valid_lines_df["book_name"].str.lower().isin(avoid_books_under_lower)
            ]
            # Recalculate hardest_line for UNDER using filtered books
            if len(valid_lines_df_under) > 0:
                hardest_line = valid_lines_df_under["over_line"].max()
        else:
            valid_lines_df_under = valid_lines_df

        # Calculate p_under (inverse of p_over)
        p_under = 1.0 - p_over

        # Determine direction based on probability
        # If p_over is high (>= 0.50), model prefers OVER
        # If p_over is low (< 0.50), model prefers UNDER (which means p_under > 0.50)
        direction = "OVER" if p_over >= 0.50 else "UNDER"

        # Try each V3 tier to find a match
        # Order: Highest WR tiers first (from Jan 16 backtest)
        # V3_HIGHLINE_UNDER: 90.9% WR | V3_EDGE_UNDER: 85.7% WR
        # V3_OPTIMAL_OVER: 82.1% WR | V3_CONFIDENT_OVER: 81.8% WR
        # JANUARY-OPTIMIZED tier order (OVER only - UNDER disabled at 50% WR)
        # UPDATED Jan 28, 2026: Disabled JAN_PRIME_OVER (40% WR) and JAN_LINE_OVER (58% WR)
        tier_order = [
            "JAN_CONFIDENT_OVER",  # 100% WR (2W/0L) - p 0.75-0.85 + edge >= 4 - ONLY REMAINING
        ]

        matched_tier = None
        matched_direction = None

        # Pre-calculate edge for OVER direction (needed for min_edge check)
        # Use filtered softest_line (excluding blacklisted books)
        edge_over = prediction - softest_line

        for tier_name in tier_order:
            tier_cfg = tiers.get(tier_name)
            if not tier_cfg:
                continue

            tier_direction = tier_cfg.get("direction", "OVER")

            # Check probability threshold (min AND max for OVER tiers)
            if tier_direction == "OVER":
                min_prob = tier_cfg.get("min_p_over", 0.50)
                max_prob = tier_cfg.get("max_p_over", 1.0)  # ★ NEW: max_p_over cap
                if p_over < min_prob or p_over > max_prob:
                    continue
            else:  # UNDER
                min_p_under = tier_cfg.get("min_p_under", 0.50)
                if p_under < min_p_under:
                    continue

            # Check line constraints
            min_line = tier_cfg.get("min_line", 0)
            max_line = tier_cfg.get("max_line", 999)

            # For OVER: check against softest line
            # For UNDER: check against hardest line
            check_line = softest_line if tier_direction == "OVER" else hardest_line
            if check_line < min_line or check_line > max_line:
                continue

            # Check spread constraint (if specified)
            min_spread = tier_cfg.get("min_spread", 0)
            if line_spread < min_spread:
                continue

            # ★ NEW: Check edge constraint (if specified)
            min_edge = tier_cfg.get("min_edge", 0)
            if min_edge > 0:
                check_edge = edge_over if tier_direction == "OVER" else (hardest_line - prediction)
                if check_edge < min_edge:
                    continue

            # This tier passed all gates
            matched_tier = tier_name
            matched_direction = tier_direction
            logger.debug(
                f"V3 tier {tier_name} PASS: {player_name} {stat_type} - "
                f"p_over {p_over:.3f} / edge {edge_over:.2f} / spread {line_spread:.1f} / line {check_line:.1f}"
            )
            break

        # If no V3 tier matched, check STAR tier for star players (OVER only)
        # Use market-specific star lists for POINTS/REBOUNDS
        if matched_tier is None:
            if stat_type == "POINTS":
                is_star_player = player_name in POINTS_STAR_PLAYERS
            elif stat_type == "REBOUNDS":
                is_star_player = player_name in REBOUNDS_STAR_PLAYERS
            else:
                is_star_player = player_name in STAR_PLAYERS
            star_config = STAR_TIER_CONFIG.get(stat_type, {})

            if is_star_player and star_config.get("enabled", False):
                star_min_p_over = star_config.get("min_p_over", 0.70)
                star_max_p_over = star_config.get("max_p_over", 1.0)
                star_min_spread = star_config.get("min_spread", 2.0)
                star_max_line = star_config.get("max_line", 29.0)
                star_min_line = star_config.get("min_line", 0)
                star_min_edge = star_config.get("min_edge", 0.5)
                star_avoid_books = star_config.get("avoid_books_softest", set())

                # ★ REGIME SHIFT FIX: Use valid_lines_df_over (excludes blacklisted books)
                # This ensures STAR tier uses same filtered book set as XL tiers
                edge = (
                    prediction - softest_line
                )  # softest_line already calculated from filtered set

                # Get softest book from FILTERED set (not all books)
                if len(valid_lines_df_over) > 0:
                    softest_book = valid_lines_df_over.loc[
                        valid_lines_df_over["over_line"].idxmin()
                    ]["book_name"]
                else:
                    softest_book = ""  # Will fail book gate

                star_prob_gate = p_over >= star_min_p_over and p_over <= star_max_p_over
                star_spread_gate = line_spread >= star_min_spread
                star_line_gate = softest_line >= star_min_line and softest_line <= star_max_line
                star_edge_gate = edge >= star_min_edge
                star_book_gate = softest_book.lower() not in {b.lower() for b in star_avoid_books}

                if (
                    star_prob_gate
                    and star_spread_gate
                    and star_line_gate
                    and star_edge_gate
                    and star_book_gate
                ):
                    matched_tier = "star_tier"
                    matched_direction = "OVER"
                    logger.info(
                        f"STAR PASS: {player_name} {stat_type} - "
                        f"p_over {p_over:.3f} / edge {edge:.2f} / spread {line_spread:.1f} / line {softest_line:.1f}"
                    )

        if matched_tier is None:
            logger.debug(
                f"V3 filter REJECT: {player_name} {stat_type} - "
                f"p_over {p_over:.3f}, p_under {p_under:.3f}, no tier match"
            )
            return None

        # Get best line and book based on direction
        if matched_direction == "OVER":
            # For OVER: softest (lowest) line is best
            # ★ Use valid_lines_df_over (excludes blacklisted books like BetRivers)
            best_row = valid_lines_df_over.loc[valid_lines_df_over["over_line"].idxmin()]
            best_book = best_row["book_name"]
            best_line = float(best_row["over_line"])
            edge = prediction - best_line
        else:  # UNDER
            # For UNDER: hardest (highest) line is best
            # ★ Use valid_lines_df_under (excludes blacklisted books like betrivers)
            if len(valid_lines_df_under) == 0:
                logger.debug(
                    f"V3 filter REJECT: {player_name} {stat_type} - all UNDER books blacklisted"
                )
                return None
            best_row = valid_lines_df_under.loc[valid_lines_df_under["over_line"].idxmax()]
            best_book = best_row["book_name"]
            best_line = float(best_row["over_line"])
            edge = best_line - prediction

        # Get expected win rate for this tier
        if matched_tier == "STAR_V3":
            expected_wr = 0.833  # 83.3% WR from backtest
            confidence = "HIGH"
        elif matched_tier in tiers:
            expected_wr = tiers[matched_tier].get("expected_wr", 0.60)
            confidence = "HIGH" if "ELITE" in matched_tier else "MEDIUM"
        else:
            expected_wr = 0.60
            confidence = "MEDIUM"

        # Get opponent/home from data
        opponent_team = lines_df["opponent_team"].iloc[0]
        is_home = lines_df["is_home"].iloc[0]

        # Build top 3 lines with edges
        if matched_direction == "OVER":
            # ★ Use valid_lines_df_over (excludes blacklisted books)
            top_3 = valid_lines_df_over.nsmallest(min(3, len(valid_lines_df_over)), "over_line")
        else:
            # ★ Use valid_lines_df_under for UNDER (excludes blacklisted books)
            top_3 = valid_lines_df_under.nlargest(min(3, len(valid_lines_df_under)), "over_line")

        top_lines = []
        for _idx, row in top_3.iterrows():
            if matched_direction == "OVER":
                line_edge = prediction - row["over_line"]
            else:
                line_edge = row["over_line"] - prediction
            top_lines.append(
                {
                    "book": row["book_name"],
                    "line": float(row["over_line"]),
                    "edge": float(line_edge),
                }
            )

        # Calculate consensus offset (how far best line is from consensus)
        consensus_offset = best_line - consensus_line

        # Build line distribution for display (group books by line value)
        from collections import defaultdict

        line_groups = defaultdict(list)
        for _idx, row in valid_lines_df.iterrows():
            line_groups[float(row["over_line"])].append(row["book_name"])

        line_distribution = []
        for line_value in sorted(line_groups.keys()):
            books = line_groups[line_value]
            if matched_direction == "OVER":
                line_edge = prediction - line_value
            else:
                line_edge = line_value - prediction
            line_distribution.append(
                {
                    "line": line_value,
                    "books": books,
                    "count": len(books),
                    "edge": float(line_edge),
                    "edge_pct": (line_edge / line_value * 100) if line_value > 0 else 0,
                }
            )

        logger.info(
            f"V3 {matched_tier} PASS: {player_name} {stat_type} {matched_direction} - "
            f"line {best_line:.1f} @ {best_book}, edge {edge:.2f}, p={p_over:.3f}"
        )

        # =============================================================================
        # VOLATILITY-AWARE STAKE SIZING (Feb 3, 2026)
        # =============================================================================
        stake_info = None
        if volatility_features and STAKE_SIZING_CONFIG.enabled:
            vol_score = volatility_features.get("usage_volatility_score", 0.0)

            # Fallback: compute from std/mean if not available
            if vol_score == 0.0:
                stat_key = stat_type.lower()
                std_key = f"{stat_key}_std_L5"
                mean_key = f"ema_{stat_key}_L5"
                if std_key in volatility_features and mean_key in volatility_features:
                    std_val = volatility_features.get(std_key, 0.0)
                    mean_val = volatility_features.get(mean_key, 1.0)
                    if mean_val > 0:
                        vol_score = std_val / mean_val

            stake_info = calculate_stake(
                p_over=p_over,
                edge=edge,
                volatility_score=vol_score,
                market=stat_type,
            )
            logger.debug(
                f"Stake sizing: {player_name} {stat_type} - "
                f"vol={vol_score:.3f} → {stake_info['stake_units']}u ({stake_info['stake_label']})"
            )

        return {
            "direction": matched_direction,
            "best_book": best_book,
            "best_line": best_line,
            "edge": float(edge),
            "consensus_line": float(consensus_line),
            "consensus_offset": float(consensus_offset),
            "line_spread": float(line_spread),
            "num_books": int(num_books),
            "confidence": confidence,
            "opponent_team": opponent_team,
            "is_home": bool(is_home),
            "p_over": float(p_over),
            "p_under": float(p_under),
            "filter_tier": matched_tier,
            "expected_wr": expected_wr,
            "top_3_lines": top_lines,
            "line_distribution": line_distribution,
            "model_version": "v3",
            "stake_sizing": stake_info,  # NEW: Volatility-aware stake recommendation
        }

    # NOTE: apply_odds_api_filter() removed - Odds API filtering is now in
    # standalone generate_odds_api_picks.py (Jan 26, 2026)

    def close(self) -> None:
        """
        Close database connection and release resources.

        Safe to call multiple times or on unconnected instance.
        Should be called when done with optimizer to prevent connection leaks.
        """
        if self.conn:
            self.conn.close()
            self.conn = None


if __name__ == "__main__":
    # Test line optimizer
    print("Testing Line Optimizer...")

    optimizer = LineOptimizer()

    # Test query (will return None if no data for this date)
    result = optimizer.get_all_book_lines(
        player_name="Luka Doncic", game_date="2025-11-07", stat_type="POINTS"
    )

    if result is not None:
        print(f"[OK] Found {len(result)} book lines for Luka Doncic")
        print(result)
    else:
        print("[WARN]  No book lines found (expected if no props for this date)")

    optimizer.close()
