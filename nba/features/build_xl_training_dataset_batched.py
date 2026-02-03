#!/usr/bin/env python3
"""
NBA XL Training Dataset Builder - BATCHED VERSION (10x faster)
===============================================================

Optimized version that eliminates N+1 query problem:
- OLD: 60,000 props × 10+ queries = 600,000+ queries (~80 minutes)
- NEW: ~15 bulk queries + in-memory assembly (~5-8 minutes)

Architecture:
1. Fetch all props from database (single query)
2. Pre-fetch ALL supporting data in bulk:
   - Player game logs (for rolling stats)
   - Team season stats (for pace/ratings)
   - Matchup history (for H2H features)
   - Prop performance history
   - Vegas context (games table)
   - Cheatsheet data
   - Book lines (for book features)
3. Build in-memory lookup dictionaries
4. Extract features using pure Python lookups (no DB calls)

Usage:
    python build_xl_training_dataset_batched.py --output datasets/
    python build_xl_training_dataset_batched.py --market POINTS --output datasets/
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from nba.utils.season_helpers import date_to_season

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": os.getenv("DB_USER", "mlb_user"),
    "password": os.getenv("DB_PASSWORD"),
}

# Port mapping for each database
DB_PORTS = {
    "players": 5536,  # nba_players - player_profile, player_game_logs
    "games": 5537,  # nba_games - games table
    "team": 5538,  # nba_team - team_season_stats, team_betting_performance
    "intelligence": 5539,  # nba_intelligence - props, matchup_history, etc.
}

# Temporal split cutoffs
TRAIN_START = date(2023, 10, 24)
TRAIN_END = date(2025, 12, 1)

# Stat types to build
STAT_TYPES = ["POINTS", "REBOUNDS"]

# Team abbreviation normalization
TEAM_ABBREV_MAP = {
    "NO": "NOP",
    "SA": "SAS",
    "UTAH": "UTA",
    "GS": "GSW",
    "NY": "NYK",
    "BKN": "BKN",
    "BRK": "BKN",
}


@dataclass
class DataCache:
    """In-memory cache for all pre-fetched data"""

    # Player data: player_name -> list of game dicts (sorted by date DESC)
    player_games: Dict[str, List[Dict]] = field(default_factory=dict)

    # Player team mapping: player_name -> team_abbrev (current team)
    player_teams: Dict[str, str] = field(default_factory=dict)

    # Player positions: player_name -> position_encoded (1=PG, 2=SG, 3=SF, 4=PF, 5=C)
    player_positions: Dict[str, float] = field(default_factory=dict)

    # Team stats: (team_abbrev, season) -> {pace, off_rating, def_rating}
    team_stats: Dict[Tuple[str, int], Dict[str, float]] = field(default_factory=dict)

    # Team betting: (team_abbrev, season) -> {ats_pct, ou_pct}
    team_betting: Dict[Tuple[str, int], Dict[str, float]] = field(default_factory=dict)

    # H2H matchup: (player_name, opponent_team, stat_type) -> features dict
    h2h_stats: Dict[Tuple[str, str, str], Dict[str, float]] = field(default_factory=dict)

    # Prop history: (player_name, stat_type, line_center, season) -> features dict
    prop_history: Dict[Tuple[str, str, float, int], Dict[str, float]] = field(default_factory=dict)

    # Vegas context: (game_date, team_abbrev) -> {total, spread, home_team, away_team}
    vegas_context: Dict[Tuple[date, str], Dict] = field(default_factory=dict)

    # Cheatsheet: (player_name, game_date, stat_type) -> features dict
    cheatsheet: Dict[Tuple[str, date, str], Dict[str, float]] = field(default_factory=dict)

    # Book lines: (player_name, game_date, stat_type) -> list of {book_name, line}
    book_lines: Dict[Tuple[str, date, str], List[Dict]] = field(default_factory=dict)

    # Historical line spreads: (player_name, stat_type) -> list of spreads (for percentile)
    historical_spreads: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)

    # Book accuracy: (book_name, market) -> {hit_rate, soft_line_rate, line_bias, sharpe_ratio}
    book_accuracy: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    # Line movement: (player_name, game_date, stat_type) -> {line_delta, line_std, snapshots, hours_to_game}
    line_movement: Dict[Tuple[str, date, str], Dict[str, float]] = field(default_factory=dict)


class BatchedDatasetBuilder:
    """Builds XL training datasets using batched queries (10x faster)"""

    def __init__(self, output_dir: str = "datasets/", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.cache = DataCache()
        self.connections: Dict[str, psycopg2.extensions.connection] = {}

        if self.verbose:
            print("=" * 80)
            print("NBA XL TRAINING DATASET BUILDER - BATCHED (10x FASTER)")
            print("=" * 80)
            print(f"Training period: {TRAIN_START} to {TRAIN_END}")
            print("=" * 80)

    def connect(self):
        """Connect to all databases"""
        db_names = {
            "players": "nba_players",
            "games": "nba_games",
            "team": "nba_team",
            "intelligence": "nba_intelligence",
        }
        for db_name, port in DB_PORTS.items():
            config = {**DB_CONFIG, "port": port, "database": db_names[db_name]}
            self.connections[db_name] = psycopg2.connect(**config)
            self.connections[db_name].autocommit = True

        if self.verbose:
            print(f"✅ Connected to {len(self.connections)} databases")

    def disconnect(self):
        """Disconnect from all databases"""
        for conn in self.connections.values():
            conn.close()
        if self.verbose:
            print("✅ Disconnected from all databases")

    def normalize_team(self, abbrev: str) -> str:
        """Normalize team abbreviation"""
        if not abbrev:
            return abbrev
        return TEAM_ABBREV_MAP.get(abbrev, abbrev)

    def calculate_ema(self, values: List[float], alpha: float = 0.4) -> float:
        """Calculate exponential moving average"""
        valid = [v for v in values if v is not None and not np.isnan(v)]
        if not valid:
            return 0.0
        ema = valid[0]
        for v in valid[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    # =========================================================================
    # BULK DATA LOADING
    # =========================================================================

    def load_all_player_games(self, player_names: Set[str], max_date: date):
        """Load all player game logs in one query"""
        if self.verbose:
            print(f"\n📊 Loading player game logs for {len(player_names)} players...")

        # Query all game logs for these players up to max_date
        # Note: nba_players uses minutes_played, fg_made, fg_attempted, ft_made, ft_attempted
        query = """
            SELECT
                pp.full_name as player_name,
                pgl.game_date,
                pgl.team_abbrev,
                pgl.opponent_abbrev,
                pgl.is_home,
                pgl.points,
                pgl.rebounds,
                pgl.assists,
                pgl.three_pointers_made as threes,
                pgl.steals,
                pgl.blocks,
                pgl.turnovers,
                pgl.minutes_played,
                pgl.fg_made,
                pgl.fg_attempted,
                pgl.ft_made,
                pgl.ft_attempted,
                pgl.plus_minus
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE pgl.game_date <= %s
            ORDER BY pp.full_name, pgl.game_date DESC
        """

        with self.connections["players"].cursor() as cur:
            cur.execute(query, (max_date,))
            rows = cur.fetchall()

        # Build lookup: player_name -> list of games (sorted by date DESC)
        for row in rows:
            player_name = row[0]
            game = {
                "game_date": row[1],
                "team_abbrev": row[2],
                "opponent_abbrev": row[3],
                "is_home": row[4],
                "points": row[5] or 0,
                "rebounds": row[6] or 0,
                "assists": row[7] or 0,
                "threes": row[8] or 0,
                "steals": row[9] or 0,
                "blocks": row[10] or 0,
                "turnovers": row[11] or 0,
                "minutes": row[12] or 0,
                "fgm": row[13] or 0,
                "fga": row[14] or 0,
                "ftm": row[15] or 0,
                "fta": row[16] or 0,
                "plus_minus": row[17] or 0,
            }

            if player_name not in self.cache.player_games:
                self.cache.player_games[player_name] = []
            self.cache.player_games[player_name].append(game)

            # Also track current team (most recent game)
            if player_name not in self.cache.player_teams:
                self.cache.player_teams[player_name] = game["team_abbrev"]

        if self.verbose:
            print(
                f"   ✅ Loaded {len(rows):,} game logs for {len(self.cache.player_games)} players"
            )

        # Also load player positions
        self._load_player_positions()

    def _load_player_positions(self):
        """Load player positions from player_profile"""
        query = """
            SELECT full_name, position
            FROM player_profile
            WHERE position IS NOT NULL
        """
        # Position encoding: PG=1, SG=2, SF=3, PF=4, C=5
        pos_map = {"PG": 1.0, "SG": 2.0, "G": 1.5, "SF": 3.0, "PF": 4.0, "F": 3.5, "C": 5.0}

        with self.connections["players"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                name, pos = row
                if pos:
                    # Handle compound positions like "SG-SF"
                    primary_pos = pos.split("-")[0].strip().upper()
                    self.cache.player_positions[name] = pos_map.get(primary_pos, 3.0)

    def load_all_team_stats(self):
        """Load all team season stats for ALL seasons (prevents data leakage)"""
        if self.verbose:
            print("\n📊 Loading team season stats (all seasons)...")

        query = """
            SELECT team_abbrev, season, pace, offensive_rating, defensive_rating,
                   def_rating_vs_pg, def_rating_vs_sg, def_rating_vs_sf,
                   def_rating_vs_pf, def_rating_vs_c
            FROM team_season_stats
        """

        with self.connections["team"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                team = row[0]
                season = row[1]
                self.cache.team_stats[(team, season)] = {
                    "pace": float(row[2]) if row[2] else 100.0,
                    "off_rating": float(row[3]) if row[3] else 112.0,
                    "def_rating": float(row[4]) if row[4] else 112.0,
                    "def_rating_vs_pg": float(row[5]) if row[5] else 112.0,
                    "def_rating_vs_sg": float(row[6]) if row[6] else 112.0,
                    "def_rating_vs_sf": float(row[7]) if row[7] else 112.0,
                    "def_rating_vs_pf": float(row[8]) if row[8] else 112.0,
                    "def_rating_vs_c": float(row[9]) if row[9] else 112.0,
                }

        if self.verbose:
            # Count unique teams and seasons
            teams = set(k[0] for k in self.cache.team_stats.keys())
            seasons = set(k[1] for k in self.cache.team_stats.keys())
            print(
                f"   ✅ Loaded stats for {len(teams)} teams across {len(seasons)} seasons ({len(self.cache.team_stats)} total records)"
            )

    def load_all_team_betting(self):
        """Load all team betting performance data"""
        if self.verbose:
            print("\n📊 Loading team betting performance...")

        query = """
            SELECT team_abbrev, season, ats_pct, ou_pct
            FROM team_betting_performance
        """

        with self.connections["team"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                key = (row[0], row[1])
                self.cache.team_betting[key] = {
                    "ats_pct": float(row[2]) if row[2] else 0.5,
                    "ou_pct": float(row[3]) if row[3] else 0.5,
                }

        if self.verbose:
            print(f"   ✅ Loaded {len(self.cache.team_betting)} team-season betting records")

    def get_team_stats(self, team: str, game_date: date) -> Dict[str, float]:
        """Get team stats for the correct season based on game date.

        This prevents data leakage by using stats from the appropriate season,
        rather than always using the latest season's stats.
        """
        season = date_to_season(game_date)
        defaults = {
            "pace": 100.0,
            "off_rating": 112.0,
            "def_rating": 112.0,
            "def_rating_vs_pg": 112.0,
            "def_rating_vs_sg": 112.0,
            "def_rating_vs_sf": 112.0,
            "def_rating_vs_pf": 112.0,
            "def_rating_vs_c": 112.0,
        }
        return self.cache.team_stats.get((team, season), defaults)

    def load_all_h2h_stats(
        self, player_opponent_pairs: Set[Tuple[str, str]], stat_types: List[str]
    ):
        """Load all H2H matchup history"""
        if self.verbose:
            print(f"\n📊 Loading H2H matchup history for {len(player_opponent_pairs)} pairs...")

        query = """
            SELECT
                player_name, opponent_team, stat_type,
                games_played, days_since_last, sample_quality, recency_weight,
                avg_points, std_points, l3_points, l5_points, l10_points, l20_points,
                home_avg_points, away_avg_points,
                avg_rebounds, std_rebounds, l3_rebounds, l5_rebounds, l10_rebounds, l20_rebounds,
                home_avg_rebounds, away_avg_rebounds,
                avg_assists, std_assists, l3_assists, l5_assists, l10_assists, l20_assists,
                home_avg_assists, away_avg_assists,
                avg_threes, std_threes, l3_threes, l5_threes, l10_threes, l20_threes,
                home_avg_threes, away_avg_threes
            FROM matchup_history
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                player_name, opponent_team, stat_type = row[0], row[1], row[2]
                key = (player_name, opponent_team, stat_type)

                # Parse all stats
                self.cache.h2h_stats[key] = {
                    "h2h_games": row[3] or 0,
                    "h2h_days_since_last": row[4] or 999,
                    "h2h_sample_quality": float(row[5]) if row[5] else 0.2,
                    "h2h_recency_weight": float(row[6]) if row[6] else 0.5,
                    # Points
                    "h2h_avg_points": float(row[7]) if row[7] else 0.0,
                    "h2h_std_points": float(row[8]) if row[8] else 0.0,
                    "h2h_L3_points": float(row[9]) if row[9] else 0.0,
                    "h2h_L5_points": float(row[10]) if row[10] else 0.0,
                    "h2h_L10_points": float(row[11]) if row[11] else 0.0,
                    "h2h_L20_points": float(row[12]) if row[12] else 0.0,
                    "h2h_home_avg_points": float(row[13]) if row[13] else 0.0,
                    "h2h_away_avg_points": float(row[14]) if row[14] else 0.0,
                    # Rebounds
                    "h2h_avg_rebounds": float(row[15]) if row[15] else 0.0,
                    "h2h_std_rebounds": float(row[16]) if row[16] else 0.0,
                    "h2h_L3_rebounds": float(row[17]) if row[17] else 0.0,
                    "h2h_L5_rebounds": float(row[18]) if row[18] else 0.0,
                    "h2h_L10_rebounds": float(row[19]) if row[19] else 0.0,
                    "h2h_L20_rebounds": float(row[20]) if row[20] else 0.0,
                    "h2h_home_avg_rebounds": float(row[21]) if row[21] else 0.0,
                    "h2h_away_avg_rebounds": float(row[22]) if row[22] else 0.0,
                    # Assists
                    "h2h_avg_assists": float(row[23]) if row[23] else 0.0,
                    "h2h_std_assists": float(row[24]) if row[24] else 0.0,
                    "h2h_L3_assists": float(row[25]) if row[25] else 0.0,
                    "h2h_L5_assists": float(row[26]) if row[26] else 0.0,
                    "h2h_L10_assists": float(row[27]) if row[27] else 0.0,
                    "h2h_L20_assists": float(row[28]) if row[28] else 0.0,
                    "h2h_home_avg_assists": float(row[29]) if row[29] else 0.0,
                    "h2h_away_avg_assists": float(row[30]) if row[30] else 0.0,
                    # Threes
                    "h2h_avg_threes": float(row[31]) if row[31] else 0.0,
                    "h2h_std_threes": float(row[32]) if row[32] else 0.0,
                    "h2h_L3_threes": float(row[33]) if row[33] else 0.0,
                    "h2h_L5_threes": float(row[34]) if row[34] else 0.0,
                    "h2h_L10_threes": float(row[35]) if row[35] else 0.0,
                    "h2h_L20_threes": float(row[36]) if row[36] else 0.0,
                    "h2h_home_avg_threes": float(row[37]) if row[37] else 0.0,
                    "h2h_away_avg_threes": float(row[38]) if row[38] else 0.0,
                }

        if self.verbose:
            print(f"   ✅ Loaded {len(self.cache.h2h_stats)} H2H records")

    def load_all_prop_history(self):
        """Load all prop performance history"""
        if self.verbose:
            print("\n📊 Loading prop performance history...")

        query = """
            SELECT
                player_name, stat_type, line_center, season,
                hit_rate_l20, hit_rate_home, hit_rate_away,
                hit_rate_vs_top10_def, hit_rate_rested,
                line_vs_season_avg, line_percentile,
                days_since_last_hit, sample_quality_score,
                bayesian_prior_weight, consecutive_overs, props_l20
            FROM prop_performance_history
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                key = (row[0], row[1], float(row[2]), row[3])
                self.cache.prop_history[key] = {
                    "prop_hit_rate_L20": float(row[4]) if row[4] else 0.5,
                    "prop_hit_rate_context": float(row[5]) if row[5] else 0.5,  # home default
                    "prop_hit_rate_away": float(row[6]) if row[6] else 0.5,
                    "prop_hit_rate_defense": float(row[7]) if row[7] else 0.5,
                    "prop_hit_rate_rest": float(row[8]) if row[8] else 0.5,
                    "prop_line_vs_season_avg": float(row[9]) if row[9] else 0.0,
                    "prop_line_percentile": float(row[10]) if row[10] else 0.5,
                    "prop_days_since_last_hit": min(int(row[11]) if row[11] else 999, 999),
                    "prop_sample_quality": float(row[12]) if row[12] else 0.2,
                    "prop_bayesian_confidence": (1.0 - float(row[13])) if row[13] else 0.2,
                    "prop_consecutive_overs": int(row[14]) if row[14] else 0,
                    "prop_sample_size_L20": int(row[15]) if row[15] else 0,
                }

        if self.verbose:
            print(f"   ✅ Loaded {len(self.cache.prop_history)} prop history records")

    def load_all_vegas_context(self, game_dates: Set[date]):
        """Load all vegas context from games table"""
        if self.verbose:
            print(f"\n📊 Loading vegas context for {len(game_dates)} game dates...")

        query = """
            SELECT game_date, home_team, away_team, vegas_total, vegas_spread
            FROM games
            WHERE game_date = ANY(%s::date[])
        """

        with self.connections["games"].cursor() as cur:
            cur.execute(query, (list(game_dates),))
            for row in cur.fetchall():
                game_date, home_team, away_team = row[0], row[1], row[2]
                vegas_total = float(row[3]) if row[3] else 220.0
                vegas_spread = float(row[4]) if row[4] else 0.0

                # Index by both home and away team
                self.cache.vegas_context[(game_date, home_team)] = {
                    "vegas_total": vegas_total,
                    "vegas_spread": -vegas_spread,  # Home team perspective (negative = favored)
                    "is_home": True,
                    "opponent": away_team,
                }
                self.cache.vegas_context[(game_date, away_team)] = {
                    "vegas_total": vegas_total,
                    "vegas_spread": vegas_spread,  # Away team perspective
                    "is_home": False,
                    "opponent": home_team,
                }

        if self.verbose:
            print(
                f"   ✅ Loaded vegas context for {len(self.cache.vegas_context)} team-game combinations"
            )

    def load_all_cheatsheet(self):
        """Load all BettingPros cheatsheet data"""
        if self.verbose:
            print("\n📊 Loading BettingPros cheatsheet data...")

        query = """
            SELECT DISTINCT ON (player_name, game_date, stat_type)
                player_name, game_date, stat_type,
                projection_diff, bet_rating, ev_pct, probability,
                opp_rank, hit_rate_l5, hit_rate_l15, hit_rate_season
            FROM cheatsheet_data
            ORDER BY player_name, game_date, stat_type, fetch_timestamp DESC
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                key = (row[0], row[1], row[2])
                self.cache.cheatsheet[key] = {
                    "bp_projection_diff": float(row[3]) if row[3] else 0.0,
                    "bp_bet_rating": float(row[4]) if row[4] else 3.0,
                    "bp_ev_pct": float(row[5]) if row[5] else 0.0,
                    "bp_probability": float(row[6]) if row[6] else 0.5,
                    "bp_opp_rank": float(row[7]) if row[7] else 15.0,
                    "bp_hit_rate_l5": float(row[8]) if row[8] else 0.5,
                    "bp_hit_rate_l15": float(row[9]) if row[9] else 0.5,
                    "bp_hit_rate_season": float(row[10]) if row[10] else 0.5,
                }

        if self.verbose:
            print(f"   ✅ Loaded {len(self.cache.cheatsheet)} cheatsheet records")

    def load_all_book_lines(self, stat_type: str):
        """Load all book lines for a stat type"""
        if self.verbose:
            print(f"\n📊 Loading book lines for {stat_type}...")

        query = """
            SELECT player_name, game_date, book_name, over_line
            FROM nba_props_xl
            WHERE stat_type = %s
              AND over_line IS NOT NULL
              AND is_active = true
              AND game_date >= %s AND game_date <= %s
            ORDER BY player_name, game_date, fetch_timestamp DESC
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query, (stat_type, TRAIN_START, TRAIN_END))
            for row in cur.fetchall():
                player_name, game_date, book_name, line = row
                key = (player_name, game_date, stat_type)

                if key not in self.cache.book_lines:
                    self.cache.book_lines[key] = []

                # Only add if we don't already have this book
                existing_books = {bl["book_name"] for bl in self.cache.book_lines[key]}
                if book_name not in existing_books:
                    self.cache.book_lines[key].append(
                        {
                            "book_name": book_name,
                            "line": float(line),
                        }
                    )

        # Also load historical spreads for percentile calculation
        spread_query = """
            SELECT player_name, line_spread
            FROM nba_props_xl
            WHERE stat_type = %s
              AND line_spread IS NOT NULL
              AND game_date >= %s AND game_date <= %s
            ORDER BY player_name, game_date DESC
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(spread_query, (stat_type, TRAIN_START, TRAIN_END))
            for row in cur.fetchall():
                player_name, spread = row
                key = (player_name, stat_type)
                if key not in self.cache.historical_spreads:
                    self.cache.historical_spreads[key] = []
                if len(self.cache.historical_spreads[key]) < 50:  # Keep last 50
                    self.cache.historical_spreads[key].append(float(spread))

        if self.verbose:
            print(f"   ✅ Loaded book lines for {len(self.cache.book_lines)} props")

    def load_all_book_accuracy(self):
        """Load book historical accuracy metrics from book_historical_accuracy table"""
        if self.verbose:
            print("\n📊 Loading book historical accuracy...")

        query = """
            SELECT book_name, market, hit_rate, soft_line_rate, line_bias, sharpe_ratio
            FROM book_historical_accuracy
            WHERE metric_window = 'season' AND player_id IS NULL
            ORDER BY as_of_date DESC
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query)
            seen = set()  # Only keep most recent per book/market
            for row in cur.fetchall():
                book_name, market = row[0], row[1]
                key = (book_name.lower(), market)
                if key not in seen:
                    self.cache.book_accuracy[key] = {
                        "hit_rate": float(row[2]) if row[2] else 0.5,
                        "soft_line_rate": float(row[3]) if row[3] else 0.1,
                        "line_bias": float(row[4]) if row[4] else 0.0,
                        "sharpe_ratio": float(row[5]) if row[5] else 0.0,
                    }
                    seen.add(key)

        if self.verbose:
            print(f"   ✅ Loaded accuracy for {len(self.cache.book_accuracy)} book-market combos")

    def load_all_line_movement(self, stat_type: str):
        """Load line movement data for market confidence features.

        Calculates:
        - line_delta: Change from first to last observed line (open → close)
        - line_std: Standard deviation of line across all snapshots
        - line_volatility: Max line - Min line across snapshots
        - snapshot_count: Number of line observations
        - hours_tracked: Time span of observations
        """
        if self.verbose:
            print(f"\n📊 Loading line movement for {stat_type}...")

        query = """
            SELECT player_name, game_date, book_name, over_line, fetch_timestamp,
                   game_time
            FROM nba_props_xl
            WHERE stat_type = %s
              AND over_line IS NOT NULL
              AND game_date >= '2023-10-01'
            ORDER BY player_name, game_date, book_name, fetch_timestamp
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query, (stat_type,))

            # Group by (player, date, stat) - aggregate across all books
            prop_snapshots = defaultdict(list)

            for row in cur.fetchall():
                player_name, game_date, book_name, line, fetch_ts, game_time = row
                key = (player_name, game_date, stat_type)
                prop_snapshots[key].append(
                    {
                        "line": float(line),
                        "timestamp": fetch_ts,
                        "book": book_name,
                        "game_time": game_time,
                    }
                )

            # Calculate movement features per prop
            for key, snapshots in prop_snapshots.items():
                if len(snapshots) < 2:
                    continue

                lines = [s["line"] for s in snapshots]
                timestamps = [s["timestamp"] for s in snapshots if s["timestamp"]]

                # Sort by timestamp to get open/close
                sorted_snaps = sorted(
                    [s for s in snapshots if s["timestamp"]], key=lambda x: x["timestamp"]
                )

                if len(sorted_snaps) >= 2:
                    open_line = sorted_snaps[0]["line"]
                    close_line = sorted_snaps[-1]["line"]
                    line_delta = close_line - open_line

                    # Hours tracked
                    first_ts = sorted_snaps[0]["timestamp"]
                    last_ts = sorted_snaps[-1]["timestamp"]
                    hours_tracked = (
                        (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else 0
                    )
                else:
                    line_delta = 0.0
                    hours_tracked = 0.0

                # Calculate stats across ALL lines (all books, all times)
                line_std = float(np.std(lines)) if len(lines) > 1 else 0.0
                line_volatility = max(lines) - min(lines)

                # Consensus movement (weighted by recency)
                unique_books = len(set(s["book"] for s in snapshots))

                self.cache.line_movement[key] = {
                    "line_delta": line_delta,
                    "line_std": line_std,
                    "line_volatility": line_volatility,
                    "snapshot_count": len(snapshots),
                    "unique_books": unique_books,
                    "hours_tracked": hours_tracked,
                    # Derived confidence metrics
                    "market_stability": 1.0 / (1.0 + line_std),  # Higher = more stable
                    "consensus_strength": unique_books / 8.0,  # 8 books = full consensus
                }

        if self.verbose:
            print(f"   ✅ Loaded line movement for {len(self.cache.line_movement)} props")

    # =========================================================================
    # FEATURE EXTRACTION (FROM CACHE - NO DB QUERIES)
    # =========================================================================

    def extract_rolling_features(
        self, player_name: str, game_date: date, stat_type: str
    ) -> Optional[Dict]:
        """Extract rolling stats from cached game logs - only for target stat"""
        games = self.cache.player_games.get(player_name, [])

        # Filter to games before this date
        prior_games = [g for g in games if g["game_date"] < game_date]

        if len(prior_games) < 20:
            return None  # Insufficient history

        features = {}

        # Map stat_type to game log column
        stat_map = {
            "POINTS": "points",
            "REBOUNDS": "rebounds",
            "ASSISTS": "assists",
            "THREES": "threes",
        }
        target_stat = stat_map.get(stat_type, "points")

        # Rolling windows
        windows = [3, 5, 10, 20]
        # Only target stat + minutes (playing time context)
        stats_to_extract = [target_stat, "minutes"]

        for window in windows:
            window_games = prior_games[:window]

            for stat in stats_to_extract:
                values = [g[stat] for g in window_games]
                ema_value = self.calculate_ema(values)
                features[f"ema_{stat}_L{window}"] = ema_value

            # FG% for this window
            fgm = sum(g["fgm"] for g in window_games)
            fga = sum(g["fga"] for g in window_games)
            features[f"fg_pct_L{window}"] = fgm / fga if fga > 0 else 0.0

            # Plus/minus (L5, L10 only)
            if window in [5, 10]:
                pm_values = [g["plus_minus"] for g in window_games]
                features[f"ema_plus_minus_L{window}"] = self.calculate_ema(pm_values)

        # FT rate and true shooting (L10)
        l10_games = prior_games[:10]
        fta = sum(g["fta"] for g in l10_games)
        fga = sum(g["fga"] for g in l10_games)
        features["ft_rate_L10"] = fta / fga if fga > 0 else 0.0

        pts = sum(g["points"] for g in l10_games)
        features["true_shooting_L10"] = pts / (2 * (fga + 0.44 * fta)) if (fga + fta) > 0 else 0.0

        # Target stat per minute (L5)
        l5_games = prior_games[:5]
        mins = sum(g["minutes"] for g in l5_games)
        stat_l5 = sum(g[target_stat] for g in l5_games)
        features[f"{target_stat}_per_minute_L5"] = stat_l5 / mins if mins > 0 else 0.0

        # Context features
        if prior_games:
            last_game = prior_games[0]
            features["player_last_game_minutes"] = last_game["minutes"]

            # Days rest
            days_rest = (game_date - last_game["game_date"]).days
            features["days_rest"] = min(days_rest, 7)
            features["is_back_to_back"] = 1.0 if days_rest == 1 else 0.0
        else:
            features["player_last_game_minutes"] = 30.0
            features["days_rest"] = 2
            features["is_back_to_back"] = 0.0

        # =========================================================================
        # USAGE VOLATILITY FEATURES
        # =========================================================================
        # High volatility = unstable role, lower prediction confidence
        # Helps penalize predictions when player's usage is drifting

        # Scoring volatility (L5 and L10)
        l5_target_values = [g[target_stat] for g in prior_games[:5]]
        l10_target_values = [g[target_stat] for g in prior_games[:10]]

        if len(l5_target_values) >= 5:
            features[f"{target_stat}_std_L5"] = float(np.std(l5_target_values))
        else:
            features[f"{target_stat}_std_L5"] = 0.0

        if len(l10_target_values) >= 10:
            features[f"{target_stat}_std_L10"] = float(np.std(l10_target_values))
        else:
            features[f"{target_stat}_std_L10"] = 0.0

        # Minutes volatility (role stability indicator)
        l5_minutes = [g["minutes"] for g in prior_games[:5]]
        l10_minutes = [g["minutes"] for g in prior_games[:10]]

        if len(l5_minutes) >= 5:
            features["minutes_std_L5"] = float(np.std(l5_minutes))
        else:
            features["minutes_std_L5"] = 0.0

        if len(l10_minutes) >= 10:
            features["minutes_std_L10"] = float(np.std(l10_minutes))
        else:
            features["minutes_std_L10"] = 0.0

        # FGA volatility (usage drift indicator)
        l5_fga = [g["fga"] for g in prior_games[:5]]
        if len(l5_fga) >= 5:
            features["fga_std_L5"] = float(np.std(l5_fga))
        else:
            features["fga_std_L5"] = 0.0

        # Minutes trend: L5 vs L20 (is playing time increasing or decreasing?)
        ema_minutes_l5 = features.get("ema_minutes_L5", 30.0)
        ema_minutes_l20 = features.get("ema_minutes_L20", 30.0)
        features["minutes_trend_ratio"] = (
            ema_minutes_l5 / ema_minutes_l20 if ema_minutes_l20 > 0 else 1.0
        )

        # Target stat trend: L5 vs L20 (is scoring increasing or decreasing?)
        ema_stat_l5 = features.get(f"ema_{target_stat}_L5", 15.0)
        ema_stat_l20 = features.get(f"ema_{target_stat}_L20", 15.0)
        features[f"{target_stat}_trend_ratio"] = (
            ema_stat_l5 / ema_stat_l20 if ema_stat_l20 > 0 else 1.0
        )

        # Combined volatility score (normalized)
        # Higher = more volatile = less reliable prediction
        stat_std_l5 = features.get(f"{target_stat}_std_L5", 0.0)
        minutes_std_l5 = features.get("minutes_std_L5", 0.0)
        avg_stat = ema_stat_l5 if ema_stat_l5 > 0 else 15.0
        avg_mins = ema_minutes_l5 if ema_minutes_l5 > 0 else 30.0

        # Coefficient of variation (std / mean) for both
        stat_cv = stat_std_l5 / avg_stat if avg_stat > 0 else 0.0
        mins_cv = minutes_std_l5 / avg_mins if avg_mins > 0 else 0.0
        features["usage_volatility_score"] = (stat_cv + mins_cv) / 2.0

        return features

    def extract_team_features(
        self, team_abbrev: str, opponent_abbrev: str, game_date: date
    ) -> Dict:
        """Extract team features from cache using season-appropriate stats.

        Args:
            team_abbrev: Player's team abbreviation
            opponent_abbrev: Opponent team abbreviation
            game_date: Date of the game (used to determine correct season)

        Returns:
            Dict of team features using stats from the correct season
        """
        team = self.normalize_team(team_abbrev)
        opp = self.normalize_team(opponent_abbrev)

        # Use season-appropriate stats to prevent data leakage
        team_stats = self.get_team_stats(team, game_date)
        opp_stats = self.get_team_stats(opp, game_date)

        pace_diff = team_stats["pace"] - opp_stats["pace"]
        projected_poss = (team_stats["pace"] + opp_stats["pace"]) / 2 * 48 / 60

        return {
            "team_pace": team_stats["pace"],
            "team_off_rating": team_stats["off_rating"],
            "team_def_rating": team_stats["def_rating"],
            "opponent_pace": opp_stats["pace"],
            "opponent_def_rating": opp_stats["def_rating"],
            "pace_diff": pace_diff,
            "projected_possessions": projected_poss,
            "expected_possessions": projected_poss,
            "projected_team_possessions": projected_poss,
        }

    def extract_team_features_with_position(
        self, team_abbrev: str, opponent_abbrev: str, game_date: date, player_name: str
    ) -> Dict:
        """Extract team features including position-specific defense from cache.

        Args:
            team_abbrev: Player's team abbreviation
            opponent_abbrev: Opponent team abbreviation
            game_date: Date of the game (used to determine correct season)
            player_name: Player name for position-specific defense features

        Returns:
            Dict of team features using stats from the correct season, plus positional defense
        """
        # Get base team features
        features = self.extract_team_features(team_abbrev, opponent_abbrev, game_date)

        # Add position-specific defense features
        opp = self.normalize_team(opponent_abbrev)
        opp_stats = self.get_team_stats(opp, game_date)

        # Get player position (default to SF=3 if unknown)
        player_pos = self.cache.player_positions.get(player_name, 3.0)
        pos_map = {1: "pg", 2: "sg", 3: "sf", 4: "pf", 5: "c"}

        # Round to nearest integer position (handles hybrid positions like G=1.5 -> SG=2)
        pos_int = int(round(player_pos))
        pos_key = f"def_rating_vs_{pos_map.get(pos_int, 'sf')}"

        # Opponent's defense rating against this player's position
        opp_positional_def = opp_stats.get(pos_key, 112.0)
        features["opp_positional_def"] = opp_positional_def

        # Matchup advantage: how much better/worse is opponent at defending this position
        # Positive = opponent is worse at defending this position (good for player)
        league_avg_def = 112.0
        features["position_matchup_advantage"] = opp_positional_def - league_avg_def

        return features

    def extract_book_features(self, player_name: str, game_date: date, stat_type: str) -> Dict:
        """Extract book features from cache"""
        key = (player_name, game_date, stat_type)
        book_data = self.cache.book_lines.get(key, [])

        if not book_data:
            return self._default_book_features()

        lines = [b["line"] for b in book_data]
        books = [b["book_name"] for b in book_data]

        line_spread = max(lines) - min(lines)
        consensus = np.mean(lines)
        line_std = np.std(lines) if len(lines) > 1 else 0.0

        # Book deviations
        book_map = {b["book_name"]: b["line"] for b in book_data}
        deviations = {}
        for book in [
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "bet365",
            "betrivers",
            "espnbet",
            "fanatics",
        ]:
            if book in book_map:
                deviations[f"{book}_deviation"] = book_map[book] - consensus
            else:
                deviations[f"{book}_deviation"] = 0.0

        # Line spread percentile
        hist_key = (player_name, stat_type)
        hist_spreads = self.cache.historical_spreads.get(hist_key, [])
        if len(hist_spreads) >= 5:
            percentile = sum(1 for s in hist_spreads if s < line_spread) / len(hist_spreads)
        else:
            percentile = 0.5

        # Softest/hardest book encoding
        book_ids = {
            "draftkings": 1,
            "fanduel": 2,
            "betmgm": 3,
            "caesars": 4,
            "bet365": 5,
            "betrivers": 6,
            "espnbet": 7,
            "fanatics": 8,
            "prizepicks": 9,
            "underdog": 10,
        }

        softest_book = min(book_data, key=lambda x: x["line"])["book_name"]
        hardest_book = max(book_data, key=lambda x: x["line"])["book_name"]
        softest_book_id = book_ids.get(softest_book.lower(), 0)

        # Get accuracy metrics for softest book from cache
        softest_book_name = softest_book.lower()
        book_acc = self.cache.book_accuracy.get((softest_book_name, stat_type), {})

        return {
            "line_spread": line_spread,
            "consensus_line": consensus,
            "line_std_dev": line_std,
            "num_books_offering": float(len(book_data)),
            "line_coef_variation": line_std / consensus if consensus > 0 else 0.0,
            **deviations,
            "softest_book_id": float(softest_book_id),
            "hardest_book_id": float(book_ids.get(hardest_book.lower(), 0)),
            "line_spread_percentile": percentile,
            "books_agree": 1.0 if line_spread < 0.5 else 0.0,
            "books_disagree": 1.0 if line_spread >= 2.0 else 0.0,
            "softest_vs_consensus": min(lines) - consensus,
            "hardest_vs_consensus": max(lines) - consensus,
            "min_line": min(lines),
            "max_line": max(lines),
            "line_std": line_std,
            # Book historical accuracy features
            "softest_book_hit_rate": book_acc.get("hit_rate", 0.5),
            "softest_book_soft_rate": book_acc.get("soft_line_rate", 0.1),
            "softest_book_line_bias": book_acc.get("line_bias", 0.0),
            "line_source_reliability": book_acc.get("sharpe_ratio", 0.0),
        }

    def _default_book_features(self) -> Dict:
        """Default book features when no data available"""
        features = {
            "line_spread": 0.0,
            "consensus_line": 0.0,
            "line_std_dev": 0.0,
            "num_books_offering": 1.0,
            "line_coef_variation": 0.0,
            "softest_book_id": 0.0,
            "hardest_book_id": 0.0,
            "line_spread_percentile": 0.5,
            "books_agree": 1.0,
            "books_disagree": 0.0,
            "softest_vs_consensus": 0.0,
            "hardest_vs_consensus": 0.0,
            "min_line": 0.0,
            "max_line": 0.0,
            "line_std": 0.0,
            # Book historical accuracy features (defaults)
            "softest_book_hit_rate": 0.5,
            "softest_book_soft_rate": 0.1,
            "softest_book_line_bias": 0.0,
            "line_source_reliability": 0.0,
        }
        for book in [
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "bet365",
            "betrivers",
            "espnbet",
            "fanatics",
        ]:
            features[f"{book}_deviation"] = 0.0
        return features

    def extract_line_movement_features(
        self, player_name: str, game_date: date, stat_type: str
    ) -> Dict:
        """Extract market confidence features from line movement data.

        These features help identify when the market is uncertain (early season)
        vs confident (late season/playoffs).
        """
        key = (player_name, game_date, stat_type)
        movement = self.cache.line_movement.get(key)

        if movement:
            return {
                # Raw movement metrics
                "line_delta": movement.get("line_delta", 0.0),
                "line_movement_std": movement.get("line_std", 0.0),
                "line_volatility": movement.get("line_volatility", 0.0),
                # Confidence metrics
                "market_stability": movement.get("market_stability", 0.5),
                "consensus_strength": movement.get("consensus_strength", 0.5),
                "snapshot_count": movement.get("snapshot_count", 1),
                # Time-based
                "hours_tracked": movement.get("hours_tracked", 0.0),
            }

        # Defaults for missing data (conservative/uncertain)
        return {
            "line_delta": 0.0,
            "line_movement_std": 0.0,
            "line_volatility": 0.0,
            "market_stability": 0.5,  # Neutral
            "consensus_strength": 0.5,  # Neutral
            "snapshot_count": 1,
            "hours_tracked": 0.0,
        }

    def extract_season_phase_features(self, game_date: date) -> Dict:
        """Extract calendar-aware season phase features.

        NBA Season phases:
        - Early (Oct-Nov): High uncertainty, player roles settling
        - Mid (Dec-Feb): Stable rotations, reliable patterns
        - Late (Mar-Apr): Playoff push, minutes management
        - Playoffs (Apr-Jun): Peak intensity, most predictable
        """
        month = game_date.month

        # Season phase encoding
        if month in [10, 11]:
            phase = 1  # Early season
            phase_name = "early"
        elif month in [12, 1, 2]:
            phase = 2  # Mid season
            phase_name = "mid"
        elif month == 3:
            phase = 3  # Late season
            phase_name = "late"
        else:  # April onwards (playoffs or off-season data)
            phase = 4  # Playoffs
            phase_name = "playoffs"

        # Days into season (Oct 1 = day 0)
        season_start = date(game_date.year if game_date.month >= 10 else game_date.year - 1, 10, 1)
        days_into_season = (game_date - season_start).days

        return {
            "season_phase_encoded": phase,
            "is_early_season": 1.0 if phase == 1 else 0.0,
            "is_mid_season": 1.0 if phase == 2 else 0.0,
            "is_late_season": 1.0 if phase == 3 else 0.0,
            "is_playoffs": 1.0 if phase == 4 else 0.0,
            "days_into_season": min(days_into_season, 250),  # Cap at ~8 months
        }

    def extract_h2h_features(self, player_name: str, opponent: str, stat_type: str) -> Dict:
        """Extract H2H features from cache - only for target stat type"""
        key = (player_name, opponent, stat_type)
        h2h = self.cache.h2h_stats.get(key)

        # Map stat_type to the stat name used in H2H columns
        stat_map = {
            "POINTS": "points",
            "REBOUNDS": "rebounds",
            "ASSISTS": "assists",
            "THREES": "threes",
        }
        target_stat = stat_map.get(stat_type, "points")

        if h2h:
            # Get raw values
            h2h_games = h2h.get("h2h_games", 0)
            days_since = h2h.get("h2h_days_since_last", 999)
            l3 = h2h.get(f"h2h_L3_{target_stat}", 0.0)
            l5 = h2h.get(f"h2h_L5_{target_stat}", 0.0)
            l10 = h2h.get(f"h2h_L10_{target_stat}", 0.0)
            l20 = h2h.get(f"h2h_L20_{target_stat}", 0.0)
            lifetime_avg = h2h.get(f"h2h_avg_{target_stat}", 0.0)

            # =========================================================
            # TIME-DECAYED H2H FEATURES
            # =========================================================
            # τ_h2h = 45 days (older matchups less relevant)
            tau_h2h = 45.0

            # Time decay factor based on days since last matchup
            # exp(-days / tau) → closer to 1 for recent, closer to 0 for old
            time_decay = np.exp(-days_since / tau_h2h) if days_since < 999 else 0.1

            # Decayed average: weight recent rolling windows more heavily
            # L3 gets highest weight, L10 gets moderate, lifetime gets low
            if h2h_games >= 3:
                # Weighted combination favoring recent matchups
                w3, w5, w10, w_life = 0.4, 0.3, 0.2, 0.1
                h2h_decayed_avg = (
                    w3 * (l3 if l3 > 0 else l5)
                    + w5 * (l5 if l5 > 0 else l10)
                    + w10 * (l10 if l10 > 0 else lifetime_avg)
                    + w_life * lifetime_avg
                )
                # Further weight by time decay
                h2h_decayed_avg = h2h_decayed_avg * time_decay + lifetime_avg * (1 - time_decay)
            else:
                h2h_decayed_avg = lifetime_avg * time_decay

            # H2H trend: L3 vs L10 (is player improving vs this opponent?)
            # Positive = getting better, Negative = getting worse
            if l3 > 0 and l10 > 0:
                h2h_trend = l3 - l10
            else:
                h2h_trend = 0.0

            # H2H recency adjusted: lifetime avg discounted by staleness
            h2h_recency_adjusted = lifetime_avg * time_decay

            # H2H reliability: low if old matchups or few games
            h2h_reliability = min(h2h_games / 5.0, 1.0) * time_decay

            # Filter to only include target stat's H2H features
            result = {
                "h2h_games": h2h_games,
                "h2h_days_since_last": days_since,
                "h2h_sample_quality": h2h.get("h2h_sample_quality", 0.2),
                "h2h_recency_weight": h2h.get("h2h_recency_weight", 0.5),
                f"h2h_avg_{target_stat}": lifetime_avg,
                f"h2h_std_{target_stat}": h2h.get(f"h2h_std_{target_stat}", 0.0),
                f"h2h_L3_{target_stat}": l3,
                f"h2h_L5_{target_stat}": l5,
                f"h2h_L10_{target_stat}": l10,
                f"h2h_L20_{target_stat}": l20,
                f"h2h_home_avg_{target_stat}": h2h.get(f"h2h_home_avg_{target_stat}", 0.0),
                f"h2h_away_avg_{target_stat}": h2h.get(f"h2h_away_avg_{target_stat}", 0.0),
                # NEW: Time-decayed H2H features
                f"h2h_decayed_avg_{target_stat}": h2h_decayed_avg,
                f"h2h_trend_{target_stat}": h2h_trend,
                f"h2h_recency_adjusted_{target_stat}": h2h_recency_adjusted,
                "h2h_time_decay_factor": time_decay,
                "h2h_reliability": h2h_reliability,
            }
            return result

        # Default H2H features - only for target stat
        return {
            "h2h_games": 0,
            "h2h_days_since_last": 999,
            "h2h_sample_quality": 0.2,
            "h2h_recency_weight": 0.5,
            f"h2h_avg_{target_stat}": 0.0,
            f"h2h_std_{target_stat}": 0.0,
            f"h2h_L3_{target_stat}": 0.0,
            f"h2h_L5_{target_stat}": 0.0,
            f"h2h_L10_{target_stat}": 0.0,
            f"h2h_L20_{target_stat}": 0.0,
            f"h2h_home_avg_{target_stat}": 0.0,
            f"h2h_away_avg_{target_stat}": 0.0,
            # NEW: Time-decayed H2H features (defaults)
            f"h2h_decayed_avg_{target_stat}": 0.0,
            f"h2h_trend_{target_stat}": 0.0,
            f"h2h_recency_adjusted_{target_stat}": 0.0,
            "h2h_time_decay_factor": 0.1,  # Old/no data
            "h2h_reliability": 0.0,
        }

    def extract_prop_features(
        self, player_name: str, stat_type: str, line: float, game_date: date, is_home: bool
    ) -> Dict:
        """Extract prop history features from cache"""
        if line is None or line == 0:
            return {
                "prop_hit_rate_L20": 0.5,
                "prop_hit_rate_context": 0.5,
                "prop_hit_rate_defense": 0.5,
                "prop_hit_rate_rest": 0.5,
                "prop_hit_rate_deviation": 0.0,
                "prop_line_vs_season_avg": 0.0,
                "prop_line_percentile": 0.5,
                "prop_days_since_last_hit": 999,
                "prop_sample_quality": 0.2,
                "prop_bayesian_confidence": 0.2,
                "prop_consecutive_overs": 0,
                "prop_sample_size_L20": 0,
            }
        line_center = round(line * 2) / 2.0
        season = date_to_season(game_date)
        key = (player_name, stat_type, line_center, season)

        prop = self.cache.prop_history.get(key)
        if prop:
            result = prop.copy()
            # Adjust context hit rate based on home/away
            away_rate = prop.get("prop_hit_rate_away", prop.get("prop_hit_rate_L20", 0.5))
            if not is_home:
                result["prop_hit_rate_context"] = away_rate
            result["prop_hit_rate_deviation"] = result["prop_hit_rate_L20"] - 0.5
            # Remove the internal away rate key (not needed in output)
            result.pop("prop_hit_rate_away", None)
            return result

        return {
            "prop_hit_rate_L20": 0.5,
            "prop_hit_rate_context": 0.5,
            "prop_hit_rate_defense": 0.5,
            "prop_hit_rate_rest": 0.5,
            "prop_hit_rate_deviation": 0.0,
            "prop_line_vs_season_avg": 0.0,
            "prop_line_percentile": 0.5,
            "prop_days_since_last_hit": 999,
            "prop_sample_quality": 0.2,
            "prop_bayesian_confidence": 0.2,
            "prop_consecutive_overs": 0,
            "prop_sample_size_L20": 0,
        }

    def extract_vegas_features(self, player_name: str, game_date: date, is_home: bool) -> Dict:
        """Extract vegas context from cache"""
        team = self.cache.player_teams.get(player_name)
        if not team:
            return {"vegas_total": 220.0, "vegas_spread": 0.0}

        key = (game_date, team)
        vegas = self.cache.vegas_context.get(key)

        if vegas:
            return {
                "vegas_total": vegas["vegas_total"],
                "vegas_spread": vegas["vegas_spread"],
            }

        return {"vegas_total": 220.0, "vegas_spread": 0.0}

    def extract_team_betting_features(
        self, player_name: str, opponent: str, game_date: date, is_home: bool
    ) -> Dict:
        """Extract team betting performance from cache"""
        team = self.cache.player_teams.get(player_name)
        season = date_to_season(game_date)

        team_key = (team, season) if team else None
        opp_key = (opponent, season) if opponent else None

        team_betting = self.cache.team_betting.get(team_key, {}) if team_key else {}
        opp_betting = self.cache.team_betting.get(opp_key, {}) if opp_key else {}

        return {
            "team_ats_pct": team_betting.get("ats_pct", 0.5),
            "opp_ats_pct": opp_betting.get("ats_pct", 0.5),
            "team_ou_pct": team_betting.get("ou_pct", 0.5),
            "opp_ou_pct": opp_betting.get("ou_pct", 0.5),
            "team_betting_available": 1.0 if team_betting else 0.0,
        }

    def extract_cheatsheet_features(
        self, player_name: str, game_date: date, stat_type: str
    ) -> Dict:
        """Extract cheatsheet features from cache"""
        key = (player_name, game_date, stat_type)
        cs = self.cache.cheatsheet.get(key)

        if cs:
            return cs

        return {
            "bp_projection_diff": 0.0,
            "bp_bet_rating": 3.0,
            "bp_ev_pct": 0.0,
            "bp_probability": 0.5,
            "bp_opp_rank": 15.0,
            "bp_hit_rate_l5": 0.5,
            "bp_hit_rate_l15": 0.5,
            "bp_hit_rate_season": 0.5,
        }

    def extract_all_features(self, prop: Dict) -> Optional[Dict]:
        """Extract all features for a prop using cached data only"""
        player_name = prop["player_name"]
        game_date = prop["game_date"]
        stat_type = prop["stat_type"]
        opponent = prop.get("opponent_team", "")
        is_home = prop.get("is_home", True)

        # Handle None or Decimal line values
        line = prop.get("consensus_line")
        if line is None:
            line = prop.get("min_over_line", 0.0)
        if line is None:
            line = 0.0
        line = float(line)

        # Convert game_date to date if needed
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, "%Y-%m-%d").date()
        elif hasattr(game_date, "date") and callable(game_date.date):
            game_date = game_date.date()

        # 1. Rolling stats (requires 20+ games)
        rolling = self.extract_rolling_features(player_name, game_date, stat_type)
        if rolling is None:
            return None

        # Get player's team from cache
        player_team = self.cache.player_teams.get(player_name, "")

        # 2. Team features (with position-specific defense)
        team_features = self.extract_team_features_with_position(
            player_team, opponent, game_date, player_name
        )

        # 3. Book features
        book_features = self.extract_book_features(player_name, game_date, stat_type)

        # 4. H2H features
        h2h_features = self.extract_h2h_features(player_name, opponent, stat_type)

        # 5. Prop history features
        prop_features = self.extract_prop_features(player_name, stat_type, line, game_date, is_home)

        # 6. Vegas context
        vegas_features = self.extract_vegas_features(player_name, game_date, is_home)

        # 7. Team betting
        betting_features = self.extract_team_betting_features(
            player_name, opponent, game_date, is_home
        )

        # 8. Cheatsheet
        cheatsheet_features = self.extract_cheatsheet_features(player_name, game_date, stat_type)

        # 9. Line movement / market confidence
        line_movement_features = self.extract_line_movement_features(
            player_name, game_date, stat_type
        )

        # 10. Season phase (calendar-aware)
        season_phase_features = self.extract_season_phase_features(game_date)

        # Combine all features
        features = {
            "is_home": 1.0 if is_home else 0.0,
            "line": line,
        }
        features.update(rolling)
        features.update(team_features)
        features.update(book_features)
        features.update(h2h_features)
        features.update(prop_features)
        features.update(vegas_features)
        features.update(betting_features)
        features.update(cheatsheet_features)
        features.update(line_movement_features)
        features.update(season_phase_features)

        # Compute derived features from existing cached data
        # (These were hardcoded before - now computed properly)

        # 1. efficiency_vs_context: (player TS% L10) / 0.55 * (opponent_def_rating / 112.0) * 100
        player_ts = features.get("true_shooting_L10", 0.55)
        opp_def = features.get("opponent_def_rating", 112.0)
        features["efficiency_vs_context"] = (player_ts / 0.55) * (opp_def / 112.0) * 100.0

        # 2. game_velocity: (team_pace + opponent_pace) / 2 / 100 * 100
        features["game_velocity"] = (
            features.get("team_pace", 100.0) + features.get("opponent_pace", 100.0)
        ) / 2.0

        # 3. season_phase: 0.0 = start of season, 0.5 = mid, 1.0 = end
        if isinstance(game_date, date):
            month = game_date.month
            # NBA season: Oct-Apr (regular), May-Jun (playoffs)
            if month >= 10:  # Oct-Dec
                features["season_phase"] = (month - 10) / 9.0  # 0.0 to 0.33
            elif month <= 6:  # Jan-Jun
                features["season_phase"] = (month + 2) / 9.0  # 0.33 to 0.89
            else:  # Jul-Sep (offseason)
                features["season_phase"] = 0.0
        else:
            features["season_phase"] = 0.5

        # Map stat_type to column name for computed features
        stat_col_map = {
            "POINTS": "points",
            "REBOUNDS": "rebounds",
            "ASSISTS": "assists",
            "THREES": "threes",
        }
        target_stat_col = stat_col_map.get(stat_type, "points")

        # 4. resistance_adjusted_L3: player L3 target stat adjusted for opponent defense
        raw_L3 = features.get(f"ema_{target_stat_col}_L3", 10.0)
        features["resistance_adjusted_L3"] = raw_L3 * (opp_def / 112.0)

        # 5. volume_proxy: estimate from stat/minutes * pace
        spm = features.get(f"{target_stat_col}_per_minute_L5", 0.3)
        mins = features.get("ema_minutes_L5", 30.0)
        pace_factor = features.get("team_pace", 100.0) / 100.0
        features["volume_proxy"] = spm * mins * pace_factor

        # 6. momentum_short_term: L3 vs L10 performance for target stat
        l3_stat = features.get(f"ema_{target_stat_col}_L3", 10.0)
        l10_stat = features.get(f"ema_{target_stat_col}_L10", 10.0)
        features["momentum_short_term"] = (l3_stat - l10_stat) / max(l10_stat, 1.0)

        # 7. starter_ratio: from minutes in recent games (not points-specific)
        games = self.cache.player_games.get(player_name, [])
        prior_games = [g for g in games if g["game_date"] < game_date][:10]
        if prior_games:
            starter_games = sum(1 for g in prior_games if g["minutes"] >= 28)
            starter_ratio = starter_games / len(prior_games)
        else:
            starter_ratio = 0.5
        features["starter_ratio"] = starter_ratio

        # 8. position_encoded: from player_profile (cached)
        position = self.cache.player_positions.get(player_name, 3.0)  # Default SF
        features["position_encoded"] = position

        # 9. matchup_advantage_score: composite of form, defense, h2h
        player_form = features.get("ema_points_L5", 15.0) / 15.0
        opp_defense = features.get("opp_def_factor", 1.1) / 1.1
        h2h_boost = 0.0
        if features.get("h2h_games", 0) > 0:
            h2h_avg = (
                features.get("h2h_avg_points", 0) or features.get("h2h_avg_rebounds", 0) or 15.0
            )
            h2h_boost = (h2h_avg - features.get("ema_points_L5", 15.0)) / 15.0
        features["matchup_advantage_score"] = player_form - opp_defense + h2h_boost

        # 10. days_since_last_30pt_game: find in game history
        days_since_30 = 999
        for g in prior_games:
            if g["points"] >= 30:
                days_since_30 = (game_date - g["game_date"]).days
                break
        features["days_since_last_30pt_game"] = min(days_since_30, 999)

        # 11. home_streak / away_streak: consecutive wins at home/away
        home_streak, away_streak = 0, 0
        for g in prior_games[:10]:
            # Simple heuristic: points > 15 = "good game" for streak
            if g["is_home"]:
                if g["points"] > 15 and home_streak >= 0:
                    home_streak += 1
                elif g["points"] <= 15 and home_streak <= 0:
                    home_streak -= 1
                else:
                    break
            else:
                if g["points"] > 15 and away_streak >= 0:
                    away_streak += 1
                elif g["points"] <= 15 and away_streak <= 0:
                    away_streak -= 1
                else:
                    break
        features["home_streak"] = home_streak
        features["away_streak"] = away_streak

        # 12. altitude_flag: 1 if playing in Denver
        features["altitude_flag"] = 1.0 if opponent == "DEN" and not is_home else 0.0

        # 13. opponent_back_to_back_flag: check if opponent played yesterday
        opp_b2b = 0.0
        if opponent:
            opp_games = [
                g
                for p, games in self.cache.player_games.items()
                for g in games
                if g.get("team_abbrev") == opponent
            ]
            for g in opp_games[:5]:
                if g["game_date"] == game_date - timedelta(days=1):
                    opp_b2b = 1.0
                    break
        features["opponent_back_to_back_flag"] = opp_b2b

        # 14. opp_def_factor: from opponent defense rating
        features["opp_def_factor"] = opp_def / 100.0

        # DROPPED FEATURES (no data available):
        # - travel_distance_km (needs city coordinates table)
        # - revenge_game_flag (needs previous matchup result lookup)
        # - avg_teammate_usage (needs teammate usage rate data)

        return features

    # =========================================================================
    # MAIN BUILD PROCESS
    # =========================================================================

    def fetch_props(self, stat_type: str) -> pd.DataFrame:
        """Fetch all props for a stat type, aggregating multi-book data"""
        if self.verbose:
            print(f"\n📊 Fetching {stat_type} props...")

        # Aggregate multi-book data per prop (similar to original builder)
        query = """
            SELECT
                player_name, game_date, stat_type,
                MAX(opponent_team) as opponent_team,
                BOOL_OR(is_home) as is_home,
                AVG(over_line) as avg_line,
                MIN(over_line) as min_line,
                MAX(over_line) as max_line,
                STDDEV(over_line) as line_std,
                COUNT(DISTINCT book_name) as num_books,
                MAX(actual_value) as actual_value
            FROM nba_props_xl
            WHERE stat_type = %s
              AND actual_value IS NOT NULL
              AND over_line IS NOT NULL
              AND game_date >= %s AND game_date <= %s
            GROUP BY player_name, game_date, stat_type
            ORDER BY player_name, game_date
        """

        with self.connections["intelligence"].cursor() as cur:
            cur.execute(query, (stat_type, TRAIN_START, TRAIN_END))
            rows = cur.fetchall()

        columns = [
            "player_name",
            "game_date",
            "stat_type",
            "opponent_team",
            "is_home",
            "consensus_line",
            "min_line",
            "max_line",
            "line_std",
            "num_books",
            "actual_value",
        ]

        df = pd.DataFrame(rows, columns=columns)

        # Compute line_spread
        df["line_spread"] = df["max_line"] - df["min_line"]

        if self.verbose:
            print(f"   ✅ Fetched {len(df):,} unique props")
            print(f"   ✅ Avg books per prop: {df['num_books'].mean():.1f}")
            null_lines = df["consensus_line"].isna().sum()
            print(f"   ✅ Props with valid line: {len(df) - null_lines:,}")

        return df

    def enrich_from_game_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich is_home and opponent_team from player_game_logs"""
        if self.verbose:
            print("\n🏠 Enriching is_home and opponent_team from game logs...")

        # Use cached player games to fill missing values
        def fill_is_home(row):
            if pd.notna(row.get("is_home")):
                return row["is_home"]
            games = self.cache.player_games.get(row["player_name"], [])
            for g in games:
                if g["game_date"] == row["game_date"]:
                    return g["is_home"]
            return True  # Default

        def fill_opponent(row):
            if pd.notna(row.get("opponent_team")) and row["opponent_team"] not in ("", "UNK"):
                return row["opponent_team"]
            games = self.cache.player_games.get(row["player_name"], [])
            for g in games:
                if g["game_date"] == row["game_date"]:
                    return g["opponent_abbrev"]
            return ""

        df["is_home"] = df.apply(fill_is_home, axis=1)
        df["opponent_team"] = df.apply(fill_opponent, axis=1)

        # Verify distribution
        home_pct = df["is_home"].sum() / len(df) * 100
        opp_coverage = (df["opponent_team"] != "").sum() / len(df) * 100

        if self.verbose:
            print(f"   ✅ is_home: {home_pct:.1f}% home games")
            print(f"   ✅ opponent_team coverage: {opp_coverage:.1f}%")

        return df

    def build_dataset(self, stat_type: str) -> Optional[pd.DataFrame]:
        """Build complete dataset for a stat type"""
        print(f"\n{'=' * 80}")
        print(f"BUILDING {stat_type} DATASET (BATCHED)")
        print(f"{'=' * 80}")

        # 1. Fetch props
        props_df = self.fetch_props(stat_type)
        if len(props_df) == 0:
            print(f"⚠️  No props found for {stat_type}")
            return None

        # 2. Collect unique keys for bulk loading
        player_names = set(props_df["player_name"].unique())
        game_dates = set(props_df["game_date"].unique())
        max_date = max(game_dates)

        # 3. Load all data in bulk
        self.load_all_player_games(player_names, max_date)
        self.load_all_team_stats()
        self.load_all_team_betting()

        # Collect player-opponent pairs
        props_df = self.enrich_from_game_logs(props_df)
        player_opponent_pairs = set(zip(props_df["player_name"], props_df["opponent_team"]))

        self.load_all_h2h_stats(player_opponent_pairs, [stat_type])
        self.load_all_prop_history()
        self.load_all_vegas_context(game_dates)
        self.load_all_cheatsheet()
        self.load_all_book_lines(stat_type)
        self.load_all_book_accuracy()
        self.load_all_line_movement(stat_type)

        # 4. Extract features (from cache - no DB queries!)
        if self.verbose:
            print(f"\n📊 Extracting features for {len(props_df):,} props (from cache)...")

        rows = []
        errors = 0

        for _, prop in tqdm(props_df.iterrows(), total=len(props_df), desc=stat_type):
            features = self.extract_all_features(prop.to_dict())

            if features is None:
                errors += 1
                continue

            # Get line and actual values
            line = prop.get("consensus_line")
            if line is None or pd.isna(line):
                line = prop.get("min_over_line", 0.0)
            if line is None or pd.isna(line):
                line = 0.0

            actual = prop.get("actual_value", 0.0)
            if actual is None or pd.isna(actual):
                errors += 1
                continue

            line = float(line)
            actual = float(actual)

            # Skip props without valid line
            if line == 0:
                errors += 1
                continue

            row = {
                "player_name": prop["player_name"],
                "game_date": prop["game_date"],
                "stat_type": stat_type,
                "opponent_team": prop.get("opponent_team", ""),
                "is_home": prop.get("is_home", True),
                "line": line,
                "source": "bettingpros",
                f"actual_{stat_type.lower()}": actual,
                "label": 1 if actual > line else 0,
                "split": "train" if prop["game_date"] <= TRAIN_END else "val",
            }
            row.update(features)
            rows.append(row)

        dataset = pd.DataFrame(rows)
        dataset = dataset.sort_values("game_date").reset_index(drop=True)

        # Drop constant columns (no predictive value)
        constant_cols = [c for c in dataset.columns if dataset[c].nunique() <= 1]
        # Keep metadata columns even if constant
        keep_meta = ["stat_type", "source", "split"]
        drop_cols = [c for c in constant_cols if c not in keep_meta]
        if drop_cols:
            dataset = dataset.drop(columns=drop_cols)
            if self.verbose:
                print(f"   Dropped {len(drop_cols)} constant columns: {drop_cols}")

        if self.verbose:
            print(f"\n✅ Dataset built: {len(dataset):,} rows")
            print(f"   Errors: {errors}")
            print(f"   Features: {len(dataset.columns) - 10}")  # Subtract metadata cols

        return dataset

    def run(self, market: Optional[str] = None):
        """Main execution"""
        try:
            self.connect()

            stat_types = [market.upper()] if market else STAT_TYPES

            for stat_type in stat_types:
                # Clear cache between markets
                self.cache = DataCache()

                dataset = self.build_dataset(stat_type)

                if dataset is not None and len(dataset) > 0:
                    output_file = self.output_dir / f"xl_training_{stat_type}_2023_2025_batched.csv"
                    dataset.to_csv(output_file, index=False)

                    print(f"\n✅ Saved: {output_file}")
                    print(f"   {len(dataset):,} props, {len(dataset.columns)} columns")

        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Build NBA XL training datasets (batched)")
    parser.add_argument("--output", type=str, default="datasets/", help="Output directory")
    parser.add_argument("--market", type=str, help="Single market (POINTS, REBOUNDS)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    builder = BatchedDatasetBuilder(output_dir=args.output, verbose=not args.quiet)
    builder.run(market=args.market)


if __name__ == "__main__":
    main()
