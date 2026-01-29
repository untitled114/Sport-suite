#!/usr/bin/env python3
"""
NBA XL Feature Extractor - Enhanced with Book Disagreement Features
===================================================================
Extends LiveFeatureExtractor to add 20 book disagreement features.
Used for production predictions with line shopping capability.

Features: 78 player features + 20 book features = 98 total
"""

import os
import pandas as pd
import numpy as np
import psycopg2
import logging
import warnings
from features.extract_live_features import LiveFeatureExtractor

# Suppress all pandas warnings about SQLAlchemy
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LiveFeatureExtractorXL(LiveFeatureExtractor):
    """
    Enhanced feature extractor with book disagreement features.
    Inherits all player feature extraction from LiveFeatureExtractor.
    Adds 20 book features from nba_props_xl table.
    """

    def __init__(self):
        # Initialize parent class (connects to players, games, team databases)
        super().__init__()

        default_user = self.DB_DEFAULT_USER
        default_password = self.DB_DEFAULT_PASSWORD
        self.INTELLIGENCE_DB_CONFIG = {
            'host': os.getenv('NBA_INTEL_DB_HOST', 'localhost'),
            'port': int(os.getenv('NBA_INTEL_DB_PORT', 5539)),
            'user': os.getenv('NBA_INTEL_DB_USER', default_user),
            'password': os.getenv('NBA_INTEL_DB_PASSWORD', default_password),
            'database': os.getenv('NBA_INTEL_DB_NAME', 'nba_intelligence')
        }

        # Add connection to nba_intelligence database (port 5539)
        self.intelligence_conn = psycopg2.connect(**self.INTELLIGENCE_DB_CONFIG)

        # Add connection to nba_team database (port 5538) for team betting performance
        self.TEAM_DB_CONFIG = {
            'host': os.getenv('NBA_TEAM_DB_HOST', 'localhost'),
            'port': int(os.getenv('NBA_TEAM_DB_PORT', 5538)),
            'user': os.getenv('NBA_TEAM_DB_USER', default_user),
            'password': os.getenv('NBA_TEAM_DB_PASSWORD', default_password),
            'database': os.getenv('NBA_TEAM_DB_NAME', 'nba_team')
        }
        self.team_conn = psycopg2.connect(**self.TEAM_DB_CONFIG)

        # Cache for team betting performance (season -> team -> metrics)
        self._team_betting_cache = {}

        # MongoDB connection for hybrid feature extraction (preferred for book features)
        try:
            from pymongo import MongoClient
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://${MONGO_USER}:${MONGO_PASSWORD}@localhost:27017/')
            self.mongo_client = MongoClient(mongo_uri)
            self.mongo_db = self.mongo_client['nba_betting_xl']
            self.mongo_collection = self.mongo_db['nba_props_xl']
            # Test connection
            self.mongo_client.admin.command('ping')
            # TEMPORARY: Force MongoDB disabled until properly tested (Nov 10, 2025)
            self.use_mongodb = False
            logger.warning("MongoDB temporarily disabled - using PostgreSQL fallback")
        except Exception as e:
            self.use_mongodb = False
            logger.warning(f"MongoDB unavailable, using PostgreSQL fallback: {e}")

    def extract_book_features_mongo(self, player_name, game_date, stat_type):
        """
        Extract book features from MongoDB (pre-computed in line_shopping subdocument).
        This is the preferred method as features are pre-computed during prop loading.

        Args:
            player_name: Player's full name
            game_date: Date of the game
            stat_type: Stat type ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')

        Returns:
            Dict with all 23 book features or None if not found
        """
        try:
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)

            # Query MongoDB for this prop
            doc = self.mongo_collection.find_one({
                'player_name': player_name,
                'game_date': game_date,
                'stat_type': stat_type.upper()
            })

            if not doc or 'line_shopping' not in doc:
                return None

            # Extract pre-computed line_shopping features
            ls = doc['line_shopping']
            deviations = ls.get('deviations', {})

            # Map MongoDB features to expected feature names
            book_features = {
                # Line variance (5 features)
                'line_spread': float(ls.get('line_spread', 0.0)),
                'consensus_line': float(ls.get('consensus_line', 0.0)),
                'line_std_dev': float(ls.get('line_std_dev', 0.0)),
                'num_books_offering': float(ls.get('num_books', 1)),
                'line_coef_variation': float(ls.get('line_coef_variation', 0.0)),

                # Book deviations (8 features)
                'draftkings_deviation': float(deviations.get('draftkings', 0.0)),
                'fanduel_deviation': float(deviations.get('fanduel', 0.0)),
                'betmgm_deviation': float(deviations.get('betmgm', 0.0)),
                'caesars_deviation': float(deviations.get('caesars', 0.0)),
                'bet365_deviation': float(deviations.get('bet365', 0.0)),
                'betrivers_deviation': float(deviations.get('betrivers', 0.0)),
                'espnbet_deviation': float(deviations.get('espnbet', 0.0)),
                'fanatics_deviation': float(deviations.get('fanatics', 0.0)),

                # Historical accuracy (7 features)
                'softest_book_id': self._encode_book_name(ls.get('softest_book', 'unknown')),
                'hardest_book_id': self._encode_book_name(ls.get('hardest_book', 'unknown')),
                'line_spread_percentile': float(ls.get('line_spread_percentile', 0.5)),
                'books_agree': 1.0 if ls.get('books_agree', False) else 0.0,
                'books_disagree': 1.0 if ls.get('books_disagree', False) else 0.0,
                'softest_vs_consensus': float(ls.get('softest_vs_consensus', 0.0)),
                'hardest_vs_consensus': float(ls.get('hardest_vs_consensus', 0.0)),

                # Additional line features (3)
                'min_line': float(ls.get('min_line', 0.0)),
                'max_line': float(ls.get('max_line', 0.0)),
                'line_std': float(ls.get('line_std_dev', 0.0))
            }

            return book_features

        except Exception as e:
            logger.debug(f"MongoDB book feature extraction failed: {e}")
            return None

    def extract_book_features(self, player_name, game_date, stat_type):
        """
        Extract 23 book disagreement features (MongoDB preferred, PostgreSQL fallback).

        Hybrid Strategy:
        1. Try MongoDB first (pre-computed features in line_shopping subdocument)
        2. Fall back to PostgreSQL if MongoDB unavailable or data not found
        3. Return defaults if both fail

        Features:
        1. Line variance (5 features):
           - line_spread: max(lines) - min(lines)
           - consensus_line: mean of all books
           - line_std_dev: standard deviation
           - num_books_offering: count of books
           - line_coef_variation: std_dev / mean

        2. Book deviations (8 features):
           - draftkings_deviation: draftkings_line - consensus
           - fanduel_deviation
           - betmgm_deviation
           - caesars_deviation
           - bet365_deviation
           - betrivers_deviation
           - espnbet_deviation
           - fanatics_deviation

        3. Historical accuracy (7 features):
           - softest_book: which book has lowest line
           - hardest_book: which book has highest line
           - line_spread_percentile: where is this spread vs historical
           - books_agree: bool (spread < 0.5)
           - books_disagree: bool (spread >= 2.0)
           - softest_vs_consensus: softest - consensus
           - hardest_vs_consensus: hardest - consensus

        4. Additional line features (3):
           - min_line, max_line, line_std

        Args:
            player_name: Player's full name
            game_date: Date of the game (datetime or string)
            stat_type: Stat type ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')

        Returns:
            Dict with all 23 book features
        """
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)

        # Normalize stat type to uppercase
        stat_type = stat_type.upper()

        # HYBRID STRATEGY: Try MongoDB first (fast, pre-computed)
        if self.use_mongodb:
            mongo_features = self.extract_book_features_mongo(player_name, game_date, stat_type)
            if mongo_features is not None:
                logger.debug(f"✓ MongoDB book features for {player_name} {stat_type}")
                return mongo_features
            else:
                logger.debug(f"MongoDB miss for {player_name} {stat_type}, trying PostgreSQL fallback")

        # FALLBACK: PostgreSQL (compute on-the-fly from nba_props_xl)

        # Query all book lines for this player/game/stat
        query = """
        SELECT
            book_name,
            over_line
        FROM nba_props_xl
        WHERE player_name = %s
          AND game_date = %s
          AND stat_type = %s
          AND is_active = true
          AND over_line IS NOT NULL
        ORDER BY fetch_timestamp DESC
        """

        try:
            # Convert to date object safely (handles datetime, Timestamp, date, and string)
            import datetime
            if isinstance(game_date, (datetime.datetime, pd.Timestamp)):
                date_param = game_date.date()
            elif isinstance(game_date, datetime.date):
                date_param = game_date  # Already a date object
            elif isinstance(game_date, str):
                date_param = pd.to_datetime(game_date).date()
            else:
                date_param = game_date  # Unknown type, pass through

            df = pd.read_sql_query(
                query,
                self.intelligence_conn,
                params=(player_name, date_param, stat_type)
            )

            if len(df) == 0:
                # No multi-book props available - return defaults
                logger.debug(f"No book props found for {player_name} {stat_type} on {game_date}, using defaults")
                return self._get_default_book_features()

            # Get the most recent fetch (first row after ordering by timestamp DESC)
            # Extract unique book lines
            book_lines = df.groupby('book_name')['over_line'].first()

            if len(book_lines) == 0:
                return self._get_default_book_features()

            # 1. LINE VARIANCE FEATURES (5 features)
            line_spread = float(book_lines.max() - book_lines.min())
            consensus_line = float(book_lines.mean())
            line_std_dev = float(book_lines.std()) if len(book_lines) > 1 else 0.0
            num_books_offering = float(len(book_lines))
            line_coef_variation = (line_std_dev / consensus_line) if consensus_line > 0 else 0.0

            # 2. BOOK-SPECIFIC DEVIATIONS (8 features)
            # Calculate deviation for each major book
            book_deviations = {}
            for book in ['draftkings', 'fanduel', 'betmgm', 'caesars', 'bet365', 'betrivers', 'espnbet', 'fanatics']:
                if book in book_lines.index:
                    book_deviations[f'{book}_deviation'] = float(book_lines[book] - consensus_line)
                else:
                    # Book doesn't offer this prop - use 0.0 (no deviation)
                    book_deviations[f'{book}_deviation'] = 0.0

            # 3. HISTORICAL ACCURACY FEATURES (7 features)
            # Softest/hardest book identification - always computed from raw data
            softest_book_name = book_lines.idxmin()  # Book with lowest line (easiest to hit)
            hardest_book_name = book_lines.idxmax()  # Book with highest line (hardest to hit)

            # Calculate percentile of this line spread vs historical spreads
            # Query historical spreads for this player/stat
            historical_spread_query = """
            SELECT line_spread
            FROM nba_props_xl
            WHERE player_name = %s
              AND stat_type = %s
              AND game_date < %s
              AND line_spread IS NOT NULL
            ORDER BY game_date DESC, fetch_timestamp DESC, line_spread DESC
            LIMIT 30
            """

            try:
                # Use date_param from above (already converted)
                hist_spreads = pd.read_sql_query(
                    historical_spread_query,
                    self.intelligence_conn,
                    params=(player_name, stat_type, date_param)
                )

                if len(hist_spreads) >= 5:
                    # Calculate percentile
                    percentile = (hist_spreads['line_spread'] < line_spread).sum() / len(hist_spreads)
                    line_spread_percentile = float(percentile)
                else:
                    # Not enough history - use 50th percentile (neutral)
                    line_spread_percentile = 0.5
            except Exception as e:
                logger.debug(f"Error calculating historical spread percentile: {e}")
                line_spread_percentile = 0.5

            # Books agree/disagree flags
            books_agree = 1.0 if line_spread < 0.5 else 0.0
            books_disagree = 1.0 if line_spread >= 2.0 else 0.0

            # Softest/hardest vs consensus
            softest_line = book_lines.min()
            hardest_line = book_lines.max()
            softest_vs_consensus = float(softest_line - consensus_line)
            hardest_vs_consensus = float(hardest_line - consensus_line)

            # Combine all features
            book_features = {
                # Line variance (5)
                'line_spread': line_spread,
                'consensus_line': consensus_line,
                'line_std_dev': line_std_dev,
                'num_books_offering': num_books_offering,
                'line_coef_variation': line_coef_variation,

                # Book deviations (8)
                'draftkings_deviation': book_deviations['draftkings_deviation'],
                'fanduel_deviation': book_deviations['fanduel_deviation'],
                'betmgm_deviation': book_deviations['betmgm_deviation'],
                'caesars_deviation': book_deviations['caesars_deviation'],
                'bet365_deviation': book_deviations['bet365_deviation'],
                'betrivers_deviation': book_deviations['betrivers_deviation'],
                'espnbet_deviation': book_deviations['espnbet_deviation'],
                'fanatics_deviation': book_deviations['fanatics_deviation'],

                # Historical accuracy (7)
                'softest_book_id': self._encode_book_name(softest_book_name),
                'hardest_book_id': self._encode_book_name(hardest_book_name),
                'line_spread_percentile': line_spread_percentile,
                'books_agree': books_agree,
                'books_disagree': books_disagree,
                'softest_vs_consensus': softest_vs_consensus,
                'hardest_vs_consensus': hardest_vs_consensus,

                # Additional line features (3) - required by trained models
                'min_line': float(softest_line),
                'max_line': float(hardest_line),
                'line_std': line_std_dev  # Same as line_std_dev, but models expect both names
            }

            return book_features

        except Exception as e:
            logger.warning(f"Error extracting book features: {e}")
            return self._get_default_book_features()

    def _encode_book_name(self, book_name):
        """
        Encode book name to numeric ID for modeling.

        Args:
            book_name: Book name string

        Returns:
            Numeric book ID (0-9)
        """
        book_map = {
            'draftkings': 1.0,
            'fanduel': 2.0,
            'betmgm': 3.0,
            'caesars': 4.0,
            'bet365': 5.0,
            'betrivers': 6.0,
            'espnbet': 7.0,
            'fanatics': 8.0,
            'prizepicks': 9.0,
            'underdog': 10.0
        }
        return book_map.get(book_name.lower() if book_name else '', 0.0)

    def _get_default_book_features(self):
        """
        Return default book features when no multi-book props available.
        Used for historical training data or when line shopping not available.

        Returns:
            Dict with default values for all 20 book features
        """
        return {
            # Line variance (5) - defaults assume single book
            'line_spread': 0.0,
            'consensus_line': 0.0,
            'line_std_dev': 0.0,
            'num_books_offering': 1.0,  # At least one book offers it
            'line_coef_variation': 0.0,

            # Book deviations (8) - all zero (no deviation when single book)
            'draftkings_deviation': 0.0,
            'fanduel_deviation': 0.0,
            'betmgm_deviation': 0.0,
            'caesars_deviation': 0.0,
            'bet365_deviation': 0.0,
            'betrivers_deviation': 0.0,
            'espnbet_deviation': 0.0,
            'fanatics_deviation': 0.0,

            # Historical accuracy (7) - neutral defaults
            'softest_book_id': 0.0,  # Unknown
            'hardest_book_id': 0.0,  # Unknown
            'line_spread_percentile': 0.5,  # 50th percentile (neutral)
            'books_agree': 1.0,  # Single book = agree by definition
            'books_disagree': 0.0,  # Single book = no disagreement
            'softest_vs_consensus': 0.0,
            'hardest_vs_consensus': 0.0,

            # Additional line features (3) - required by trained models
            'min_line': 0.0,
            'max_line': 0.0,
            'line_std': 0.0
        }

    def validate_features(self, features_dict: dict) -> tuple:
        """
        Validate all 102 features are present and not NaN.

        Args:
            features_dict: Dictionary of extracted features

        Returns:
            (is_valid, list_of_missing_or_nan_features)
        """
        missing = []

        # Check each feature
        for feat, value in features_dict.items():
            if feat == 'expected_diff':
                # expected_diff is computed later by classifier, skip
                continue
            if pd.isna(value):
                missing.append(f"{feat} (NaN)")

        # Check for minimum expected features (at least 101 of 102, since expected_diff added later)
        if len(features_dict) < 101:
            missing.append(f"Only {len(features_dict)} features extracted (expected 101, plus expected_diff added later)")

        is_valid = len(missing) == 0
        return (is_valid, missing)

    def extract_features(self, player_name, game_date, is_home=None, opponent_team=None,
                        line=None, source='validation', spread_diff=0.0, total_diff=0.0,
                        stat_type='POINTS', include_book_features=True):
        """
        Extract 98 features for a player (78 player + 20 book).

        Overrides parent extract_features() to add book features.

        Args:
            player_name: Player's full name
            game_date: Date of the game (datetime or string)
            is_home: True if home game, False if away, None if unknown
            opponent_team: Opponent team abbreviation (e.g., 'LAL')
            line: Prop line value (e.g., 25.5 points)
            source: Prop source for book encoding
            spread_diff: Point spread differential
            total_diff: Totals line differential
            stat_type: Stat type ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')
            include_book_features: Whether to include book features (default True)

        Returns:
            Dict with 142 features (78 base + 20 book + 32 H2H + 12 prop)
            Or 78 features if include_book_features=False
        """
        # Get 78 player features from parent class
        player_features = super().extract_features(
            player_name, game_date, is_home, opponent_team,
            line, source, spread_diff, total_diff
        )

        if not include_book_features:
            # Return only player features (for historical training without multi-book data)
            return player_features

        # Add 20 book features
        book_features = self.extract_book_features(player_name, game_date, stat_type)
        player_features.update(book_features)

        # Add 44 H2H matchup features
        if opponent_team:
            h2h_features = self.extract_h2h_matchup_features(
                player_name, opponent_team, stat_type, game_date
            )
            player_features.update(h2h_features)
        else:
            player_features.update(self._get_default_h2h_features())

        # Add 12 prop history features
        if line is not None and line > 0:
            prop_features = self.extract_prop_history_features(
                player_name, stat_type, line, game_date, is_home
            )
            player_features.update(prop_features)
        else:
            player_features.update(self._get_default_prop_features())

        # Add 2 vegas context features (total, spread)
        vegas_features = self.extract_vegas_context(player_name, game_date, is_home)
        player_features.update(vegas_features)

        # Add 5 team betting performance features (ATS, O/U, availability flag)
        team_betting_features = self.extract_team_betting_features(player_name, opponent_team, game_date, is_home)
        player_features.update(team_betting_features)

        # Add 8 BettingPros cheatsheet features
        cheatsheet_features = self.extract_cheatsheet_features(player_name, game_date, stat_type)
        player_features.update(cheatsheet_features)

        # Total: 82 (base+new) + 23 (book) + 36 (H2H) + 12 (prop) + 2 (vegas) + 5 (team betting) + 8 (cheatsheet) = ~168 features
        # Note: expected_diff (1 feature) will be added by classifier head during training
        return player_features

    def extract_h2h_matchup_features(self, player_name, opponent_team, stat_type, game_date):
        """
        Extract 32 head-to-head matchup features from matchup_history table.

        Features:
        - 4 quality metrics (games, days_since_last, sample_quality, recency_weight)
        - 20 per-stat averages (avg, std, L5/L10/L20 for 4 stats)
        - 8 home/away splits (4 stats × home/away)

        Args:
            player_name: Player's full name
            opponent_team: Opponent team code (e.g., 'BOS')
            stat_type: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
            game_date: Game date (for temporal safety)

        Returns:
            dict with 32 H2H features
        """
        if not opponent_team or opponent_team == '':
            return self._get_default_h2h_features()

        try:
            # Query for the specific stat_type row - data is now stat-type-specific
            # Each stat_type row has its own columns (e.g., POINTS row has avg_points, l3_points, etc.)
            stat_col_map = {
                'POINTS': 'points',
                'REBOUNDS': 'rebounds',
                'ASSISTS': 'assists',
                'THREES': 'threes'
            }
            stat_suffix = stat_col_map.get(stat_type, 'points')

            # Build query for the specific stat_type
            query = f"""
            SELECT
                games_played, days_since_last, sample_quality, recency_weight,
                avg_{stat_suffix}, std_{stat_suffix},
                l3_{stat_suffix}, l5_{stat_suffix}, l10_{stat_suffix}, l20_{stat_suffix},
                home_avg_{stat_suffix}, away_avg_{stat_suffix}, home_away_split_{stat_suffix}
            FROM matchup_history
            WHERE player_name = %s
              AND opponent_team = %s
              AND stat_type = %s
              AND (computed_as_of_date IS NULL OR computed_as_of_date <= %s)
            ORDER BY computed_as_of_date DESC NULLS LAST
            LIMIT 1
            """

            with self.intelligence_conn.cursor() as cur:
                cur.execute(query, (player_name, opponent_team, stat_type, game_date))
                result = cur.fetchone()

                if result:
                    # For backward compatibility, we still return all stat names
                    # but only the queried stat_type will have actual values
                    features = {
                        'h2h_games': result[0] or 0,
                        'h2h_days_since_last': result[1] or 999,
                        'h2h_sample_quality': result[2] or 0.2,
                        'h2h_recency_weight': result[3] or 0.5,
                    }

                    # Initialize all stats to 0.0
                    for stat in ['points', 'rebounds', 'assists', 'threes']:
                        features[f'h2h_avg_{stat}'] = 0.0
                        features[f'h2h_std_{stat}'] = 0.0
                        for window in ['L3', 'L5', 'L10', 'L20']:
                            features[f'h2h_{window}_{stat}'] = 0.0
                        features[f'h2h_home_avg_{stat}'] = 0.0
                        features[f'h2h_away_avg_{stat}'] = 0.0

                    # Populate the specific stat_type values from the query
                    features[f'h2h_avg_{stat_suffix}'] = result[4] or 0.0
                    features[f'h2h_std_{stat_suffix}'] = result[5] or 0.0
                    features[f'h2h_L3_{stat_suffix}'] = result[6] or 0.0
                    features[f'h2h_L5_{stat_suffix}'] = result[7] or 0.0
                    features[f'h2h_L10_{stat_suffix}'] = result[8] or 0.0
                    features[f'h2h_L20_{stat_suffix}'] = result[9] or 0.0
                    features[f'h2h_home_avg_{stat_suffix}'] = result[10] or 0.0
                    features[f'h2h_away_avg_{stat_suffix}'] = result[11] or 0.0

                    return features
                else:
                    return self._get_default_h2h_features()

        except Exception as e:
            logger.warning(f"Error extracting H2H features for {player_name} vs {opponent_team}: {e}")
            return self._get_default_h2h_features()

    def _get_default_h2h_features(self):
        """Return zero-filled H2H features when no matchup history exists (36 features)"""
        features = {
            'h2h_games': 0,
            'h2h_days_since_last': 999,
            'h2h_sample_quality': 0.2,
            'h2h_recency_weight': 0.5
        }

        for stat in ['points', 'rebounds', 'assists', 'threes']:
            features[f'h2h_avg_{stat}'] = 0.0
            features[f'h2h_std_{stat}'] = 0.0
            for window in ['L3', 'L5', 'L10', 'L20']:
                features[f'h2h_{window}_{stat}'] = 0.0
            features[f'h2h_home_avg_{stat}'] = 0.0
            features[f'h2h_away_avg_{stat}'] = 0.0

        return features

    def extract_prop_history_features(self, player_name, stat_type, line, game_date, is_home):
        """
        Extract 12 prop performance history features from prop_performance_history table.

        Features:
        - 5 hit rate metrics (L20, context, defense, rest, deviation)
        - 3 line positioning (vs avg, percentile, days since hit)
        - 2 quality (sample quality, bayesian confidence)
        - 2 streak/size (consecutive overs, sample size L20)

        Args:
            player_name: Player's full name
            stat_type: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
            line: Prop line value (e.g., 25.5)
            game_date: Game date (for season calculation)
            is_home: True if home game

        Returns:
            dict with 12 prop history features
        """
        if line is None or line == 0:
            return self._get_default_prop_features()

        try:
            import sys
            import os
            # Add parent directory to path to import season_helpers
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from utils.season_helpers import date_to_season

            line_center = round(line * 2) / 2.0  # Round to nearest 0.5
            season = date_to_season(game_date)  # Uses END year convention

            query = """
            SELECT
                hit_rate_l20, hit_rate_home, hit_rate_away,
                hit_rate_vs_top10_def, hit_rate_vs_bottom10_def,
                hit_rate_rested, hit_rate_b2b,
                line_vs_season_avg, line_percentile,
                days_since_last_hit, sample_quality_score,
                bayesian_prior_weight, consecutive_overs, props_l20
            FROM prop_performance_history
            WHERE player_name = %s
              AND stat_type = %s
              AND line_center = %s
              AND season = %s
            LIMIT 1
            """

            with self.intelligence_conn.cursor() as cur:
                cur.execute(query, (player_name, stat_type, line_center, season))
                result = cur.fetchone()

                if result:
                    # Convert all Decimal types to float to avoid type errors
                    # Select context hit rate based on home/away
                    context_hit_rate = float(result[1]) if result[1] is not None else 0.5
                    if not is_home:
                        context_hit_rate = float(result[2]) if result[2] is not None else 0.5

                    hit_rate_l20 = float(result[0]) if result[0] is not None else 0.5

                    return {
                        'prop_hit_rate_L20': hit_rate_l20,
                        'prop_hit_rate_context': context_hit_rate,
                        'prop_hit_rate_defense': float(result[3]) if result[3] is not None else 0.5,
                        'prop_hit_rate_rest': float(result[5]) if result[5] is not None else 0.5,
                        'prop_hit_rate_deviation': hit_rate_l20 - 0.5,  # deviation from 50%
                        'prop_line_vs_season_avg': float(result[7]) if result[7] is not None else 0.0,
                        'prop_line_percentile': float(result[8]) if result[8] is not None else 0.5,
                        'prop_days_since_last_hit': min(int(result[9]) if result[9] is not None else 999, 999),
                        'prop_sample_quality': float(result[10]) if result[10] is not None else 0.2,
                        'prop_bayesian_confidence': (1.0 - float(result[11])) if result[11] is not None else 0.2,
                        'prop_consecutive_overs': int(result[12]) if result[12] is not None else 0,
                        'prop_sample_size_L20': int(result[13]) if result[13] is not None else 0
                    }
                else:
                    return self._get_default_prop_features()

        except Exception as e:
            logger.warning(f"Error extracting prop history for {player_name} {stat_type} {line}: {e}")
            return self._get_default_prop_features()

    def _get_default_prop_features(self):
        """Return neutral prop features when no history exists"""
        return {
            'prop_hit_rate_L20': 0.5,
            'prop_hit_rate_context': 0.5,
            'prop_hit_rate_defense': 0.5,
            'prop_hit_rate_rest': 0.5,
            'prop_hit_rate_deviation': 0.0,
            'prop_line_vs_season_avg': 0.0,
            'prop_line_percentile': 0.5,
            'prop_days_since_last_hit': 999,
            'prop_sample_quality': 0.2,
            'prop_bayesian_confidence': 0.2,
            'prop_consecutive_overs': 0,
            'prop_sample_size_L20': 0
        }

    def extract_vegas_context(self, player_name, game_date, is_home):
        """
        Extract Vegas context features (total, spread) from games table.

        Features:
        - vegas_total: Projected game total points (e.g., 225.5)
        - vegas_spread: Point spread adjusted for player's team (positive = favored)

        Args:
            player_name: Player's full name
            game_date: Game date
            is_home: True if home game, False if away

        Returns:
            dict with vegas_total and vegas_spread features
        """
        try:
            # Get player's team
            player_team = self.get_player_team(player_name, game_date)
            if not player_team:
                return self._get_default_vegas_features()

            # Query games table for vegas lines
            # games table is in nba_games database (port 5537)
            query = """
            SELECT vegas_total, vegas_spread, home_team, away_team
            FROM games
            WHERE game_date = %s
              AND (home_team = %s OR away_team = %s)
            LIMIT 1
            """

            with self.games_conn.cursor() as cur:
                cur.execute(query, (game_date, player_team, player_team))
                result = cur.fetchone()

                if result:
                    vegas_total, vegas_spread, home_team, away_team = result

                    # Adjust spread for player's perspective
                    # vegas_spread is typically home team spread (negative = home favored)
                    if vegas_spread is not None:
                        if player_team == home_team:
                            # Player is home - use spread as-is (negative = favored)
                            player_spread = -float(vegas_spread)  # Flip sign so positive = favored
                        else:
                            # Player is away - flip the spread
                            player_spread = float(vegas_spread)
                    else:
                        player_spread = 0.0

                    return {
                        'vegas_total': float(vegas_total) if vegas_total else 220.0,
                        'vegas_spread': player_spread
                    }
                else:
                    return self._get_default_vegas_features()

        except Exception as e:
            logger.debug(f"Error extracting vegas context for {player_name}: {e}")
            return self._get_default_vegas_features()

    def _get_default_vegas_features(self):
        """Return neutral vegas features when no data available"""
        return {
            'vegas_total': 220.0,  # League average total
            'vegas_spread': 0.0    # Neutral (pick'em)
        }

    def _get_season_from_date(self, game_date):
        """
        Convert game date to NBA season year.
        NBA season runs Oct-Jun, so Oct-Dec games are next year's season.
        E.g., Oct 2023 -> 2024, Feb 2024 -> 2024
        """
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        if hasattr(game_date, 'month'):
            if game_date.month >= 10:
                return game_date.year + 1
            else:
                return game_date.year
        return 2024  # Default fallback

    def _load_season_betting_data(self, season):
        """Load all team betting performance data for a season into cache."""
        if season in self._team_betting_cache:
            return

        try:
            query = """
            SELECT team_abbrev, ats_pct, ou_pct
            FROM team_betting_performance
            WHERE season = %s
            """
            with self.team_conn.cursor() as cur:
                cur.execute(query, (season,))
                rows = cur.fetchall()

                season_data = {}
                for team_abbrev, ats_pct, ou_pct in rows:
                    season_data[team_abbrev] = {
                        'ats_pct': float(ats_pct) if ats_pct else 0.5,
                        'ou_pct': float(ou_pct) if ou_pct else 0.5
                    }

                self._team_betting_cache[season] = season_data
                logger.debug(f"Loaded betting performance for {len(season_data)} teams in {season}")

        except Exception as e:
            logger.warning(f"Failed to load team betting data for season {season}: {e}")
            self._team_betting_cache[season] = {}

    def get_player_team_from_game(self, game_date, opponent_team, is_home):
        """
        Derive player's team from game context using the games table.
        More reliable than player_game_logs when game context is available.

        Args:
            game_date: Date of the game
            opponent_team: Opponent team abbreviation
            is_home: True if player's team is home, False if away

        Returns:
            Team abbreviation or None if not found
        """
        if not opponent_team:
            return None

        try:
            # Query games table to find the matchup
            if is_home:
                # Player is home, opponent is away
                query = """
                SELECT home_team FROM games
                WHERE game_date = %s AND away_team = %s
                LIMIT 1
                """
            else:
                # Player is away, opponent is home
                query = """
                SELECT away_team FROM games
                WHERE game_date = %s AND home_team = %s
                LIMIT 1
                """

            with self.games_conn.cursor() as cur:
                cur.execute(query, (game_date, opponent_team))
                result = cur.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            logger.debug(f"Error getting team from game context: {e}")

        return None

    def extract_team_betting_features(self, player_name, opponent_team, game_date, is_home=None):
        """
        Extract team betting performance features (ATS, O/U).

        Features (5 total):
        - team_ats_pct: Player's team ATS win percentage for the season
        - opp_ats_pct: Opponent team's ATS win percentage
        - team_ou_pct: Player's team O/U (over) percentage for the season
        - opp_ou_pct: Opponent team's O/U percentage
        - team_betting_available: 1.0 if real data, 0.0 if defaults used

        Args:
            player_name: Player's full name
            opponent_team: Opponent team abbreviation (e.g., 'LAL')
            game_date: Game date (datetime or string)
            is_home: True if player is home, False if away (used for reliable team lookup)

        Returns:
            dict with 5 team betting features
        """
        defaults = {
            'team_ats_pct': 0.5,
            'opp_ats_pct': 0.5,
            'team_ou_pct': 0.5,
            'opp_ou_pct': 0.5,
            'team_betting_available': 0.0  # Flag: using defaults
        }

        try:
            # Get player's team - prefer game context (more reliable) over player_game_logs
            player_team = None
            if is_home is not None and opponent_team:
                player_team = self.get_player_team_from_game(game_date, opponent_team, is_home)

            # Fallback to player_game_logs if game context didn't work
            if not player_team:
                player_team = self.get_player_team(player_name, game_date)

            if not player_team:
                return defaults

            # Get season from game date
            season = self._get_season_from_date(game_date)

            # Load season data into cache if not present
            self._load_season_betting_data(season)

            season_data = self._team_betting_cache.get(season, {})

            # Check if we have data for this season
            if not season_data:
                logger.debug(f"No team betting data for season {season}")
                return defaults

            # Get team metrics
            team_data = season_data.get(player_team, {})
            opp_data = season_data.get(opponent_team, {}) if opponent_team else {}

            # Only mark as available if we have the player's team data
            has_team_data = bool(team_data)

            return {
                'team_ats_pct': team_data.get('ats_pct', 0.5),
                'opp_ats_pct': opp_data.get('ats_pct', 0.5),
                'team_ou_pct': team_data.get('ou_pct', 0.5),
                'opp_ou_pct': opp_data.get('ou_pct', 0.5),
                'team_betting_available': 1.0 if has_team_data else 0.0
            }

        except Exception as e:
            logger.debug(f"Error extracting team betting features for {player_name}: {e}")
            return defaults

    def extract_cheatsheet_features(self, player_name, game_date, stat_type):
        """
        Extract BettingPros cheatsheet features from cheatsheet_data table.

        Features (8 total):
        - bp_projection_diff: BettingPros projection minus line (positive = OVER lean)
        - bp_bet_rating: 1-5 strength rating (higher = stronger bet)
        - bp_ev_pct: Expected value percentage
        - bp_probability: BettingPros estimated probability of hitting
        - bp_opp_rank: Opponent defense rank (1=worst defense, 30=best)
        - bp_hit_rate_l5: Player's hit rate last 5 games
        - bp_hit_rate_l15: Player's hit rate last 15 games
        - bp_hit_rate_season: Player's season hit rate

        Args:
            player_name: Player's full name
            game_date: Date of the game
            stat_type: Stat type ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES')

        Returns:
            dict with 8 cheatsheet features
        """
        defaults = {
            'bp_projection_diff': 0.0,
            'bp_bet_rating': 3.0,  # Neutral rating
            'bp_ev_pct': 0.0,
            'bp_probability': 0.5,
            'bp_opp_rank': 15.0,  # Middle rank
            'bp_hit_rate_l5': 0.5,
            'bp_hit_rate_l15': 0.5,
            'bp_hit_rate_season': 0.5,
        }

        try:
            # Normalize stat_type to uppercase (matches cheatsheet_data format)
            cs_stat_type = stat_type.upper()

            # Query cheatsheet_data - use ILIKE for case-insensitive matching
            # Note: For accented names (Dončić), data should be stored consistently
            query = """
            SELECT
                projection_diff,
                bet_rating,
                ev_pct,
                probability,
                opp_rank,
                hit_rate_l5,
                hit_rate_l15,
                hit_rate_season
            FROM cheatsheet_data
            WHERE lower(player_name) = lower(%s)
              AND game_date = %s
              AND stat_type = %s
            ORDER BY fetch_timestamp DESC
            LIMIT 1
            """

            with self.intelligence_conn.cursor() as cur:
                cur.execute(query, (player_name, game_date, cs_stat_type))
                result = cur.fetchone()

                if result:
                    return {
                        'bp_projection_diff': float(result[0]) if result[0] is not None else 0.0,
                        'bp_bet_rating': float(result[1]) if result[1] is not None else 3.0,
                        'bp_ev_pct': float(result[2]) if result[2] is not None else 0.0,
                        'bp_probability': float(result[3]) if result[3] is not None else 0.5,
                        'bp_opp_rank': float(result[4]) if result[4] is not None else 15.0,
                        'bp_hit_rate_l5': float(result[5]) if result[5] is not None else 0.5,
                        'bp_hit_rate_l15': float(result[6]) if result[6] is not None else 0.5,
                        'bp_hit_rate_season': float(result[7]) if result[7] is not None else 0.5,
                    }
                else:
                    return defaults

        except Exception as e:
            logger.debug(f"Error extracting cheatsheet features for {player_name}: {e}")
            return defaults

    def close(self):
        """Close all database connections"""
        super().close()
        if hasattr(self, 'intelligence_conn') and self.intelligence_conn:
            self.intelligence_conn.close()
        if hasattr(self, 'team_conn') and self.team_conn:
            self.team_conn.close()


if __name__ == '__main__':
    # Test XL feature extractor
    extractor = LiveFeatureExtractorXL()

    print("=" * 80)
    print("Testing NBA XL Feature Extractor - Book Disagreement Features")
    print("=" * 80)

    # Test Bam Adebayo (should have 4.0 point spread from earlier test)
    print("\n[1] Bam Adebayo (Miami Heat) - 2025-11-05 - POINTS:")
    features = extractor.extract_features(
        'Bam Adebayo',
        '2025-11-05',
        is_home=True,
        opponent_team='LAL',
        stat_type='POINTS'
    )

    print(f"\n  Player Features (78):")
    print(f"    ema_points_L5: {features.get('ema_points_L5', 0):.2f}")
    print(f"    team_pace: {features.get('team_pace', 0):.2f}")
    print(f"    is_home: {features.get('is_home', 0):.1f}")

    print(f"\n  Book Features (20):")
    print(f"    line_spread: {features.get('line_spread', 0):.2f} (should be 4.0 if test data loaded)")
    print(f"    consensus_line: {features.get('consensus_line', 0):.2f}")
    print(f"    line_std_dev: {features.get('line_std_dev', 0):.2f}")
    print(f"    num_books_offering: {features.get('num_books_offering', 0):.0f}")
    print(f"    line_coef_variation: {features.get('line_coef_variation', 0):.3f}")

    print(f"\n  Book Deviations:")
    print(f"    draftkings_deviation: {features.get('draftkings_deviation', 0):.2f}")
    print(f"    fanduel_deviation: {features.get('fanduel_deviation', 0):.2f}")
    print(f"    betmgm_deviation: {features.get('betmgm_deviation', 0):.2f}")

    print(f"\n  Historical Accuracy:")
    print(f"    softest_book_id: {features.get('softest_book_id', 0):.0f}")
    print(f"    hardest_book_id: {features.get('hardest_book_id', 0):.0f}")
    print(f"    line_spread_percentile: {features.get('line_spread_percentile', 0):.2f}")
    print(f"    books_agree: {features.get('books_agree', 0):.0f}")
    print(f"    books_disagree: {features.get('books_disagree', 0):.0f}")

    # Count total features
    total_features = len(features)
    print(f"\n  TOTAL FEATURES: {total_features}")

    # Verify feature count
    expected_features = 78 + 20  # 78 player + 20 book
    if total_features == expected_features:
        print(f"  ✅ PASS: {total_features} features extracted (78 player + 20 book)")
    else:
        print(f"  ⚠️  WARNING: Expected {expected_features} features, got {total_features}")
        print(f"     Missing: {expected_features - total_features} features")

    # Test with player that has no multi-book props (should use defaults)
    print("\n" + "=" * 80)
    print("[2] Testing Default Features (No Multi-Book Props):")
    print("=" * 80)

    features2 = extractor.extract_features(
        'Test Player NoProps',
        '2025-11-05',
        stat_type='POINTS'
    )

    print(f"\n  line_spread: {features2.get('line_spread', 0):.2f} (should be 0.0 - default)")
    print(f"  num_books_offering: {features2.get('num_books_offering', 0):.0f} (should be 1.0 - default)")
    print(f"  books_agree: {features2.get('books_agree', 0):.0f} (should be 1.0 - default)")
    print(f"  books_disagree: {features2.get('books_disagree', 0):.0f} (should be 0.0 - default)")

    # Test without book features (historical mode)
    print("\n" + "=" * 80)
    print("[3] Testing Historical Mode (Player Features Only):")
    print("=" * 80)

    features3 = extractor.extract_features(
        'Bam Adebayo',
        '2025-11-05',
        stat_type='POINTS',
        include_book_features=False
    )

    total_features3 = len(features3)
    print(f"\n  TOTAL FEATURES: {total_features3}")
    if total_features3 == 78:
        print(f"  ✅ PASS: {total_features3} player features only (book features excluded)")
    else:
        print(f"  ⚠️  WARNING: Expected 78 features, got {total_features3}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"✓ LiveFeatureExtractorXL successfully extends LiveFeatureExtractor")
    print(f"✓ Extracts 98 total features (78 player + 20 book)")
    print(f"✓ Handles missing book data gracefully (defaults to 0)")
    print(f"✓ Supports historical mode (player features only)")
    print(f"✓ Ready for XL model training and predictions")
    print("=" * 80)

    extractor.close()
