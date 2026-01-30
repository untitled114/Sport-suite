"""
Book Disagreement Feature Extractor
====================================
Extracts 23 features related to sportsbook line disagreement.

Features capture market inefficiencies by measuring how different
sportsbooks price the same prop, which signals betting edge opportunities.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class BookFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts 23 book disagreement features from prop lines.

    Feature Groups:
    1. Line Variance (5 features): spread, consensus, std_dev, num_books, coef_variation
    2. Book Deviations (8 features): Individual book deviation from consensus
    3. Line Metadata (7 features): softest/hardest book, agreement signals
    4. Additional (3 features): min/max line, std
    """

    FEATURE_NAMES = (
        # Line variance (5)
        "line_spread",
        "consensus_line",
        "line_std_dev",
        "num_books_offering",
        "line_coef_variation",
        # Book deviations (8)
        "draftkings_deviation",
        "fanduel_deviation",
        "betmgm_deviation",
        "caesars_deviation",
        "bet365_deviation",
        "betrivers_deviation",
        "espnbet_deviation",
        "fanatics_deviation",
        # Line metadata (7)
        "softest_book_id",
        "hardest_book_id",
        "line_spread_percentile",
        "books_agree",
        "books_disagree",
        "softest_vs_consensus",
        "hardest_vs_consensus",
        # Additional (3)
        "min_line",
        "max_line",
        "line_std",
    )

    # Book name normalization
    BOOK_NAME_MAP = {
        "DraftKings": "draftkings",
        "draftkings": "draftkings",
        "FanDuel": "fanduel",
        "fanduel": "fanduel",
        "BetMGM": "betmgm",
        "betmgm": "betmgm",
        "Caesars": "caesars",
        "caesars": "caesars",
        "Bet365": "bet365",
        "bet365": "bet365",
        "BetRivers": "betrivers",
        "betrivers": "betrivers",
        "ESPNBet": "espnbet",
        "espnbet": "espnbet",
        "Fanatics": "fanatics",
        "fanatics": "fanatics",
        "Underdog": "underdog",
        "underdog": "underdog",
    }

    def __init__(self, conn: Any):
        """
        Initialize book feature extractor.

        Args:
            conn: Database connection to nba_intelligence database
        """
        super().__init__(conn, name="BookFeatures")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for all 23 book features."""
        return {
            # Line variance
            "line_spread": 0.0,
            "consensus_line": 0.0,
            "line_std_dev": 0.0,
            "num_books_offering": 1.0,
            "line_coef_variation": 0.0,
            # Book deviations (0 = no data)
            "draftkings_deviation": 0.0,
            "fanduel_deviation": 0.0,
            "betmgm_deviation": 0.0,
            "caesars_deviation": 0.0,
            "bet365_deviation": 0.0,
            "betrivers_deviation": 0.0,
            "espnbet_deviation": 0.0,
            "fanatics_deviation": 0.0,
            # Line metadata
            "softest_book_id": 0.0,
            "hardest_book_id": 0.0,
            "line_spread_percentile": 0.5,
            "books_agree": 0.0,
            "books_disagree": 0.0,
            "softest_vs_consensus": 0.0,
            "hardest_vs_consensus": 0.0,
            # Additional
            "min_line": 0.0,
            "max_line": 0.0,
            "line_std": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract book disagreement features from prop lines.

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: POINTS, REBOUNDS, ASSISTS, or THREES

        Returns:
            Dict with 23 book disagreement features
        """
        game_date_str = self._normalize_date(game_date)

        # Query all book lines for this prop
        query = """
        SELECT book_name, over_line
        FROM nba_props_xl
        WHERE player_name = %s
          AND game_date = %s
          AND stat_type = %s
          AND is_active = true
          AND over_line IS NOT NULL
        """
        df = self._safe_query(query, (player_name, game_date_str, stat_type.upper()))

        if df is None or len(df) == 0:
            return self.get_defaults()

        return self._compute_features(df)

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all book features from lines DataFrame.

        Args:
            df: DataFrame with book_name and over_line columns

        Returns:
            Dict with computed features
        """
        features = self.get_defaults()

        lines = df["over_line"].astype(float)

        # Line variance features
        features["line_spread"] = float(lines.max() - lines.min())
        features["consensus_line"] = float(lines.mean())
        features["line_std_dev"] = float(lines.std()) if len(lines) > 1 else 0.0
        features["num_books_offering"] = float(len(df))
        features["line_coef_variation"] = (
            features["line_std_dev"] / features["consensus_line"]
            if features["consensus_line"] > 0
            else 0.0
        )

        # Additional line features
        features["min_line"] = float(lines.min())
        features["max_line"] = float(lines.max())
        features["line_std"] = features["line_std_dev"]

        # Book deviations
        consensus = features["consensus_line"]
        for _, row in df.iterrows():
            book_name = self.BOOK_NAME_MAP.get(row["book_name"], "")
            if book_name:
                deviation_key = f"{book_name}_deviation"
                if deviation_key in features:
                    features[deviation_key] = float(row["over_line"]) - consensus

        # Softest and hardest book
        softest_idx = lines.idxmin()
        hardest_idx = lines.idxmax()
        softest_book = df.loc[softest_idx, "book_name"]
        hardest_book = df.loc[hardest_idx, "book_name"]

        features["softest_book_id"] = self._encode_book_name(softest_book)
        features["hardest_book_id"] = self._encode_book_name(hardest_book)

        # Agreement signals
        spread = features["line_spread"]
        features["books_agree"] = 1.0 if spread < 0.5 else 0.0
        features["books_disagree"] = 1.0 if spread >= 2.0 else 0.0

        # Distance from consensus
        features["softest_vs_consensus"] = float(lines.min()) - consensus
        features["hardest_vs_consensus"] = float(lines.max()) - consensus

        # Line spread percentile (0.5 default, updated if historical data available)
        features["line_spread_percentile"] = self._compute_spread_percentile(spread, stat_type=None)

        return features

    def _compute_spread_percentile(
        self,
        spread: float,
        stat_type: Optional[str] = None,
    ) -> float:
        """
        Compute percentile of this spread vs historical spreads.

        Args:
            spread: Current line spread
            stat_type: Optional stat type for filtering

        Returns:
            Percentile (0.0 to 1.0)
        """
        # Typical spread distribution based on historical data
        # These are approximate percentiles
        if spread < 0.5:
            return 0.15
        elif spread < 1.0:
            return 0.35
        elif spread < 1.5:
            return 0.55
        elif spread < 2.0:
            return 0.70
        elif spread < 2.5:
            return 0.82
        elif spread < 3.0:
            return 0.90
        else:
            return 0.95

    def extract_from_lines_df(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from a pre-queried lines DataFrame.

        Useful when lines are already available from line shopping.

        Args:
            df: DataFrame with book_name and over_line columns

        Returns:
            Dict with 23 book features
        """
        if df is None or len(df) == 0:
            return self.get_defaults()
        return self._compute_features(df)
