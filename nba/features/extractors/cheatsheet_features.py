"""
BettingPros Cheatsheet Feature Extractor
========================================
Extracts 8 features from BettingPros cheatsheet data.

Features capture expert consensus and projection data
that provides additional signal for prop predictions.
"""

import logging
from typing import Any, Dict, Optional

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class CheatsheetExtractor(BaseFeatureExtractor):
    """
    Extracts 8 BettingPros cheatsheet features.

    Features:
    1. Expert projections: Average expert projection
    2. Consensus: Number of experts agreeing
    3. Edge signals: Projection vs line differences
    4. Confidence: Expert confidence level
    """

    FEATURE_NAMES = (
        "bp_projection",
        "bp_projection_vs_line",
        "bp_expert_count",
        "bp_expert_consensus",
        "bp_over_pct",
        "bp_under_pct",
        "bp_grade",
        "bp_value_rating",
    )

    def __init__(self, conn: Any):
        """
        Initialize cheatsheet extractor.

        Args:
            conn: Database connection to nba_intelligence database
        """
        super().__init__(conn, name="Cheatsheet")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for cheatsheet features."""
        return {
            "bp_projection": 0.0,
            "bp_projection_vs_line": 0.0,
            "bp_expert_count": 0.0,
            "bp_expert_consensus": 0.5,
            "bp_over_pct": 0.5,
            "bp_under_pct": 0.5,
            "bp_grade": 0.5,  # Neutral grade
            "bp_value_rating": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        line: float = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract cheatsheet features for a prop.

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: POINTS, REBOUNDS, ASSISTS, or THREES
            line: Current prop line (for edge calculation)

        Returns:
            Dict with 8 cheatsheet features
        """
        game_date_str = self._normalize_date(game_date)

        # Query cheatsheet data
        query = """
        SELECT
            projection,
            expert_count,
            over_percentage,
            under_percentage,
            grade,
            value_rating
        FROM bp_cheatsheet
        WHERE player_name = %s
          AND game_date = %s
          AND stat_type = %s
        LIMIT 1
        """
        df = self._safe_query(query, (player_name, game_date_str, stat_type.upper()))

        if df is None or len(df) == 0:
            return self.get_defaults()

        return self._compute_features(df, line)

    def _compute_features(self, df, line: Optional[float] = None) -> Dict[str, float]:
        """
        Compute cheatsheet features from query result.

        Args:
            df: DataFrame with cheatsheet data
            line: Optional line for edge calculation

        Returns:
            Dict with computed features
        """
        features = self.get_defaults()

        row = df.iloc[0]

        # Projection
        projection = self._safe_float(row.get("projection"))
        features["bp_projection"] = projection

        if line and projection > 0:
            features["bp_projection_vs_line"] = projection - line

        # Expert consensus
        features["bp_expert_count"] = self._safe_float(row.get("expert_count"))

        over_pct = self._safe_float(row.get("over_percentage"), 0.5)
        under_pct = self._safe_float(row.get("under_percentage"), 0.5)
        features["bp_over_pct"] = over_pct
        features["bp_under_pct"] = under_pct

        # Expert consensus (how much experts agree)
        features["bp_expert_consensus"] = max(over_pct, under_pct)

        # Grade (convert letter to numeric if needed)
        grade = row.get("grade")
        features["bp_grade"] = self._encode_grade(grade)

        # Value rating
        features["bp_value_rating"] = self._safe_float(row.get("value_rating"))

        return features

    def _encode_grade(self, grade: Any) -> float:
        """
        Encode letter grade to numeric value.

        Args:
            grade: Letter grade (A, B, C, D, F) or numeric

        Returns:
            Numeric grade (0.0 to 1.0)
        """
        if grade is None:
            return 0.5

        if isinstance(grade, (int, float)):
            return self._safe_float(grade)

        grade_map = {
            "A+": 1.0,
            "A": 0.95,
            "A-": 0.90,
            "B+": 0.85,
            "B": 0.80,
            "B-": 0.75,
            "C+": 0.70,
            "C": 0.65,
            "C-": 0.60,
            "D+": 0.55,
            "D": 0.50,
            "D-": 0.45,
            "F": 0.30,
        }

        return grade_map.get(str(grade).upper(), 0.5)
