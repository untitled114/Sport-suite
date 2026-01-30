"""
Base Feature Extractor
======================
Abstract base class for all feature extractors.

Provides common functionality for database access, logging,
and default value handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All feature extractors should inherit from this class and implement:
    - extract(): Main extraction method
    - get_defaults(): Returns default values for all features
    - FEATURE_NAMES: Class attribute listing all feature names

    Attributes:
        conn: Database connection (psycopg2 or similar)
        name: Extractor name for logging
    """

    # Subclasses must define this
    FEATURE_NAMES: tuple = ()

    def __init__(self, conn: Any, name: str = None):
        """
        Initialize the feature extractor.

        Args:
            conn: Database connection object
            name: Optional extractor name for logging (defaults to class name)
        """
        self.conn = conn
        self.name = name or self.__class__.__name__

    @abstractmethod
    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract features for a player/game/stat combination.

        Args:
            player_name: Player's full name
            game_date: Game date (datetime or string)
            stat_type: Stat type (POINTS, REBOUNDS, etc.)
            **kwargs: Additional context (opponent_team, is_home, etc.)

        Returns:
            Dict mapping feature names to float values
        """
        pass

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> Dict[str, float]:
        """
        Get default values for all features.

        Returns:
            Dict mapping feature names to default float values
        """
        pass

    def _safe_query(
        self,
        query: str,
        params: tuple,
        columns: list = None,
    ) -> Optional[pd.DataFrame]:
        """
        Execute a query with error handling.

        Args:
            query: SQL query string
            params: Query parameters
            columns: Optional column names for result DataFrame

        Returns:
            DataFrame with results, or None on error
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df if len(df) > 0 else None
        except Exception as e:
            logger.debug(f"{self.name}: Query failed: {e}")
            return None

    def _normalize_date(self, game_date: Any) -> str:
        """
        Normalize game_date to string format YYYY-MM-DD.

        Args:
            game_date: Date as string, datetime, or pd.Timestamp

        Returns:
            Date string in YYYY-MM-DD format
        """
        if isinstance(game_date, str):
            return game_date[:10]  # Handle 'YYYY-MM-DD HH:MM:SS' format
        elif hasattr(game_date, "strftime"):
            return game_date.strftime("%Y-%m-%d")
        else:
            return str(game_date)[:10]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float.

        Args:
            value: Value to convert
            default: Default if conversion fails

        Returns:
            Float value or default
        """
        if value is None:
            return default
        try:
            result = float(value)
            if pd.isna(result):
                return default
            return result
        except (ValueError, TypeError):
            return default

    def _encode_book_name(self, book_name: str) -> float:
        """
        Encode book name as numeric ID for ML features.

        Args:
            book_name: Sportsbook name

        Returns:
            Numeric encoding (0-10 scale)
        """
        book_encoding = {
            "draftkings": 1.0,
            "fanduel": 2.0,
            "betmgm": 3.0,
            "caesars": 4.0,
            "bet365": 5.0,
            "betrivers": 6.0,
            "espnbet": 7.0,
            "fanatics": 8.0,
            "underdog": 9.0,
            "prizepicks": 10.0,
        }
        return book_encoding.get(book_name.lower(), 0.0) if book_name else 0.0

    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Validate extracted features and fill missing with defaults.

        Args:
            features: Extracted feature dict

        Returns:
            Validated feature dict with all expected features
        """
        defaults = self.get_defaults()
        result = defaults.copy()

        for key, value in features.items():
            if key in result:
                result[key] = self._safe_float(value, defaults.get(key, 0.0))

        return result
