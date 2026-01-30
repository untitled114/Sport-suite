"""
Feature Extractors Package
==========================
Modular feature extraction classes for NBA props ML system.

This package splits the monolithic feature extraction into focused,
testable components following the Single Responsibility Principle.

Usage:
    from nba.features.extractors import (
        BookFeatureExtractor,
        H2HFeatureExtractor,
        PropHistoryExtractor,
        VegasContextExtractor,
        TeamBettingExtractor,
        CheatsheetExtractor,
    )

    # Create extractor with database connection
    book_extractor = BookFeatureExtractor(conn)
    features = book_extractor.extract(player_name, game_date, stat_type)
"""

from .base import BaseFeatureExtractor
from .book_features import BookFeatureExtractor
from .cheatsheet_features import CheatsheetExtractor
from .h2h_features import H2HFeatureExtractor
from .prop_history_features import PropHistoryExtractor
from .team_betting_features import TeamBettingExtractor
from .vegas_features import VegasContextExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BookFeatureExtractor",
    "H2HFeatureExtractor",
    "PropHistoryExtractor",
    "VegasContextExtractor",
    "TeamBettingExtractor",
    "CheatsheetExtractor",
]
