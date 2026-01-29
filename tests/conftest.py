"""
Pytest Configuration and Fixtures
=================================
Provides database mocking, sample data, and common test utilities.

Note: Project paths are configured via pyproject.toml [tool.pytest.ini_options] pythonpath.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# DATABASE MOCKING FIXTURES
# =============================================================================


class MockCursor:
    """Mock database cursor that returns predefined results."""

    def __init__(self, results: Optional[List] = None):
        self.results = results or []
        self.description = None
        self._index = 0

    def execute(self, query: str, params: tuple = None):
        """Mock execute - does nothing but stores query for inspection."""
        self.last_query = query
        self.last_params = params

    def fetchone(self):
        """Return first result or None."""
        if self.results:
            return self.results[0]
        return None

    def fetchall(self):
        """Return all results."""
        return self.results

    def fetchmany(self, size: int = 1):
        """Return up to size results."""
        return self.results[:size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnection:
    """Mock database connection."""

    def __init__(self, cursor_results: Optional[Dict[str, List]] = None):
        self.cursor_results = cursor_results or {}
        self._cursors = []

    def cursor(self):
        cursor = MockCursor()
        self._cursors.append(cursor)
        return cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def mock_db_connection():
    """Provides a mock database connection."""
    return MockConnection()


@pytest.fixture
def mock_player_db():
    """Mock player database with sample player data."""
    conn = MockConnection()
    return conn


@pytest.fixture
def mock_games_db():
    """Mock games database with sample game data."""
    conn = MockConnection()
    return conn


@pytest.fixture
def mock_team_db():
    """Mock team database with sample team data."""
    conn = MockConnection()
    return conn


@pytest.fixture
def mock_intelligence_db():
    """Mock intelligence database with sample prop data."""
    conn = MockConnection()
    return conn


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_player_profile():
    """Sample player profile data."""
    return {
        "player_id": 201566,
        "full_name": "Russell Westbrook",
        "first_name": "Russell",
        "last_name": "Westbrook",
        "team_abbreviation": "DEN",
        "position": "G",
        "height": "6-3",
        "weight": 200,
        "birth_date": "1988-11-12",
        "is_active": True,
    }


@pytest.fixture
def sample_game_logs():
    """Sample player game logs for testing rolling stats."""
    base_date = date(2025, 1, 15)
    logs = []
    for i in range(20):
        game_date = base_date - timedelta(days=i * 2)
        logs.append(
            {
                "player_id": 201566,
                "game_date": game_date,
                "team_abbreviation": "DEN",
                "opponent": "LAL" if i % 2 == 0 else "GSW",
                "is_home": i % 2 == 0,
                "pts": 20 + np.random.randint(-5, 10),
                "reb": 5 + np.random.randint(-2, 4),
                "ast": 7 + np.random.randint(-3, 5),
                "fg3m": 2 + np.random.randint(-2, 3),
                "stl": 1 + np.random.randint(0, 2),
                "blk": np.random.randint(0, 2),
                "tov": 3 + np.random.randint(-1, 3),
                "min": 30 + np.random.randint(-5, 8),
                "fgm": 8 + np.random.randint(-3, 4),
                "fga": 18 + np.random.randint(-4, 6),
            }
        )
    return pd.DataFrame(logs)


@pytest.fixture
def sample_team_stats():
    """Sample team statistics."""
    return {
        "team_abbreviation": "LAL",
        "pace": 100.5,
        "offensive_rating": 115.2,
        "defensive_rating": 112.8,
        "pts_allowed_per_game": 112.5,
        "reb_allowed_per_game": 44.2,
        "ast_allowed_per_game": 25.8,
    }


@pytest.fixture
def sample_prop_lines():
    """Sample prop lines from multiple sportsbooks."""
    return [
        {
            "book_name": "draftkings",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 25.5,
        },
        {
            "book_name": "fanduel",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 26.0,
        },
        {
            "book_name": "betmgm",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 25.5,
        },
        {
            "book_name": "caesars",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 26.5,
        },
        {
            "book_name": "betrivers",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 25.5,
        },
        {
            "book_name": "espnbet",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 26.0,
        },
        {
            "book_name": "underdog",
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "over_line": 24.5,
        },
    ]


@pytest.fixture
def sample_book_features():
    """Sample book features for a prop."""
    return {
        "line_spread": 2.0,
        "consensus_line": 25.7,
        "line_std_dev": 0.63,
        "num_books_offering": 7,
        "line_coef_variation": 0.025,
        "draftkings_deviation": -0.2,
        "fanduel_deviation": 0.3,
        "betmgm_deviation": -0.2,
        "caesars_deviation": 0.8,
        "bet365_deviation": 0.0,
        "betrivers_deviation": -0.2,
        "espnbet_deviation": 0.3,
        "fanatics_deviation": 0.0,
        "softest_book_id": 7,  # underdog
        "hardest_book_id": 4,  # caesars
        "line_spread_percentile": 0.75,
        "books_agree": 0.0,
        "books_disagree": 1.0,
        "softest_vs_consensus": -1.2,
        "hardest_vs_consensus": 0.8,
        "min_line": 24.5,
        "max_line": 26.5,
        "line_std": 0.63,
    }


@pytest.fixture
def sample_prediction_output():
    """Sample model prediction output."""
    return {
        "player_name": "LeBron James",
        "stat_type": "POINTS",
        "side": "OVER",
        "prediction": 28.5,
        "p_over": 0.72,
        "confidence": "HIGH",
        "filter_tier": "STAR_V3",
        "consensus_line": 25.7,
        "line_spread": 2.0,
        "num_books": 7,
        "opponent_team": "GSW",
        "is_home": True,
        "best_book": "underdog",
        "best_line": 24.5,
        "edge": 4.0,
        "edge_pct": 16.3,
    }


# =============================================================================
# FEATURE EXTRACTION FIXTURES
# =============================================================================


@pytest.fixture
def sample_rolling_stats():
    """Sample rolling statistics for a player."""
    return {
        "ema_points_L3": 24.5,
        "ema_points_L5": 23.8,
        "ema_points_L10": 22.9,
        "ema_points_L20": 22.1,
        "ema_rebounds_L3": 7.2,
        "ema_rebounds_L5": 6.8,
        "ema_rebounds_L10": 6.5,
        "ema_rebounds_L20": 6.3,
        "ema_assists_L3": 8.1,
        "ema_assists_L5": 7.9,
        "ema_assists_L10": 7.5,
        "ema_assists_L20": 7.2,
        "ema_threes_L3": 2.3,
        "ema_threes_L5": 2.1,
        "ema_threes_L10": 2.0,
        "ema_threes_L20": 1.9,
        "ema_minutes_L5": 35.2,
        "ema_fg_pct_L5": 0.52,
    }


@pytest.fixture
def sample_feature_vector():
    """Sample 102-dimension feature vector."""
    # Create a feature vector with realistic values
    np.random.seed(42)
    features = {
        # Player features (78)
        "is_home": 1.0,
        "ema_points_L3": 24.5,
        "ema_points_L5": 23.8,
        "ema_points_L10": 22.9,
        "ema_points_L20": 22.1,
        "ema_rebounds_L3": 7.2,
        "ema_rebounds_L5": 6.8,
        "ema_rebounds_L10": 6.5,
        "ema_rebounds_L20": 6.3,
        "ema_assists_L3": 8.1,
        "ema_assists_L5": 7.9,
        "ema_assists_L10": 7.5,
        "ema_assists_L20": 7.2,
        "ema_threes_L3": 2.3,
        "ema_threes_L5": 2.1,
        "ema_threes_L10": 2.0,
        "ema_threes_L20": 1.9,
        "ema_steals_L5": 1.2,
        "ema_blocks_L5": 0.5,
        "ema_turnovers_L5": 3.1,
        "ema_minutes_L5": 35.2,
        "ema_fg_pct_L5": 0.52,
        "team_pace": 100.5,
        "team_off_rating": 115.2,
        "team_def_rating": 112.8,
        "opp_pace": 98.7,
        "opp_def_rating": 110.5,
        "opp_pts_allowed": 112.5,
        "rest_days": 2,
        "is_back_to_back": 0,
        "games_played_season": 45,
        "season_phase": 0.6,
    }
    # Fill remaining with random values
    for i in range(len(features), 102):
        features[f"feature_{i}"] = np.random.randn()
    return features


# =============================================================================
# MODEL FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """Mock LightGBM model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([25.5])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def mock_calibrator():
    """Mock isotonic calibrator."""
    calibrator = MagicMock()
    calibrator.transform.return_value = np.array([0.68])
    return calibrator


@pytest.fixture
def mock_imputer():
    """Mock sklearn imputer."""
    imputer = MagicMock()
    imputer.transform.return_value = np.zeros((1, 102))
    return imputer


@pytest.fixture
def mock_scaler():
    """Mock sklearn scaler."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 102))
    return scaler


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("BETTINGPROS_API_KEY", "test_api_key")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_mock_cursor_with_data(data: List[tuple]) -> MockCursor:
    """Create a mock cursor that returns specified data."""
    cursor = MockCursor(results=data)
    return cursor


def assert_features_valid(features: Dict[str, Any], expected_count: int = 102):
    """Assert that a feature dictionary is valid."""
    assert features is not None, "Features should not be None"
    assert (
        len(features) >= expected_count
    ), f"Expected {expected_count} features, got {len(features)}"
    for key, value in features.items():
        assert (
            value is not None or np.isnan(value) if isinstance(value, float) else True
        ), f"Feature {key} has invalid value"
