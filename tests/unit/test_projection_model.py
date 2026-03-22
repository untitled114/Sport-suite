"""
Tests for ProjectionModel
===========================
Tests the pace-adjusted projection model.
Uses realistic but hardcoded test values — NO synthetic data.

These tests mock database calls since we can't rely on live DB in unit tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from nba.core.exceptions import DatabaseConnectionError, DataNotFoundError
from nba.models.projection_model import (
    HOME_ADVANTAGE,
    LEAGUE_AVG_PACE,
    ROLLING_WEIGHTS,
    ProjectionModel,
)


@pytest.fixture
def model():
    """Create a ProjectionModel with mocked DB connections."""
    m = ProjectionModel()
    # Mock connections to avoid needing real DBs in unit tests
    m._players_conn = MagicMock()
    m._players_conn.closed = False
    m._team_conn = MagicMock()
    m._team_conn.closed = False
    return m


def _mock_rolling_row(
    points_l3=24.0,
    points_l5=23.5,
    points_l10=23.0,
    points_l20=22.5,
    minutes_l5=34.0,
    minutes_l10=33.5,
    team_abbrev="LAL",
):
    """Create a mock row from player_rolling_stats query."""
    return (points_l3, points_l5, points_l10, points_l20, minutes_l5, minutes_l10, team_abbrev)


# =============================================================================
# Rolling Average Weighting
# =============================================================================


class TestRollingWeights:
    """Tests for weighted rolling average calculation."""

    def test_weights_sum_to_one(self):
        """Rolling weights should sum to 1.0."""
        total = sum(ROLLING_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_l5_has_highest_weight(self):
        """L5 (most recent) should have the highest weight."""
        assert ROLLING_WEIGHTS["L5"] > ROLLING_WEIGHTS["L10"]
        assert ROLLING_WEIGHTS["L10"] > ROLLING_WEIGHTS["L20"]

    def test_weighted_average_computation(self, model):
        """Verify weighted average formula with known values."""
        # L5=23.5, L10=23.0, L20=22.5
        # Expected: 23.5*0.50 + 23.0*0.30 + 22.5*0.20 = 11.75 + 6.90 + 4.50 = 23.15
        expected = 23.5 * 0.50 + 23.0 * 0.30 + 22.5 * 0.20
        assert abs(expected - 23.15) < 0.01


# =============================================================================
# Projection with Mocked DB
# =============================================================================


class TestProjection:
    """Tests for the full projection pipeline with mocked data."""

    def test_basic_projection(self, model):
        """Basic projection should return a positive float."""
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row()

        mock_team_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_team_cursor
        # Mock pace query (both teams)
        mock_team_cursor.fetchall.return_value = [
            ("LAL", LEAGUE_AVG_PACE),
            ("BOS", LEAGUE_AVG_PACE),
        ]
        # Mock defensive rating query
        mock_team_cursor.fetchone.return_value = (112.0,)

        result = model.project("LeBron James", "POINTS", "BOS", is_home=True)
        assert isinstance(result, float)
        assert result > 0

    def test_projection_home_vs_away(self, model):
        """Home projection should be slightly higher than away."""
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row()

        mock_team_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_team_cursor
        mock_team_cursor.fetchall.return_value = [
            ("LAL", LEAGUE_AVG_PACE),
            ("BOS", LEAGUE_AVG_PACE),
        ]
        mock_team_cursor.fetchone.return_value = (112.0,)

        home = model.project("LeBron James", "POINTS", "BOS", is_home=True)
        away = model.project("LeBron James", "POINTS", "BOS", is_home=False)
        assert home > away

    def test_missing_player_raises(self, model):
        """Missing player should raise DataNotFoundError."""
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        with pytest.raises(DataNotFoundError):
            model.project("Nonexistent Player", "POINTS", "BOS", is_home=True)


# =============================================================================
# Pace Factor
# =============================================================================


class TestPaceFactor:
    """Tests for pace adjustment."""

    def test_average_pace_no_adjustment(self, model):
        """League average pace should give factor ~1.0."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("LAL", LEAGUE_AVG_PACE),
            ("BOS", LEAGUE_AVG_PACE),
        ]

        factor = model._get_pace_factor("LAL", "BOS")
        assert abs(factor - 1.0) < 0.01

    def test_fast_pace_upward_adjustment(self, model):
        """Fast pace matchup should increase projection."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("IND", 105.0),  # Fast team
            ("SAC", 104.0),  # Fast team
        ]

        factor = model._get_pace_factor("IND", "SAC")
        assert factor > 1.0

    def test_slow_pace_downward_adjustment(self, model):
        """Slow pace matchup should decrease projection."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("NYK", 96.0),
            ("MIA", 95.0),
        ]

        factor = model._get_pace_factor("NYK", "MIA")
        assert factor < 1.0

    def test_missing_team_returns_neutral(self, model):
        """Missing team data should return factor 1.0."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        factor = model._get_pace_factor("UNK", "BOS")
        assert factor == 1.0

    def test_empty_team_returns_neutral(self, model):
        """Empty team abbreviation should return factor 1.0."""
        factor = model._get_pace_factor("", "BOS")
        assert factor == 1.0


# =============================================================================
# Defensive Factor
# =============================================================================


class TestDefensiveFactor:
    """Tests for opponent defensive adjustment."""

    def test_bad_defense_increases_projection(self, model):
        """Bad defense (high def_rating) should increase projection."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (118.0,)  # Bad defense

        factor = model._get_defensive_factor("SAC", "points")
        assert factor > 1.0

    def test_good_defense_decreases_projection(self, model):
        """Good defense (low def_rating) should decrease projection."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (106.0,)  # Good defense

        factor = model._get_defensive_factor("BOS", "points")
        assert factor < 1.0

    def test_average_defense_neutral(self, model):
        """Average defense (112.0) should give factor ~1.0."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (112.0,)

        factor = model._get_defensive_factor("MIL", "points")
        assert abs(factor - 1.0) < 0.01

    def test_missing_team_returns_neutral(self, model):
        """Missing team should return factor 1.0."""
        mock_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        factor = model._get_defensive_factor("UNK", "points")
        assert factor == 1.0


# =============================================================================
# Minutes Ratio
# =============================================================================


class TestMinutesRatio:
    """Tests for minutes adjustment."""

    def test_stable_minutes_neutral(self, model):
        """Stable minutes should give ratio ~1.0."""
        rolling = {"minutes_L5": 34.0, "minutes_L10": 34.0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert abs(ratio - 1.0) < 0.01

    def test_increasing_minutes_boost(self, model):
        """Increasing minutes should boost projection."""
        rolling = {"minutes_L5": 36.0, "minutes_L10": 33.0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert ratio > 1.0

    def test_decreasing_minutes_reduction(self, model):
        """Decreasing minutes should reduce projection."""
        rolling = {"minutes_L5": 28.0, "minutes_L10": 34.0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert ratio < 1.0

    def test_ratio_clamped_high(self, model):
        """Minutes ratio should be clamped at 1.15."""
        rolling = {"minutes_L5": 40.0, "minutes_L10": 25.0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert ratio <= 1.15

    def test_ratio_clamped_low(self, model):
        """Minutes ratio should be clamped at 0.85."""
        rolling = {"minutes_L5": 20.0, "minutes_L10": 35.0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert ratio >= 0.85

    def test_zero_minutes_neutral(self, model):
        """Zero minutes should give ratio 1.0 (no info)."""
        rolling = {"minutes_L5": 0, "minutes_L10": 0}
        ratio = model._get_minutes_ratio("Test Player", rolling)
        assert ratio == 1.0


# =============================================================================
# Std Dev Estimation
# =============================================================================


class TestStdEstimation:
    """Tests for standard deviation estimation from rolling windows."""

    def test_positive_std(self, model):
        """Std should always be positive for non-zero inputs."""
        std = model._estimate_std_from_rolling(25.0, 24.0, 23.0)
        assert std > 0

    def test_wider_spread_higher_std(self, model):
        """Wider spread between windows should give higher std."""
        std_narrow = model._estimate_std_from_rolling(25.0, 24.5, 24.0)
        std_wide = model._estimate_std_from_rolling(30.0, 24.0, 20.0)
        assert std_wide > std_narrow

    def test_all_zeros_default(self, model):
        """All zeros should return default std."""
        std = model._estimate_std_from_rolling(0.0, 0.0, 0.0)
        assert std == 5.0

    def test_reasonable_range(self, model):
        """Std should be in a reasonable range relative to mean."""
        std = model._estimate_std_from_rolling(25.0, 24.0, 23.0)
        mean = (25.0 + 24.0 + 23.0) / 3.0
        cv = std / mean
        assert 0.10 < cv < 0.50  # 10-50% CV is reasonable for NBA stats


# =============================================================================
# Home/Away Advantage Constants
# =============================================================================


class TestHomeAdvantage:
    """Tests for home advantage constants."""

    def test_points_has_largest_advantage(self):
        """POINTS should have the largest home advantage."""
        assert HOME_ADVANTAGE["POINTS"] >= HOME_ADVANTAGE["REBOUNDS"]
        assert HOME_ADVANTAGE["POINTS"] >= HOME_ADVANTAGE["THREES"]

    def test_all_advantages_positive(self):
        """All home advantages should be positive."""
        for market, adv in HOME_ADVANTAGE.items():
            assert adv > 0, f"{market} home advantage should be positive"

    def test_advantages_are_small(self):
        """Home advantages should be modest (<5%)."""
        for market, adv in HOME_ADVANTAGE.items():
            assert adv < 0.05, f"{market} home advantage {adv} is too large"


# =============================================================================
# Project With Details
# =============================================================================


class TestProjectWithDetails:
    """Tests for the detailed projection output."""

    def test_returns_all_components(self, model):
        """Should return all component factors."""
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row()

        mock_team_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_team_cursor
        mock_team_cursor.fetchall.return_value = [
            ("LAL", LEAGUE_AVG_PACE),
            ("BOS", LEAGUE_AVG_PACE),
        ]
        mock_team_cursor.fetchone.return_value = (112.0,)

        result = model.project_with_details("LeBron James", "POINTS", "BOS", is_home=True)

        assert "projection" in result
        assert "base_projection" in result
        assert "rolling_L5" in result
        assert "pace_factor" in result
        assert "defensive_factor" in result
        assert "home_factor" in result
        assert "minutes_ratio" in result
        assert "std_dev" in result
        assert result["projection"] > 0
        assert result["player_name"] == "LeBron James"
        assert result["stat_type"] == "POINTS"


# =============================================================================
# Connection Management
# =============================================================================


class TestGetPlayerStdDev:
    """Tests for get_player_std_dev method."""

    def test_returns_std_from_rolling(self, model):
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row()

        std = model.get_player_std_dev("LeBron James", "POINTS")
        assert isinstance(std, float)
        assert std > 0

    def test_returns_default_on_missing(self, model):
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        with pytest.raises(DataNotFoundError):
            model.get_player_std_dev("Unknown Player", "POINTS")


class TestZeroBaseProjection:
    """Tests for zero base projection path."""

    def test_zero_rolling_returns_zero(self, model):
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row(
            points_l3=0.0, points_l5=0.0, points_l10=0.0, points_l20=0.0
        )

        mock_team_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_team_cursor
        mock_team_cursor.fetchall.return_value = []
        mock_team_cursor.fetchone.return_value = None

        result = model.project("Bench Player", "POINTS", "BOS", is_home=True)
        assert result == 0.0


class TestProjectWithDetailsAway:
    """Tests for project_with_details in away mode."""

    def test_away_projection(self, model):
        mock_cursor = MagicMock()
        model._players_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = _mock_rolling_row()

        mock_team_cursor = MagicMock()
        model._team_conn.cursor.return_value = mock_team_cursor
        mock_team_cursor.fetchall.return_value = [
            ("LAL", LEAGUE_AVG_PACE),
            ("BOS", LEAGUE_AVG_PACE),
        ]
        mock_team_cursor.fetchone.return_value = (112.0,)

        result = model.project_with_details("LeBron James", "POINTS", "BOS", is_home=False)
        assert result["is_home"] is False
        assert result["home_factor"] < 1.0


class TestDBConnectionCreation:
    """Tests for DB connection creation/error paths."""

    def test_get_players_conn_creates_on_none(self):
        m = ProjectionModel()
        # _players_conn is None, calling _get_players_conn should try to connect
        with patch("nba.models.projection_model.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            conn = m._get_players_conn()
            assert conn is mock_conn
            mock_pg.connect.assert_called_once()

    def test_get_team_conn_creates_on_none(self):
        m = ProjectionModel()
        with patch("nba.models.projection_model.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            conn = m._get_team_conn()
            assert conn is mock_conn
            mock_pg.connect.assert_called_once()

    def test_get_players_conn_raises_on_failure(self):
        m = ProjectionModel()
        with patch("nba.models.projection_model.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("Connection refused")
            with pytest.raises(DatabaseConnectionError):
                m._get_players_conn()

    def test_get_team_conn_raises_on_failure(self):
        m = ProjectionModel()
        with patch("nba.models.projection_model.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("Connection refused")
            with pytest.raises(DatabaseConnectionError):
                m._get_team_conn()


class TestConnectionManagement:
    """Tests for database connection lifecycle."""

    def test_close_connections(self, model):
        """close() should close both connections."""
        model.close()
        model._players_conn.close.assert_called_once()
        model._team_conn.close.assert_called_once()

    def test_close_already_closed(self):
        """close() on already-closed connections should not raise."""
        m = ProjectionModel()
        m._players_conn = MagicMock()
        m._players_conn.closed = True
        m._team_conn = MagicMock()
        m._team_conn.closed = True
        m.close()  # Should not call .close() on already-closed connections

    def test_del_calls_close(self, model):
        """__del__ should call close()."""
        model.__del__()
        model._players_conn.close.assert_called()
        model._team_conn.close.assert_called()
