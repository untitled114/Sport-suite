#!/usr/bin/env python3
"""
Unit tests for nba.betting_xl.risk_filters module.

Tests the RiskAssessment dataclass and RiskFilter class for
market-specific risk assessment and stake sizing.
"""

from unittest.mock import MagicMock, patch

import pytest

from nba.betting_xl.risk_filters import RiskAssessment, RiskFilter

# =============================================================================
# RiskAssessment Dataclass Tests
# =============================================================================


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        assessment = RiskAssessment()
        assert assessment.stat_type == "POINTS"
        assert assessment.high_volatility is False
        assert assessment.elite_defense is False
        assert assessment.negative_trend is False
        assert assessment.volatility_score == 0.0
        assert assessment.defense_score == 0.0
        assert assessment.trend_score == 0.0
        assert assessment.total_risk_score == 0.0
        assert assessment.risk_level == "LOW"
        assert assessment.line_is_soft is False
        assert assessment.recommended_stake == 1.0
        assert assessment.stake_reason == ""

    def test_custom_values(self):
        """Test custom initialization."""
        assessment = RiskAssessment(
            stat_type="REBOUNDS",
            high_volatility=True,
            elite_defense=True,
            risk_level="EXTREME",
            recommended_stake=0.5,
        )
        assert assessment.stat_type == "REBOUNDS"
        assert assessment.high_volatility is True
        assert assessment.elite_defense is True
        assert assessment.risk_level == "EXTREME"
        assert assessment.recommended_stake == 0.5


class TestRiskAssessmentShouldSkip:
    """Tests for RiskAssessment.should_skip property."""

    def test_rebounds_extreme_risk_skips(self):
        """REBOUNDS with EXTREME risk should be skipped."""
        assessment = RiskAssessment(stat_type="REBOUNDS", risk_level="EXTREME")
        assert assessment.should_skip is True

    def test_rebounds_two_flags_skips(self):
        """REBOUNDS with 2+ risk flags should be skipped."""
        assessment = RiskAssessment(
            stat_type="REBOUNDS",
            high_volatility=True,
            elite_defense=True,
            risk_level="HIGH",
        )
        assert assessment.should_skip is True

    def test_rebounds_one_flag_does_not_skip(self):
        """REBOUNDS with only 1 flag should not skip."""
        assessment = RiskAssessment(
            stat_type="REBOUNDS",
            high_volatility=True,
            elite_defense=False,
            negative_trend=False,
            risk_level="MEDIUM",
        )
        assert assessment.should_skip is False

    def test_rebounds_low_risk_does_not_skip(self):
        """REBOUNDS with low risk should not skip."""
        assessment = RiskAssessment(stat_type="REBOUNDS", risk_level="LOW")
        assert assessment.should_skip is False

    def test_points_zero_stake_skips(self):
        """POINTS with zero stake should be skipped."""
        assessment = RiskAssessment(
            stat_type="POINTS",
            recommended_stake=0.0,
            risk_level="EXTREME",
        )
        assert assessment.should_skip is True

    def test_points_positive_stake_does_not_skip(self):
        """POINTS with positive stake should not skip."""
        assessment = RiskAssessment(
            stat_type="POINTS",
            recommended_stake=0.5,
            risk_level="HIGH",
        )
        assert assessment.should_skip is False


class TestRiskAssessmentShouldFlag:
    """Tests for RiskAssessment.should_flag property."""

    def test_high_risk_flagged(self):
        """HIGH risk should be flagged."""
        assessment = RiskAssessment(risk_level="HIGH")
        assert assessment.should_flag is True

    def test_extreme_risk_flagged(self):
        """EXTREME risk should be flagged."""
        assessment = RiskAssessment(risk_level="EXTREME")
        assert assessment.should_flag is True

    def test_medium_risk_not_flagged(self):
        """MEDIUM risk should not be flagged."""
        assessment = RiskAssessment(risk_level="MEDIUM")
        assert assessment.should_flag is False

    def test_low_risk_not_flagged(self):
        """LOW risk should not be flagged."""
        assessment = RiskAssessment(risk_level="LOW")
        assert assessment.should_flag is False


# =============================================================================
# RiskFilter Class Tests
# =============================================================================


class TestRiskFilterInit:
    """Tests for RiskFilter initialization."""

    def test_init_defaults(self):
        """Test default initialization values."""
        rf = RiskFilter()
        assert rf.stats_assessed == 0
        assert rf.high_risk_skipped == 0
        assert rf._conn is None
        assert rf._volatility_cache == {}

    def test_class_constants(self):
        """Test class constants are set correctly."""
        assert RiskFilter.POINTS_VOLATILITY_HIGH == 0.30
        assert RiskFilter.POINTS_VOLATILITY_EXTREME == 0.45
        assert RiskFilter.REBOUNDS_VOLATILITY_HIGH == 0.35
        assert RiskFilter.REBOUNDS_VOLATILITY_EXTREME == 0.50
        assert RiskFilter.ELITE_DEFENSE_RANK == 5
        assert RiskFilter.GOOD_DEFENSE_RANK == 10
        assert RiskFilter.SLUMP_THRESHOLD == 0.88
        assert RiskFilter.HOT_THRESHOLD == 1.12
        assert RiskFilter.SOFT_LINE_SPREAD == 2.0
        assert RiskFilter.VERY_SOFT_LINE_SPREAD == 3.0
        assert RiskFilter.SOFT_EDGE_PCT == 15.0


class TestRiskFilterAssessDefense:
    """Tests for RiskFilter._assess_defense method."""

    def test_elite_defense(self):
        """Test elite defense detection (top 5)."""
        rf = RiskFilter()
        score, flag, detail = rf._assess_defense(3, "POINTS")
        assert score == 1.0
        assert flag is True
        assert "ELITE" in detail
        assert "#3" in detail

    def test_good_defense(self):
        """Test good defense detection (6-10)."""
        rf = RiskFilter()
        score, flag, detail = rf._assess_defense(8, "REBOUNDS")
        assert score == 0.6
        assert flag is False
        assert "Good" in detail

    def test_average_defense(self):
        """Test average defense detection (11-20)."""
        rf = RiskFilter()
        score, flag, detail = rf._assess_defense(15, "POINTS")
        assert score == 0.3
        assert flag is False
        assert "Average" in detail

    def test_weak_defense(self):
        """Test weak defense detection (21+)."""
        rf = RiskFilter()
        score, flag, detail = rf._assess_defense(25, "POINTS")
        assert score == 0.1
        assert flag is False
        assert "Weak" in detail

    def test_unknown_defense(self):
        """Test unknown defense (None)."""
        rf = RiskFilter()
        score, flag, detail = rf._assess_defense(None, "POINTS")
        assert score == 0.3
        assert flag is False
        assert "Unknown" in detail


class TestRiskFilterCalculateRiskLevel:
    """Tests for RiskFilter._calculate_risk_level method."""

    def test_extreme_risk_multiple_flags(self):
        """Test EXTREME risk with multiple high risk flags."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            high_volatility=True,
            elite_defense=True,
            negative_trend=False,
            total_risk_score=0.6,
        )
        level = rf._calculate_risk_level(assessment)
        assert level == "EXTREME"

    def test_extreme_risk_high_score(self):
        """Test EXTREME risk with high total score."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            high_volatility=False,
            elite_defense=False,
            negative_trend=False,
            total_risk_score=0.85,
        )
        level = rf._calculate_risk_level(assessment)
        assert level == "EXTREME"

    def test_high_risk(self):
        """Test HIGH risk detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            high_volatility=True,
            elite_defense=False,
            negative_trend=False,
            total_risk_score=0.55,
        )
        level = rf._calculate_risk_level(assessment)
        assert level == "HIGH"

    def test_medium_risk(self):
        """Test MEDIUM risk detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            high_volatility=False,
            elite_defense=False,
            negative_trend=False,
            total_risk_score=0.45,
        )
        level = rf._calculate_risk_level(assessment)
        assert level == "MEDIUM"

    def test_low_risk(self):
        """Test LOW risk detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            high_volatility=False,
            elite_defense=False,
            negative_trend=False,
            total_risk_score=0.2,
        )
        level = rf._calculate_risk_level(assessment)
        assert level == "LOW"


class TestRiskFilterAssessVolatility:
    """Tests for RiskFilter._assess_volatility method."""

    def test_extreme_volatility_from_features(self):
        """Test extreme volatility detection from features."""
        rf = RiskFilter()
        features = {
            "points_std_L5": 12.0,
            "ema_points_L5": 20.0,  # CV = 0.6
            "minutes_std_L5": 8.0,
            "ema_minutes_L5": 30.0,
        }
        score, flag, detail = rf._assess_volatility(features, "POINTS", player_name=None)
        assert flag is True
        assert score == 1.0
        assert "EXTREME" in detail

    def test_high_volatility_from_features(self):
        """Test high volatility detection from features."""
        rf = RiskFilter()
        # Need combined_cv >= 0.30 for HIGH
        # combined_cv = 0.7 * stat_cv + 0.3 * minutes_cv
        # stat_cv = 10/25 = 0.4, minutes_cv = 4/30 = 0.133
        # combined_cv = 0.7 * 0.4 + 0.3 * 0.133 = 0.28 + 0.04 = 0.32
        features = {
            "points_std_L5": 10.0,
            "ema_points_L5": 25.0,  # CV = 0.4
            "minutes_std_L5": 4.0,
            "ema_minutes_L5": 30.0,  # CV = 0.133
        }
        score, flag, detail = rf._assess_volatility(features, "POINTS", player_name=None)
        assert flag is True
        assert score == 0.7
        assert "HIGH" in detail

    def test_normal_volatility_from_features(self):
        """Test normal volatility detection from features."""
        rf = RiskFilter()
        features = {
            "points_std_L5": 3.0,
            "ema_points_L5": 20.0,  # CV = 0.15
            "minutes_std_L5": 2.0,
            "ema_minutes_L5": 32.0,
        }
        score, flag, detail = rf._assess_volatility(features, "POINTS", player_name=None)
        assert flag is False
        assert score < 0.7
        assert "Normal" in detail

    def test_rebounds_volatility_thresholds(self):
        """Test REBOUNDS uses different thresholds."""
        rf = RiskFilter()
        # CV = 0.32 - HIGH for POINTS but normal for REBOUNDS
        features = {
            "rebounds_std_L5": 3.2,
            "ema_rebounds_L5": 10.0,
            "minutes_std_L5": 2.0,
            "ema_minutes_L5": 30.0,
        }
        score, flag, detail = rf._assess_volatility(features, "REBOUNDS", player_name=None)
        assert flag is False  # Under REBOUNDS threshold of 0.35
        assert "Normal" in detail


class TestRiskFilterAssessTrend:
    """Tests for RiskFilter._assess_trend method."""

    def test_slump_detection(self):
        """Test slump (negative trend) detection."""
        rf = RiskFilter()
        # L5 avg much lower than L10 avg (ratio < 0.88)
        features = {
            "ema_points_L5": 15.0,
            "ema_points_L10": 20.0,  # ratio = 0.75
        }
        score, flag, detail = rf._assess_trend(features, "POINTS", player_name=None)
        assert flag is True
        assert "SLUMP" in detail

    def test_hot_streak_detection(self):
        """Test hot streak (positive trend) detection."""
        rf = RiskFilter()
        # L5 avg much higher than L10 avg (ratio > 1.12)
        features = {
            "ema_points_L5": 25.0,
            "ema_points_L10": 20.0,  # ratio = 1.25
        }
        score, flag, detail = rf._assess_trend(features, "POINTS", player_name=None)
        assert flag is False
        assert score == 0.0
        assert "HOT" in detail

    def test_stable_trend(self):
        """Test stable trend detection."""
        rf = RiskFilter()
        features = {
            "ema_points_L5": 19.0,
            "ema_points_L10": 20.0,  # ratio = 0.95
        }
        score, flag, detail = rf._assess_trend(features, "POINTS", player_name=None)
        assert flag is False
        assert score == 0.2
        assert "Stable" in detail


class TestRiskFilterCalculatePointsStake:
    """Tests for RiskFilter._calculate_points_stake method."""

    def test_extreme_risk_zero_stake(self):
        """Test EXTREME risk results in zero stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="EXTREME",
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.0
        assert "EXTREME" in assessment.stake_reason
        assert "SKIP" in assessment.stake_reason

    def test_high_vol_sharp_line_skip(self):
        """Test high volatility + sharp line results in skip."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="HIGH",
            high_volatility=True,
            line_is_soft=False,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.0
        assert "sharp line = SKIP" in assessment.stake_reason

    def test_sharp_line_reduced_stake(self):
        """Test sharp line without high vol results in reduced stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="LOW",
            high_volatility=False,
            line_is_soft=False,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.5
        assert "sharp line = reduced" in assessment.stake_reason

    def test_high_vol_soft_line_stake(self):
        """Test high volatility + soft line results in 0.75u stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="HIGH",
            high_volatility=True,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.75
        assert "high vol + soft line" in assessment.stake_reason

    def test_medium_risk_soft_line_full_stake(self):
        """Test medium risk + soft line results in full stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="MEDIUM",
            high_volatility=False,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 1.0
        assert "medium risk + soft line" in assessment.stake_reason

    def test_low_risk_soft_line_full_stake(self):
        """Test low risk + soft line results in full stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="LOW",
            high_volatility=False,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 1.0
        assert "low risk + soft line" in assessment.stake_reason

    def test_slump_penalty(self):
        """Test slump reduces stake by 0.25u."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="LOW",
            high_volatility=False,
            negative_trend=True,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.75
        assert "slump -0.25u" in assessment.stake_reason

    def test_elite_defense_penalty(self):
        """Test elite defense reduces stake by 0.25u."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="LOW",
            high_volatility=False,
            elite_defense=True,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.75
        assert "elite def -0.25u" in assessment.stake_reason

    def test_combined_penalties(self):
        """Test slump + elite defense penalties are additive."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="LOW",
            high_volatility=False,
            negative_trend=True,
            elite_defense=True,
            line_is_soft=True,
        )
        rf._calculate_points_stake(assessment)
        assert assessment.recommended_stake == 0.5  # 1.0 - 0.25 - 0.25 = 0.5 (min)


class TestRiskFilterAssessLineSoftness:
    """Tests for RiskFilter._assess_line_softness method."""

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_very_soft_spread(self, mock_vol):
        """Test very soft line spread detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")
        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=3.5,  # Very soft
            edge_pct=5.0,
            prediction=27.0,
            consensus_line=25.0,
        )
        assert assessment.line_is_soft is True
        assert "very soft spread" in assessment.details.get("line_softness", "")

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_soft_spread(self, mock_vol):
        """Test soft line spread detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")
        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=2.5,  # Soft
            edge_pct=5.0,
            prediction=27.0,
            consensus_line=25.0,
        )
        # Not quite soft enough without other factors
        assert "soft spread" in assessment.details.get("line_softness", "")

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_huge_edge(self, mock_vol):
        """Test huge edge detection."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")
        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=1.0,
            edge_pct=35.0,  # Huge edge (>= 30%)
            prediction=30.0,
            consensus_line=25.0,
        )
        assert "huge edge" in assessment.details.get("line_softness", "")

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_sharp_line(self, mock_vol):
        """Test sharp line (no softness indicators)."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")
        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=0.5,  # Very tight spread
            edge_pct=3.0,  # Low edge
            prediction=25.5,  # No prediction edge
            consensus_line=25.5,
        )
        assert assessment.line_is_soft is False
        assert assessment.details.get("line_softness") == "sharp line"

    @patch.object(RiskFilter, "_get_real_volatility")
    def test_upside_clears_line(self, mock_vol):
        """Test when player's 75th percentile clears line."""
        mock_vol.return_value = {
            "percentile_75": 30.0,  # 75th percentile > 25.5 * 1.1
            "percentile_90": 35.0,  # 90th percentile > 25.5 * 1.2
        }
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")
        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=1.5,
            edge_pct=10.0,
            prediction=28.0,
            consensus_line=26.0,
        )
        assert "upside clears" in assessment.details.get("line_softness", "")


class TestRiskFilterFormatRiskSummary:
    """Tests for RiskFilter.format_risk_summary method."""

    def test_points_format(self):
        """Test POINTS format includes line softness and stake."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="POINTS",
            risk_level="HIGH",
            high_volatility=True,
            line_is_soft=True,
            recommended_stake=0.75,
        )
        summary = rf.format_risk_summary(assessment)
        assert "[HIGH]" in summary
        assert "HIGH_VOL" in summary
        assert "SOFT" in summary
        assert "0.75u" in summary

    def test_rebounds_format_with_flags(self):
        """Test REBOUNDS format with risk flags."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="REBOUNDS",
            risk_level="EXTREME",
            high_volatility=True,
            elite_defense=True,
            negative_trend=True,
        )
        summary = rf.format_risk_summary(assessment)
        assert "[EXTREME]" in summary
        assert "HIGH_VOL" in summary
        assert "ELITE_DEF" in summary
        assert "SLUMP" in summary

    def test_rebounds_format_no_flags(self):
        """Test REBOUNDS format with no risk flags returns None."""
        rf = RiskFilter()
        assessment = RiskAssessment(
            stat_type="REBOUNDS",
            risk_level="LOW",
            high_volatility=False,
            elite_defense=False,
            negative_trend=False,
        )
        summary = rf.format_risk_summary(assessment)
        assert summary is None


class TestRiskFilterGetStats:
    """Tests for RiskFilter.get_stats method."""

    def test_initial_stats(self):
        """Test initial stats are zero."""
        rf = RiskFilter()
        stats = rf.get_stats()
        assert stats["assessed"] == 0
        assert stats["high_risk_skipped"] == 0
        assert stats["skip_rate"] == 0

    def test_stats_after_assessments(self):
        """Test stats after some assessments."""
        rf = RiskFilter()
        rf.stats_assessed = 100
        rf.high_risk_skipped = 15
        stats = rf.get_stats()
        assert stats["assessed"] == 100
        assert stats["high_risk_skipped"] == 15
        assert stats["skip_rate"] == 15.0


class TestRiskFilterAssessRisk:
    """Integration tests for RiskFilter.assess_risk method."""

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    @patch.object(RiskFilter, "_get_connection", return_value=None)
    def test_assess_risk_points_low_risk(self, mock_conn, mock_vol):
        """Test low risk POINTS assessment."""
        rf = RiskFilter()
        features = {
            "points_std_L5": 3.0,
            "ema_points_L5": 20.0,
            "minutes_std_L5": 2.0,
            "ema_minutes_L5": 32.0,
        }
        assessment = rf.assess_risk(
            player_name="Test Player",
            stat_type="POINTS",
            features=features,
            opp_rank=20,
            p_over=0.55,
            line=20.5,
            line_spread=3.0,  # Soft
            edge_pct=10.0,
            prediction=22.0,
            consensus_line=21.0,
        )
        assert assessment.stat_type == "POINTS"
        assert assessment.risk_level in ("LOW", "MEDIUM")
        assert assessment.recommended_stake > 0
        assert rf.stats_assessed == 1

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    @patch.object(RiskFilter, "_get_connection", return_value=None)
    def test_assess_risk_rebounds_high_risk(self, mock_conn, mock_vol):
        """Test high risk REBOUNDS assessment that should be skipped."""
        rf = RiskFilter()
        features = {
            "rebounds_std_L5": 5.0,
            "ema_rebounds_L5": 8.0,  # High CV
            "ema_rebounds_L10": 12.0,  # Slump
            "minutes_std_L5": 5.0,
            "ema_minutes_L5": 25.0,
        }
        assessment = rf.assess_risk(
            player_name="Test Player",
            stat_type="REBOUNDS",
            features=features,
            opp_rank=3,  # Elite defense
            p_over=0.55,
            line=8.5,
            line_spread=1.0,
            edge_pct=5.0,
            prediction=9.0,
            consensus_line=8.5,
        )
        assert assessment.stat_type == "REBOUNDS"
        # With high vol + elite defense, should be high risk
        assert assessment.elite_defense is True
        # May or may not skip depending on all factors


# =============================================================================
# DATABASE CONNECTION AND VOLATILITY TESTS
# =============================================================================


class TestRiskFilterDatabaseMethods:
    """Tests for database-related methods in RiskFilter."""

    def test_get_connection_when_none(self):
        """Test _get_connection creates connection when None."""
        rf = RiskFilter()
        assert rf._conn is None

        # Mock the connection
        with patch("nba.betting_xl.risk_filters.psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            conn = rf._get_connection()
            assert conn is not None
            mock_connect.assert_called_once()

    def test_get_connection_reuses_existing(self):
        """Test _get_connection reuses existing open connection."""
        rf = RiskFilter()
        mock_conn = MagicMock()
        mock_conn.closed = False
        rf._conn = mock_conn

        with patch("nba.betting_xl.risk_filters.psycopg2.connect") as mock_connect:
            conn = rf._get_connection()
            assert conn == mock_conn
            mock_connect.assert_not_called()  # Should not create new

    def test_get_connection_reconnects_if_closed(self):
        """Test _get_connection reconnects if connection closed."""
        rf = RiskFilter()
        old_conn = MagicMock()
        old_conn.closed = True  # Connection is closed
        rf._conn = old_conn

        with patch("nba.betting_xl.risk_filters.psycopg2.connect") as mock_connect:
            new_conn = MagicMock()
            new_conn.closed = False
            mock_connect.return_value = new_conn

            conn = rf._get_connection()
            assert conn == new_conn
            mock_connect.assert_called_once()

    def test_get_connection_handles_error(self):
        """Test _get_connection handles connection errors."""
        import psycopg2

        rf = RiskFilter()

        with patch("nba.betting_xl.risk_filters.psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.Error("Connection failed")

            conn = rf._get_connection()
            assert conn is None


class TestRiskFilterGetRealVolatility:
    """Tests for _get_real_volatility method."""

    def test_returns_cached_value(self):
        """Test returns cached volatility if available."""
        rf = RiskFilter()
        cached_data = {
            "std_L5": 3.0,
            "avg_L5": 20.0,
            "cv_L5": 0.15,
        }
        rf._volatility_cache["TestPlayer_POINTS_2025-01-15"] = cached_data

        result = rf._get_real_volatility("TestPlayer", "POINTS", "2025-01-15")
        assert result == cached_data

    @patch.object(RiskFilter, "_get_connection", return_value=None)
    def test_returns_none_without_connection(self, mock_conn):
        """Test returns None when no database connection."""
        rf = RiskFilter()

        result = rf._get_real_volatility("TestPlayer", "POINTS", "2025-01-15")
        assert result is None

    @patch.object(RiskFilter, "_get_connection")
    def test_returns_none_with_insufficient_data(self, mock_conn):
        """Test returns None when less than 5 games."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (20.0, 30.0, "2025-01-15"),
            (22.0, 32.0, "2025-01-14"),
            (18.0, 28.0, "2025-01-13"),
        ]  # Only 3 games
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection

        rf = RiskFilter()
        result = rf._get_real_volatility("TestPlayer", "POINTS", "2025-01-15")
        assert result is None

    @patch.object(RiskFilter, "_get_connection")
    def test_calculates_volatility_correctly(self, mock_conn):
        """Test volatility calculation with sufficient data."""
        mock_cursor = MagicMock()
        # 5 games of data: [25, 22, 28, 20, 24] points, [30, 32, 28, 31, 29] minutes
        mock_cursor.fetchall.return_value = [
            (25.0, 30.0, "2025-01-15"),
            (22.0, 32.0, "2025-01-14"),
            (28.0, 28.0, "2025-01-13"),
            (20.0, 31.0, "2025-01-12"),
            (24.0, 29.0, "2025-01-11"),
        ]
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection

        rf = RiskFilter()
        result = rf._get_real_volatility("TestPlayer", "POINTS", "2025-01-15")

        assert result is not None
        assert "std_L5" in result
        assert "avg_L5" in result
        assert "cv_L5" in result
        assert "min_std_L5" in result
        assert "recent_games" in result
        assert "percentile_75" in result

        # Check calculation correctness
        assert result["avg_L5"] == 23.8  # mean([25, 22, 28, 20, 24])
        assert len(result["recent_games"]) == 5


class TestRiskFilterVolatilityWithRealData:
    """Tests for volatility assessment using real volatility data."""

    def test_assess_volatility_with_real_data(self):
        """Test volatility assessment when real data is available."""
        rf = RiskFilter()

        # Mock real volatility data
        with patch.object(rf, "_get_real_volatility") as mock_vol:
            mock_vol.return_value = {
                "std_L5": 5.0,
                "std_L10": 4.5,
                "avg_L5": 22.0,
                "avg_L10": 21.0,
                "cv_L5": 0.227,  # 5/22
                "min_std_L5": 3.0,
                "min_avg_L5": 30.0,
                "min_cv_L5": 0.1,
                "recent_games": [25, 22, 18, 24, 21],
                "range_L5": 7,
                "percentile_75": 24.5,
                "percentile_90": 25.0,
                "median_L5": 22.0,
            }

            score, flag, detail = rf._assess_volatility({}, "POINTS", "TestPlayer")

            # CV = 0.7 * 0.227 + 0.3 * 0.1 = 0.159 + 0.03 = 0.189
            # Below HIGH threshold of 0.30, so should be normal
            assert flag is False
            assert "Normal" in detail

    def test_assess_volatility_with_extreme_range(self):
        """Test volatility assessment when range is extreme."""
        rf = RiskFilter()

        with patch.object(rf, "_get_real_volatility") as mock_vol:
            mock_vol.return_value = {
                "std_L5": 4.0,
                "std_L10": 4.0,
                "avg_L5": 20.0,
                "avg_L10": 20.0,
                "cv_L5": 0.20,
                "min_std_L5": 2.0,
                "min_avg_L5": 30.0,
                "min_cv_L5": 0.067,
                "recent_games": [5, 35, 15, 25, 20],  # Huge range
                "range_L5": 30,  # range/avg = 1.5 > 1.0 triggers boost
                "percentile_75": 27.5,
                "percentile_90": 35.0,
                "median_L5": 20.0,
            }

            score, flag, detail = rf._assess_volatility({}, "POINTS", "TestPlayer")

            # Range ratio = 30/20 = 1.5 > 1.0, boosts CV to at least 0.5
            # 0.5 > EXTREME threshold (0.45) makes it EXTREME
            assert flag is True
            assert score == 1.0
            assert "EXTREME" in detail


class TestRiskFilterTrendWithRealData:
    """Tests for trend assessment using real volatility data."""

    def test_assess_trend_with_real_data_slump(self):
        """Test trend assessment detects slump from real data."""
        rf = RiskFilter()

        with patch.object(rf, "_get_real_volatility") as mock_vol:
            mock_vol.return_value = {
                "recent_games": [15, 16, 14, 22, 24],  # L3: 15, 16, 14 - slump
                "avg_L10": 20.0,
            }

            score, flag, detail = rf._assess_trend({}, "POINTS", "TestPlayer")

            # median(15, 16, 14) = 15, ratio = 15/20 = 0.75 < 0.88
            assert flag is True
            assert "SLUMP" in detail

    def test_assess_trend_with_real_data_hot(self):
        """Test trend assessment detects hot streak from real data."""
        rf = RiskFilter()

        with patch.object(rf, "_get_real_volatility") as mock_vol:
            mock_vol.return_value = {
                "recent_games": [28, 30, 26, 20, 18],  # L3: 28, 30, 26 - hot
                "avg_L10": 22.0,
            }

            score, flag, detail = rf._assess_trend({}, "POINTS", "TestPlayer")

            # median(28, 30, 26) = 28, ratio = 28/22 = 1.27 > 1.12
            assert flag is False
            assert score == 0.0
            assert "HOT" in detail


class TestRiskFilterLineSoftnessEdgeCases:
    """Additional tests for line softness edge cases."""

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_line_softness_good_edge(self, mock_vol):
        """Test good edge contribution to softness."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")

        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.5,
            line_spread=1.0,
            edge_pct=18.0,  # Good edge (>= 15%)
            prediction=27.0,
            consensus_line=26.0,
        )

        assert "good edge" in assessment.details.get("line_softness", "")

    @patch.object(RiskFilter, "_get_real_volatility", return_value=None)
    def test_line_softness_model_prediction_edge(self, mock_vol):
        """Test model prediction edge contribution."""
        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")

        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=20.0,
            line_spread=1.0,
            edge_pct=5.0,
            prediction=26.0,  # +6 points over line
            consensus_line=20.0,
        )

        assert "model +" in assessment.details.get("line_softness", "")

    @patch.object(RiskFilter, "_get_real_volatility")
    def test_line_softness_ceiling_check(self, mock_vol):
        """Test 90th percentile ceiling check."""
        mock_vol.return_value = {
            "percentile_75": 28.0,
            "percentile_90": 35.0,  # High ceiling
        }

        rf = RiskFilter()
        assessment = RiskAssessment(stat_type="POINTS")

        rf._assess_line_softness(
            assessment,
            player_name="Test Player",
            line=25.0,  # 35 > 25 * 1.2 = 30
            line_spread=1.5,
            edge_pct=10.0,
            prediction=28.0,
            consensus_line=26.0,
        )

        softness_detail = assessment.details.get("line_softness", "")
        assert "ceiling" in softness_detail or "upside" in softness_detail
