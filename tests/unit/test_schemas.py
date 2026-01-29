"""
Unit Tests for Pydantic Schemas
===============================
Tests for data validation schemas.
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError


class TestBookName:
    """Tests for BookName enum."""

    def test_book_name_values(self):
        """Test BookName has expected values."""
        from nba.core.schemas import BookName

        assert BookName.DRAFTKINGS.value == "draftkings"
        assert BookName.FANDUEL.value == "fanduel"
        assert BookName.BETMGM.value == "betmgm"
        assert BookName.UNDERDOG.value == "underdog"

    def test_book_name_count(self):
        """Test BookName has all expected books."""
        from nba.core.schemas import BookName

        assert len(BookName) == 10


class TestPropLineSchema:
    """Tests for PropLine Pydantic model."""

    def test_valid_prop_line(self):
        """Test creating a valid prop line."""
        from nba.core.schemas import MarketType, PropLine

        prop = PropLine(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name="draftkings",
            game_date=date(2025, 11, 6),
        )

        assert prop.player_name == "LeBron James"
        assert prop.line == 25.5
        assert prop.book_name == "draftkings"

    def test_prop_line_normalizes_player_name(self):
        """Test player name whitespace is normalized."""
        from nba.core.schemas import MarketType, PropLine

        prop = PropLine(
            player_name="  LeBron   James  ",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name="draftkings",
            game_date=date(2025, 11, 6),
        )

        assert prop.player_name == "LeBron James"

    def test_prop_line_normalizes_book_name(self):
        """Test book name string is converted to BookName enum."""
        from nba.core.schemas import BookName, MarketType, PropLine

        prop = PropLine(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name="DraftKings",
            game_date=date(2025, 11, 6),
        )

        assert prop.book_name == BookName.DRAFTKINGS

    def test_prop_line_accepts_book_name_enum(self):
        """Test PropLine accepts BookName enum directly."""
        from nba.core.schemas import BookName, MarketType, PropLine

        prop = PropLine(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name=BookName.FANDUEL,
            game_date=date(2025, 11, 6),
        )

        assert prop.book_name == BookName.FANDUEL

    def test_prop_line_accepts_book_abbreviations(self):
        """Test PropLine accepts common book abbreviations."""
        from nba.core.schemas import BookName, MarketType, PropLine

        # Test DK abbreviation
        prop = PropLine(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name="dk",
            game_date=date(2025, 11, 6),
        )
        assert prop.book_name == BookName.DRAFTKINGS

    def test_prop_line_rejects_invalid_book(self):
        """Test PropLine rejects unknown book names."""
        from nba.core.schemas import MarketType, PropLine

        with pytest.raises(ValidationError) as exc_info:
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name="invalid_book_xyz",
                game_date=date(2025, 11, 6),
            )

        assert "Unknown book name" in str(exc_info.value)

    def test_prop_line_parses_date_string(self):
        """Test date string is parsed correctly."""
        from nba.core.schemas import MarketType, PropLine

        prop = PropLine(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            line=25.5,
            book_name="draftkings",
            game_date="2025-11-06",
        )

        assert prop.game_date == date(2025, 11, 6)

    def test_prop_line_rejects_negative_line(self):
        """Test negative line is rejected."""
        from nba.core.schemas import MarketType, PropLine

        with pytest.raises(ValidationError):
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=-5.0,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            )

    def test_prop_line_rejects_excessive_line(self):
        """Test line > 200 is rejected."""
        from nba.core.schemas import MarketType, PropLine

        with pytest.raises(ValidationError):
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=250.0,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            )

    def test_prop_line_rejects_empty_player_name(self):
        """Test empty player name is rejected."""
        from nba.core.schemas import MarketType, PropLine

        with pytest.raises(ValidationError):
            PropLine(
                player_name="",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            )


class TestPredictionSchema:
    """Tests for Prediction Pydantic model."""

    def test_valid_prediction(self):
        """Test creating a valid prediction."""
        from nba.core.schemas import Confidence, MarketType, Prediction, Side

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        assert pred.prediction == 28.5
        assert pred.p_over == 0.72
        assert pred.side == Side.OVER
        assert pred.edge == 3.0

    def test_prediction_validates_side_over(self):
        """Test side OVER requires p_over >= 0.5."""
        from nba.core.schemas import MarketType, Prediction, Side

        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                prediction=28.5,
                p_over=0.3,  # Low probability but OVER side
                side=Side.OVER,
                edge=3.0,
                best_book="underdog",
                best_line=25.5,
            )

        assert "Side is OVER but p_over < 0.5" in str(exc_info.value)

    def test_prediction_validates_side_under(self):
        """Test side UNDER requires p_over <= 0.5."""
        from nba.core.schemas import MarketType, Prediction, Side

        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                prediction=28.5,
                p_over=0.7,  # High probability but UNDER side
                side=Side.UNDER,
                edge=3.0,
                best_book="underdog",
                best_line=25.5,
            )

        assert "Side is UNDER but p_over > 0.5" in str(exc_info.value)

    def test_prediction_p_over_bounds(self):
        """Test p_over must be between 0 and 1."""
        from nba.core.schemas import MarketType, Prediction, Side

        with pytest.raises(ValidationError):
            Prediction(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                prediction=28.5,
                p_over=1.5,  # Invalid
                side=Side.OVER,
                edge=3.0,
                best_book="underdog",
                best_line=25.5,
            )


class TestPropLineCollection:
    """Tests for PropLineCollection."""

    def test_collection_softest_line(self):
        """Test softest line is minimum."""
        from nba.core.schemas import MarketType, PropLine, PropLineCollection

        lines = [
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=24.5,
                book_name="underdog",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=26.0,
                book_name="fanduel",
                game_date=date(2025, 11, 6),
            ),
        ]

        collection = PropLineCollection(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            game_date=date(2025, 11, 6),
            lines=lines,
        )

        assert collection.softest_line.line == 24.5
        assert collection.softest_line.book_name == "underdog"

    def test_collection_hardest_line(self):
        """Test hardest line is maximum."""
        from nba.core.schemas import MarketType, PropLine, PropLineCollection

        lines = [
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=24.5,
                book_name="underdog",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=26.0,
                book_name="fanduel",
                game_date=date(2025, 11, 6),
            ),
        ]

        collection = PropLineCollection(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            game_date=date(2025, 11, 6),
            lines=lines,
        )

        assert collection.hardest_line.line == 26.0
        assert collection.hardest_line.book_name == "fanduel"

    def test_collection_line_spread(self):
        """Test line spread calculation."""
        from nba.core.schemas import MarketType, PropLine, PropLineCollection

        lines = [
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name="draftkings",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=24.5,
                book_name="underdog",
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=26.5,
                book_name="fanduel",
                game_date=date(2025, 11, 6),
            ),
        ]

        collection = PropLineCollection(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            game_date=date(2025, 11, 6),
            lines=lines,
        )

        assert collection.line_spread == 2.0  # 26.5 - 24.5

    def test_collection_consensus_odd(self):
        """Test consensus line with odd number of books."""
        from nba.core.schemas import BookName, MarketType, PropLine, PropLineCollection

        lines = [
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=24.5,
                book_name=BookName.DRAFTKINGS,
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=25.5,
                book_name=BookName.FANDUEL,
                game_date=date(2025, 11, 6),
            ),
            PropLine(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                line=26.5,
                book_name=BookName.BETMGM,
                game_date=date(2025, 11, 6),
            ),
        ]

        collection = PropLineCollection(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            game_date=date(2025, 11, 6),
            lines=lines,
        )

        assert collection.consensus_line == 25.5  # Median of 3


class TestMarketType:
    """Tests for MarketType enum."""

    def test_market_type_values(self):
        """Test MarketType has expected values."""
        from nba.core.schemas import MarketType

        assert MarketType.POINTS.value == "POINTS"
        assert MarketType.REBOUNDS.value == "REBOUNDS"
        assert MarketType.ASSISTS.value == "ASSISTS"
        assert MarketType.THREES.value == "THREES"


class TestFeatureVector:
    """Tests for FeatureVector schema."""

    def test_feature_vector_creation(self):
        """Test creating a feature vector."""
        from nba.core.schemas import FeatureVector

        fv = FeatureVector(
            is_home=1.0,
            line=25.5,
            ema_points_L3=24.0,
            ema_points_L5=23.5,
            team_pace=102.5,
        )

        assert fv.is_home == 1.0
        assert fv.line == 25.5
        assert fv.ema_points_L3 == 24.0

    def test_feature_vector_allows_extra_fields(self):
        """Test feature vector allows additional fields."""
        from nba.core.schemas import FeatureVector

        fv = FeatureVector(
            is_home=1.0,
            line=25.5,
            custom_feature=10.0,  # Extra field
        )

        assert fv.custom_feature == 10.0

    def test_feature_vector_to_array(self):
        """Test converting feature vector to array."""
        from nba.core.schemas import FeatureVector

        fv = FeatureVector(
            is_home=1.0,
            line=25.5,
            ema_points_L3=24.0,
        )

        arr = fv.to_array(["is_home", "line", "ema_points_L3"])
        assert arr == [1.0, 25.5, 24.0]

    def test_feature_vector_to_array_missing_fills_zero(self):
        """Test missing features filled with zero."""
        from nba.core.schemas import FeatureVector

        fv = FeatureVector(
            is_home=1.0,
            line=25.5,
        )

        arr = fv.to_array(["is_home", "line", "missing_feature"])
        assert arr == [1.0, 25.5, 0.0]


class TestPredictionBatch:
    """Tests for PredictionBatch schema."""

    def test_valid_prediction_batch(self):
        """Test creating a valid prediction batch."""
        from nba.core.schemas import BookName, MarketType, Prediction, PredictionBatch, Side

        picks = [
            Prediction(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                prediction=28.5,
                p_over=0.72,
                side=Side.OVER,
                edge=3.0,
                best_book="underdog",
                best_line=25.5,
            ),
            Prediction(
                player_name="Stephen Curry",
                stat_type=MarketType.THREES,
                prediction=4.5,
                p_over=0.65,
                side=Side.OVER,
                edge=2.5,
                best_book="draftkings",
                best_line=4.0,
            ),
        ]

        batch = PredictionBatch(
            date=date(2025, 11, 6),
            markets_enabled=[MarketType.POINTS, MarketType.THREES],
            total_picks=2,
            picks=picks,
        )

        assert batch.total_picks == 2
        assert len(batch.picks) == 2
        assert batch.strategy == "XL Line Shopping"

    def test_prediction_batch_auto_corrects_total_picks(self):
        """Test total_picks is auto-corrected to match picks length."""
        from nba.core.schemas import MarketType, Prediction, PredictionBatch, Side

        picks = [
            Prediction(
                player_name="LeBron James",
                stat_type=MarketType.POINTS,
                prediction=28.5,
                p_over=0.72,
                side=Side.OVER,
                edge=3.0,
                best_book="underdog",
                best_line=25.5,
            ),
        ]

        # Pass wrong total_picks
        batch = PredictionBatch(
            date=date(2025, 11, 6),
            markets_enabled=[MarketType.POINTS],
            total_picks=999,  # Wrong!
            picks=picks,
        )

        # Should be auto-corrected
        assert batch.total_picks == 1

    def test_prediction_batch_empty_picks(self):
        """Test prediction batch with no picks."""
        from nba.core.schemas import MarketType, PredictionBatch

        batch = PredictionBatch(
            date=date(2025, 11, 6),
            markets_enabled=[MarketType.POINTS],
            total_picks=0,
            picks=[],
        )

        assert batch.total_picks == 0
        assert len(batch.picks) == 0

    def test_prediction_batch_generated_at_default(self):
        """Test generated_at is auto-populated."""
        from nba.core.schemas import MarketType, PredictionBatch

        batch = PredictionBatch(
            date=date(2025, 11, 6),
            markets_enabled=[MarketType.POINTS],
            total_picks=0,
        )

        assert batch.generated_at is not None
        assert isinstance(batch.generated_at, datetime)


class TestValidationResult:
    """Tests for ValidationResult schema."""

    def test_valid_validation_result_win(self):
        """Test creating a winning validation result."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        result = ValidationResult(
            prediction=pred,
            actual_value=30.0,  # Over 25.5, OVER wins
            result="WIN",
            profit=1.0,
        )

        assert result.result == "WIN"
        assert result.hit_over is True
        assert result.profit == 1.0

    def test_valid_validation_result_loss(self):
        """Test creating a losing validation result."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        result = ValidationResult(
            prediction=pred,
            actual_value=20.0,  # Under 25.5, OVER loses
            result="LOSS",
            profit=-1.0,
        )

        assert result.result == "LOSS"
        assert result.hit_over is False
        assert result.profit == -1.0

    def test_valid_validation_result_push(self):
        """Test creating a push validation result."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        result = ValidationResult(
            prediction=pred,
            actual_value=25.5,  # Exactly line
            result="PUSH",
            profit=0.0,
        )

        assert result.result == "PUSH"
        assert result.profit == 0.0

    def test_validation_result_rejects_invalid_result(self):
        """Test ValidationResult rejects invalid result strings."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        with pytest.raises(ValidationError):
            ValidationResult(
                prediction=pred,
                actual_value=30.0,
                result="INVALID",  # Not WIN/LOSS/PUSH
                profit=1.0,
            )

    def test_validation_result_detects_result_mismatch(self):
        """Test ValidationResult detects inconsistent result vs actual."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=28.5,
            p_over=0.72,
            side=Side.OVER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        # actual > line with OVER bet should be WIN, not LOSS
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(
                prediction=pred,
                actual_value=30.0,  # Over wins
                result="LOSS",  # But claiming loss
                profit=-1.0,
            )

        assert "Result mismatch" in str(exc_info.value)

    def test_validation_result_under_bet_win(self):
        """Test ValidationResult for UNDER bet that wins."""
        from nba.core.schemas import MarketType, Prediction, Side, ValidationResult

        pred = Prediction(
            player_name="LeBron James",
            stat_type=MarketType.POINTS,
            prediction=22.0,
            p_over=0.35,
            side=Side.UNDER,
            edge=3.0,
            best_book="underdog",
            best_line=25.5,
        )

        result = ValidationResult(
            prediction=pred,
            actual_value=20.0,  # Under 25.5, UNDER wins
            result="WIN",
            profit=1.0,
        )

        assert result.result == "WIN"
        assert result.hit_over is False  # Did not hit over


class TestTrainingMetrics:
    """Tests for TrainingMetrics schema."""

    def test_valid_training_metrics(self):
        """Test creating valid training metrics."""
        from nba.core.schemas import MarketType, TrainingMetrics

        metrics = TrainingMetrics(
            market=MarketType.POINTS,
            samples_train=15000,
            samples_test=5000,
            rmse_train=6.0,
            rmse_test=6.5,
            mae_test=5.0,
            r2_test=0.41,
            accuracy_train=0.90,
            accuracy_test=0.88,
            auc_test=0.76,
            auc_calibrated=0.77,
            auc_blended=0.765,
            brier_before=0.22,
            brier_after=0.20,
            feature_count=102,
        )

        assert metrics.market == MarketType.POINTS
        assert metrics.auc_test == 0.76
        assert metrics.feature_count == 102

    def test_training_metrics_r2_bounds(self):
        """Test R2 can be negative (poor model)."""
        from nba.core.schemas import MarketType, TrainingMetrics

        # R2 can be negative for very poor models
        metrics = TrainingMetrics(
            market=MarketType.ASSISTS,
            samples_train=15000,
            samples_test=5000,
            rmse_train=6.0,
            rmse_test=6.5,
            mae_test=5.0,
            r2_test=-0.5,  # Negative R2 is valid
            accuracy_train=0.55,
            accuracy_test=0.52,
            auc_test=0.55,
            auc_calibrated=0.55,
            auc_blended=0.55,
            brier_before=0.25,
            brier_after=0.25,
            feature_count=102,
        )

        assert metrics.r2_test == -0.5
