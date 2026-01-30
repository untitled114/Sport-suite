"""
Unit Tests for Custom Exceptions
================================
Tests for the custom exception hierarchy.
"""

import pytest


class TestBaseExceptions:
    """Tests for base exception classes."""

    def test_nba_props_error_is_exception(self):
        """Test NBAPropsError is an Exception."""
        from nba.core.exceptions import NBAPropsError

        assert issubclass(NBAPropsError, Exception)

    def test_nba_props_error_message(self):
        """Test NBAPropsError stores message."""
        from nba.core.exceptions import NBAPropsError

        error = NBAPropsError("Test error message")
        assert str(error) == "Test error message"


class TestModelExceptions:
    """Tests for model-related exceptions."""

    def test_model_not_found_error(self):
        """Test ModelNotFoundError with path."""
        from nba.core.exceptions import ModelNotFoundError

        error = ModelNotFoundError("/path/to/model.pkl")
        assert "Model not found" in str(error)
        assert "/path/to/model.pkl" in str(error)

    def test_model_not_found_error_with_market(self):
        """Test ModelNotFoundError with market."""
        from nba.core.exceptions import ModelNotFoundError

        error = ModelNotFoundError("/path/to/model.pkl", market="POINTS")
        assert "POINTS" in str(error)
        assert error.market == "POINTS"

    def test_model_load_error(self):
        """Test ModelLoadError."""
        from nba.core.exceptions import ModelLoadError

        error = ModelLoadError("/path/to/model.pkl", reason="Version mismatch")
        assert "Failed to load model" in str(error)
        assert "Version mismatch" in str(error)

    def test_model_prediction_error(self):
        """Test ModelPredictionError."""
        from nba.core.exceptions import ModelPredictionError

        error = ModelPredictionError(
            "Prediction failed", player_name="LeBron James", stat_type="POINTS"
        )
        assert "Prediction failed" in str(error)
        assert "LeBron James" in str(error)
        assert "POINTS" in str(error)


class TestFeatureExceptions:
    """Tests for feature-related exceptions."""

    def test_feature_extraction_error(self):
        """Test FeatureExtractionError."""
        from nba.core.exceptions import FeatureExtractionError

        error = FeatureExtractionError("Extraction failed", player_name="Stephen Curry")
        assert "Extraction failed" in str(error)
        assert "Stephen Curry" in str(error)

    def test_missing_feature_error(self):
        """Test MissingFeatureError."""
        from nba.core.exceptions import MissingFeatureError

        error = MissingFeatureError("ema_points_L5", player_name="Kevin Durant")
        assert "Missing required feature" in str(error)
        assert "ema_points_L5" in str(error)
        assert "Kevin Durant" in str(error)

    def test_feature_drift_error(self):
        """Test FeatureDriftError."""
        from nba.core.exceptions import FeatureDriftError

        drifted = ["feature_1", "feature_2", "feature_3"]
        error = FeatureDriftError(drifted_features=drifted, drift_percentage=15.5)
        assert "Feature drift detected" in str(error)
        assert "15.5%" in str(error)
        assert "3 features" in str(error)


class TestDataValidationExceptions:
    """Tests for data validation exceptions."""

    def test_invalid_prop_line_error(self):
        """Test InvalidPropLineError."""
        from nba.core.exceptions import InvalidPropLineError

        error = InvalidPropLineError("Line out of range", player_name="Giannis", line=250.0)
        assert "Line out of range" in str(error)
        assert "Giannis" in str(error)
        assert "250" in str(error)

    def test_invalid_prediction_error(self):
        """Test InvalidPredictionError."""
        from nba.core.exceptions import InvalidPredictionError

        error = InvalidPredictionError("Probability out of bounds", field="p_over")
        assert "Probability out of bounds" in str(error)
        assert "p_over" in str(error)


class TestDatabaseExceptions:
    """Tests for database-related exceptions."""

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        from nba.core.exceptions import DatabaseConnectionError

        error = DatabaseConnectionError("nba_players", host="localhost", port=5536)
        assert "Failed to connect" in str(error)
        assert "nba_players" in str(error)
        assert "localhost:5536" in str(error)

    def test_database_query_error(self):
        """Test DatabaseQueryError."""
        from nba.core.exceptions import DatabaseQueryError

        error = DatabaseQueryError("Query timeout", query="SELECT * FROM players")
        assert "Query timeout" in str(error)
        assert "SELECT" in str(error)

    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        from nba.core.exceptions import DataNotFoundError

        error = DataNotFoundError("Player", identifier="LeBron James")
        assert "Player not found" in str(error)
        assert "LeBron James" in str(error)


class TestAPIExceptions:
    """Tests for API-related exceptions."""

    def test_api_connection_error(self):
        """Test APIConnectionError."""
        from nba.core.exceptions import APIConnectionError

        error = APIConnectionError("BettingPros", status_code=503)
        assert "Failed to connect" in str(error)
        assert "BettingPros" in str(error)
        assert "503" in str(error)

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError."""
        from nba.core.exceptions import APIRateLimitError

        error = APIRateLimitError("ESPN", retry_after=60)
        assert "rate limit exceeded" in str(error)
        assert "ESPN" in str(error)
        assert "60s" in str(error)

    def test_api_response_error(self):
        """Test APIResponseError."""
        from nba.core.exceptions import APIResponseError

        error = APIResponseError("BettingPros", "Invalid JSON response")
        assert "BettingPros API error" in str(error)
        assert "Invalid JSON" in str(error)


class TestConfigurationExceptions:
    """Tests for configuration exceptions."""

    def test_missing_config_error(self):
        """Test MissingConfigError."""
        from nba.core.exceptions import MissingConfigError

        error = MissingConfigError("API_KEY", source=".env")
        assert "Missing required configuration" in str(error)
        assert "API_KEY" in str(error)
        assert ".env" in str(error)

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        from nba.core.exceptions import InvalidConfigError

        error = InvalidConfigError("THRESHOLD", value="-0.5", reason="must be positive")
        assert "Invalid configuration" in str(error)
        assert "THRESHOLD=-0.5" in str(error)
        assert "must be positive" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_model_errors_inherit_from_nba_props_error(self):
        """Test model errors inherit from NBAPropsError."""
        from nba.core.exceptions import (
            ModelError,
            ModelLoadError,
            ModelNotFoundError,
            NBAPropsError,
        )

        assert issubclass(ModelError, NBAPropsError)
        assert issubclass(ModelNotFoundError, ModelError)
        assert issubclass(ModelLoadError, ModelError)

    def test_feature_errors_inherit_from_nba_props_error(self):
        """Test feature errors inherit from NBAPropsError."""
        from nba.core.exceptions import (
            FeatureError,
            FeatureExtractionError,
            MissingFeatureError,
            NBAPropsError,
        )

        assert issubclass(FeatureError, NBAPropsError)
        assert issubclass(FeatureExtractionError, FeatureError)
        assert issubclass(MissingFeatureError, FeatureError)

    def test_can_catch_all_with_base_exception(self):
        """Test catching all exceptions with NBAPropsError."""
        from nba.core.exceptions import ModelNotFoundError, NBAPropsError

        with pytest.raises(NBAPropsError):
            raise ModelNotFoundError("/path/model.pkl")


class TestSerializationExceptions:
    """Tests for serialization-related exceptions."""

    def test_pickle_load_error(self):
        """Test PickleLoadError with file path and reason."""
        from nba.core.exceptions import PickleLoadError, SerializationError

        error = PickleLoadError("/models/model.pkl", reason="corrupted file")
        assert error.file_path == "/models/model.pkl"
        assert error.reason == "corrupted file"
        assert "Failed to load pickle file" in str(error)
        assert "/models/model.pkl" in str(error)
        assert "corrupted file" in str(error)
        assert isinstance(error, SerializationError)

    def test_pickle_load_error_without_reason(self):
        """Test PickleLoadError without reason."""
        from nba.core.exceptions import PickleLoadError

        error = PickleLoadError("/models/model.pkl")
        assert "Failed to load pickle file" in str(error)
        assert error.reason is None

    def test_json_load_error(self):
        """Test JSONLoadError with file path and reason."""
        from nba.core.exceptions import JSONLoadError, SerializationError

        error = JSONLoadError("/config/settings.json", reason="invalid JSON")
        assert error.file_path == "/config/settings.json"
        assert error.reason == "invalid JSON"
        assert "Failed to load JSON file" in str(error)
        assert isinstance(error, SerializationError)


class TestCalibrationExceptions:
    """Tests for calibration-related exceptions."""

    def test_calibration_error_is_base(self):
        """Test CalibrationError is NBAPropsError."""
        from nba.core.exceptions import CalibrationError, NBAPropsError

        assert issubclass(CalibrationError, NBAPropsError)

    def test_calibration_data_error(self):
        """Test CalibrationDataError with market, reason, samples."""
        from nba.core.exceptions import CalibrationDataError, CalibrationError

        error = CalibrationDataError(market="POINTS", reason="insufficient samples", samples=10)
        assert error.market == "POINTS"
        assert error.reason == "insufficient samples"
        assert error.samples == 10
        assert "POINTS" in str(error)
        assert "insufficient samples" in str(error)
        assert "samples=10" in str(error)
        assert isinstance(error, CalibrationError)

    def test_calibration_data_error_without_samples(self):
        """Test CalibrationDataError without samples."""
        from nba.core.exceptions import CalibrationDataError

        error = CalibrationDataError(market="REBOUNDS", reason="invalid data")
        assert error.samples is None
        assert "REBOUNDS" in str(error)

    def test_calibration_fit_error(self):
        """Test CalibrationFitError with market and reason."""
        from nba.core.exceptions import CalibrationError, CalibrationFitError

        error = CalibrationFitError(market="POINTS", reason="non-monotonic data")
        assert error.market == "POINTS"
        assert error.reason == "non-monotonic data"
        assert "Failed to fit calibrator" in str(error)
        assert "POINTS" in str(error)
        assert isinstance(error, CalibrationError)

    def test_calibration_fit_error_without_reason(self):
        """Test CalibrationFitError without reason."""
        from nba.core.exceptions import CalibrationFitError

        error = CalibrationFitError(market="ASSISTS")
        assert error.reason is None


class TestMongoDBExceptions:
    """Tests for MongoDB-related exceptions."""

    def test_mongodb_error_inherits_database_error(self):
        """Test MongoDBError inherits from DatabaseError."""
        from nba.core.exceptions import DatabaseError, MongoDBError

        assert issubclass(MongoDBError, DatabaseError)

    def test_mongodb_connection_error(self):
        """Test MongoDBConnectionError with host, port, reason."""
        from nba.core.exceptions import MongoDBConnectionError, MongoDBError

        error = MongoDBConnectionError(host="localhost", port=27017, reason="authentication failed")
        assert error.host == "localhost"
        assert error.port == 27017
        assert error.reason == "authentication failed"
        assert "Failed to connect to MongoDB" in str(error)
        assert "localhost:27017" in str(error)
        assert "authentication failed" in str(error)
        assert isinstance(error, MongoDBError)

    def test_mongodb_connection_error_minimal(self):
        """Test MongoDBConnectionError with minimal args."""
        from nba.core.exceptions import MongoDBConnectionError

        error = MongoDBConnectionError()
        assert "Failed to connect to MongoDB" in str(error)
        assert error.host is None
        assert error.port is None
        assert error.reason is None

    def test_mongodb_query_error(self):
        """Test MongoDBQueryError with collection and operation."""
        from nba.core.exceptions import MongoDBError, MongoDBQueryError

        error = MongoDBQueryError(
            collection="player_stats", message="invalid filter", operation="find"
        )
        assert error.collection == "player_stats"
        assert error.operation == "find"
        assert "MongoDB query error" in str(error)
        assert "player_stats" in str(error)
        assert "operation=find" in str(error)
        assert isinstance(error, MongoDBError)


class TestLineShoppingExceptions:
    """Tests for line shopping exceptions."""

    def test_line_shopping_error_inheritance(self):
        """Test LineShoppingError inherits from NBAPropsError."""
        from nba.core.exceptions import LineShoppingError, NBAPropsError

        assert issubclass(LineShoppingError, NBAPropsError)

    def test_no_lines_found_error(self):
        """Test NoLinesFoundError with player, stat, date."""
        from nba.core.exceptions import LineShoppingError, NoLinesFoundError

        error = NoLinesFoundError(
            player_name="Stephen Curry", stat_type="POINTS", game_date="2024-01-15"
        )
        assert error.player_name == "Stephen Curry"
        assert error.stat_type == "POINTS"
        assert error.game_date == "2024-01-15"
        assert "No book lines found" in str(error)
        assert "Stephen Curry" in str(error)
        assert "POINTS" in str(error)
        assert "2024-01-15" in str(error)
        assert isinstance(error, LineShoppingError)

    def test_no_lines_found_error_without_date(self):
        """Test NoLinesFoundError without game_date."""
        from nba.core.exceptions import NoLinesFoundError

        error = NoLinesFoundError(player_name="Kevin Durant", stat_type="REBOUNDS")
        assert error.game_date is None
        assert "2024" not in str(error)

    def test_all_books_blacklisted_error(self):
        """Test AllBooksBlacklistedError with player, stat, books."""
        from nba.core.exceptions import AllBooksBlacklistedError, LineShoppingError

        error = AllBooksBlacklistedError(
            player_name="LeBron James", stat_type="ASSISTS", books=["FanDuel", "BetRivers"]
        )
        assert error.player_name == "LeBron James"
        assert error.stat_type == "ASSISTS"
        assert error.books == ["FanDuel", "BetRivers"]
        assert "blacklisted" in str(error)
        assert "LeBron James" in str(error)
        assert "FanDuel" in str(error)
        assert isinstance(error, LineShoppingError)

    def test_all_books_blacklisted_error_without_books(self):
        """Test AllBooksBlacklistedError without books list."""
        from nba.core.exceptions import AllBooksBlacklistedError

        error = AllBooksBlacklistedError(player_name="Giannis Antetokounmpo", stat_type="THREES")
        assert error.books is None
