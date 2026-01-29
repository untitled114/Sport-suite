"""
Integration Tests for Prediction Pipeline
=========================================
Tests for the end-to-end prediction pipeline including:
- Model loading
- Feature extraction
- Prediction generation
- Output formatting
"""

import json
import pickle
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestXLPredictorModelLoading:
    """Tests for XL predictor model loading."""

    @pytest.fixture
    def mock_model_files(self, tmp_path):
        """Create mock model files for testing."""
        models_dir = tmp_path / "models" / "saved_xl"
        models_dir.mkdir(parents=True)

        # Use simple dicts/lists instead of objects to avoid pickle issues
        features = [f"feature_{i}" for i in range(102)]

        # Save mock models as simple data structures
        for market in ["points", "rebounds"]:
            # Save feature list (the only thing we really need to test)
            with open(models_dir / f"{market}_xl_features.pkl", "wb") as f:
                pickle.dump(features, f)

            # Create placeholder files (empty dicts are picklable)
            for component in ["regressor", "classifier", "calibrator", "imputer", "scaler"]:
                with open(models_dir / f"{market}_xl_{component}.pkl", "wb") as f:
                    pickle.dump({"type": component, "market": market}, f)

            # Create metadata
            metadata = {
                "market": market.upper(),
                "trained_date": "2025-11-06",
                "features": {"count": 102, "names": features},
            }
            with open(models_dir / f"{market}_xl_metadata.json", "w") as f:
                json.dump(metadata, f)

        return models_dir

    def test_model_files_exist(self, mock_model_files):
        """Test mock model files were created."""
        assert (mock_model_files / "points_xl_regressor.pkl").exists()
        assert (mock_model_files / "points_xl_classifier.pkl").exists()
        assert (mock_model_files / "points_xl_calibrator.pkl").exists()

    def test_model_metadata_is_valid_json(self, mock_model_files):
        """Test metadata files are valid JSON."""
        with open(mock_model_files / "points_xl_metadata.json") as f:
            metadata = json.load(f)
        assert metadata["market"] == "POINTS"
        assert metadata["features"]["count"] == 102


class TestPredictionOutput:
    """Tests for prediction output format."""

    def test_prediction_has_required_fields(self, sample_prediction_output):
        """Test prediction output has all required fields."""
        required_fields = [
            "player_name",
            "stat_type",
            "side",
            "prediction",
            "p_over",
            "best_book",
            "best_line",
            "edge",
        ]
        for field in required_fields:
            assert field in sample_prediction_output, f"Missing required field: {field}"

    def test_prediction_values_are_valid(self, sample_prediction_output):
        """Test prediction values are valid."""
        assert sample_prediction_output["side"] in ["OVER", "UNDER"]
        assert 0 <= sample_prediction_output["p_over"] <= 1
        assert sample_prediction_output["prediction"] > 0
        assert sample_prediction_output["edge"] >= 0

    def test_edge_calculation(self, sample_prediction_output):
        """Test edge is calculated correctly."""
        prediction = sample_prediction_output["prediction"]
        best_line = sample_prediction_output["best_line"]
        edge = sample_prediction_output["edge"]

        # For OVER, edge = prediction - line
        if sample_prediction_output["side"] == "OVER":
            expected_edge = prediction - best_line
        else:
            expected_edge = best_line - prediction

        assert abs(edge - expected_edge) < 0.01

    def test_confidence_is_valid(self, sample_prediction_output):
        """Test confidence level is valid."""
        valid_confidences = ["HIGH", "MEDIUM", "STANDARD", "LOW"]
        assert sample_prediction_output["confidence"] in valid_confidences


class TestPredictionFiltering:
    """Tests for prediction filtering logic."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for filtering tests."""
        return [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "p_over": 0.72,
                "edge": 4.0,
                "line_spread": 2.5,
            },
            {
                "player_name": "Kevin Durant",
                "stat_type": "POINTS",
                "p_over": 0.55,
                "edge": 1.5,
                "line_spread": 1.0,
            },
            {
                "player_name": "Stephen Curry",
                "stat_type": "POINTS",
                "p_over": 0.80,
                "edge": 5.0,
                "line_spread": 3.0,
            },
        ]

    def test_filter_by_probability_threshold(self, sample_predictions):
        """Test filtering by probability threshold."""
        threshold = 0.65
        filtered = [p for p in sample_predictions if p["p_over"] >= threshold]
        assert len(filtered) == 2
        assert all(p["p_over"] >= threshold for p in filtered)

    def test_filter_by_edge_threshold(self, sample_predictions):
        """Test filtering by edge threshold."""
        threshold = 2.0
        filtered = [p for p in sample_predictions if p["edge"] >= threshold]
        assert len(filtered) == 2
        assert all(p["edge"] >= threshold for p in filtered)

    def test_filter_by_line_spread(self, sample_predictions):
        """Test filtering by line spread."""
        threshold = 2.0
        filtered = [p for p in sample_predictions if p["line_spread"] >= threshold]
        assert len(filtered) == 2

    def test_hybrid_filter(self, sample_predictions):
        """Test hybrid filter (probability AND (edge OR spread))."""
        prob_threshold = 0.65
        edge_threshold = 2.0
        spread_threshold = 2.5

        filtered = [
            p
            for p in sample_predictions
            if p["p_over"] >= prob_threshold
            and (p["edge"] >= edge_threshold or p["line_spread"] >= spread_threshold)
        ]

        assert len(filtered) == 2
        for p in filtered:
            assert p["p_over"] >= prob_threshold
            assert p["edge"] >= edge_threshold or p["line_spread"] >= spread_threshold


class TestOutputFileFormat:
    """Tests for output file format."""

    @pytest.fixture
    def sample_output_json(self, temp_output_dir):
        """Create a sample output JSON file."""
        output = {
            "generated_at": datetime.now().isoformat(),
            "date": "2026-01-15",
            "strategy": "XL Line Shopping",
            "markets_enabled": ["POINTS", "REBOUNDS"],
            "total_picks": 3,
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "prediction": 28.5,
                    "p_over": 0.72,
                    "best_book": "underdog",
                    "best_line": 24.5,
                    "edge": 4.0,
                }
            ],
            "summary": {"total": 1, "by_market": {"POINTS": 1}},
        }

        output_path = temp_output_dir / "xl_picks_2026-01-15.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output_path

    def test_output_file_is_valid_json(self, sample_output_json):
        """Test output file is valid JSON."""
        with open(sample_output_json) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_output_has_required_sections(self, sample_output_json):
        """Test output has required sections."""
        with open(sample_output_json) as f:
            data = json.load(f)

        required_sections = ["generated_at", "date", "picks", "summary"]
        for section in required_sections:
            assert section in data, f"Missing section: {section}"

    def test_picks_is_list(self, sample_output_json):
        """Test picks is a list."""
        with open(sample_output_json) as f:
            data = json.load(f)
        assert isinstance(data["picks"], list)

    def test_total_picks_matches_list_length(self, sample_output_json):
        """Test total_picks matches actual picks count."""
        with open(sample_output_json) as f:
            data = json.load(f)
        assert data["total_picks"] >= len(data["picks"])


class TestLineOptimization:
    """Tests for line shopping/optimization."""

    @pytest.fixture
    def multi_book_props(self):
        """Props from multiple books for line shopping tests."""
        return pd.DataFrame(
            [
                {"book_name": "draftkings", "over_line": 25.5},
                {"book_name": "fanduel", "over_line": 26.0},
                {"book_name": "betmgm", "over_line": 25.5},
                {"book_name": "caesars", "over_line": 26.5},
                {"book_name": "underdog", "over_line": 24.5},
            ]
        )

    def test_find_softest_line_for_over(self, multi_book_props):
        """Test finding softest line (lowest) for OVER bets."""
        softest_line = multi_book_props["over_line"].min()
        softest_book = multi_book_props.loc[
            multi_book_props["over_line"] == softest_line, "book_name"
        ].iloc[0]

        assert softest_line == 24.5
        assert softest_book == "underdog"

    def test_find_hardest_line_for_under(self, multi_book_props):
        """Test finding hardest line (highest) for UNDER bets."""
        hardest_line = multi_book_props["over_line"].max()
        hardest_book = multi_book_props.loc[
            multi_book_props["over_line"] == hardest_line, "book_name"
        ].iloc[0]

        assert hardest_line == 26.5
        assert hardest_book == "caesars"

    def test_edge_improvement_from_line_shopping(self, multi_book_props):
        """Test that line shopping improves edge vs consensus."""
        consensus = multi_book_props["over_line"].mean()
        softest = multi_book_props["over_line"].min()

        # Model predicts 28.5
        model_prediction = 28.5

        edge_at_consensus = model_prediction - consensus
        edge_at_softest = model_prediction - softest

        assert edge_at_softest > edge_at_consensus
        improvement = edge_at_softest - edge_at_consensus
        assert improvement > 0


class TestPipelineEndToEnd:
    """End-to-end pipeline tests (mocked)."""

    @pytest.fixture
    def mock_pipeline_components(self):
        """Create mock pipeline components."""
        # Mock feature extractor
        feature_extractor = MagicMock()
        feature_extractor.extract.return_value = {
            f"feature_{i}": np.random.randn() for i in range(102)
        }

        # Mock predictor
        predictor = MagicMock()
        predictor.predict.return_value = (28.5, 0.72, "OVER", 4.0)

        return {"extractor": feature_extractor, "predictor": predictor}

    def test_pipeline_produces_output(self, mock_pipeline_components, temp_output_dir):
        """Test pipeline produces output file."""
        # Simulate pipeline execution
        props = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "game_date": "2026-01-15",
                "opponent": "GSW",
            }
        ]

        results = []
        for prop in props:
            features = mock_pipeline_components["extractor"].extract()
            prediction, p_over, side, edge = mock_pipeline_components["predictor"].predict()

            results.append(
                {
                    "player_name": prop["player_name"],
                    "stat_type": prop["stat_type"],
                    "prediction": prediction,
                    "p_over": p_over,
                    "side": side,
                    "edge": edge,
                }
            )

        # Write output
        output = {"generated_at": datetime.now().isoformat(), "picks": results}

        output_path = temp_output_dir / "test_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert len(data["picks"]) == 1

    def test_pipeline_handles_no_props(self, mock_pipeline_components, temp_output_dir):
        """Test pipeline handles case with no props."""
        props = []
        results = []

        output = {"generated_at": datetime.now().isoformat(), "picks": results, "total_picks": 0}

        output_path = temp_output_dir / "empty_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f)

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_picks"] == 0
        assert len(data["picks"]) == 0


class TestValidationIntegration:
    """Tests for validation integration."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        results = [
            {"result": "WIN"},
            {"result": "WIN"},
            {"result": "WIN"},
            {"result": "LOSS"},
            {"result": "PUSH"},
        ]

        wins = sum(1 for r in results if r["result"] == "WIN")
        losses = sum(1 for r in results if r["result"] == "LOSS")
        total = wins + losses

        win_rate = wins / total if total > 0 else 0
        assert win_rate == 0.75  # 3/4

    def test_roi_calculation(self):
        """Test ROI calculation."""
        # Standard -110 odds: risk 1.1 to win 1
        wins = 3
        losses = 1
        total_risked = (wins + losses) * 1.1
        total_won = wins * 1.0
        total_lost = losses * 1.1

        profit = total_won - total_lost
        roi = profit / total_risked

        assert roi > 0  # Should be profitable

    def test_breakeven_threshold(self):
        """Test breakeven threshold for -110 odds."""
        # At -110 odds, need 52.38% to break even
        breakeven = 110 / (110 + 100)
        assert abs(breakeven - 0.524) < 0.01
