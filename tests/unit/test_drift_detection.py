"""
Unit Tests for Drift Detection Module
=====================================
Tests for feature drift detection using KS test.
"""

import numpy as np
import pandas as pd
import pytest


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_drift_detector_initialization(self):
        """Test drift detector initializes with defaults."""
        from nba.core.drift_detection import DriftDetector

        detector = DriftDetector()

        assert detector.drift_threshold == 0.05
        assert detector.min_samples == 100

    def test_drift_detector_custom_threshold(self):
        """Test drift detector accepts custom threshold."""
        from nba.core.drift_detection import DriftDetector

        detector = DriftDetector(drift_threshold=0.01)

        assert detector.drift_threshold == 0.01

    def test_set_reference_data(self):
        """Test setting reference data."""
        from nba.core.drift_detection import DriftDetector

        detector = DriftDetector()

        reference = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 100),
                "feature_2": np.random.normal(5, 1, 100),
            }
        )

        detector.set_reference(reference)

        assert detector._reference_data is not None
        assert len(detector._reference_data) == 100

    def test_check_drift_no_drift(self):
        """Test no drift detected when distributions are same."""
        from nba.core.drift_detection import DriftDetector

        np.random.seed(42)

        reference = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 500),
                "feature_2": np.random.normal(5, 1, 500),
            }
        )

        current = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 500),
                "feature_2": np.random.normal(5, 1, 500),
            }
        )

        detector = DriftDetector(drift_threshold=0.05)
        report = detector.check_drift(current_data=current, reference_data=reference)

        # With same distributions, should have few or no drifted features
        assert report.drifted_count <= 1  # Allow for random chance

    def test_check_drift_with_drift(self):
        """Test drift detected when distributions differ."""
        from nba.core.drift_detection import DriftDetector

        np.random.seed(42)

        reference = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 500),
                "feature_2": np.random.normal(5, 1, 500),
            }
        )

        # Shifted distribution
        current = pd.DataFrame(
            {
                "feature_1": np.random.normal(15, 3, 500),  # Mean 10->15, std 2->3
                "feature_2": np.random.normal(5, 1, 500),  # Same as reference
            }
        )

        detector = DriftDetector(drift_threshold=0.05)
        report = detector.check_drift(current_data=current, reference_data=reference)

        # feature_1 should be drifted
        assert "feature_1" in report.drifted_features
        assert report.drifted_features["feature_1"].is_drifted

        # feature_2 should NOT be drifted
        assert "feature_2" not in report.drifted_features

    def test_check_drift_uses_stored_reference(self):
        """Test check_drift uses stored reference if not provided."""
        from nba.core.drift_detection import DriftDetector

        np.random.seed(42)

        reference = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 500),
            }
        )

        current = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 2, 500),
            }
        )

        detector = DriftDetector()
        detector.set_reference(reference)

        # Call without reference_data
        report = detector.check_drift(current_data=current)

        assert report.reference_samples == 500


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    def test_drift_result_to_dict(self):
        """Test DriftResult converts to dict."""
        from nba.core.drift_detection import DriftResult

        result = DriftResult(
            feature_name="feature_1",
            ks_statistic=0.15,
            p_value=0.03,
            is_drifted=True,
            reference_mean=10.0,
            reference_std=2.0,
            current_mean=12.0,
            current_std=2.5,
            mean_shift=2.0,
            std_ratio=1.25,
        )

        d = result.to_dict()

        assert d["feature_name"] == "feature_1"
        assert d["ks_statistic"] == 0.15
        assert d["is_drifted"] is True
        assert d["mean_shift"] == 2.0


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_drift_report_significant_drift(self):
        """Test significant drift detection (>10% of features)."""
        from datetime import datetime

        from nba.core.drift_detection import DriftReport, DriftResult

        # Create results with 20% drift (2 of 10 features)
        results = {}
        for i in range(10):
            is_drifted = i < 2  # First 2 drifted
            results[f"feature_{i}"] = DriftResult(
                feature_name=f"feature_{i}",
                ks_statistic=0.2 if is_drifted else 0.05,
                p_value=0.01 if is_drifted else 0.5,
                is_drifted=is_drifted,
                reference_mean=10.0,
                reference_std=2.0,
                current_mean=12.0 if is_drifted else 10.0,
                current_std=2.0,
                mean_shift=2.0 if is_drifted else 0.0,
                std_ratio=1.0,
            )

        report = DriftReport(
            timestamp=datetime.now(),
            reference_samples=500,
            current_samples=500,
            total_features=10,
            drifted_count=2,
            drift_threshold=0.05,
            feature_results=results,
        )

        assert report.has_significant_drift is True
        assert report.drift_percentage == 20.0

    def test_drift_report_no_significant_drift(self):
        """Test no significant drift (<10% of features)."""
        from datetime import datetime

        from nba.core.drift_detection import DriftReport, DriftResult

        # Create results with 5% drift (1 of 20 features)
        results = {}
        for i in range(20):
            is_drifted = i == 0  # Only first drifted
            results[f"feature_{i}"] = DriftResult(
                feature_name=f"feature_{i}",
                ks_statistic=0.2 if is_drifted else 0.05,
                p_value=0.01 if is_drifted else 0.5,
                is_drifted=is_drifted,
                reference_mean=10.0,
                reference_std=2.0,
                current_mean=10.0,
                current_std=2.0,
                mean_shift=0.0,
                std_ratio=1.0,
            )

        report = DriftReport(
            timestamp=datetime.now(),
            reference_samples=500,
            current_samples=500,
            total_features=20,
            drifted_count=1,
            drift_threshold=0.05,
            feature_results=results,
        )

        assert report.has_significant_drift is False
        assert report.drift_percentage == 5.0

    def test_drift_report_top_drifted(self):
        """Test top_drifted returns sorted by KS statistic."""
        from datetime import datetime

        from nba.core.drift_detection import DriftReport, DriftResult

        results = {
            "feature_a": DriftResult(
                feature_name="feature_a",
                ks_statistic=0.3,
                p_value=0.001,
                is_drifted=True,
                reference_mean=10.0,
                reference_std=2.0,
                current_mean=15.0,
                current_std=2.0,
                mean_shift=5.0,
                std_ratio=1.0,
            ),
            "feature_b": DriftResult(
                feature_name="feature_b",
                ks_statistic=0.5,
                p_value=0.0001,
                is_drifted=True,
                reference_mean=10.0,
                reference_std=2.0,
                current_mean=20.0,
                current_std=2.0,
                mean_shift=10.0,
                std_ratio=1.0,
            ),
        }

        report = DriftReport(
            timestamp=datetime.now(),
            reference_samples=500,
            current_samples=500,
            total_features=2,
            drifted_count=2,
            drift_threshold=0.05,
            feature_results=results,
        )

        top = report.top_drifted
        assert top[0][0] == "feature_b"  # Highest KS statistic first
        assert top[1][0] == "feature_a"

    def test_drift_report_summary(self):
        """Test summary generation."""
        from datetime import datetime

        from nba.core.drift_detection import DriftReport

        report = DriftReport(
            timestamp=datetime.now(),
            reference_samples=500,
            current_samples=500,
            total_features=10,
            drifted_count=0,
            drift_threshold=0.05,
            feature_results={},
        )

        summary = report.summary()

        assert "FEATURE DRIFT REPORT" in summary
        assert "Reference samples: 500" in summary
        assert "No significant drift detected" in summary


class TestPopulationStabilityIndex:
    """Tests for PSI calculation."""

    def test_psi_no_shift(self):
        """Test PSI close to 0 when no shift."""
        from nba.core.drift_detection import PopulationStabilityIndex

        np.random.seed(42)
        reference = np.random.normal(10, 2, 1000)
        current = np.random.normal(10, 2, 1000)

        psi = PopulationStabilityIndex.calculate(reference, current)

        assert psi < 0.1  # No significant shift

    def test_psi_significant_shift(self):
        """Test PSI >= 0.2 when significant shift."""
        from nba.core.drift_detection import PopulationStabilityIndex

        np.random.seed(42)
        reference = np.random.normal(10, 2, 1000)
        current = np.random.normal(20, 2, 1000)  # Large mean shift

        psi = PopulationStabilityIndex.calculate(reference, current)

        assert psi >= 0.2  # Significant shift

    def test_psi_interpret(self):
        """Test PSI interpretation."""
        from nba.core.drift_detection import PopulationStabilityIndex

        assert "No significant" in PopulationStabilityIndex.interpret(0.05)
        assert "Moderate" in PopulationStabilityIndex.interpret(0.15)
        assert "Significant" in PopulationStabilityIndex.interpret(0.25)


class TestCheckSingleFeature:
    """Tests for single feature drift check."""

    def test_check_single_feature_drift(self):
        """Test checking drift for single feature."""
        from nba.core.drift_detection import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(10, 2, 500)
        current = np.random.normal(15, 3, 500)  # Different distribution

        detector = DriftDetector(drift_threshold=0.05)
        result = detector.check_single_feature("test_feature", reference, current)

        assert result.feature_name == "test_feature"
        assert result.is_drifted  # numpy bool, use truthiness not identity
        assert result.ks_statistic > 0.1

    def test_check_single_feature_no_drift(self):
        """Test checking single feature with no drift."""
        from nba.core.drift_detection import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(10, 2, 500)
        current = np.random.normal(10, 2, 500)  # Same distribution

        detector = DriftDetector(drift_threshold=0.05)
        result = detector.check_single_feature("test_feature", reference, current)

        # Likely no drift (though random chance can sometimes cause it)
        assert result.p_value > 0.01
