"""
Feature Drift Detection Module
==============================
Detects distribution shifts in features using statistical tests.

Key Concepts:
- Feature drift: When the statistical properties of input features change
- Model staleness: Drift can cause model performance degradation
- Monitoring: Regular drift checks help identify when retraining is needed

Usage:
    from nba.core.drift_detection import DriftDetector

    detector = DriftDetector()

    # Compare training vs production distributions
    drift_report = detector.check_drift(
        reference_data=training_features,
        current_data=production_features,
        feature_names=feature_list
    )

    if drift_report.has_significant_drift:
        print("Warning: Feature drift detected!")
        for feature, info in drift_report.drifted_features.items():
            print(f"  {feature}: KS stat={info['ks_stat']:.3f}, p={info['p_value']:.4f}")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a drift test for a single feature."""

    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    mean_shift: float  # Absolute change in mean
    std_ratio: float  # Ratio of current/reference std

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "ks_statistic": self.ks_statistic,
            "p_value": self.p_value,
            "is_drifted": self.is_drifted,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "current_mean": self.current_mean,
            "current_std": self.current_std,
            "mean_shift": self.mean_shift,
            "std_ratio": self.std_ratio,
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""

    timestamp: datetime
    reference_samples: int
    current_samples: int
    total_features: int
    drifted_count: int
    drift_threshold: float
    significant_drift_pct: float = 0.10  # Configurable threshold for "significant" drift
    bonferroni_corrected: bool = False  # Whether Bonferroni correction was applied
    feature_results: Dict[str, DriftResult] = field(default_factory=dict)

    @property
    def has_significant_drift(self) -> bool:
        """Check if significant drift was detected."""
        return self.drifted_count / max(self.total_features, 1) > self.significant_drift_pct

    @property
    def drift_percentage(self) -> float:
        """Percentage of features with drift."""
        return self.drifted_count / max(self.total_features, 1) * 100

    @property
    def drifted_features(self) -> Dict[str, DriftResult]:
        """Get only the features that drifted."""
        return {name: result for name, result in self.feature_results.items() if result.is_drifted}

    @property
    def top_drifted(self) -> List[Tuple[str, DriftResult]]:
        """Get features sorted by drift severity (KS statistic)."""
        return sorted(
            self.drifted_features.items(),
            key=lambda x: x[1].ks_statistic,
            reverse=True,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
            "total_features": self.total_features,
            "drifted_count": self.drifted_count,
            "drift_percentage": self.drift_percentage,
            "has_significant_drift": self.has_significant_drift,
            "drift_threshold": self.drift_threshold,
            "significant_drift_pct": self.significant_drift_pct,
            "bonferroni_corrected": self.bonferroni_corrected,
            "feature_results": {
                name: result.to_dict() for name, result in self.feature_results.items()
            },
        }

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        bonferroni_note = " (Bonferroni corrected)" if self.bonferroni_corrected else ""
        lines = [
            "=" * 60,
            "FEATURE DRIFT REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Reference samples: {self.reference_samples:,}",
            f"Current samples: {self.current_samples:,}",
            f"Total features: {self.total_features}",
            f"Drift threshold (p-value): {self.drift_threshold:.6f}{bonferroni_note}",
            f"Significant drift threshold: {self.significant_drift_pct:.0%} of features",
            "",
            f"DRIFT DETECTED: {self.drifted_count} features ({self.drift_percentage:.1f}%)",
            "",
        ]

        if self.has_significant_drift:
            lines.append("⚠️  SIGNIFICANT DRIFT - Consider retraining the model")
        else:
            lines.append("✓ No significant drift detected")

        if self.drifted_count > 0:
            lines.append("")
            lines.append("Top Drifted Features:")
            lines.append("-" * 60)
            for name, result in self.top_drifted[:10]:
                lines.append(
                    f"  {name:30s} KS={result.ks_statistic:.3f} "
                    f"p={result.p_value:.4f} shift={result.mean_shift:+.3f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class DriftDetector:
    """
    Kolmogorov-Smirnov based feature drift detector.

    Uses the two-sample KS test to compare distributions between
    reference (training) and current (production) data.

    The KS test:
    - Compares empirical cumulative distribution functions
    - Non-parametric (makes no distributional assumptions)
    - Sensitive to both location and shape changes
    - Returns p-value indicating probability distributions are same

    Bonferroni Correction:
    When testing many features simultaneously, the probability of false positives
    increases (multiple comparisons problem). Bonferroni correction adjusts the
    threshold by dividing by the number of tests: adjusted_threshold = alpha / n_tests

    For 102 features with alpha=0.05: adjusted_threshold = 0.05/102 ≈ 0.00049
    This is conservative but reduces false positives significantly.
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        min_samples: int = 100,
        use_bonferroni: bool = True,
        significant_drift_pct: float = 0.10,
    ):
        """
        Initialize drift detector.

        Args:
            drift_threshold: Base p-value threshold for drift detection
                            (lower = more confident about drift)
            min_samples: Minimum samples required for reliable testing
            use_bonferroni: Apply Bonferroni correction for multiple comparisons
                           (recommended when testing many features)
            significant_drift_pct: Percentage of drifted features above which
                                   drift is considered "significant" (default: 10%)
        """
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.use_bonferroni = use_bonferroni
        self.significant_drift_pct = significant_drift_pct
        self._reference_data: Optional[pd.DataFrame] = None
        self._reference_stats: Optional[dict] = None

    def set_reference(self, data: pd.DataFrame):
        """
        Set reference distribution (typically training data).

        Args:
            data: DataFrame with features as columns
        """
        if len(data) < self.min_samples:
            logger.warning(
                f"Reference data has {len(data)} samples, "
                f"minimum recommended is {self.min_samples}"
            )

        self._reference_data = data.copy()
        self._reference_stats = self._compute_stats(data)
        logger.info(f"Set reference data: {len(data)} samples, {len(data.columns)} features")

    def _compute_stats(self, data: pd.DataFrame) -> dict:
        """Compute basic statistics for each feature."""
        stats_dict = {}
        for col in data.columns:
            values = data[col].dropna()
            if len(values) > 0:
                stats_dict[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                }
        return stats_dict

    def check_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Check for feature drift between reference and current data.

        Args:
            current_data: Current/production data to check
            reference_data: Reference/training data (uses stored if None)
            feature_names: Specific features to check (all if None)

        Returns:
            DriftReport with detailed results
        """
        # Use provided reference or stored reference
        if reference_data is not None:
            ref_data = reference_data
        elif self._reference_data is not None:
            ref_data = self._reference_data
        else:
            raise ValueError("No reference data provided. Call set_reference() first.")

        # Determine features to check
        if feature_names is None:
            # Check intersection of features
            common_features = list(set(ref_data.columns) & set(current_data.columns))
        else:
            common_features = [
                f for f in feature_names if f in ref_data.columns and f in current_data.columns
            ]

        if not common_features:
            raise ValueError("No common features found between reference and current data")

        # Calculate adjusted threshold for Bonferroni correction
        n_features = len(common_features)
        if self.use_bonferroni and n_features > 1:
            adjusted_threshold = self.drift_threshold / n_features
            logger.info(
                f"Bonferroni correction: {self.drift_threshold:.4f} / {n_features} = {adjusted_threshold:.6f}"
            )
        else:
            adjusted_threshold = self.drift_threshold

        # Run KS test on each feature
        results = {}
        drifted_count = 0

        for feature in common_features:
            ref_values = ref_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values

            # Skip if insufficient data
            if len(ref_values) < 10 or len(cur_values) < 10:
                logger.debug(f"Skipping {feature}: insufficient data")
                continue

            # Run two-sample KS test
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)

            # Compute statistics
            ref_mean = float(np.mean(ref_values))
            ref_std = float(np.std(ref_values))
            cur_mean = float(np.mean(cur_values))
            cur_std = float(np.std(cur_values))

            is_drifted = p_value < adjusted_threshold

            results[feature] = DriftResult(
                feature_name=feature,
                ks_statistic=float(ks_stat),
                p_value=float(p_value),
                is_drifted=is_drifted,
                reference_mean=ref_mean,
                reference_std=ref_std,
                current_mean=cur_mean,
                current_std=cur_std,
                mean_shift=cur_mean - ref_mean,
                std_ratio=cur_std / ref_std if ref_std > 0 else 0.0,
            )

            if is_drifted:
                drifted_count += 1

        report = DriftReport(
            timestamp=datetime.now(),
            reference_samples=len(ref_data),
            current_samples=len(current_data),
            total_features=len(results),
            drifted_count=drifted_count,
            drift_threshold=adjusted_threshold,  # Use adjusted threshold
            significant_drift_pct=self.significant_drift_pct,
            bonferroni_corrected=self.use_bonferroni and n_features > 1,
            feature_results=results,
        )

        return report

    def check_single_feature(
        self,
        feature_name: str,
        reference_values: np.ndarray,
        current_values: np.ndarray,
    ) -> DriftResult:
        """
        Check drift for a single feature.

        Args:
            feature_name: Name of the feature
            reference_values: Reference distribution values
            current_values: Current distribution values

        Returns:
            DriftResult for the feature
        """
        ref_values = np.array(reference_values).flatten()
        cur_values = np.array(current_values).flatten()

        # Remove NaN/inf
        ref_values = ref_values[np.isfinite(ref_values)]
        cur_values = cur_values[np.isfinite(cur_values)]

        ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)

        ref_mean = float(np.mean(ref_values))
        ref_std = float(np.std(ref_values))
        cur_mean = float(np.mean(cur_values))
        cur_std = float(np.std(cur_values))

        return DriftResult(
            feature_name=feature_name,
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            is_drifted=p_value < self.drift_threshold,
            reference_mean=ref_mean,
            reference_std=ref_std,
            current_mean=cur_mean,
            current_std=cur_std,
            mean_shift=cur_mean - ref_mean,
            std_ratio=cur_std / ref_std if ref_std > 0 else 0.0,
        )


class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) calculator.

    PSI measures how much a population has shifted between two samples.
    Common interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.2: Moderate shift, monitor
    - PSI >= 0.2: Significant shift, action required

    Formula: PSI = Sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """

    @staticmethod
    def calculate(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate PSI between reference and current distributions.

        Args:
            reference: Reference distribution values
            current: Current distribution values
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Add small epsilon to avoid division by zero
        eps = 1e-10

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_proportions = ref_counts / len(reference) + eps
        cur_proportions = cur_counts / len(current) + eps

        # Calculate PSI
        psi = np.sum(
            (cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions)
        )

        return float(psi)

    @staticmethod
    def interpret(psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant shift"
        elif psi < 0.2:
            return "Moderate shift - monitor"
        else:
            return "Significant shift - action required"


def run_drift_check(
    training_data_path: str,
    production_data_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.05,
) -> DriftReport:
    """
    Convenience function to run drift check from files.

    Args:
        training_data_path: Path to training data CSV
        production_data_path: Path to production data CSV
        output_path: Optional path to save report
        threshold: P-value threshold for drift detection

    Returns:
        DriftReport
    """
    logger.info(f"Loading training data from {training_data_path}")
    training_data = pd.read_csv(training_data_path)

    logger.info(f"Loading production data from {production_data_path}")
    production_data = pd.read_csv(production_data_path)

    # Exclude non-feature columns
    exclude_cols = [
        "player_name",
        "game_date",
        "actual_result",
        "actual_points",
        "actual_rebounds",
        "actual_assists",
        "actual_threes",
        "stat_type",
        "source",
        "prop_id",
        "label",
        "hit_over",
    ]

    feature_cols = [
        col
        for col in training_data.columns
        if col not in exclude_cols and col in production_data.columns
    ]

    detector = DriftDetector(drift_threshold=threshold)
    report = detector.check_drift(
        current_data=production_data[feature_cols],
        reference_data=training_data[feature_cols],
    )

    print(report.summary())

    if output_path:
        report.save(output_path)
        logger.info(f"Report saved to {output_path}")

    return report


if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np

    print("Feature Drift Detection Demo")
    print("=" * 60)

    # Create synthetic reference data
    np.random.seed(42)
    n_samples = 1000
    reference = pd.DataFrame(
        {
            "feature_1": np.random.normal(10, 2, n_samples),
            "feature_2": np.random.normal(5, 1, n_samples),
            "feature_3": np.random.uniform(0, 1, n_samples),
        }
    )

    # Create current data with drift in feature_1
    current = pd.DataFrame(
        {
            "feature_1": np.random.normal(12, 2.5, n_samples),  # Shifted mean and std
            "feature_2": np.random.normal(5, 1, n_samples),  # No drift
            "feature_3": np.random.uniform(0, 1, n_samples),  # No drift
        }
    )

    detector = DriftDetector(drift_threshold=0.05)
    report = detector.check_drift(current_data=current, reference_data=reference)

    print(report.summary())
