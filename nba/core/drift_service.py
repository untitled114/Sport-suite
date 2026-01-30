"""
Production Drift Detection Service
===================================
Provides real-time drift detection for production predictions.

Integrates with reference distributions to detect feature drift
and alert when model inputs deviate from training distribution.

Usage:
    from nba.core.drift_service import DriftService

    service = DriftService("POINTS")
    result = service.check_drift(current_features)

    if result.has_drift:
        logger.warning(f"Drift detected: {result.drifted_features}")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import psycopg2

from nba.core.reference_distributions import ReferenceDistributions

logger = logging.getLogger(__name__)


# Default reference distributions path
REFERENCE_DIR = Path(__file__).parent.parent / "models" / "reference_distributions"


@dataclass
class DriftCheckResult:
    """Result of a drift check."""

    market: str
    timestamp: str
    total_features: int
    checked_features: int
    drifted_features: List[str] = field(default_factory=list)
    drift_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_features: List[str] = field(default_factory=list)
    severity: str = "none"  # none, low, medium, high

    @property
    def has_drift(self) -> bool:
        """Check if any drift was detected."""
        return len(self.drifted_features) > 0

    @property
    def drift_percentage(self) -> float:
        """Percentage of features that drifted."""
        if self.checked_features == 0:
            return 0.0
        return len(self.drifted_features) / self.checked_features * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp,
            "has_drift": self.has_drift,
            "total_features": self.total_features,
            "checked_features": self.checked_features,
            "drifted_count": len(self.drifted_features),
            "drift_percentage": round(self.drift_percentage, 2),
            "drifted_features": self.drifted_features,
            "drift_details": self.drift_details,
            "missing_features": self.missing_features,
            "severity": self.severity,
        }


class DriftService:
    """
    Production drift detection service.

    Compares incoming feature distributions against reference
    distributions built from training data.
    """

    def __init__(
        self,
        market: str,
        reference_path: str = None,
        z_threshold: float = 3.0,
        missing_threshold: float = 0.1,
        range_multiplier: float = 1.5,
    ):
        """
        Initialize drift service.

        Args:
            market: Market name (POINTS, REBOUNDS, etc.)
            reference_path: Path to reference distributions JSON
            z_threshold: Z-score threshold for drift detection
            missing_threshold: Threshold for excessive missing values
            range_multiplier: Multiplier for acceptable range expansion
        """
        self.market = market.upper()
        self.z_threshold = z_threshold
        self.missing_threshold = missing_threshold
        self.range_multiplier = range_multiplier

        # Load reference distributions
        if reference_path:
            ref_path = Path(reference_path)
        else:
            ref_path = REFERENCE_DIR / f"{self.market.lower()}_reference.json"

        self.reference = None
        self._load_reference(ref_path)

    def _load_reference(self, path: Path) -> None:
        """Load reference distributions from file."""
        if path.exists():
            try:
                self.reference = ReferenceDistributions.load(str(path))
                logger.info(
                    f"Loaded reference distributions for {self.market} "
                    f"({len(self.reference.features)} features)"
                )
            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to load reference distributions: {e}")
        else:
            logger.warning(f"Reference distributions not found: {path}")

    def check_drift(
        self,
        features: Dict[str, float],
        check_range: bool = True,
        check_missing: bool = True,
        check_zscore: bool = True,
    ) -> DriftCheckResult:
        """
        Check for drift in feature values.

        Args:
            features: Dict of feature name -> value
            check_range: Check if values are outside expected range
            check_missing: Check for excessive missing values
            check_zscore: Check for z-score violations

        Returns:
            DriftCheckResult with drift detection results
        """
        if self.reference is None:
            return DriftCheckResult(
                market=self.market,
                timestamp=datetime.now().isoformat(),
                total_features=len(features),
                checked_features=0,
                severity="unknown",
            )

        drifted = []
        details = {}
        missing = []
        checked = 0

        for name, value in features.items():
            ref = self.reference.get_feature(name)

            if ref is None:
                missing.append(name)
                continue

            checked += 1
            drift_reasons = []

            # Check for missing/NaN
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if check_missing and ref.missing_rate < self.missing_threshold:
                    drift_reasons.append("unexpected_missing")
                continue

            value = float(value)

            # Check range
            if check_range:
                range_buffer = (ref.max - ref.min) * (self.range_multiplier - 1)
                expected_min = ref.min - range_buffer
                expected_max = ref.max + range_buffer

                if value < expected_min or value > expected_max:
                    drift_reasons.append("out_of_range")

            # Check z-score
            if check_zscore and ref.std > 0:
                z_score = abs(value - ref.mean) / ref.std
                if z_score > self.z_threshold:
                    drift_reasons.append(f"high_zscore_{z_score:.1f}")

            if drift_reasons:
                drifted.append(name)
                details[name] = {
                    "value": value,
                    "reference_mean": ref.mean,
                    "reference_std": ref.std,
                    "reference_range": [ref.min, ref.max],
                    "reasons": drift_reasons,
                }

        # Determine severity
        drift_pct = len(drifted) / checked * 100 if checked > 0 else 0
        if drift_pct >= 20:
            severity = "high"
        elif drift_pct >= 10:
            severity = "medium"
        elif drift_pct > 0:
            severity = "low"
        else:
            severity = "none"

        return DriftCheckResult(
            market=self.market,
            timestamp=datetime.now().isoformat(),
            total_features=len(features),
            checked_features=checked,
            drifted_features=drifted,
            drift_details=details,
            missing_features=missing,
            severity=severity,
        )

    def check_batch_drift(
        self,
        feature_df: pd.DataFrame,
    ) -> DriftCheckResult:
        """
        Check for drift in a batch of feature samples.

        Args:
            feature_df: DataFrame with feature columns

        Returns:
            DriftCheckResult aggregated across samples
        """
        if self.reference is None or len(feature_df) == 0:
            return DriftCheckResult(
                market=self.market,
                timestamp=datetime.now().isoformat(),
                total_features=len(feature_df.columns),
                checked_features=0,
                severity="unknown",
            )

        drifted = []
        details = {}
        missing = []
        checked = 0

        for col in feature_df.columns:
            ref = self.reference.get_feature(col)

            if ref is None:
                missing.append(col)
                continue

            checked += 1
            series = feature_df[col].dropna()
            drift_reasons = []

            if len(series) == 0:
                continue

            # Compare batch mean to reference
            batch_mean = float(series.mean())
            batch_std = float(series.std())

            # Check if batch mean is significantly different
            if ref.std > 0:
                z_score = abs(batch_mean - ref.mean) / ref.std
                if z_score > self.z_threshold:
                    drift_reasons.append(f"mean_shift_{z_score:.1f}")

            # Check if batch std changed significantly
            if ref.std > 0:
                std_ratio = batch_std / ref.std
                if std_ratio < 0.5 or std_ratio > 2.0:
                    drift_reasons.append(f"variance_change_{std_ratio:.2f}")

            if drift_reasons:
                drifted.append(col)
                details[col] = {
                    "batch_mean": batch_mean,
                    "batch_std": batch_std,
                    "reference_mean": ref.mean,
                    "reference_std": ref.std,
                    "reasons": drift_reasons,
                }

        # Determine severity
        drift_pct = len(drifted) / checked * 100 if checked > 0 else 0
        if drift_pct >= 20:
            severity = "high"
        elif drift_pct >= 10:
            severity = "medium"
        elif drift_pct > 0:
            severity = "low"
        else:
            severity = "none"

        return DriftCheckResult(
            market=self.market,
            timestamp=datetime.now().isoformat(),
            total_features=len(feature_df.columns),
            checked_features=checked,
            drifted_features=drifted,
            drift_details=details,
            missing_features=missing,
            severity=severity,
        )

    def should_alert(self, result: DriftCheckResult) -> bool:
        """
        Determine if drift result should trigger an alert.

        Args:
            result: Drift check result

        Returns:
            True if alert should be sent
        """
        return result.severity in ("medium", "high")

    def get_status(self) -> Dict[str, Any]:
        """Get current drift service status."""
        if self.reference is None:
            return {
                "market": self.market,
                "status": "no_reference",
                "features": 0,
            }

        return {
            "market": self.market,
            "status": "ready",
            "features": len(self.reference.features),
            "reference_created": self.reference.created_at,
            "training_samples": self.reference.training_samples,
            "thresholds": {
                "z_threshold": self.z_threshold,
                "missing_threshold": self.missing_threshold,
                "range_multiplier": self.range_multiplier,
            },
        }
