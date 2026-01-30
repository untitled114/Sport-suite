"""
Reference Distribution Builder
==============================
Builds reference distributions from training data for drift detection.

Usage:
    from nba.core.reference_distributions import build_reference_distributions

    # Build reference from training data
    reference = build_reference_distributions(training_df, feature_names)

    # Save for production use
    reference.save("reference_distributions.json")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureDistribution:
    """Distribution statistics for a single feature."""

    name: str
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    missing_rate: float
    unique_count: int
    is_categorical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "is_categorical": self.is_categorical,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDistribution":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ReferenceDistributions:
    """Collection of reference distributions for all features."""

    market: str
    created_at: str
    training_samples: int
    features: Dict[str, FeatureDistribution] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "created_at": self.created_at,
            "training_samples": self.training_samples,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceDistributions":
        """Create from dictionary."""
        features = {
            k: FeatureDistribution.from_dict(v) for k, v in data.get("features", {}).items()
        }
        return cls(
            market=data["market"],
            created_at=data["created_at"],
            training_samples=data["training_samples"],
            features=features,
        )

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved reference distributions to {path}")

    @classmethod
    def load(cls, path: str) -> "ReferenceDistributions":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_feature(self, name: str) -> Optional[FeatureDistribution]:
        """Get distribution for a specific feature."""
        return self.features.get(name)


def compute_feature_distribution(
    series: pd.Series,
    name: str,
) -> FeatureDistribution:
    """
    Compute distribution statistics for a single feature.

    Args:
        series: Pandas Series with feature values
        name: Feature name

    Returns:
        FeatureDistribution with computed statistics
    """
    # Handle missing values
    valid = series.dropna()
    missing_rate = 1.0 - (len(valid) / len(series)) if len(series) > 0 else 1.0

    if len(valid) == 0:
        return FeatureDistribution(
            name=name,
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            missing_rate=missing_rate,
            unique_count=0,
            is_categorical=False,
        )

    # Check if categorical
    unique_count = valid.nunique()
    is_categorical = unique_count <= 10 or valid.dtype == "object"

    return FeatureDistribution(
        name=name,
        mean=float(valid.mean()),
        std=float(valid.std()),
        min=float(valid.min()),
        max=float(valid.max()),
        p25=float(valid.quantile(0.25)),
        p50=float(valid.quantile(0.50)),
        p75=float(valid.quantile(0.75)),
        missing_rate=missing_rate,
        unique_count=unique_count,
        is_categorical=is_categorical,
    )


def build_reference_distributions(
    df: pd.DataFrame,
    feature_names: List[str],
    market: str = "UNKNOWN",
) -> ReferenceDistributions:
    """
    Build reference distributions from training data.

    Args:
        df: Training data DataFrame
        feature_names: List of feature column names
        market: Market name (POINTS, REBOUNDS, etc.)

    Returns:
        ReferenceDistributions object
    """
    logger.info(f"Building reference distributions for {len(feature_names)} features")

    features = {}
    for name in feature_names:
        if name in df.columns:
            features[name] = compute_feature_distribution(df[name], name)
        else:
            logger.warning(f"Feature {name} not found in DataFrame")

    return ReferenceDistributions(
        market=market,
        created_at=datetime.now().isoformat(),
        training_samples=len(df),
        features=features,
    )


def build_from_training_dataset(
    csv_path: str,
    market: str,
    output_path: str = None,
) -> ReferenceDistributions:
    """
    Build reference distributions from a training CSV file.

    Args:
        csv_path: Path to training CSV
        market: Market name
        output_path: Optional output path for JSON

    Returns:
        ReferenceDistributions object
    """
    logger.info(f"Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Exclude metadata columns
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
        "season",
        "hit_over",
        "residual",
    ]
    feature_names = [c for c in df.columns if c not in exclude_cols]

    reference = build_reference_distributions(df, feature_names, market)

    if output_path:
        reference.save(output_path)

    return reference
