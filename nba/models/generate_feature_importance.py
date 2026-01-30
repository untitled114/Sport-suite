#!/usr/bin/env python3
"""
SHAP Feature Importance Generator
=================================
Generates SHAP-based feature importance analysis for XL models.

Outputs:
- SHAP summary plots (beeswarm and bar charts)
- Feature importance tables in markdown
- Updates to model cards with feature importance sections

Usage:
    python -m nba.models.generate_feature_importance --market POINTS
    python -m nba.models.generate_feature_importance --all
"""

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available for plotting")


# Project paths
MODELS_DIR = Path(__file__).parent / "saved_xl"
MODEL_CARDS_DIR = Path(__file__).parent / "model_cards"
IMAGES_DIR = MODEL_CARDS_DIR / "images"


def load_model_components(market: str) -> Tuple[object, object, List[str], Dict]:
    """
    Load model components for a market.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)

    Returns:
        Tuple of (regressor, classifier, features, metadata)
    """
    market_lower = market.lower()

    # Try v3 models first, then xl models
    for prefix in [f"{market_lower}_market", f"{market_lower}_xl"]:
        try:
            with open(MODELS_DIR / f"{prefix}_regressor.pkl", "rb") as f:
                regressor = pickle.load(f)
            with open(MODELS_DIR / f"{prefix}_classifier.pkl", "rb") as f:
                classifier = pickle.load(f)
            with open(MODELS_DIR / f"{prefix}_features.pkl", "rb") as f:
                features = pickle.load(f)

            # Load metadata
            metadata = {}
            try:
                with open(MODELS_DIR / f"{prefix}_metadata.json", "r") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                pass

            logger.info(f"Loaded {prefix} models for {market}")
            return regressor, classifier, features, metadata

        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"No models found for {market}")


def generate_sample_data(features: List[str], n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample data for SHAP analysis.

    In production, this would load real training data.
    For now, generates synthetic data matching feature distributions.

    Args:
        features: List of feature names
        n_samples: Number of samples to generate

    Returns:
        DataFrame with sample feature data
    """
    np.random.seed(42)

    data = {}
    for feat in features:
        if "is_home" in feat:
            data[feat] = np.random.choice([0, 1], size=n_samples)
        elif "line" in feat.lower():
            data[feat] = np.random.uniform(10, 35, size=n_samples)
        elif "ema_points" in feat:
            data[feat] = np.random.normal(20, 5, size=n_samples)
        elif "ema_rebounds" in feat:
            data[feat] = np.random.normal(6, 2, size=n_samples)
        elif "ema_assists" in feat:
            data[feat] = np.random.normal(5, 2, size=n_samples)
        elif "pace" in feat:
            data[feat] = np.random.normal(100, 3, size=n_samples)
        elif "_pct" in feat or "rate" in feat:
            data[feat] = np.random.uniform(0.3, 0.7, size=n_samples)
        elif "deviation" in feat:
            data[feat] = np.random.normal(0, 0.5, size=n_samples)
        elif "spread" in feat:
            data[feat] = np.random.uniform(0, 3, size=n_samples)
        else:
            data[feat] = np.random.randn(n_samples)

    return pd.DataFrame(data)


def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    model_type: str = "tree",
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for a model.

    Args:
        model: Trained model (LightGBM regressor or classifier)
        X: Feature data
        model_type: Type of explainer ('tree' for LightGBM)

    Returns:
        SHAP values array or None on error
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP not available")
        return None

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)

        shap_values = explainer.shap_values(X)

        # Handle binary classifier output (returns list of [class0, class1])
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use class 1 (OVER) SHAP values

        return shap_values

    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.error(f"SHAP computation failed: {e}")
        return None


def get_feature_importance(
    shap_values: np.ndarray,
    features: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.

    Args:
        shap_values: SHAP values array
        features: Feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance rankings
    """
    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame(
        {
            "feature": features,
            "importance": importance,
        }
    )

    df = df.sort_values("importance", ascending=False).head(top_n)
    df["rank"] = range(1, len(df) + 1)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)

    return df[["rank", "feature", "importance", "importance_pct"]]


def generate_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    market: str,
    model_type: str,
) -> None:
    """
    Generate SHAP summary plots.

    Args:
        shap_values: SHAP values array
        X: Feature data
        market: Market name
        model_type: 'regressor' or 'classifier'
    """
    if not MATPLOTLIB_AVAILABLE or not SHAP_AVAILABLE:
        logger.warning("Plotting not available")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    market_lower = market.lower()

    # Beeswarm plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"{market} {model_type.title()} - SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{market_lower}_{model_type}_shap_summary.png", dpi=150)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"{market} {model_type.title()} - Mean |SHAP| Value")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{market_lower}_{model_type}_shap_bar.png", dpi=150)
    plt.close()

    logger.info(f"Saved plots to {IMAGES_DIR}")


def generate_markdown_table(
    importance_df: pd.DataFrame,
    market: str,
    model_type: str,
) -> str:
    """
    Generate markdown table of feature importance.

    Args:
        importance_df: Feature importance DataFrame
        market: Market name
        model_type: 'regressor' or 'classifier'

    Returns:
        Markdown table string
    """
    lines = [
        f"### {market} {model_type.title()} - Top 20 Features\n",
        "| Rank | Feature | Importance | % of Total |",
        "|------|---------|------------|------------|",
    ]

    for _, row in importance_df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['feature']}` | "
            f"{row['importance']:.4f} | {row['importance_pct']:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def update_model_card(market: str, importance_tables: Dict[str, str]) -> None:
    """
    Update model card with feature importance section.

    Args:
        market: Market name
        importance_tables: Dict of model_type -> markdown table
    """
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    card_path = MODEL_CARDS_DIR / f"{market}.md"

    # Create or update model card
    if card_path.exists():
        content = card_path.read_text()
    else:
        content = f"# {market} Model Card\n\n"

    # Add or update feature importance section
    section_marker = "## Feature Importance (SHAP)"
    if section_marker in content:
        # Replace existing section
        start = content.find(section_marker)
        end = content.find("\n## ", start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + content[end:]

    # Add new section
    importance_section = [
        f"\n{section_marker}\n",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
    ]

    for model_type, table in importance_tables.items():
        importance_section.append(f"\n{table}\n")

        # Add plot references
        market_lower = market.lower()
        importance_section.append(
            f"\n![{model_type} SHAP Summary](images/{market_lower}_{model_type}_shap_summary.png)\n"
        )

    content += "\n".join(importance_section)

    card_path.write_text(content)
    logger.info(f"Updated model card: {card_path}")


def analyze_market(market: str) -> None:
    """
    Run full SHAP analysis for a market.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)
    """
    logger.info(f"Analyzing {market}...")

    try:
        regressor, classifier, features, metadata = load_model_components(market)
    except FileNotFoundError as e:
        logger.error(f"Failed to load models: {e}")
        return

    # Generate sample data
    X = generate_sample_data(features)

    importance_tables = {}

    # Analyze regressor
    logger.info(f"Computing SHAP values for {market} regressor...")
    reg_shap = compute_shap_values(regressor, X)
    if reg_shap is not None:
        reg_importance = get_feature_importance(reg_shap, features)
        generate_plots(reg_shap, X, market, "regressor")
        importance_tables["regressor"] = generate_markdown_table(
            reg_importance, market, "regressor"
        )
        logger.info(f"Top 5 regressor features: {reg_importance['feature'].head().tolist()}")

    # Analyze classifier (add expected_diff feature)
    X_cls = X.copy()
    X_cls["expected_diff"] = np.random.uniform(-3, 3, len(X))
    cls_features = features + ["expected_diff"]

    logger.info(f"Computing SHAP values for {market} classifier...")
    cls_shap = compute_shap_values(classifier, X_cls)
    if cls_shap is not None:
        cls_importance = get_feature_importance(cls_shap, cls_features)
        generate_plots(cls_shap, X_cls, market, "classifier")
        importance_tables["classifier"] = generate_markdown_table(
            cls_importance, market, "classifier"
        )
        logger.info(f"Top 5 classifier features: {cls_importance['feature'].head().tolist()}")

    # Update model card
    if importance_tables:
        update_model_card(market, importance_tables)

    logger.info(f"Completed analysis for {market}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP feature importance analysis for XL models"
    )
    parser.add_argument(
        "--market",
        choices=["POINTS", "REBOUNDS", "ASSISTS", "THREES"],
        help="Market to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all markets",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not SHAP_AVAILABLE:
        logger.error("SHAP is required. Install with: pip install shap")
        return 1

    if args.all:
        for market in ["POINTS", "REBOUNDS"]:  # Only active markets
            analyze_market(market)
    elif args.market:
        analyze_market(args.market)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
