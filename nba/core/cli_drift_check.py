#!/usr/bin/env python3
"""
Drift Detection CLI Tool
========================
Manual drift checking and reference distribution generation.

Usage:
    # Generate reference distributions from training data
    python -m nba.core.cli_drift_check --market POINTS --generate-reference

    # Check current predictions for drift
    python -m nba.core.cli_drift_check --market POINTS --check-latest

    # Get drift service status
    python -m nba.core.cli_drift_check --status
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
REFERENCE_DIR = PROJECT_ROOT / "nba" / "models" / "reference_distributions"
TRAINING_DIR = PROJECT_ROOT / "nba" / "features" / "datasets"


def generate_reference(market: str) -> int:
    """Generate reference distributions from training data."""
    from nba.core.reference_distributions import build_from_training_dataset

    market_upper = market.upper()
    market_lower = market.lower()

    # Find training data
    training_file = TRAINING_DIR / f"xl_training_{market_upper}_2023_2025.csv"

    if not training_file.exists():
        logger.error(f"Training data not found: {training_file}")
        return 1

    # Create output directory
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REFERENCE_DIR / f"{market_lower}_reference.json"

    logger.info(f"Building reference distributions for {market_upper}...")
    reference = build_from_training_dataset(
        str(training_file),
        market_upper,
        str(output_path),
    )

    print(f"\nReference Distributions Generated:")
    print(f"  Market: {reference.market}")
    print(f"  Features: {len(reference.features)}")
    print(f"  Training Samples: {reference.training_samples}")
    print(f"  Output: {output_path}")

    return 0


def check_latest(market: str) -> int:
    """Check latest predictions for drift."""
    from nba.core.drift_service import DriftService

    service = DriftService(market)
    status = service.get_status()

    if status["status"] != "ready":
        logger.error(f"Drift service not ready: {status['status']}")
        logger.info("Run with --generate-reference first")
        return 1

    print(f"\nDrift Service Status:")
    print(f"  Market: {status['market']}")
    print(f"  Features: {status['features']}")
    print(f"  Reference Created: {status['reference_created']}")
    print(f"  Training Samples: {status['training_samples']}")

    # TODO: Load recent predictions and check drift
    print("\nNote: Full drift check requires integration with prediction pipeline")

    return 0


def show_status() -> int:
    """Show drift service status for all markets."""
    from nba.core.drift_service import DriftService

    print("\nDrift Detection Status:")
    print("=" * 60)

    for market in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
        service = DriftService(market)
        status = service.get_status()

        status_icon = "\u2705" if status["status"] == "ready" else "\u274c"
        print(f"\n{status_icon} {market}")
        print(f"   Status: {status['status']}")

        if status["status"] == "ready":
            print(f"   Features: {status['features']}")
            print(f"   Reference: {status['reference_created']}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Drift detection CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--market",
        choices=["POINTS", "REBOUNDS", "ASSISTS", "THREES"],
        help="Market to analyze",
    )

    parser.add_argument(
        "--generate-reference",
        action="store_true",
        help="Generate reference distributions from training data",
    )

    parser.add_argument(
        "--check-latest",
        action="store_true",
        help="Check latest predictions for drift",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show drift service status",
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
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.status:
        return show_status()

    if args.generate_reference:
        if not args.market:
            logger.error("--market required with --generate-reference")
            return 1
        return generate_reference(args.market)

    if args.check_latest:
        if not args.market:
            logger.error("--market required with --check-latest")
            return 1
        return check_latest(args.market)

    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
