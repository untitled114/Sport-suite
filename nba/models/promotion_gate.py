"""
Promotion Gate — Performance Efficiency Pillar
================================================
Stores walk-forward validation results and makes promotion decisions
based on data, not judgment calls.

Flow:
    1. Train new model
    2. Run walk-forward validation → store results in axiom.validation_runs
    3. Query promotion gate: does new model beat production?
    4. If yes → shadow for 3 days → promote
    5. If 7d WR drops post-promotion → auto-rollback via ModelRegistry

Usage:
    gate = PromotionGate()

    # After walk-forward validation
    run_id = gate.store_validation("v4_POINTS_20260322", "POINTS", summary)

    # Promotion decision
    decision = gate.evaluate("v4_POINTS_20260322", "POINTS")
    if decision["promote"]:
        registry.start_shadow("v4_POINTS_20260322")
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from nba.config.database import get_connection

EST = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class PromotionGate:
    """Stores validation results and makes promotion decisions."""

    def __init__(self, conn=None):
        self._external_conn = conn is not None
        self._conn = conn

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = get_connection("axiom")
        return self._conn

    def close(self):
        if self._conn and not self._external_conn and not self._conn.closed:
            self._conn.close()

    def store_validation(
        self,
        model_version: str,
        market: str,
        auc_mean: float,
        auc_std: float,
        wr_mean: float,
        roi_mean: float,
        fold_count: int,
        beats_baseline: bool = False,
        raw_results: Optional[Dict] = None,
    ) -> Optional[int]:
        """
        Store walk-forward validation results.

        Returns the validation_run id, or None on failure.
        """
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO validation_runs
                        (model_version, market, auc_mean, auc_std, wr_mean, roi_mean,
                         fold_count, beats_baseline, raw_results)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        model_version,
                        market.upper(),
                        round(auc_mean, 4),
                        round(auc_std, 4),
                        round(wr_mean, 4),
                        round(roi_mean, 4),
                        fold_count,
                        beats_baseline,
                        json.dumps(raw_results) if raw_results else None,
                    ),
                )
                row = cur.fetchone()
                run_id = row[0] if row else None
                logger.info(
                    "Stored validation run %s for %s/%s: AUC=%.4f WR=%.1f%%",
                    run_id,
                    model_version,
                    market,
                    auc_mean,
                    wr_mean * 100,
                )
                return run_id
        except Exception:
            logger.exception("Failed to store validation for %s", model_version)
            return None

    def get_validation(self, model_version: str, market: str) -> Optional[Dict[str, Any]]:
        """Get the most recent validation run for a model version."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, auc_mean, auc_std, wr_mean, roi_mean, fold_count,
                           beats_baseline, raw_results, run_date
                    FROM validation_runs
                    WHERE model_version = %s AND market = %s
                    ORDER BY run_date DESC
                    LIMIT 1
                    """,
                    (model_version, market.upper()),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "auc_mean": float(row[1]) if row[1] else None,
                        "auc_std": float(row[2]) if row[2] else None,
                        "wr_mean": float(row[3]) if row[3] else None,
                        "roi_mean": float(row[4]) if row[4] else None,
                        "fold_count": row[5],
                        "beats_baseline": row[6],
                        "raw_results": row[7],
                        "run_date": str(row[8]),
                    }
                return None
        except Exception:
            logger.exception("Failed to get validation for %s", model_version)
            return None

    def evaluate(
        self,
        new_version: str,
        market: str,
        production_auc: Optional[float] = None,
        production_wr: Optional[float] = None,
        min_auc_improvement: float = 0.005,
        max_auc_std: float = 0.03,
        min_wr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether a new model should be promoted.

        Checks:
        1. AUC improvement over production (>= min_auc_improvement)
        2. AUC standard deviation (< max_auc_std) — reject high-variance models
        3. Win rate meets minimum (if specified)
        4. Win rate >= production win rate (if production_wr given)

        Returns dict with promote (bool), reason, and comparison metrics.
        """
        new_val = self.get_validation(new_version, market)
        if not new_val:
            return {
                "promote": False,
                "reason": f"No validation results found for {new_version}",
                "new": None,
            }

        reasons = []
        checks_passed = True

        # Check AUC improvement
        if production_auc is not None:
            auc_improvement = new_val["auc_mean"] - production_auc
            if auc_improvement < min_auc_improvement:
                checks_passed = False
                reasons.append(
                    f"AUC improvement {auc_improvement:+.4f} < required {min_auc_improvement:+.4f}"
                )
            else:
                reasons.append(f"AUC improvement {auc_improvement:+.4f} OK")

        # Check AUC variance
        if new_val["auc_std"] is not None and new_val["auc_std"] > max_auc_std:
            checks_passed = False
            reasons.append(
                f"AUC std {new_val['auc_std']:.4f} > max {max_auc_std:.4f} (high variance)"
            )
        else:
            reasons.append(f"AUC std {new_val['auc_std']:.4f} OK")

        # Check win rate vs production
        if production_wr is not None and new_val["wr_mean"] is not None:
            if new_val["wr_mean"] < production_wr:
                checks_passed = False
                reasons.append(f"WR {new_val['wr_mean']:.1%} < production {production_wr:.1%}")
            else:
                reasons.append(f"WR {new_val['wr_mean']:.1%} >= production {production_wr:.1%}")

        # Check minimum win rate
        if min_wr is not None and new_val["wr_mean"] is not None:
            if new_val["wr_mean"] < min_wr:
                checks_passed = False
                reasons.append(f"WR {new_val['wr_mean']:.1%} < minimum {min_wr:.1%}")

        return {
            "promote": checks_passed,
            "reason": "; ".join(reasons),
            "new_version": new_version,
            "new_auc": new_val["auc_mean"],
            "new_wr": new_val["wr_mean"],
            "new_auc_std": new_val["auc_std"],
            "production_auc": production_auc,
            "production_wr": production_wr,
        }

    def list_validations(self, market: str, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent validation runs for a market."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT model_version, auc_mean, auc_std, wr_mean, roi_mean,
                           fold_count, beats_baseline, run_date
                    FROM validation_runs
                    WHERE market = %s
                    ORDER BY run_date DESC
                    LIMIT %s
                    """,
                    (market.upper(), limit),
                )
                return [
                    {
                        "model_version": r[0],
                        "auc_mean": float(r[1]) if r[1] else None,
                        "auc_std": float(r[2]) if r[2] else None,
                        "wr_mean": float(r[3]) if r[3] else None,
                        "roi_mean": float(r[4]) if r[4] else None,
                        "fold_count": r[5],
                        "beats_baseline": r[6],
                        "run_date": str(r[7]),
                    }
                    for r in cur.fetchall()
                ]
        except Exception:
            logger.exception("Failed to list validations for %s", market)
            return []
