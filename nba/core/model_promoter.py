"""
Model Promoter — validation-gated promotion and auto-rollback.

Stores walk-forward results, gates promotion on metrics, and rolls back
when 7-day WR drops below threshold after promotion.

Usage:
    promoter = ModelPromoter()

    # After walk-forward validation
    promoter.record_validation("v4", "POINTS", summary)

    # Check if candidate beats production
    if promoter.is_better_than_production("v4", "POINTS"):
        promoter.promote("v4", "POINTS")

    # Nightly check — auto-rollback if WR dropped
    promoter.check_rollback("POINTS")
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from nba.config.database import get_axiom_db_config

logger = logging.getLogger(__name__)

# Promotion thresholds
_AUC_IMPROVEMENT_MIN = 0.005  # Must beat production AUC by at least this
_AUC_STD_MAX = 0.03  # Reject high-variance models
_WR_ROLLBACK_THRESHOLD = 0.52  # 7-day WR below this triggers rollback
_MIN_PICKS_FOR_ROLLBACK = 20  # Need enough data before rolling back


@dataclass
class ValidationResult:
    """Parsed validation run from the database."""

    id: int
    model_version: str
    market: str
    auc_mean: float
    auc_std: float
    wr_mean: float
    roi_mean: float
    fold_count: int
    promoted: bool


def _connect():
    import psycopg2

    config = get_axiom_db_config()
    config["connect_timeout"] = 5
    conn = psycopg2.connect(**config)
    conn.autocommit = True
    return conn


class ModelPromoter:
    """Validation-gated model promotion with auto-rollback."""

    def __init__(self, conn=None):
        self._external_conn = conn is not None
        self._conn = conn

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = _connect()
        return self._conn

    def close(self):
        if self._conn and not self._external_conn and not self._conn.closed:
            self._conn.close()

    # ──────────────────────────────────────────────────────────
    # Record validation results
    # ──────────────────────────────────────────────────────────

    def record_validation(
        self,
        model_version: str,
        market: str,
        auc_mean: float,
        auc_std: float,
        wr_mean: float,
        roi_mean: float,
        fold_count: int,
        raw_results: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Store walk-forward validation results. Returns the validation_run id."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # Check if candidate beats current production
                prod = self.get_production_validation(market)
                beats = False
                if prod is not None:
                    beats = (
                        auc_mean > prod.auc_mean + _AUC_IMPROVEMENT_MIN
                        and wr_mean > prod.wr_mean
                        and auc_std < _AUC_STD_MAX
                    )

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
                        auc_mean,
                        auc_std,
                        wr_mean,
                        roi_mean,
                        fold_count,
                        beats,
                        json.dumps(raw_results) if raw_results else None,
                    ),
                )
                row = cur.fetchone()
                run_id = row[0] if row else None

                logger.info(
                    "Recorded validation: %s/%s AUC=%.4f WR=%.4f beats_baseline=%s",
                    model_version,
                    market,
                    auc_mean,
                    wr_mean,
                    beats,
                )
                return run_id
        except Exception:
            logger.exception("Failed to record validation for %s/%s", model_version, market)
            return None

    # ──────────────────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────────────────

    def get_production_version(self, market: str) -> Optional[str]:
        """Get the current production model version for a market."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT production_version FROM model_registry WHERE market = %s",
                    (market.upper(),),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            logger.exception("Failed to get production version for %s", market)
            return None

    def get_production_validation(self, market: str) -> Optional[ValidationResult]:
        """Get the validation results for the current production model."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vr.id, vr.model_version, vr.market, vr.auc_mean, vr.auc_std,
                           vr.wr_mean, vr.roi_mean, vr.fold_count, vr.promoted
                    FROM validation_runs vr
                    JOIN model_registry mr ON mr.market = vr.market
                        AND mr.production_version = vr.model_version
                    WHERE vr.market = %s AND vr.promoted = TRUE
                    ORDER BY vr.run_date DESC
                    LIMIT 1
                    """,
                    (market.upper(),),
                )
                row = cur.fetchone()
                if row:
                    return ValidationResult(*row)
                return None
        except Exception:
            logger.exception("Failed to get production validation for %s", market)
            return None

    def get_validation(self, model_version: str, market: str) -> Optional[ValidationResult]:
        """Get the latest validation run for a specific version/market."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, model_version, market, auc_mean, auc_std,
                           wr_mean, roi_mean, fold_count, promoted
                    FROM validation_runs
                    WHERE model_version = %s AND market = %s
                    ORDER BY run_date DESC
                    LIMIT 1
                    """,
                    (model_version, market.upper()),
                )
                row = cur.fetchone()
                if row:
                    return ValidationResult(*row)
                return None
        except Exception:
            logger.exception("Failed to get validation for %s/%s", model_version, market)
            return None

    # ──────────────────────────────────────────────────────────
    # Promotion gate
    # ──────────────────────────────────────────────────────────

    def is_better_than_production(self, candidate_version: str, market: str) -> bool:
        """Check if candidate model beats production on all criteria.

        Criteria:
        - AUC improvement >= 0.005 (meaningful, not noise)
        - Win rate higher than production
        - AUC std < 0.03 (stable across folds)
        """
        candidate = self.get_validation(candidate_version, market)
        if candidate is None:
            return False

        prod = self.get_production_validation(market)
        if prod is None:
            # No production model validated yet — candidate wins by default
            return candidate.auc_std < _AUC_STD_MAX

        return (
            candidate.auc_mean > prod.auc_mean + _AUC_IMPROVEMENT_MIN
            and candidate.wr_mean > prod.wr_mean
            and candidate.auc_std < _AUC_STD_MAX
        )

    # ──────────────────────────────────────────────────────────
    # Promote
    # ──────────────────────────────────────────────────────────

    def promote(self, model_version: str, market: str) -> bool:
        """Promote a model version to production for a market.

        Updates model_registry and marks the validation run as promoted.
        """
        try:
            conn = self._get_conn()
            candidate = self.get_validation(model_version, market)
            if candidate is None:
                logger.warning(
                    "Cannot promote %s/%s — no validation run found", model_version, market
                )
                return False

            with conn.cursor() as cur:
                # Get current production version for rollback tracking
                cur.execute(
                    "SELECT production_version FROM model_registry WHERE market = %s",
                    (market.upper(),),
                )
                row = cur.fetchone()
                previous = row[0] if row else None

                # Update registry
                cur.execute(
                    """
                    INSERT INTO model_registry (market, production_version, promoted_at, validation_run_id, previous_version)
                    VALUES (%s, %s, NOW(), %s, %s)
                    ON CONFLICT (market) DO UPDATE SET
                        production_version = EXCLUDED.production_version,
                        promoted_at = NOW(),
                        validation_run_id = EXCLUDED.validation_run_id,
                        previous_version = model_registry.production_version
                    """,
                    (market.upper(), model_version, candidate.id, previous),
                )

                # Mark validation run as promoted
                cur.execute(
                    "UPDATE validation_runs SET promoted = TRUE, promoted_at = NOW() WHERE id = %s",
                    (candidate.id,),
                )

            logger.info(
                "Promoted %s → %s for %s (previous: %s)",
                previous,
                model_version,
                market,
                previous,
            )
            return True
        except Exception:
            logger.exception("Failed to promote %s/%s", model_version, market)
            return False

    # ──────────────────────────────────────────────────────────
    # Auto-rollback
    # ──────────────────────────────────────────────────────────

    def check_rollback(self, market: str) -> bool:
        """Check if current production model should be rolled back.

        Triggers when 7-day rolling WR drops below threshold after promotion.
        Rolls back to previous_version from model_registry.

        Returns True if rollback was executed.
        """
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # Get current production info
                cur.execute(
                    "SELECT production_version, previous_version FROM model_registry WHERE market = %s",
                    (market.upper(),),
                )
                row = cur.fetchone()
                if not row or not row[1]:
                    return False  # No previous version to roll back to

                current_version, previous_version = row

                # Compute 7-day rolling WR for current version
                cur.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN is_hit = TRUE THEN 1 ELSE 0 END) AS wins
                    FROM nba_prediction_history
                    WHERE run_date >= CURRENT_DATE - INTERVAL '7 days'
                      AND model_version = %s
                      AND stat_type = %s
                      AND is_hit IS NOT NULL
                    """,
                    (current_version, market.upper()),
                )
                row = cur.fetchone()
                total = row[0] if row else 0
                wins = row[1] if row and row[1] else 0

                if total < _MIN_PICKS_FOR_ROLLBACK:
                    return False  # Not enough data

                wr_7d = wins / total
                if wr_7d >= _WR_ROLLBACK_THRESHOLD:
                    return False  # WR is fine

                # Rollback
                reason = f"7d WR {wr_7d:.1%} < {_WR_ROLLBACK_THRESHOLD:.0%} ({wins}/{total} picks)"

                cur.execute(
                    """
                    UPDATE model_registry
                    SET production_version = previous_version,
                        previous_version = production_version,
                        rollback_count = rollback_count + 1
                    WHERE market = %s
                    """,
                    (market.upper(),),
                )

                # Mark the validation run as rolled back
                cur.execute(
                    """
                    UPDATE validation_runs
                    SET rolled_back = TRUE, rolled_back_at = NOW(), rollback_reason = %s
                    WHERE model_version = %s AND market = %s AND promoted = TRUE
                    ORDER BY run_date DESC
                    LIMIT 1
                    """,
                    (reason, current_version, market.upper()),
                )

                logger.warning(
                    "ROLLBACK %s → %s for %s: %s",
                    current_version,
                    previous_version,
                    market,
                    reason,
                )
                return True
        except Exception:
            logger.exception("Failed to check rollback for %s", market)
            return False
