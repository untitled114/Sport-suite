"""
Model Registry — Reliability Pillar
=====================================
Tracks which model version is production, shadow, or rolled back.
Handles promotion, rollback, and degradation detection.

Every trained model gets registered. Only one version per market can be
'production' at a time. The registry is the source of truth for which
pkl files the predictor should load.

Usage:
    registry = ModelRegistry()

    # Register after training
    registry.register("v4_POINTS_20260322", "POINTS", auc=0.78,
                       pkl_path="nba/models/saved_xl/points_v4_*.pkl")

    # Promote after shadow period
    registry.promote("v4_POINTS_20260322")

    # Auto-rollback if new model degrades
    registry.rollback("POINTS", reason="7d WR 54% < 60% threshold")

    # Query current production model
    current = registry.get_production("POINTS")
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from nba.config.database import get_connection

EST = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

# Status lifecycle: training → shadow → production → rolled_back
VALID_STATUSES = {"training", "shadow", "production", "rolled_back", "failed"}


class ModelRegistry:
    """Read/write interface for axiom.model_registry."""

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

    def register(
        self,
        version: str,
        market: str,
        *,
        auc: Optional[float] = None,
        r2: Optional[float] = None,
        feature_count: Optional[int] = None,
        pkl_path: str = "",
        training_samples: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a newly trained model. Status starts as 'training'."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_registry
                        (version, market, status, auc, r2, feature_count,
                         pkl_path, training_samples, metadata, created_at)
                    VALUES (%s, %s, 'training', %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (version) DO UPDATE SET
                        auc = EXCLUDED.auc,
                        r2 = EXCLUDED.r2,
                        feature_count = EXCLUDED.feature_count,
                        pkl_path = EXCLUDED.pkl_path,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        version,
                        market.upper(),
                        auc,
                        r2,
                        feature_count,
                        pkl_path,
                        training_samples,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
            return True
        except Exception:
            logger.exception("Failed to register model %s", version)
            return False

    def promote(self, version: str) -> bool:
        """
        Promote a model to production. Demotes the current production model
        for the same market to 'rolled_back'.
        """
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # Get the market for this version
                cur.execute("SELECT market FROM model_registry WHERE version = %s", (version,))
                row = cur.fetchone()
                if not row:
                    logger.error("Cannot promote %s — not found in registry", version)
                    return False
                market = row[0]

                # Demote current production model
                cur.execute(
                    """
                    UPDATE model_registry
                    SET status = 'rolled_back', rolled_back_at = NOW()
                    WHERE market = %s AND status = 'production'
                    """,
                    (market,),
                )

                # Promote the new one
                cur.execute(
                    """
                    UPDATE model_registry
                    SET status = 'production', promoted_at = NOW()
                    WHERE version = %s
                    """,
                    (version,),
                )
            logger.info("Promoted %s to production for %s", version, market)
            return True
        except Exception:
            logger.exception("Failed to promote %s", version)
            return False

    def start_shadow(self, version: str) -> bool:
        """Move a model from 'training' to 'shadow' mode."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE model_registry SET status = 'shadow' WHERE version = %s",
                    (version,),
                )
            return True
        except Exception:
            logger.exception("Failed to start shadow for %s", version)
            return False

    def rollback(self, market: str, reason: str = "") -> Optional[str]:
        """
        Roll back the current production model and restore the previous one.

        Returns the version that was restored, or None if no previous model exists.
        """
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # Find current production model
                cur.execute(
                    "SELECT version FROM model_registry "
                    "WHERE market = %s AND status = 'production'",
                    (market.upper(),),
                )
                current = cur.fetchone()

                if current:
                    cur.execute(
                        """
                        UPDATE model_registry
                        SET status = 'rolled_back', rolled_back_at = NOW(),
                            metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                        WHERE version = %s
                        """,
                        (json.dumps({"rollback_reason": reason}), current[0]),
                    )

                # Find the most recent rolled_back model to restore
                cur.execute(
                    """
                    SELECT version FROM model_registry
                    WHERE market = %s AND status = 'rolled_back'
                      AND version != COALESCE(%s, '')
                    ORDER BY promoted_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (market.upper(), current[0] if current else None),
                )
                previous = cur.fetchone()

                if previous:
                    cur.execute(
                        "UPDATE model_registry SET status = 'production', promoted_at = NOW() "
                        "WHERE version = %s",
                        (previous[0],),
                    )
                    logger.info(
                        "Rolled back %s → restored %s (reason: %s)",
                        current[0] if current else "none",
                        previous[0],
                        reason,
                    )
                    return previous[0]
                else:
                    logger.warning("No previous model to restore for %s", market)
                    return None
        except Exception:
            logger.exception("Failed to rollback %s", market)
            return None

    def get_production(self, market: str) -> Optional[Dict[str, Any]]:
        """Get the current production model for a market."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT version, market, status, auc, r2, feature_count,
                           pkl_path, promoted_at, training_samples, metadata
                    FROM model_registry
                    WHERE market = %s AND status = 'production'
                    """,
                    (market.upper(),),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "version": row[0],
                        "market": row[1],
                        "status": row[2],
                        "auc": float(row[3]) if row[3] else None,
                        "r2": float(row[4]) if row[4] else None,
                        "feature_count": row[5],
                        "pkl_path": row[6],
                        "promoted_at": row[7].isoformat() if row[7] else None,
                        "training_samples": row[8],
                        "metadata": row[9],
                    }
                return None
        except Exception:
            logger.exception("Failed to get production model for %s", market)
            return None

    def list_models(
        self, market: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List models with optional filters."""
        try:
            conn = self._get_conn()
            conditions = []
            params = []
            if market:
                conditions.append("market = %s")
                params.append(market.upper())
            if status:
                conditions.append("status = %s")
                params.append(status)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT version, market, status, auc, pkl_path, promoted_at "
                    f"FROM model_registry {where} ORDER BY created_at DESC",
                    params,
                )
                return [
                    {
                        "version": r[0],
                        "market": r[1],
                        "status": r[2],
                        "auc": float(r[3]) if r[3] else None,
                        "pkl_path": r[4],
                        "promoted_at": r[5].isoformat() if r[5] else None,
                    }
                    for r in cur.fetchall()
                ]
        except Exception:
            logger.exception("Failed to list models")
            return []

    def should_retrain(self, market: str, win_rate_7d: float, threshold: float = 0.60) -> bool:
        """Check if 7-day win rate is below threshold, triggering retrain."""
        if win_rate_7d < threshold:
            logger.warning(
                "Retrain triggered for %s: 7d WR %.1f%% < %.0f%% threshold",
                market,
                win_rate_7d * 100,
                threshold * 100,
            )
            return True
        return False

    def should_promote_shadow(
        self,
        shadow_version: str,
        shadow_auc: float,
        production_auc: float,
        min_improvement: float = 0.005,
    ) -> bool:
        """
        Decide whether to promote shadow model to production.
        Requires the shadow model to beat production by min_improvement.
        """
        improvement = shadow_auc - production_auc
        if improvement >= min_improvement:
            logger.info(
                "Shadow %s beats production: AUC %.4f vs %.4f (+%.4f)",
                shadow_version,
                shadow_auc,
                production_auc,
                improvement,
            )
            return True
        else:
            logger.info(
                "Shadow %s does NOT beat production: AUC %.4f vs %.4f (need +%.4f)",
                shadow_version,
                shadow_auc,
                production_auc,
                min_improvement,
            )
            return False
