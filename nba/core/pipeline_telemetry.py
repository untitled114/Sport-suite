"""
Pipeline Telemetry — Operational Excellence
=============================================
Structured observability for every pipeline run. Captures what ran,
what it produced, what it skipped, and why.

Usage:
    ctx = PipelineContext(run_type="full")

    with ctx.task("fetch_props") as t:
        props = fetch()
        t.record(props_count=len(props), books=7)

    with ctx.task("extract_features") as t:
        features = extract()
        t.record(feature_count=len(features.columns), players=len(features))

    ctx.finalize(picks_generated=len(picks))
    ctx.persist()  # writes to axiom.pipeline_runs
"""

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from nba.config.database import get_connection

EST = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class TaskTelemetry:
    """Captures telemetry for a single pipeline task."""

    def __init__(self, name: str):
        self.name = name
        self.started_at = datetime.now(EST)
        self.ended_at: Optional[datetime] = None
        self.duration_ms: Optional[int] = None
        self.status = "running"
        self.metrics: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def record(self, **kwargs):
        for k, v in kwargs.items():
            self.metrics[k] = v

    def succeed(self):
        self.status = "success"
        self.ended_at = datetime.now(EST)
        self.duration_ms = int((self.ended_at - self.started_at).total_seconds() * 1000)

    def fail(self, error: str):
        self.status = "failed"
        self.error = error
        self.ended_at = datetime.now(EST)
        self.duration_ms = int((self.ended_at - self.started_at).total_seconds() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metrics": self.metrics,
            "error": self.error,
        }


class PipelineContext:
    """
    Thread-safe context for a single pipeline execution.

    Generates a pipeline_run_id (UUID), tracks task-level telemetry,
    detects anomalies, and persists the full run record to the DB.
    """

    def __init__(self, run_type: str = "full", run_date: Optional[str] = None, run_number: int = 1):
        self.run_id = str(uuid.uuid4())
        self.run_type = run_type
        self.run_date = run_date or datetime.now(EST).strftime("%Y-%m-%d")
        self.run_number = run_number
        self.started_at = datetime.now(EST)
        self.ended_at: Optional[datetime] = None
        self.status = "running"
        self.tasks: List[TaskTelemetry] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    @contextmanager
    def task(self, name: str):
        """Context manager for tracking a pipeline task."""
        t = TaskTelemetry(name)
        self.tasks.append(t)
        logger.info(
            "pipeline_task_start",
            extra={"pipeline_run_id": self.run_id, "task": name},
        )
        try:
            yield t
            t.succeed()
            logger.info(
                "pipeline_task_end",
                extra={
                    "pipeline_run_id": self.run_id,
                    "task": name,
                    "duration_ms": t.duration_ms,
                    **t.metrics,
                },
            )
        except Exception as e:
            t.fail(str(e))
            logger.error(
                "pipeline_task_failed",
                extra={"pipeline_run_id": self.run_id, "task": name, "error": str(e)},
            )
            raise

    def check_anomalies(
        self,
        picks_generated: int,
        feature_count: Optional[int] = None,
        expected_feature_count: int = 102,
        min_picks: int = 1,
        props_fetched: int = 0,
        min_props: int = 50,
    ):
        """Detect operational anomalies after the run completes."""
        if picks_generated == 0 and props_fetched > min_props:
            self.anomalies.append(
                {
                    "type": "zero_picks",
                    "severity": "critical",
                    "message": f"Pipeline produced 0 picks from {props_fetched} props",
                }
            )

        if feature_count is not None and feature_count < expected_feature_count:
            self.anomalies.append(
                {
                    "type": "feature_count_regression",
                    "severity": "warning",
                    "message": f"Feature count {feature_count} < expected {expected_feature_count}",
                    "actual": feature_count,
                    "expected": expected_feature_count,
                }
            )

        if props_fetched < min_props and props_fetched > 0:
            self.anomalies.append(
                {
                    "type": "low_props",
                    "severity": "warning",
                    "message": f"Only {props_fetched} props fetched (min: {min_props})",
                }
            )

        if picks_generated < min_picks and props_fetched >= min_props:
            self.anomalies.append(
                {
                    "type": "low_picks",
                    "severity": "warning",
                    "message": f"Only {picks_generated} picks from {props_fetched} props",
                }
            )

        return self.anomalies

    def finalize(self, **summary_kwargs):
        """Mark the run as complete with summary metrics."""
        self.ended_at = datetime.now(EST)
        self.status = "failed" if any(t.status == "failed" for t in self.tasks) else "success"
        self.summary = summary_kwargs

        duration_ms = int((self.ended_at - self.started_at).total_seconds() * 1000)
        self.summary["total_duration_ms"] = duration_ms

        logger.info(
            "pipeline_run_complete",
            extra={
                "pipeline_run_id": self.run_id,
                "status": self.status,
                "duration_ms": duration_ms,
                "anomaly_count": len(self.anomalies),
                **{
                    k: v
                    for k, v in summary_kwargs.items()
                    if isinstance(v, (int, float, str, bool))
                },
            },
        )

    def persist(self) -> bool:
        """Write the full run record to axiom.pipeline_runs."""
        try:
            conn = get_connection("axiom")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs
                        (run_id, run_date, run_number, run_type, started_at, ended_at,
                         status, duration_ms, tasks, anomalies, summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        ended_at = EXCLUDED.ended_at,
                        status = EXCLUDED.status,
                        duration_ms = EXCLUDED.duration_ms,
                        tasks = EXCLUDED.tasks,
                        anomalies = EXCLUDED.anomalies,
                        summary = EXCLUDED.summary
                    """,
                    (
                        self.run_id,
                        self.run_date,
                        self.run_number,
                        self.run_type,
                        self.started_at,
                        self.ended_at,
                        self.status,
                        self.summary.get("total_duration_ms"),
                        json.dumps([t.to_dict() for t in self.tasks]),
                        json.dumps(self.anomalies) if self.anomalies else None,
                        json.dumps(self.summary),
                    ),
                )
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"Failed to persist pipeline run (non-critical): {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "run_number": self.run_number,
            "run_type": self.run_type,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "tasks": [t.to_dict() for t in self.tasks],
            "anomalies": self.anomalies,
            "summary": self.summary,
        }
