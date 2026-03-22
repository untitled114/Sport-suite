"""Tests for nba.core.pipeline_telemetry — operational excellence."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestTaskTelemetry:
    def test_init(self):
        from nba.core.pipeline_telemetry import TaskTelemetry

        t = TaskTelemetry("fetch_props")
        assert t.name == "fetch_props"
        assert t.status == "running"
        assert t.metrics == {}

    def test_record(self):
        from nba.core.pipeline_telemetry import TaskTelemetry

        t = TaskTelemetry("extract")
        t.record(feature_count=102, players=50)
        assert t.metrics["feature_count"] == 102
        assert t.metrics["players"] == 50

    def test_succeed(self):
        from nba.core.pipeline_telemetry import TaskTelemetry

        t = TaskTelemetry("predict")
        t.succeed()
        assert t.status == "success"
        assert t.duration_ms is not None
        assert t.duration_ms >= 0

    def test_fail(self):
        from nba.core.pipeline_telemetry import TaskTelemetry

        t = TaskTelemetry("predict")
        t.fail("DB timeout")
        assert t.status == "failed"
        assert t.error == "DB timeout"
        assert t.duration_ms is not None

    def test_to_dict(self):
        from nba.core.pipeline_telemetry import TaskTelemetry

        t = TaskTelemetry("fetch")
        t.record(count=100)
        t.succeed()
        d = t.to_dict()
        assert d["name"] == "fetch"
        assert d["status"] == "success"
        assert d["metrics"]["count"] == 100
        assert d["duration_ms"] >= 0


class TestPipelineContext:
    def test_init(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="full", run_date="2026-03-22", run_number=3)
        assert ctx.run_type == "full"
        assert ctx.run_date == "2026-03-22"
        assert ctx.run_number == 3
        assert len(ctx.run_id) == 36  # UUID
        assert ctx.status == "running"

    def test_init_defaults(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        assert ctx.run_type == "full"
        assert ctx.run_number == 1

    def test_task_context_manager_success(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        with ctx.task("fetch_props") as t:
            t.record(props_count=500, books=7)

        assert len(ctx.tasks) == 1
        assert ctx.tasks[0].status == "success"
        assert ctx.tasks[0].metrics["props_count"] == 500

    def test_task_context_manager_failure(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        with pytest.raises(ValueError):
            with ctx.task("bad_task") as t:
                raise ValueError("something broke")

        assert ctx.tasks[0].status == "failed"
        assert "something broke" in ctx.tasks[0].error

    def test_multiple_tasks(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        with ctx.task("fetch") as t:
            t.record(count=100)
        with ctx.task("extract") as t:
            t.record(features=102)
        with ctx.task("predict") as t:
            t.record(picks=12)

        assert len(ctx.tasks) == 3
        assert all(t.status == "success" for t in ctx.tasks)

    def test_finalize_success(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        with ctx.task("fetch") as t:
            t.record(count=100)

        ctx.finalize(picks_generated=12, feature_count=102)
        assert ctx.status == "success"
        assert ctx.summary["picks_generated"] == 12
        assert ctx.summary["total_duration_ms"] >= 0

    def test_finalize_failed(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        try:
            with ctx.task("bad") as t:
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        ctx.finalize(picks_generated=0)
        assert ctx.status == "failed"


class TestAnomalyDetection:
    def test_zero_picks(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(picks_generated=0, props_fetched=500)
        types = [a["type"] for a in anomalies]
        assert "zero_picks" in types

    def test_feature_count_regression(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(
            picks_generated=10, feature_count=98, expected_feature_count=102
        )
        assert any(a["type"] == "feature_count_regression" for a in anomalies)

    def test_low_props(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(picks_generated=5, props_fetched=30, min_props=50)
        assert any(a["type"] == "low_props" for a in anomalies)

    def test_low_picks(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(picks_generated=0, props_fetched=500, min_picks=1)
        assert any(a["type"] == "zero_picks" for a in anomalies)

    def test_no_anomalies(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(picks_generated=15, feature_count=102, props_fetched=500)
        assert len(anomalies) == 0

    def test_feature_count_ok(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext()
        anomalies = ctx.check_anomalies(
            picks_generated=10, feature_count=136, expected_feature_count=102
        )
        assert not any(a["type"] == "feature_count_regression" for a in anomalies)


class TestToDict:
    def test_full_run(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="refresh", run_date="2026-03-22", run_number=4)
        with ctx.task("fetch") as t:
            t.record(props=200)
        ctx.check_anomalies(picks_generated=0, props_fetched=200)
        ctx.finalize(picks_generated=0)

        d = ctx.to_dict()
        assert d["run_id"] == ctx.run_id
        assert d["run_type"] == "refresh"
        assert len(d["tasks"]) == 1
        assert len(d["anomalies"]) >= 1
        assert d["status"] == "success"


class TestPersist:
    def test_persist_success(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="full", run_date="2026-03-22", run_number=1)
        ctx.finalize(picks_generated=5)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.pipeline_telemetry.get_connection", return_value=mock_conn):
            result = ctx.persist()

        assert result is True
        mock_cur.execute.assert_called_once()
        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO pipeline_runs" in sql
        mock_conn.close.assert_called_once()

    def test_persist_writes_correct_params(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="refresh", run_date="2026-03-22", run_number=3)
        with ctx.task("fetch") as t:
            t.record(props=100)
        ctx.finalize(picks_generated=10)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.pipeline_telemetry.get_connection", return_value=mock_conn):
            ctx.persist()

        params = mock_cur.execute.call_args[0][1]
        assert params[0] == ctx.run_id
        assert params[1] == "2026-03-22"
        assert params[2] == 3
        assert params[3] == "refresh"
        tasks_json = json.loads(params[8])
        assert len(tasks_json) == 1
        assert tasks_json[0]["name"] == "fetch"

    def test_persist_with_anomalies(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="full", run_date="2026-03-22", run_number=1)
        ctx.check_anomalies(picks_generated=0, props_fetched=200)
        ctx.finalize(picks_generated=0)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.pipeline_telemetry.get_connection", return_value=mock_conn):
            ctx.persist()

        params = mock_cur.execute.call_args[0][1]
        anomalies_json = json.loads(params[9])
        assert len(anomalies_json) >= 1

    def test_persist_failure_returns_false(self):
        from nba.core.pipeline_telemetry import PipelineContext

        ctx = PipelineContext(run_type="full", run_date="2026-03-22", run_number=1)
        ctx.finalize(picks_generated=5)

        with patch("nba.core.pipeline_telemetry.get_connection", side_effect=Exception("refused")):
            result = ctx.persist()

        assert result is False
