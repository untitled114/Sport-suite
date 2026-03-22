"""Tests for nba.models.model_registry — reliability pillar."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_conn():
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


class TestRegister:
    @patch("nba.models.model_registry.get_connection")
    def test_register_success(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.register(
            "v4_POINTS_20260322",
            "POINTS",
            auc=0.78,
            r2=0.55,
            feature_count=188,
            pkl_path="nba/models/saved_xl/points_v4_*.pkl",
        )
        assert ok is True
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO model_registry" in sql

    @patch("nba.models.model_registry.get_connection")
    def test_register_with_metadata(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.register(
            "v4_POINTS_20260322",
            "POINTS",
            pkl_path="test.pkl",
            metadata={"training_config": {"lr": 0.05}},
        )
        assert ok is True
        params = cur.execute.call_args[0][1]
        assert json.loads(params[7])["training_config"]["lr"] == 0.05

    @patch("nba.models.model_registry.get_connection")
    def test_register_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.register("bad", "POINTS")
        assert ok is False


class TestPromote:
    @patch("nba.models.model_registry.get_connection")
    def test_promote_success(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.return_value = ("POINTS",)
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.promote("v4_POINTS_20260322")
        assert ok is True
        assert cur.execute.call_count == 3  # get market, demote old, promote new

    @patch("nba.models.model_registry.get_connection")
    def test_promote_not_found(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.promote("nonexistent")
        assert ok is False

    @patch("nba.models.model_registry.get_connection")
    def test_promote_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.promote("bad")
        assert ok is False


class TestStartShadow:
    @patch("nba.models.model_registry.get_connection")
    def test_shadow_success(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.start_shadow("v4_POINTS_20260322")
        assert ok is True

    @patch("nba.models.model_registry.get_connection")
    def test_shadow_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        ok = reg.start_shadow("bad")
        assert ok is False


class TestRollback:
    @patch("nba.models.model_registry.get_connection")
    def test_rollback_restores_previous(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            ("v4_POINTS_20260322",),  # current production
            ("v3_POINTS_20260203",),  # previous to restore
        ]
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        restored = reg.rollback("POINTS", reason="7d WR below 60%")
        assert restored == "v3_POINTS_20260203"

    @patch("nba.models.model_registry.get_connection")
    def test_rollback_no_previous(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            ("v4_POINTS_20260322",),  # current
            None,  # no previous
        ]
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        restored = reg.rollback("POINTS")
        assert restored is None

    @patch("nba.models.model_registry.get_connection")
    def test_rollback_no_current(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [None, None]
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        restored = reg.rollback("POINTS")
        assert restored is None

    @patch("nba.models.model_registry.get_connection")
    def test_rollback_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        restored = reg.rollback("POINTS")
        assert restored is None


class TestGetProduction:
    @patch("nba.models.model_registry.get_connection")
    def test_found(self, mock_gc):
        from datetime import datetime

        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (
            "v3_POINTS_20260203",
            "POINTS",
            "production",
            0.740,
            0.548,
            136,
            "nba/models/saved_xl/points_v3_*.pkl",
            datetime(2026, 2, 3),
            24000,
            None,
        )
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        result = reg.get_production("POINTS")
        assert result["version"] == "v3_POINTS_20260203"
        assert result["auc"] == 0.740
        assert result["feature_count"] == 136

    @patch("nba.models.model_registry.get_connection")
    def test_not_found(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        assert reg.get_production("THREES") is None

    @patch("nba.models.model_registry.get_connection")
    def test_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        assert reg.get_production("POINTS") is None


class TestListModels:
    @patch("nba.models.model_registry.get_connection")
    def test_list_all(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [
            ("v3_POINTS", "POINTS", "production", 0.740, "path", None),
            ("xl_POINTS", "POINTS", "rolled_back", 0.767, "path", None),
        ]
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        models = reg.list_models()
        assert len(models) == 2

    @patch("nba.models.model_registry.get_connection")
    def test_list_filtered(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [("v3_POINTS", "POINTS", "production", 0.74, "p", None)]
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        models = reg.list_models(market="POINTS", status="production")
        assert len(models) == 1
        sql = cur.execute.call_args[0][0]
        assert "market" in sql
        assert "status" in sql

    @patch("nba.models.model_registry.get_connection")
    def test_list_db_error(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        reg = ModelRegistry(conn=conn)
        assert reg.list_models() == []


class TestShouldRetrain:
    def test_below_threshold(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.should_retrain("POINTS", win_rate_7d=0.54) is True

    def test_above_threshold(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.should_retrain("POINTS", win_rate_7d=0.65) is False

    def test_custom_threshold(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.should_retrain("POINTS", win_rate_7d=0.54, threshold=0.50) is False


class TestShouldPromoteShadow:
    def test_shadow_beats_production(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.should_promote_shadow("v4", shadow_auc=0.78, production_auc=0.74) is True

    def test_shadow_does_not_beat(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.should_promote_shadow("v4", shadow_auc=0.741, production_auc=0.740) is False

    def test_exact_threshold(self):
        from nba.models.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert (
            reg.should_promote_shadow(
                "v4",
                shadow_auc=0.745,
                production_auc=0.740,
                min_improvement=0.005,
            )
            is True
        )


class TestConnectionManagement:
    def test_external_conn_not_closed(self):
        from nba.models.model_registry import ModelRegistry

        conn = MagicMock()
        conn.closed = False
        reg = ModelRegistry(conn=conn)
        reg.close()
        conn.close.assert_not_called()

    @patch("nba.models.model_registry.get_connection")
    def test_creates_conn_on_demand(self, mock_gc):
        from nba.models.model_registry import ModelRegistry

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_gc.return_value = mock_conn
        reg = ModelRegistry()
        conn = reg._get_conn()
        assert conn is mock_conn
        mock_gc.assert_called_once_with("axiom")

    def test_close_own_conn(self):
        from nba.models.model_registry import ModelRegistry

        conn = MagicMock()
        conn.closed = False
        reg = ModelRegistry()
        reg._conn = conn
        reg.close()
        conn.close.assert_called_once()
