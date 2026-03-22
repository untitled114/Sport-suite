"""Tests for nba.models.promotion_gate — performance efficiency pillar."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


def _mock_conn():
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


class TestStoreValidation:
    @patch("nba.models.promotion_gate.get_connection")
    def test_store_success(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (42,)
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        run_id = gate.store_validation(
            "v4_POINTS_20260322",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.02,
            wr_mean=0.62,
            roi_mean=0.08,
            fold_count=6,
            beats_baseline=True,
        )
        assert run_id == 42
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO validation_runs" in sql

    @patch("nba.models.promotion_gate.get_connection")
    def test_store_with_raw_results(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1,)
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        raw = {"folds": [{"auc": 0.78}, {"auc": 0.76}]}
        gate.store_validation("v4", "POINTS", 0.77, 0.01, 0.60, 0.05, 2, raw_results=raw)
        params = cur.execute.call_args[0][1]
        assert json.loads(params[8])["folds"][0]["auc"] == 0.78

    @patch("nba.models.promotion_gate.get_connection")
    def test_store_db_error(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)
        assert gate.store_validation("v4", "POINTS", 0.78, 0.02, 0.62, 0.08, 6) is None


class TestGetValidation:
    @patch("nba.models.promotion_gate.get_connection")
    def test_found(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (42, 0.78, 0.02, 0.62, 0.08, 6, True, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        val = gate.get_validation("v4_POINTS", "POINTS")
        assert val["auc_mean"] == 0.78
        assert val["wr_mean"] == 0.62
        assert val["fold_count"] == 6

    @patch("nba.models.promotion_gate.get_connection")
    def test_not_found(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)
        assert gate.get_validation("nonexistent", "POINTS") is None

    @patch("nba.models.promotion_gate.get_connection")
    def test_db_error(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)
        assert gate.get_validation("v4", "POINTS") is None


class TestEvaluate:
    @patch("nba.models.promotion_gate.get_connection")
    def test_promote_all_checks_pass(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.78, 0.02, 0.65, 0.10, 6, True, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate(
            "v4_POINTS",
            "POINTS",
            production_auc=0.74,
            production_wr=0.60,
        )
        assert result["promote"] is True
        assert result["new_auc"] == 0.78

    @patch("nba.models.promotion_gate.get_connection")
    def test_reject_insufficient_auc(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.742, 0.02, 0.62, 0.05, 6, False, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("v4", "POINTS", production_auc=0.740)
        assert result["promote"] is False
        assert "AUC improvement" in result["reason"]

    @patch("nba.models.promotion_gate.get_connection")
    def test_reject_high_variance(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.78, 0.05, 0.62, 0.08, 6, True, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("v4", "POINTS", production_auc=0.74, max_auc_std=0.03)
        assert result["promote"] is False
        assert "high variance" in result["reason"]

    @patch("nba.models.promotion_gate.get_connection")
    def test_reject_wr_below_production(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.78, 0.02, 0.55, 0.03, 6, False, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("v4", "POINTS", production_auc=0.74, production_wr=0.60)
        assert result["promote"] is False
        assert "WR" in result["reason"]

    @patch("nba.models.promotion_gate.get_connection")
    def test_reject_wr_below_minimum(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.78, 0.02, 0.50, 0.03, 6, False, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("v4", "POINTS", min_wr=0.55)
        assert result["promote"] is False
        assert "minimum" in result["reason"]

    @patch("nba.models.promotion_gate.get_connection")
    def test_no_validation_found(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("nonexistent", "POINTS")
        assert result["promote"] is False
        assert "No validation results" in result["reason"]

    @patch("nba.models.promotion_gate.get_connection")
    def test_no_production_comparison(self, mock_gc):
        """When no production metrics given, only variance check runs."""
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (1, 0.78, 0.02, 0.65, 0.10, 6, True, None, date(2026, 3, 22))
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        result = gate.evaluate("v4", "POINTS")
        assert result["promote"] is True


class TestListValidations:
    @patch("nba.models.promotion_gate.get_connection")
    def test_list(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [
            ("v4", 0.78, 0.02, 0.62, 0.08, 6, True, date(2026, 3, 22)),
            ("v3", 0.74, 0.01, 0.60, 0.05, 6, True, date(2026, 2, 3)),
        ]
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        runs = gate.list_validations("POINTS")
        assert len(runs) == 2
        assert runs[0]["model_version"] == "v4"

    @patch("nba.models.promotion_gate.get_connection")
    def test_list_empty(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.fetchall.return_value = []
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)
        assert gate.list_validations("THREES") == []

    @patch("nba.models.promotion_gate.get_connection")
    def test_list_db_error(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("err")
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)
        assert gate.list_validations("POINTS") == []


class TestConnectionManagement:
    def test_external_conn_not_closed(self):
        from nba.models.promotion_gate import PromotionGate

        conn = MagicMock()
        conn.closed = False
        gate = PromotionGate(conn=conn)
        gate.close()
        conn.close.assert_not_called()

    def test_close_own_conn(self):
        from nba.models.promotion_gate import PromotionGate

        conn = MagicMock()
        conn.closed = False
        gate = PromotionGate()
        gate._conn = conn
        gate.close()
        conn.close.assert_called_once()

    def test_close_no_conn(self):
        from nba.models.promotion_gate import PromotionGate

        gate = PromotionGate()
        gate.close()  # should not raise


class TestFullLifecycle:
    """Integration-style test: train → validate → evaluate → promote/reject."""

    @patch("nba.models.promotion_gate.get_connection")
    def test_promote_lifecycle(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        mock_gc.return_value = conn

        gate = PromotionGate(conn=conn)

        # Store validation
        cur.fetchone.return_value = (1,)
        run_id = gate.store_validation(
            "v4_POINTS_20260322",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.018,
            wr_mean=0.63,
            roi_mean=0.09,
            fold_count=6,
            beats_baseline=True,
        )
        assert run_id == 1

        # Evaluate promotion
        cur.fetchone.return_value = (1, 0.78, 0.018, 0.63, 0.09, 6, True, None, date(2026, 3, 22))
        decision = gate.evaluate(
            "v4_POINTS_20260322",
            "POINTS",
            production_auc=0.740,
            production_wr=0.60,
        )
        assert decision["promote"] is True

    @patch("nba.models.promotion_gate.get_connection")
    def test_reject_lifecycle(self, mock_gc):
        from nba.models.promotion_gate import PromotionGate

        conn, cur = _mock_conn()
        mock_gc.return_value = conn
        gate = PromotionGate(conn=conn)

        # Validation shows marginal improvement with high variance
        cur.fetchone.return_value = (2, 0.742, 0.04, 0.58, 0.02, 6, False, None, date(2026, 3, 22))
        decision = gate.evaluate(
            "v4_bad",
            "POINTS",
            production_auc=0.740,
            production_wr=0.60,
        )
        assert decision["promote"] is False
        assert "high variance" in decision["reason"] or "WR" in decision["reason"]
