"""Tests for nba.core.model_promoter — validation-gated promotion + auto-rollback."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_conn():
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


# ──────────────────────────────────────────────────────────
# record_validation
# ──────────────────────────────────────────────────────────


class TestRecordValidation:
    def test_records_and_returns_id(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            None,
            (42,),
        ]  # get_production_validation returns None, INSERT returns id
        promoter = ModelPromoter(conn=conn)

        result = promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.02,
            wr_mean=0.65,
            roi_mean=0.12,
            fold_count=6,
        )

        assert result == 42

    def test_beats_baseline_when_better(self):
        from nba.core.model_promoter import ModelPromoter, ValidationResult

        conn, cur = _mock_conn()
        # get_production_validation returns existing prod
        prod_row = (1, "v3", "POINTS", 0.74, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [prod_row, (99,)]
        promoter = ModelPromoter(conn=conn)

        promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.02,
            wr_mean=0.65,
            roi_mean=0.15,
            fold_count=6,
        )

        # Check the INSERT call params
        insert_args = cur.execute.call_args_list[-1][0][1]
        beats_baseline = insert_args[7]
        assert beats_baseline is True

    def test_does_not_beat_when_auc_too_close(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        prod_row = (1, "v3", "POINTS", 0.74, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [prod_row, (99,)]
        promoter = ModelPromoter(conn=conn)

        promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.742,
            auc_std=0.02,  # only +0.002, below 0.005 threshold
            wr_mean=0.65,
            roi_mean=0.15,
            fold_count=6,
        )

        insert_args = cur.execute.call_args_list[-1][0][1]
        beats_baseline = insert_args[7]
        assert beats_baseline is False

    def test_does_not_beat_when_high_variance(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        prod_row = (1, "v3", "POINTS", 0.74, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [prod_row, (99,)]
        promoter = ModelPromoter(conn=conn)

        promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.05,  # std > 0.03 threshold
            wr_mean=0.65,
            roi_mean=0.15,
            fold_count=6,
        )

        insert_args = cur.execute.call_args_list[-1][0][1]
        beats_baseline = insert_args[7]
        assert beats_baseline is False

    def test_does_not_beat_when_wr_lower(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        prod_row = (1, "v3", "POINTS", 0.74, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [prod_row, (99,)]
        promoter = ModelPromoter(conn=conn)

        promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.02,
            wr_mean=0.58,
            roi_mean=0.15,
            fold_count=6,  # WR < prod's 0.60
        )

        insert_args = cur.execute.call_args_list[-1][0][1]
        beats_baseline = insert_args[7]
        assert beats_baseline is False

    def test_db_error_returns_none(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        promoter = ModelPromoter(conn=conn)

        result = promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.78,
            auc_std=0.02,
            wr_mean=0.65,
            roi_mean=0.12,
            fold_count=6,
        )
        assert result is None

    def test_with_raw_results(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [None, (10,)]
        promoter = ModelPromoter(conn=conn)

        raw = {"folds": [{"auc": 0.78}, {"auc": 0.76}]}
        promoter.record_validation(
            "v4",
            "POINTS",
            auc_mean=0.77,
            auc_std=0.01,
            wr_mean=0.62,
            roi_mean=0.10,
            fold_count=2,
            raw_results=raw,
        )

        insert_args = cur.execute.call_args_list[-1][0][1]
        assert '"folds"' in insert_args[8]  # JSON string


# ──────────────────────────────────────────────────────────
# get_production_version
# ──────────────────────────────────────────────────────────


class TestGetProductionVersion:
    def test_returns_version(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = ("v3",)
        promoter = ModelPromoter(conn=conn)

        assert promoter.get_production_version("POINTS") == "v3"

    def test_returns_none_when_missing(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        promoter = ModelPromoter(conn=conn)

        assert promoter.get_production_version("ASSISTS") is None

    def test_db_error_returns_none(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("fail")
        promoter = ModelPromoter(conn=conn)

        assert promoter.get_production_version("POINTS") is None


# ──────────────────────────────────────────────────────────
# is_better_than_production
# ──────────────────────────────────────────────────────────


class TestIsBetterThanProduction:
    def test_no_candidate_returns_false(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        promoter = ModelPromoter(conn=conn)

        assert promoter.is_better_than_production("v4", "POINTS") is False

    def test_no_production_candidate_wins_if_stable(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        # get_validation returns candidate, get_production_validation returns None
        candidate_row = (5, "v4", "POINTS", 0.78, 0.02, 0.65, 0.12, 6, False)
        cur.fetchone.side_effect = [candidate_row, None]
        promoter = ModelPromoter(conn=conn)

        assert promoter.is_better_than_production("v4", "POINTS") is True

    def test_no_production_candidate_loses_if_unstable(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        candidate_row = (5, "v4", "POINTS", 0.78, 0.05, 0.65, 0.12, 6, False)  # std > 0.03
        cur.fetchone.side_effect = [candidate_row, None]
        promoter = ModelPromoter(conn=conn)

        assert promoter.is_better_than_production("v4", "POINTS") is False

    def test_candidate_beats_production(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        candidate = (5, "v4", "POINTS", 0.78, 0.02, 0.65, 0.12, 6, False)
        production = (1, "v3", "POINTS", 0.74, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [candidate, production]
        promoter = ModelPromoter(conn=conn)

        assert promoter.is_better_than_production("v4", "POINTS") is True

    def test_candidate_fails_auc_threshold(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        candidate = (5, "v4", "POINTS", 0.743, 0.02, 0.65, 0.12, 6, False)  # +0.003 < 0.005
        production = (1, "v3", "POINTS", 0.740, 0.015, 0.60, 0.10, 6, True)
        cur.fetchone.side_effect = [candidate, production]
        promoter = ModelPromoter(conn=conn)

        assert promoter.is_better_than_production("v4", "POINTS") is False


# ──────────────────────────────────────────────────────────
# promote
# ──────────────────────────────────────────────────────────


class TestPromote:
    def test_promote_success(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        # get_validation returns candidate
        candidate = (5, "v4", "POINTS", 0.78, 0.02, 0.65, 0.12, 6, False)
        # SELECT production_version returns v3
        cur.fetchone.side_effect = [candidate, ("v3",)]
        promoter = ModelPromoter(conn=conn)

        result = promoter.promote("v4", "POINTS")
        assert result is True
        # Should have INSERT/UPDATE for registry + UPDATE for validation_runs
        assert cur.execute.call_count >= 3

    def test_promote_no_validation_returns_false(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        promoter = ModelPromoter(conn=conn)

        assert promoter.promote("v4", "POINTS") is False

    def test_promote_db_error_returns_false(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        candidate = (5, "v4", "POINTS", 0.78, 0.02, 0.65, 0.12, 6, False)
        cur.fetchone.side_effect = [candidate, Exception("fail")]
        promoter = ModelPromoter(conn=conn)

        assert promoter.promote("v4", "POINTS") is False


# ──────────────────────────────────────────────────────────
# check_rollback
# ──────────────────────────────────────────────────────────


class TestCheckRollback:
    def test_no_previous_version_no_rollback(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = ("v4", None)  # no previous
        promoter = ModelPromoter(conn=conn)

        assert promoter.check_rollback("POINTS") is False

    def test_not_enough_picks_no_rollback(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            ("v4", "v3"),  # registry
            (10, 5),  # total=10 < 20 minimum
        ]
        promoter = ModelPromoter(conn=conn)

        assert promoter.check_rollback("POINTS") is False

    def test_wr_above_threshold_no_rollback(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            ("v4", "v3"),  # registry
            (50, 30),  # 60% WR, above 52% threshold
        ]
        promoter = ModelPromoter(conn=conn)

        assert promoter.check_rollback("POINTS") is False

    def test_wr_below_threshold_triggers_rollback(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.side_effect = [
            ("v4", "v3"),  # registry
            (50, 20),  # 40% WR, below 52% threshold
        ]
        promoter = ModelPromoter(conn=conn)

        result = promoter.check_rollback("POINTS")
        assert result is True
        # Should UPDATE model_registry and validation_runs
        assert cur.execute.call_count >= 3

    def test_no_registry_row(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        promoter = ModelPromoter(conn=conn)

        assert promoter.check_rollback("POINTS") is False

    def test_db_error_returns_false(self):
        from nba.core.model_promoter import ModelPromoter

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        promoter = ModelPromoter(conn=conn)

        assert promoter.check_rollback("POINTS") is False


# ──────────────────────────────────────────────────────────
# Connection lifecycle
# ──────────────────────────────────────────────────────────


class TestConnectionLifecycle:
    def test_external_conn_not_closed(self):
        from nba.core.model_promoter import ModelPromoter

        conn = MagicMock()
        conn.closed = False
        promoter = ModelPromoter(conn=conn)
        promoter.close()
        conn.close.assert_not_called()

    def test_own_conn_closed(self):
        from nba.core.model_promoter import ModelPromoter

        promoter = ModelPromoter()
        conn = MagicMock()
        conn.closed = False
        promoter._conn = conn
        promoter.close()
        conn.close.assert_called_once()

    @patch("nba.core.model_promoter.get_axiom_db_config")
    @patch("psycopg2.connect")
    def test_get_conn_creates_connection(self, mock_connect, mock_config):
        from nba.core.model_promoter import ModelPromoter

        mock_config.return_value = {"host": "localhost", "port": 5500}
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        promoter = ModelPromoter()
        conn = promoter._get_conn()
        assert conn is mock_conn
        assert mock_conn.autocommit is True
