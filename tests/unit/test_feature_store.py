"""Tests for nba.features.feature_store — feature store ETL."""

import json
from datetime import date, datetime
from unittest.mock import MagicMock, call, patch

import pytest

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _mock_conn():
    """Create a mock psycopg2 connection with cursor context manager."""
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


SAMPLE_FEATURES = {
    "is_home": 1,
    "line": 25.5,
    "ema_points_L3": 24.1,
    "ema_points_L5": 23.8,
    "opponent_team": "LAL",
}


# ──────────────────────────────────────────────────────────────
# FeatureStore.__init__ and connection
# ──────────────────────────────────────────────────────────────


class TestFeatureStoreInit:

    def test_init_with_external_conn(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        store = FeatureStore(conn=conn)
        assert store._external_conn is True
        assert store._conn is conn

    def test_init_without_conn(self):
        from nba.features.feature_store import FeatureStore

        store = FeatureStore()
        assert store._external_conn is False
        assert store._conn is None

    @patch("nba.features.feature_store.get_features_db_config")
    @patch("psycopg2.connect")
    def test_get_conn_creates_connection(self, mock_connect, mock_config):
        from nba.features.feature_store import FeatureStore

        mock_config.return_value = {
            "host": "localhost",
            "port": 5432,
            "database": "sportsuite",
            "options": "-c search_path=features,public",
        }
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        store = FeatureStore()
        conn = store._get_conn()

        assert conn is mock_conn
        mock_connect.assert_called_once()
        assert mock_conn.autocommit is True

    @patch("nba.features.feature_store.get_features_db_config")
    @patch("psycopg2.connect")
    def test_get_conn_reuses_existing(self, mock_connect, mock_config):
        from nba.features.feature_store import FeatureStore

        existing = MagicMock()
        existing.closed = False
        store = FeatureStore(conn=existing)

        conn = store._get_conn()
        assert conn is existing
        mock_connect.assert_not_called()

    @patch("nba.features.feature_store.get_features_db_config")
    @patch("psycopg2.connect")
    def test_get_conn_reconnects_if_closed(self, mock_connect, mock_config):
        from nba.features.feature_store import FeatureStore

        mock_config.return_value = {"host": "localhost"}
        new_conn = MagicMock()
        new_conn.closed = False
        mock_connect.return_value = new_conn

        store = FeatureStore()
        store._conn = MagicMock()
        store._conn.closed = True

        conn = store._get_conn()
        assert conn is new_conn

    def test_close_external_conn_not_closed(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        conn.closed = False
        store = FeatureStore(conn=conn)
        store.close()
        conn.close.assert_not_called()

    def test_close_own_conn(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        conn.closed = False
        store = FeatureStore()
        store._conn = conn
        store.close()
        conn.close.assert_called_once()

    def test_close_already_closed(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        conn.closed = True
        store = FeatureStore()
        store._conn = conn
        store.close()
        conn.close.assert_not_called()

    def test_close_no_conn(self):
        from nba.features.feature_store import FeatureStore

        store = FeatureStore()
        store.close()  # should not raise


# ──────────────────────────────────────────────────────────────
# write_features
# ──────────────────────────────────────────────────────────────


class TestWriteFeatures:

    def test_write_success(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.write_features(
            "LeBron James", "2026-03-20", "POINTS", "xl_v1", SAMPLE_FEATURES
        )

        assert result is True
        cur.execute.assert_called_once()
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO computed_features" in sql
        assert "ON CONFLICT" in sql

    def test_write_with_date_object(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.write_features(
            "LeBron James", date(2026, 3, 20), "POINTS", "xl_v1", SAMPLE_FEATURES
        )
        assert result is True
        params = cur.execute.call_args[0][1]
        assert params[2] == "2026-03-20"

    def test_write_with_player_id(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.write_features(
            "LeBron James", "2026-03-20", "POINTS", "xl_v1", SAMPLE_FEATURES, player_id=2544
        )
        assert result is True
        params = cur.execute.call_args[0][1]
        assert params[1] == 2544

    def test_write_empty_features_skipped(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.write_features("LeBron James", "2026-03-20", "POINTS", "xl_v1", {})
        assert result is False
        cur.execute.assert_not_called()

    def test_write_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        result = store.write_features(
            "LeBron James", "2026-03-20", "POINTS", "xl_v1", SAMPLE_FEATURES
        )
        assert result is False


# ──────────────────────────────────────────────────────────────
# write_batch
# ──────────────────────────────────────────────────────────────


class TestWriteBatch:

    def test_batch_empty(self):
        from nba.features.feature_store import FeatureStore

        store = FeatureStore(conn=MagicMock())
        assert store.write_batch([], "xl_v1") == 0

    def test_batch_multiple_rows(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        rows = [
            {
                "player_name": "LeBron James",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 1},
            },
            {
                "player_name": "Steph Curry",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 2},
            },
        ]

        count = store.write_batch(rows, "xl_v1")
        assert count == 2
        assert cur.execute.call_count == 2

    def test_batch_skips_empty_features(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        rows = [
            {
                "player_name": "LeBron James",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 1},
            },
            {
                "player_name": "Empty Player",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {},
            },
        ]

        count = store.write_batch(rows, "xl_v1")
        assert count == 1

    def test_batch_with_player_id(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        rows = [
            {
                "player_name": "LeBron James",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 1},
                "player_id": 2544,
            },
        ]

        count = store.write_batch(rows, "xl_v1")
        assert count == 1
        params = cur.execute.call_args[0][1]
        assert params[1] == 2544

    def test_batch_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = [None, Exception("DB error")]
        store = FeatureStore(conn=conn)

        rows = [
            {
                "player_name": "P1",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 1},
            },
            {
                "player_name": "P2",
                "game_date": "2026-03-20",
                "stat_type": "POINTS",
                "feature_values": {"a": 2},
            },
        ]

        count = store.write_batch(rows, "xl_v1")
        assert count == 1  # first succeeded before error


# ──────────────────────────────────────────────────────────────
# read_player_features
# ──────────────────────────────────────────────────────────────


class TestReadPlayerFeatures:

    def test_read_found_json_string(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (json.dumps(SAMPLE_FEATURES),)
        store = FeatureStore(conn=conn)

        result = store.read_player_features("LeBron James", "2026-03-20", "POINTS", "xl_v1")
        assert result == SAMPLE_FEATURES

    def test_read_found_dict(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (SAMPLE_FEATURES,)
        store = FeatureStore(conn=conn)

        result = store.read_player_features("LeBron James", "2026-03-20", "POINTS", "xl_v1")
        assert result == SAMPLE_FEATURES

    def test_read_not_found(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        store = FeatureStore(conn=conn)

        result = store.read_player_features("Unknown", "2026-03-20", "POINTS", "xl_v1")
        assert result is None

    def test_read_with_date_object(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (SAMPLE_FEATURES,)
        store = FeatureStore(conn=conn)

        result = store.read_player_features("LeBron James", date(2026, 3, 20), "POINTS", "xl_v1")
        assert result is not None
        params = cur.execute.call_args[0][1]
        assert params[1] == "2026-03-20"

    def test_read_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        result = store.read_player_features("LeBron James", "2026-03-20", "POINTS", "xl_v1")
        assert result is None


# ──────────────────────────────────────────────────────────────
# read_features (bulk)
# ──────────────────────────────────────────────────────────────


class TestReadFeatures:

    def test_read_basic(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [
            ("LeBron James", date(2026, 3, 20), "POINTS", {"a": 1}),
            ("Steph Curry", date(2026, 3, 20), "POINTS", {"a": 2}),
        ]
        store = FeatureStore(conn=conn)

        results = store.read_features("xl_v1")
        assert len(results) == 2
        assert results[0]["player_name"] == "LeBron James"
        assert results[0]["a"] == 1
        assert results[1]["stat_type"] == "POINTS"

    def test_read_with_date_range(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = []
        store = FeatureStore(conn=conn)

        store.read_features("xl_v1", start_date="2024-01-01", end_date="2025-12-31")
        sql = cur.execute.call_args[0][0]
        assert "game_date >=" in sql
        assert "game_date <=" in sql

    def test_read_with_stat_type_filter(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = []
        store = FeatureStore(conn=conn)

        store.read_features("xl_v1", stat_type="REBOUNDS")
        params = cur.execute.call_args[0][1]
        assert "REBOUNDS" in params

    def test_read_with_limit(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = []
        store = FeatureStore(conn=conn)

        store.read_features("xl_v1", limit=100)
        sql = cur.execute.call_args[0][0]
        assert "LIMIT" in sql
        params = cur.execute.call_args[0][1]
        assert 100 in params

    def test_read_json_string_values(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [
            ("LeBron James", date(2026, 3, 20), "POINTS", json.dumps({"a": 1})),
        ]
        store = FeatureStore(conn=conn)

        results = store.read_features("xl_v1")
        assert results[0]["a"] == 1

    def test_read_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        results = store.read_features("xl_v1")
        assert results == []

    def test_read_all_filters(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = []
        store = FeatureStore(conn=conn)

        store.read_features(
            "xl_v1", start_date="2024-01-01", end_date="2025-12-31", stat_type="POINTS", limit=50
        )
        sql = cur.execute.call_args[0][0]
        assert "game_date >=" in sql
        assert "game_date <=" in sql
        assert "stat_type" in sql
        assert "LIMIT" in sql


# ──────────────────────────────────────────────────────────────
# get_feature_set
# ──────────────────────────────────────────────────────────────


class TestGetFeatureSet:

    def test_found(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = ("xl_v1", 1, 102, ["is_home", "line"], "XL feature set", True)
        store = FeatureStore(conn=conn)

        result = store.get_feature_set("xl_v1")
        assert result["name"] == "xl_v1"
        assert result["version"] == 1
        assert result["feature_count"] == 102
        assert result["is_active"] is True

    def test_not_found(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = None
        store = FeatureStore(conn=conn)

        result = store.get_feature_set("nonexistent")
        assert result is None

    def test_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        result = store.get_feature_set("xl_v1")
        assert result is None


# ──────────────────────────────────────────────────────────────
# list_feature_sets
# ──────────────────────────────────────────────────────────────


class TestListFeatureSets:

    def test_active_only(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [("xl_v1", 1, 102, True), ("v3_v1", 1, 136, True)]
        store = FeatureStore(conn=conn)

        results = store.list_feature_sets(active_only=True)
        assert len(results) == 2
        sql = cur.execute.call_args[0][0]
        assert "is_active = TRUE" in sql

    def test_all(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchall.return_value = [("xl_v1", 1, 102, True)]
        store = FeatureStore(conn=conn)

        results = store.list_feature_sets(active_only=False)
        sql = cur.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        results = store.list_feature_sets()
        assert results == []


# ──────────────────────────────────────────────────────────────
# get_coverage
# ──────────────────────────────────────────────────────────────


class TestGetCoverage:

    def test_with_data(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (5000, date(2024, 1, 1), date(2025, 12, 31))
        store = FeatureStore(conn=conn)

        result = store.get_coverage("xl_v1")
        assert result["count"] == 5000
        assert result["min_date"] == "2024-01-01"
        assert result["max_date"] == "2025-12-31"

    def test_empty(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (None, None, None)
        store = FeatureStore(conn=conn)

        result = store.get_coverage("xl_v1")
        assert result["count"] == 0
        assert result["min_date"] is None

    def test_with_stat_type(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.fetchone.return_value = (100, date(2024, 6, 1), date(2024, 12, 31))
        store = FeatureStore(conn=conn)

        store.get_coverage("xl_v1", stat_type="REBOUNDS")
        params = cur.execute.call_args[0][1]
        assert "REBOUNDS" in params

    def test_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        result = store.get_coverage("xl_v1")
        assert result["count"] == 0


# ──────────────────────────────────────────────────────────────
# update_feature_metadata
# ──────────────────────────────────────────────────────────────


class TestUpdateFeatureMetadata:

    def test_success(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.update_feature_metadata(
            "xl_v1", "ema_points_L3", train_mean=22.5, train_std=5.1, shap_importance=0.032
        )
        assert result is True
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO feature_metadata" in sql
        assert "ON CONFLICT" in sql

    def test_with_all_params(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        store = FeatureStore(conn=conn)

        result = store.update_feature_metadata(
            "xl_v1",
            "is_home",
            train_mean=0.545,
            train_std=0.498,
            train_min=0.0,
            train_max=1.0,
            shap_importance=0.015,
            importance_rank=12,
        )
        assert result is True
        params = cur.execute.call_args[0][1]
        assert params[0] == "xl_v1"
        assert params[1] == "is_home"
        assert params[2] == 0.545

    def test_db_error(self):
        from nba.features.feature_store import FeatureStore

        conn, cur = _mock_conn()
        cur.execute.side_effect = Exception("DB error")
        store = FeatureStore(conn=conn)

        result = store.update_feature_metadata("xl_v1", "bad_feature")
        assert result is False


# ──────────────────────────────────────────────────────────────
# _json_default
# ──────────────────────────────────────────────────────────────


class TestJsonDefault:

    def test_numpy_int(self):
        import numpy as np

        from nba.features.feature_store import _json_default

        assert _json_default(np.int64(42)) == 42
        assert isinstance(_json_default(np.int64(42)), int)

    def test_numpy_float(self):
        import numpy as np

        from nba.features.feature_store import _json_default

        assert _json_default(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_json_default(np.float64(3.14)), float)

    def test_numpy_array(self):
        import numpy as np

        from nba.features.feature_store import _json_default

        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_bool(self):
        import numpy as np

        from nba.features.feature_store import _json_default

        assert _json_default(np.bool_(True)) is True
        assert _json_default(np.bool_(False)) is False

    def test_datetime(self):
        from nba.features.feature_store import _json_default

        result = _json_default(datetime(2026, 3, 20, 15, 30))
        assert "2026-03-20" in result

    def test_date(self):
        from nba.features.feature_store import _json_default

        result = _json_default(date(2026, 3, 20))
        assert result == "2026-03-20"

    def test_unsupported_type(self):
        from nba.features.feature_store import _json_default

        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(set([1, 2, 3]))
