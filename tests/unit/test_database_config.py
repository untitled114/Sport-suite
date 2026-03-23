"""Tests for nba.config.database — legacy per-port + consolidated schema interface."""

import os
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────
# LEGACY INTERFACE (per-port functions)
# ──────────────────────────────────────────────────────────────


class TestGetDbConfig:
    def test_default_values(self):
        from nba.config.database import get_db_config

        config = get_db_config("TEST", 9999, "test_db")
        assert config["host"] == "localhost"
        assert config["port"] == 9999
        assert config["database"] == "test_db"
        assert config["connect_timeout"] == 10

    @patch.dict(
        os.environ, {"TEST_DB_HOST": "remote", "TEST_DB_PORT": "6000", "TEST_DB_NAME": "alt"}
    )
    def test_env_overrides(self):
        from nba.config.database import get_db_config

        config = get_db_config("TEST", 9999, "test_db")
        assert config["host"] == "remote"
        assert config["port"] == 6000
        assert config["database"] == "alt"

    @patch.dict(os.environ, {"TEST_DB_USER": "custom_user"})
    def test_user_override(self):
        from nba.config.database import get_db_config

        config = get_db_config("TEST", 9999, "test_db")
        assert config["user"] == "custom_user"

    def test_custom_default_host(self):
        from nba.config.database import get_db_config

        config = get_db_config("TEST", 9999, "test_db", default_host="192.168.1.1")
        assert config["host"] == "192.168.1.1"

    @patch.dict(os.environ, {"NBA_DB_CONNECT_TIMEOUT": "30"})
    def test_timeout_override(self):
        from nba.config.database import get_db_config

        config = get_db_config("TEST", 9999, "test_db")
        assert config["connect_timeout"] == 30


class TestLegacyAliases:
    """Legacy config functions now route to consolidated DB (port 5500)."""

    def test_players(self):
        from nba.config.database import get_players_db_config

        config = get_players_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"
        assert "search_path=players" in config["options"]

    def test_games(self):
        from nba.config.database import get_games_db_config

        config = get_games_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"
        assert "search_path=games" in config["options"]

    def test_team(self):
        from nba.config.database import get_team_db_config

        config = get_team_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"
        assert "search_path=teams" in config["options"]

    def test_intelligence(self):
        from nba.config.database import get_intelligence_db_config

        config = get_intelligence_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"
        assert "search_path=intelligence" in config["options"]

    def test_all_have_required_keys(self):
        from nba.config.database import (
            get_games_db_config,
            get_intelligence_db_config,
            get_players_db_config,
            get_team_db_config,
        )

        for fn in [
            get_players_db_config,
            get_games_db_config,
            get_team_db_config,
            get_intelligence_db_config,
        ]:
            config = fn()
            for key in ("host", "port", "database", "user", "password", "connect_timeout"):
                assert key in config, f"{fn.__name__} missing {key}"


class TestMongoConfig:
    def test_default_no_auth(self):
        from nba.config.database import get_mongo_config

        config = get_mongo_config()
        assert "uri" in config
        assert config["database"] == "nba_betting_xl"
        assert config["collection"] == "nba_props_xl"
        assert config["timeout_ms"] == 8000

    @patch.dict(os.environ, {"MONGO_USER": "user", "MONGO_PASSWORD": "pa$$"})
    def test_auth_uri(self):
        from nba.config.database import get_mongo_config

        config = get_mongo_config()
        assert "user" in config["uri"]
        assert "pa%24%24" in config["uri"]  # URL-encoded

    @patch.dict(os.environ, {"NBA_MONGO_URI": "mongodb://custom:27018/"})
    def test_uri_override(self):
        from nba.config.database import get_mongo_config

        config = get_mongo_config()
        assert config["uri"] == "mongodb://custom:27018/"


class TestCredentials:
    def test_default_user(self):
        from nba.config.database import DB_DEFAULT_USER

        assert isinstance(DB_DEFAULT_USER, str)
        assert len(DB_DEFAULT_USER) > 0

    def test_password_warning(self):
        import importlib

        env = os.environ.copy()
        env.pop("DB_PASSWORD", None)
        env.pop("NBA_DB_PASSWORD", None)
        with patch.dict(os.environ, env, clear=True):
            import nba.config.database

            with pytest.warns(RuntimeWarning, match="DB_PASSWORD"):
                importlib.reload(nba.config.database)


# ──────────────────────────────────────────────────────────────
# CONSOLIDATED INTERFACE (schema-based)
# ──────────────────────────────────────────────────────────────


class TestSchemas:
    def test_all_six_present(self):
        from nba.config.database import SCHEMAS

        assert set(SCHEMAS.keys()) == {
            "players",
            "games",
            "teams",
            "intelligence",
            "axiom",
            "features",
        }


class TestGetSchemaConfig:
    @pytest.mark.parametrize(
        "schema", ["players", "games", "teams", "intelligence", "axiom", "features"]
    )
    def test_valid_schemas(self, schema):
        from nba.config.database import get_schema_config

        config = get_schema_config(schema)
        assert config["database"] == "sportsuite"
        assert config["port"] == 5500
        assert f"search_path={schema}" in config["options"]

    def test_has_all_psycopg2_keys(self):
        from nba.config.database import get_schema_config

        config = get_schema_config("intelligence")
        for key in ("host", "port", "database", "user", "password", "connect_timeout", "options"):
            assert key in config

    def test_invalid_schema_raises(self):
        from nba.config.database import get_schema_config

        with pytest.raises(ValueError, match="Unknown schema"):
            get_schema_config("nonexistent")

    def test_error_lists_valid_schemas(self):
        from nba.config.database import get_schema_config

        with pytest.raises(ValueError, match="players"):
            get_schema_config("bad")

    @patch.dict(os.environ, {"DB_HOST": "db.prod", "DB_PORT": "6000", "DB_NAME": "prod_db"})
    def test_env_overrides(self):
        from nba.config.database import get_schema_config

        config = get_schema_config("intelligence")
        assert config["host"] == "db.prod"
        assert config["port"] == 6000
        assert config["database"] == "prod_db"


class TestGetConnection:
    @patch("psycopg2.connect")
    def test_returns_connection(self, mock_connect):
        from nba.config.database import get_connection

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        conn = get_connection("intelligence")
        assert conn is mock_conn
        mock_connect.assert_called_once()

    @patch("psycopg2.connect")
    def test_autocommit_default_true(self, mock_connect):
        from nba.config.database import get_connection

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        get_connection("players")
        assert mock_conn.autocommit is True

    @patch("psycopg2.connect")
    def test_autocommit_false(self, mock_connect):
        from nba.config.database import get_connection

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        get_connection("games", autocommit=False)
        assert mock_conn.autocommit is False

    @patch("psycopg2.connect")
    def test_passes_schema_options(self, mock_connect):
        from nba.config.database import get_connection

        mock_connect.return_value = MagicMock()
        get_connection("axiom")
        kwargs = mock_connect.call_args[1]
        assert "axiom" in kwargs["options"]

    def test_invalid_schema_raises(self):
        from nba.config.database import get_connection

        with pytest.raises(ValueError):
            get_connection("invalid")


class TestAxiomConfig:
    def test_returns_schema_config(self):
        from nba.config.database import get_axiom_db_config

        config = get_axiom_db_config()
        assert "axiom" in config["options"]
        assert config["database"] == "sportsuite"


class TestFeaturesConfig:
    def test_returns_schema_config(self):
        from nba.config.database import get_features_db_config

        config = get_features_db_config()
        assert "features" in config["options"]
        assert config["database"] == "sportsuite"
