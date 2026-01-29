"""
Centralized Database Configuration for NBA Suite
=================================================
All database connection configs should be imported from this module.
Credentials are read from environment variables with sensible fallbacks.

Environment Variables:
    DB_USER: Default database username (fallback: 'nba_user')
    DB_PASSWORD: Default database password (REQUIRED - no hardcoded default)
    NBA_*_DB_*: Per-database overrides (host, port, user, password, name)
    MONGO_URI: MongoDB connection string
    MONGO_USER: MongoDB username
    MONGO_PASSWORD: MongoDB password

Usage:
    from nba.config import get_intelligence_db_config

    config = get_intelligence_db_config()
    conn = psycopg2.connect(**config)
"""

import os
from typing import Any, Dict

# Default credentials from environment (REQUIRED - no hardcoded passwords)
DB_DEFAULT_USER = os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
DB_DEFAULT_PASSWORD = os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))

if DB_DEFAULT_PASSWORD is None:
    import warnings

    warnings.warn(
        "DB_PASSWORD environment variable not set. "
        "Database connections will fail until credentials are configured.",
        RuntimeWarning,
        stacklevel=2,
    )


def get_db_config(
    env_prefix: str, default_port: int, default_database: str, default_host: str = "localhost"
) -> Dict[str, Any]:
    """
    Build database config dict from environment variables.

    Args:
        env_prefix: Prefix for env vars (e.g., 'NBA_INT' -> NBA_INT_DB_HOST)
        default_port: Default port if not in env
        default_database: Default database name if not in env
        default_host: Default host if not in env

    Returns:
        Dict with psycopg2 connection parameters
    """
    return {
        "host": os.getenv(f"{env_prefix}_DB_HOST", default_host),
        "port": int(os.getenv(f"{env_prefix}_DB_PORT", default_port)),
        "database": os.getenv(f"{env_prefix}_DB_NAME", default_database),
        "user": os.getenv(f"{env_prefix}_DB_USER", DB_DEFAULT_USER),
        "password": os.getenv(f"{env_prefix}_DB_PASSWORD", DB_DEFAULT_PASSWORD),
        "connect_timeout": int(os.getenv("NBA_DB_CONNECT_TIMEOUT", 10)),
    }


def get_players_db_config() -> Dict[str, Any]:
    """Get nba_players database config (port 5536)."""
    return get_db_config("NBA_PLAYERS", 5536, "nba_players")


def get_games_db_config() -> Dict[str, Any]:
    """Get nba_games database config (port 5537)."""
    return get_db_config("NBA_GAMES", 5537, "nba_games")


def get_team_db_config() -> Dict[str, Any]:
    """Get nba_team database config (port 5538)."""
    return get_db_config("NBA_TEAM", 5538, "nba_team")


def get_intelligence_db_config() -> Dict[str, Any]:
    """
    Get nba_intelligence database config (port 5539).

    IMPORTANT: This is the ORIGINAL/LEGACY database for props.
    Do NOT use port 5540 (nba_reference) - it gives mixed/bad predictions.
    """
    return get_db_config("NBA_INT", 5539, "nba_intelligence")


def get_mongo_config() -> Dict[str, Any]:
    """
    Get MongoDB connection config.

    Returns dict with 'uri', 'database', 'collection', and 'timeout_ms'.
    """
    mongo_user = os.getenv("MONGO_USER", "")
    mongo_password = os.getenv("MONGO_PASSWORD", "")

    # Build URI - only include auth if credentials provided
    if mongo_user and mongo_password:
        default_uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27017/"
    else:
        default_uri = "mongodb://localhost:27017/"

    return {
        "uri": os.getenv("NBA_MONGO_URI", os.getenv("MONGO_URI", default_uri)),
        "database": os.getenv("NBA_MONGO_DB", "nba_betting_xl"),
        "collection": os.getenv("NBA_MONGO_COLLECTION", "nba_props_xl"),
        "timeout_ms": int(os.getenv("NBA_MONGO_TIMEOUT_MS", 8000)),
    }
