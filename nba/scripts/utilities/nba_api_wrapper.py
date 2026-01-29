"""
NBA API Wrapper
Provides robust interface to NBA Stats API with rate limiting, error handling, and caching.

Usage:
    from nba_api_wrapper import NBAApiWrapper

    api = NBAApiWrapper()
    players = api.get_all_players(season="2023-24")
    games = api.get_season_games(season="2023-24")
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints.boxscoretraditionalv2 import BoxScoreTraditionalV2
from nba_api.stats.endpoints.commonplayerinfo import CommonPlayerInfo
from nba_api.stats.endpoints.leaguedashplayerstats import LeagueDashPlayerStats
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.endpoints.playergamelog import PlayerGameLog
from nba_api.stats.endpoints.teamgamelog import TeamGameLog

# NBA API imports
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAApiWrapper:
    """Wrapper for NBA Stats API with rate limiting and error handling"""

    def __init__(self, requests_per_minute: int = 20, cache_ttl: int = 3600):
        """
        Initialize NBA API wrapper

        Args:
            requests_per_minute: Rate limit for API calls (default: 20)
            cache_ttl: Cache time-to-live in seconds (default: 3600)
        """
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.cache = {}
        self.cache_ttl = cache_ttl

        logger.info(f"Initialized NBA API wrapper (rate limit: {requests_per_minute} req/min)")

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit: {key}")
                return data
            else:
                del self.cache[key]
        return None

    def _save_to_cache(self, key: str, data: Any):
        """Save data to cache with timestamp"""
        self.cache[key] = (data, time.time())

    def _api_call_with_retry(self, endpoint_func, max_retries: int = 3, **kwargs) -> pd.DataFrame:
        """
        Make API call with exponential backoff retry logic

        Args:
            endpoint_func: NBA API endpoint function
            max_retries: Maximum number of retry attempts
            **kwargs: Arguments to pass to endpoint function

        Returns:
            DataFrame with API response data
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")

                endpoint = endpoint_func(**kwargs)
                df = endpoint.get_data_frames()[0]

                logger.info(f"API call successful: {endpoint_func.__name__} ({len(df)} records)")
                return df

            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries} attempts")
                    raise

        return pd.DataFrame()

    # ========== PLAYER METHODS ==========

    def get_all_players(
        self, season: Optional[str] = None, active_only: bool = True
    ) -> pd.DataFrame:
        """
        Get all NBA players

        Args:
            season: Season (e.g., "2023-24") - if None, returns all-time players
            active_only: If True, return only currently active players

        Returns:
            DataFrame with player info (id, full_name, is_active)
        """
        cache_key = f"all_players_{season}_{active_only}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching all players (season={season}, active_only={active_only})")

        # Get static player list (fast, no rate limiting needed)
        player_list = static_players.get_players()
        df = pd.DataFrame(player_list)

        if active_only:
            df = df[df["is_active"]]

        self._save_to_cache(cache_key, df)
        logger.info(f"Retrieved {len(df)} players")
        return df

    def get_player_season_stats(self, season: str = "2023-24") -> pd.DataFrame:
        """
        Get per-game stats for all players in a season

        Args:
            season: Season (e.g., "2023-24")

        Returns:
            DataFrame with player season stats
        """
        cache_key = f"player_season_stats_{season}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching player season stats for {season}")

        df = self._api_call_with_retry(
            LeagueDashPlayerStats,
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",  # Fix: Include advanced stats (USG_PCT)
            season_type_all_star="Regular Season",
        )

        self._save_to_cache(cache_key, df)
        return df

    def get_player_game_logs(
        self, player_id: int, season: str = "2023-24", season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get game logs for a specific player

        Args:
            player_id: NBA player ID
            season: Season (e.g., "2023-24")
            season_type: "Regular Season", "Playoffs", "All Star"

        Returns:
            DataFrame with game-by-game stats
        """
        cache_key = f"player_game_logs_{player_id}_{season}_{season_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching game logs for player {player_id} ({season})")

        df = self._api_call_with_retry(
            PlayerGameLog, player_id=player_id, season=season, season_type_all_star=season_type
        )

        self._save_to_cache(cache_key, df)
        return df

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        """
        Get detailed player profile info

        Args:
            player_id: NBA player ID

        Returns:
            Dictionary with player profile data
        """
        cache_key = f"player_info_{player_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching player info for {player_id}")

        df = self._api_call_with_retry(CommonPlayerInfo, player_id=player_id)

        if len(df) > 0:
            info = df.iloc[0].to_dict()
            self._save_to_cache(cache_key, info)
            return info

        return {}

    # ========== TEAM METHODS ==========

    def get_all_teams(self) -> pd.DataFrame:
        """
        Get all NBA teams

        Returns:
            DataFrame with team info (id, full_name, abbreviation, city)
        """
        cache_key = "all_teams"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("Fetching all teams")

        # Get static team list (fast, no rate limiting needed)
        team_list = static_teams.get_teams()
        df = pd.DataFrame(team_list)

        self._save_to_cache(cache_key, df)
        logger.info(f"Retrieved {len(df)} teams")
        return df

    def get_team_game_logs(
        self, team_id: int, season: str = "2023-24", season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get game logs for a specific team

        Args:
            team_id: NBA team ID
            season: Season (e.g., "2023-24")
            season_type: "Regular Season", "Playoffs"

        Returns:
            DataFrame with team game logs
        """
        cache_key = f"team_game_logs_{team_id}_{season}_{season_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching team game logs for team {team_id} ({season})")

        df = self._api_call_with_retry(
            TeamGameLog, team_id=team_id, season=season, season_type_all_star=season_type
        )

        self._save_to_cache(cache_key, df)
        return df

    # ========== GAME METHODS ==========

    def get_season_games(
        self, season: str = "2023-24", season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get all games for a season

        Args:
            season: Season (e.g., "2023-24")
            season_type: "Regular Season", "Playoffs"

        Returns:
            DataFrame with all games
        """
        cache_key = f"season_games_{season}_{season_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching all games for {season} {season_type}")

        df = self._api_call_with_retry(
            LeagueGameFinder,
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",
        )

        self._save_to_cache(cache_key, df)
        logger.info(f"Retrieved {len(df)} game records")
        return df

    def get_box_score(self, game_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get detailed box score for a specific game

        Args:
            game_id: NBA game ID (format: "0022300001")

        Returns:
            Dictionary with player_stats, team_stats DataFrames
        """
        cache_key = f"box_score_{game_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Fetching box score for game {game_id}")

        endpoint = self._api_call_with_retry(BoxScoreTraditionalV2, game_id=game_id)

        # Box score returns multiple DataFrames
        data_frames = endpoint.get_data_frames() if hasattr(endpoint, "get_data_frames") else []

        result = {
            "player_stats": data_frames[0] if len(data_frames) > 0 else pd.DataFrame(),
            "team_stats": data_frames[1] if len(data_frames) > 1 else pd.DataFrame(),
        }

        self._save_to_cache(cache_key, result)
        return result

    # ========== UTILITY METHODS ==========

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {"total_entries": len(self.cache), "cache_ttl_seconds": self.cache_ttl}


# Convenience functions for quick access
def get_players_wrapper(season: str = "2023-24") -> NBAApiWrapper:
    """Get pre-configured API wrapper for player data"""
    return NBAApiWrapper(requests_per_minute=20)


def get_games_wrapper(season: str = "2023-24") -> NBAApiWrapper:
    """Get pre-configured API wrapper for game data"""
    return NBAApiWrapper(requests_per_minute=15)  # More conservative for game data


if __name__ == "__main__":
    # Test the wrapper
    print("Testing NBA API Wrapper...")

    api = NBAApiWrapper()

    # Test 1: Get all active players
    print("\n1. Fetching active players...")
    players = api.get_all_players(active_only=True)
    print(f"Found {len(players)} active players")
    print(players.head())

    # Test 2: Get all teams
    print("\n2. Fetching teams...")
    teams = api.get_all_teams()
    print(f"Found {len(teams)} teams")
    print(teams.head())

    # Test 3: Get season games
    print("\n3. Fetching games for 2023-24...")
    games = api.get_season_games(season="2023-24")
    print(f"Found {len(games)} game records")
    print(games.head())

    # Test 4: Cache stats
    print("\n4. Cache statistics:")
    print(api.get_cache_stats())

    print("\n✅ NBA API Wrapper test complete!")
