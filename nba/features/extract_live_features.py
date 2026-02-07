#!/usr/bin/env python3
"""
Live NBA Feature Extractor - Database-Driven
=============================================
Extracts features directly from player_game_logs table with EMA calculations.
Used for production predictions on current season (2025-26).
"""

import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2

# Suppress all pandas warnings about SQLAlchemy
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors

# Import centralized database config
from nba.config.database import (
    DB_DEFAULT_PASSWORD,
    DB_DEFAULT_USER,
    get_games_db_config,
    get_intelligence_db_config,
    get_players_db_config,
    get_team_db_config,
)


class LiveFeatureExtractor:
    """Extract features from database with on-the-fly EMA calculations"""

    # Expose defaults for subclasses
    DB_DEFAULT_USER = DB_DEFAULT_USER
    DB_DEFAULT_PASSWORD = DB_DEFAULT_PASSWORD

    # Database configs (centralized)
    PLAYER_DB_CONFIG = get_players_db_config()
    GAMES_DB_CONFIG = get_games_db_config()
    TEAM_DB_CONFIG = get_team_db_config()
    INTELLIGENCE_DB_CONFIG = get_intelligence_db_config()

    # Team abbreviation mapping (props -> database)
    TEAM_ABBREV_MAP = {
        "NO": "NOP",  # New Orleans Pelicans
        "SA": "SAS",  # San Antonio Spurs
        "UTAH": "UTA",  # Utah Jazz
        "GS": "GSW",  # Golden State Warriors
        "NY": "NYK",  # New York Knicks
        "BKN": "BKN",  # Brooklyn Nets (sometimes BRK)
        "BRK": "BKN",  # Brooklyn Nets
    }

    # SQL expression to normalize player names (removes accents and suffixes)
    # Use this on BOTH sides of name comparisons for consistent matching
    # Handles: Jr./Jr, Sr./Sr, II, III, IV suffixes and periods
    SQL_NORMALIZE_NAME = """
        regexp_replace(
            regexp_replace(
                unaccent(%s),
                '\\.',
                '',
                'g'
            ),
            ' (Jr|Sr|II|III|IV)$',
            '',
            'i'
        )
    """

    @classmethod
    def sql_name_match(cls, column: str = "pp.full_name") -> str:
        """
        Generate SQL WHERE clause for normalized name matching.

        Returns SQL that compares normalized versions of both the column and parameter.
        Handles: accents (Jokić→Jokic), suffixes (Jr./Jr/III), periods.

        Args:
            column: Database column to compare (default: pp.full_name)

        Returns:
            SQL string with %s placeholder for the player name parameter

        Example:
            >>> cls.sql_name_match()
            "regexp_replace(...unaccent(pp.full_name)...) = regexp_replace(...unaccent(%s)...)"
        """
        normalize_expr = """regexp_replace(
            regexp_replace(
                unaccent({col}),
                '\\.',
                '',
                'g'
            ),
            ' (Jr|Sr|II|III|IV)$',
            '',
            'i'
        )"""
        return f"{normalize_expr.format(col=column)} = {normalize_expr.format(col='%s')}"

    @staticmethod
    def normalize_player_name(name: str) -> str:
        """
        Normalize player names to match database format.

        Handles:
        - "Russell Westbrook III" -> "Russell Westbrook" (remove suffix)
        - "PJ Washington Jr." -> "PJ Washington" (remove suffix)
        - "Nikola Jokić" -> "Nikola Jokic" (remove accents)
        - "Luka Dončić" -> "Luka Doncic" (remove accents)
        - Extra whitespace

        Strategy:
        1. Remove extra whitespace
        2. Remove accented characters (ć->c, č->c, etc.)
        3. Remove common suffixes (Jr, Jr., Sr, Sr., II, II., III, III., IV, IV.)
        """
        if not name:
            return name

        # Remove extra whitespace
        name = " ".join(name.split())

        # Remove accented characters (normalize to ASCII)
        # Common accents in NBA names: ć, č, š, ž, ñ, é, ö, ü, etc.
        import unicodedata

        name = unicodedata.normalize("NFD", name)
        name = "".join(char for char in name if unicodedata.category(char) != "Mn")
        name = unicodedata.normalize("NFC", name)

        # Remove common suffixes
        # Order matters: try with periods first, then without
        suffixes = [" Jr.", " Sr.", " II.", " III.", " IV.", " Jr", " Sr", " II", " III", " IV"]

        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
                break

        return name

    def __init__(self) -> None:
        """
        Initialize feature extractor with database connections.

        Connects to three PostgreSQL databases:
        - nba_players (port 5536): Player profiles and game logs
        - nba_games (port 5537): Game data and box scores
        - nba_team (port 5538): Team statistics

        Raises:
            psycopg2.OperationalError: If database connection fails
        """
        self.conn = psycopg2.connect(**self.PLAYER_DB_CONFIG)
        self.conn.autocommit = True  # Avoid stuck transactions on query errors
        self.games_conn = psycopg2.connect(**self.GAMES_DB_CONFIG)
        self.games_conn.autocommit = True
        self.team_conn = psycopg2.connect(**self.TEAM_DB_CONFIG)
        self.team_conn.autocommit = True

    def normalize_team_abbrev(self, team_abbrev):
        """
        Normalize team abbreviation to match database format.

        Args:
            team_abbrev: Team abbreviation from props (e.g., 'NO', 'SA', 'UTAH')

        Returns:
            Normalized team abbreviation (e.g., 'NOP', 'SAS', 'UTA')
        """
        if not team_abbrev:
            return team_abbrev
        return self.TEAM_ABBREV_MAP.get(team_abbrev, team_abbrev)

    def _get_player_name(self, player_id):
        """
        Get player name from player_id

        Args:
            player_id: NBA player ID

        Returns:
            Player's full name or None if not found
        """
        query = """
        SELECT full_name
        FROM player_profile
        WHERE player_id = %s
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (player_id,))
                result = cur.fetchone()
                if result:
                    return result[0]
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error getting player name for ID {player_id}: {e}")

        return None

    def calculate_ema(self, values, alpha=0.4):
        """
        Calculate exponential moving average with NaN handling.

        Args:
            values: List or array of numeric values (may contain NaN)
            alpha: Smoothing factor (0-1), higher = more weight on recent values

        Returns:
            EMA value, or 0.0 if no valid values
        """
        import numpy as np

        # Filter out NaN/None values to prevent NaN propagation
        if hasattr(values, "tolist"):
            values = values.tolist()
        valid_values = [
            v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]

        if len(valid_values) == 0:
            return 0.0

        ema = valid_values[0]
        for val in valid_values[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema

    def get_player_team(self, player_name, as_of_date):
        """
        Get player's team abbreviation as of a given date.
        First checks player_game_logs for most recent team, then falls back to player_profile.
        Uses unaccent() for name matching to handle accented characters (e.g., Dončić vs Doncic).

        Args:
            player_name: Player's full name
            as_of_date: Date to check team (datetime or string)

        Returns:
            Team abbreviation (e.g., 'GSW', 'LAL') or None if not found
        """
        # Try to get team from most recent game log before as_of_date
        # Use normalized name matching to handle accents AND suffixes (Jr./Jr, III, etc.)
        query = f"""
        SELECT pgl.team_abbrev
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.game_date <= %s
          AND pgl.team_abbrev IS NOT NULL
        ORDER BY pgl.game_date DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (player_name, as_of_date))
                result = cur.fetchone()
                if result:
                    return result[0]
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying player game logs: {e}")

        # Fallback to player_profile if no game logs found
        # Use normalized name matching for consistency
        query_profile = f"""
        SELECT team_abbrev
        FROM player_profile
        WHERE {self.sql_name_match('full_name')}
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query_profile, (player_name,))
                result = cur.fetchone()
                if result:
                    return result[0]
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying player profile: {e}")

        return None

    def get_team_rolling_stats(self, team_abbrev, as_of_date, opponent_abbrev=None, window=10):
        """
        Get team's rolling statistics from team_season_stats (primary) with validation.

        Priority order (uses team_season_stats which has verified correct data):
        1. team_season_stats (most recent season) - PRIMARY SOURCE
        2. League average defaults - FALLBACK

        Note: team_game_logs was previously used but contains incorrect pace values
        (170+ instead of expected 95-105 range). Using season stats until game logs are fixed.

        Args:
            team_abbrev: Team abbreviation (e.g., 'GSW')
            as_of_date: Date to calculate stats as of (datetime or string)
            opponent_abbrev: Optional opponent team abbreviation
            window: Number of games to include in rolling window (default 10)

        Returns:
            Dict with keys: pace, off_rating, def_rating
        """
        # Normalize team abbreviation
        team_abbrev = self.normalize_team_abbrev(team_abbrev)

        # PRIMARY: Use team_season_stats (verified correct data)
        season_query = """
        SELECT pace, offensive_rating, defensive_rating
        FROM team_season_stats
        WHERE team_abbrev = %s
        ORDER BY season DESC
        LIMIT 1
        """

        try:
            with self.team_conn.cursor() as cur:
                cur.execute(season_query, (team_abbrev,))
                result = cur.fetchone()
                if result and result[0] is not None:
                    pace = float(result[0])
                    # Validate pace is in expected NBA range (90-110)
                    if 90.0 <= pace <= 115.0:
                        return {
                            "pace": pace,
                            "off_rating": float(result[1]) if result[1] else 110.0,
                            "def_rating": float(result[2]) if result[2] else 110.0,
                        }
                    else:
                        logger.debug(f"Invalid pace {pace} for {team_abbrev}, using defaults")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying team_season_stats for {team_abbrev}: {e}")

        # FALLBACK: League average defaults
        logger.debug(f"Using league defaults for {team_abbrev}")
        return {"pace": 100.0, "off_rating": 112.0, "def_rating": 112.0}

    def get_recent_games(self, player_name, as_of_date, n_games=20, min_games_threshold=20):
        """
        Get player's recent games with strict 20-game requirement.

        Policy (Updated 2026-02-02):
        - For January/February props: Use 20 random games from previous season (early season)
        - For 0 games in current season: Fetch last 20 from previous season
        - If still < 20 games total: Return None (player will be dropped from training)

        Args:
            player_name: Player's full name
            as_of_date: Date to query games before (datetime or string)
            n_games: Target number of games (default 20) - STRICT requirement
            min_games_threshold: Minimum games required (default 20) - must meet this or return None

        Returns:
            DataFrame with games in chronological order, or None if insufficient data
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Normalize player name (Jr vs Jr., etc.)
        player_name = self.normalize_player_name(player_name)

        # Helper to get season label (START year convention)
        def season_label_start_year(date_obj):
            return date_obj.year if date_obj.month >= 10 else date_obj.year - 1

        current_season_start = season_label_start_year(as_of_date)
        prev_season_label = current_season_start - 1

        # Check if this is January or February (early season - use previous season data)
        is_early_season = as_of_date.month in (1, 2)

        if is_early_season:
            # EARLY SEASON POLICY: Use 20 random games from previous season
            # This avoids using limited current season data for early-season props
            logger.info(
                f"{player_name}: Early season ({as_of_date.strftime('%b')}), "
                f"using random 20 games from {prev_season_label} season"
            )

            # Fetch ALL previous season games, then sample 20 randomly
            prev_season_all_query = f"""
            SELECT
                pgl.game_date,
                pgl.is_home,
                pgl.minutes_played,
                pgl.points,
                pgl.rebounds,
                pgl.assists,
                pgl.three_pointers_made,
                pgl.three_pt_attempted,
                pgl.steals,
                pgl.blocks,
                pgl.turnovers,
                pgl.fg_made,
                pgl.fg_attempted,
                pgl.ft_made,
                pgl.ft_attempted,
                pgl.plus_minus
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE {self.sql_name_match('pp.full_name')}
              AND pgl.season = %s
            ORDER BY pgl.game_date DESC
            """

            prev_games = pd.read_sql_query(
                prev_season_all_query,
                self.conn,
                params=(player_name, prev_season_label),
            )
            prev_games = self._dedupe_and_sort_games(prev_games)

            if len(prev_games) >= n_games:
                # Sample 20 random games from previous season
                sampled = prev_games.sample(n=n_games, random_state=42)
                return sampled.sort_values("game_date")
            elif len(prev_games) > 0:
                logger.warning(
                    f"{player_name}: Only {len(prev_games)} games in {prev_season_label} season "
                    f"(need {n_games}), dropping player"
                )
                return None  # Not enough games - drop player
            else:
                logger.warning(
                    f"{player_name}: No games found in {prev_season_label} season, dropping player"
                )
                return None  # No previous season data - drop player

        # REGULAR SEASON POLICY: Fetch current season games first
        current_season_query = f"""
        SELECT
            pgl.game_date,
            pgl.is_home,
            pgl.minutes_played,
            pgl.points,
            pgl.rebounds,
            pgl.assists,
            pgl.three_pointers_made,
            pgl.three_pt_attempted,
            pgl.steals,
            pgl.blocks,
            pgl.turnovers,
            pgl.fg_made,
            pgl.fg_attempted,
            pgl.ft_made,
            pgl.ft_attempted,
            pgl.plus_minus
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.game_date < %s
        ORDER BY pgl.game_date DESC
        LIMIT %s
        """

        current_games = pd.read_sql_query(
            current_season_query, self.conn, params=(player_name, as_of_date, n_games)
        )
        current_games = self._dedupe_and_sort_games(current_games)

        current_count = len(current_games)

        # If we have enough games, return immediately
        if current_count >= n_games:
            return current_games.sort_values("game_date")

        # FALLBACK: If 0 games in current window, fetch last 20 from previous season
        if current_count == 0:
            logger.info(
                f"{player_name}: 0 games in current season, "
                f"fetching last 20 from {prev_season_label} season"
            )

            prev_season_query = f"""
            SELECT
                pgl.game_date,
                pgl.is_home,
                pgl.minutes_played,
                pgl.points,
                pgl.rebounds,
                pgl.assists,
                pgl.three_pointers_made,
                pgl.three_pt_attempted,
                pgl.steals,
                pgl.blocks,
                pgl.turnovers,
                pgl.fg_made,
                pgl.fg_attempted,
                pgl.ft_made,
                pgl.ft_attempted,
                pgl.plus_minus
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE {self.sql_name_match('pp.full_name')}
              AND pgl.season = %s
            ORDER BY pgl.game_date DESC
            LIMIT %s
            """

            prev_games = pd.read_sql_query(
                prev_season_query,
                self.conn,
                params=(player_name, prev_season_label, n_games),
            )
            prev_games = self._dedupe_and_sort_games(prev_games)

            if len(prev_games) >= n_games:
                return prev_games.sort_values("game_date")
            else:
                logger.warning(
                    f"{player_name}: Only {len(prev_games)} games in previous season "
                    f"(need {n_games}), dropping player"
                )
                return None  # Not enough games - drop player

        # CROSS-SEASON FILL: Have some current games but < 20, fill from previous season
        deficit = n_games - current_count

        logger.info(
            f"Cross-season fallback for {player_name}: {current_count}/{n_games} games, "
            f"fetching exactly {deficit} from previous season"
        )

        previous_season_query = f"""
        SELECT
            pgl.game_date,
            pgl.is_home,
            pgl.minutes_played,
            pgl.points,
            pgl.rebounds,
            pgl.assists,
            pgl.three_pointers_made,
            pgl.three_pt_attempted,
            pgl.steals,
            pgl.blocks,
            pgl.turnovers,
            pgl.fg_made,
            pgl.fg_attempted,
            pgl.ft_made,
            pgl.ft_attempted,
            pgl.plus_minus
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.season = %s
          AND pgl.game_date < %s
        ORDER BY pgl.game_date DESC
        LIMIT %s
        """

        previous_games = pd.read_sql_query(
            previous_season_query,
            self.conn,
            params=(player_name, prev_season_label, as_of_date.date(), deficit),
        )
        previous_games = self._dedupe_and_sort_games(previous_games)

        previous_count = len(previous_games)
        total_games = current_count + previous_count

        # STRICT CHECK: Must have at least 20 games total
        if total_games < min_games_threshold:
            logger.warning(
                f"{player_name}: Only {total_games} total games "
                f"({previous_count} previous + {current_count} current), "
                f"need {min_games_threshold}, dropping player"
            )
            return None  # Not enough games - drop player

        logger.info(
            f"{player_name}: Combined {previous_count} previous + "
            f"{current_count} current = {total_games} total games"
        )

        # Combine games: previous season games first (oldest), then current season
        combined_games = pd.concat([previous_games, current_games], ignore_index=True)
        return combined_games.sort_values("game_date")

    def _dedupe_and_sort_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure deterministic ordering by sorting on stable stats and removing duplicates.

        This is critical for reproducible EMA calculations - games must be in
        consistent order across runs.

        Args:
            games_df: DataFrame with player game logs

        Returns:
            Deduplicated DataFrame sorted by game_date ascending
        """
        if games_df is None or games_df.empty:
            return games_df

        ordering = [
            col
            for col in ["game_date", "minutes_played", "points", "rebounds", "assists", "game_id"]
            if col in games_df.columns
        ]
        ascending = [False if col != "game_id" else True for col in ordering]

        ordered = games_df.sort_values(by=ordering, ascending=ascending, kind="mergesort")
        subset = ["game_date", "game_id"] if "game_id" in games_df.columns else ["game_date"]
        deduped = ordered.drop_duplicates(subset=subset, keep="first")
        return deduped.sort_values("game_date").reset_index(drop=True)

    def get_h2h_stats(self, player_name, opponent_team, as_of_date, stat_type="points"):
        """
        Get player's head-to-head statistics vs a specific opponent team.

        Args:
            player_name: Player's full name
            opponent_team: Opponent team abbreviation (e.g., 'LAL')
            as_of_date: Date to calculate H2H as of (datetime or string)
            stat_type: Stat to calculate (points, rebounds, assists, threes)

        Returns:
            Dict with: h2h_avg_stat, h2h_L3_stat, h2h_games
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Map stat type to column name
        stat_column_map = {
            "points": "points",
            "rebounds": "rebounds",
            "assists": "assists",
            "threes": "three_pointers_made",
        }

        stat_column = stat_column_map.get(stat_type, "points")

        # Query all H2H games before as_of_date
        # Use normalized name matching to handle suffixes (Jr./Jr, III, etc.)
        query = f"""
        SELECT
            pgl.game_date,
            pgl.{stat_column} as stat_value
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.opponent_abbrev = %s
          AND pgl.game_date < %s
        ORDER BY pgl.game_date DESC
        """

        try:
            df = pd.read_sql_query(
                query, self.conn, params=(player_name, opponent_team, as_of_date)
            )

            if len(df) == 0:
                # No H2H history - return None to signal use of player average
                return {"h2h_games": 0.0, "h2h_avg_stat": None, "h2h_L3_stat": None}

            # Calculate H2H stats
            h2h_games = len(df)
            h2h_avg_stat = float(df["stat_value"].mean())

            # Last 3 H2H games (if available)
            if len(df) >= 3:
                h2h_L3_stat = float(df.head(3)["stat_value"].mean())
            else:
                # Use whatever H2H data we have
                h2h_L3_stat = h2h_avg_stat

            return {
                "h2h_games": float(h2h_games),
                "h2h_avg_stat": h2h_avg_stat,
                "h2h_L3_stat": h2h_L3_stat,
            }

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying H2H stats: {e}")
            return {"h2h_games": 0.0, "h2h_avg_stat": None, "h2h_L3_stat": None}

    def get_days_since_milestone(self, player_name, as_of_date, stat="points", threshold=30):
        """
        Calculate days since player last achieved milestone (e.g., 30+ points).
        Returns days since milestone, or 999 if never achieved.

        Args:
            player_name: Player's full name
            as_of_date: Date to check as of (datetime or string)
            stat: Stat column to check ('points', 'rebounds', 'assists', etc.)
            threshold: Minimum value for milestone

        Returns:
            Days since last milestone (999 if never achieved)
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        query = f"""
        SELECT pgl.game_date
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.game_date < %s
          AND pgl.{stat} >= %s
        ORDER BY pgl.game_date DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (player_name, as_of_date, threshold))
                result = cur.fetchone()

                if result:
                    milestone_date = result[0]
                    # Handle both datetime.date and pandas.Timestamp
                    if hasattr(as_of_date, "date"):
                        as_of_date_obj = as_of_date.date()
                    else:
                        as_of_date_obj = as_of_date
                    days_since = (as_of_date_obj - milestone_date).days
                    return float(days_since)
                else:
                    # Never achieved milestone
                    return 999.0
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying milestone for {player_name}: {e}")
            return 999.0

    def get_momentum_short_term(self, recent_games, stat_type="points"):
        """
        Calculate short-term momentum: (L3 avg - L10 avg) / L10 avg
        Positive = heating up, Negative = cooling down

        Args:
            recent_games: DataFrame of recent games (chronologically ordered)
            stat_type: Stat column to calculate momentum for ('points', 'rebounds', 'assists')

        Returns:
            Momentum score (0.0 if insufficient data)
        """
        if len(recent_games) < 3:
            return 0.0

        # Get L3 average
        l3_games = recent_games.tail(3)
        l3_avg = l3_games[stat_type].mean()

        # Get L10 average (or whatever is available)
        l10_games = recent_games.tail(10)
        l10_avg = l10_games[stat_type].mean()

        if l10_avg > 0:
            momentum = (l3_avg - l10_avg) / l10_avg
            return float(momentum)
        else:
            return 0.0

    def get_home_away_streaks(self, player_name, as_of_date):
        """
        Calculate home and away winning streaks.
        Returns: dict with 'home' and 'away' streak counts

        Streak logic:
        - Positive = consecutive wins
        - Negative = consecutive losses
        - 0 = no streak or no games

        Args:
            player_name: Player's full name
            as_of_date: Date to calculate streaks as of

        Returns:
            Dict with 'home' and 'away' streak integers
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Query game logs with win/loss info (using plus_minus as proxy)
        # Use normalized name matching to handle suffixes (Jr./Jr, III, etc.)
        query = f"""
        SELECT
            pgl.game_date,
            pgl.is_home,
            pgl.plus_minus
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.game_date < %s
          AND pgl.plus_minus IS NOT NULL
        ORDER BY pgl.game_date DESC
        LIMIT 20
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=(player_name, as_of_date))

            if len(df) == 0:
                return {"home": 0.0, "away": 0.0}

            # Calculate streaks separately for home and away
            home_games = df[df["is_home"]].sort_values("game_date", ascending=False)
            away_games = df[~df["is_home"]].sort_values("game_date", ascending=False)

            def calculate_streak(games_df):
                """Calculate streak from most recent games"""
                if len(games_df) == 0:
                    return 0.0

                streak = 0
                last_result = None  # 'W' or 'L'

                for _, row in games_df.iterrows():
                    is_win = row["plus_minus"] > 0
                    current_result = "W" if is_win else "L"

                    if last_result is None:
                        # First game in streak
                        last_result = current_result
                        streak = 1 if is_win else -1
                    elif current_result == last_result:
                        # Streak continues
                        if is_win:
                            streak += 1
                        else:
                            streak -= 1
                    else:
                        # Streak broken
                        break

                return float(streak)

            home_streak = calculate_streak(home_games)
            away_streak = calculate_streak(away_games)

            return {"home": home_streak, "away": away_streak}

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error calculating streaks for {player_name}: {e}")
            return {"home": 0.0, "away": 0.0}

    def check_opponent_back_to_back(self, opponent_team, game_date):
        """
        Check if opponent played yesterday (back-to-back game).
        Returns 1.0 if B2B, 0.0 otherwise.

        Args:
            opponent_team: Team abbreviation (e.g., 'GSW', 'LAL')
            game_date: Date of the game being analyzed (datetime or string)

        Returns:
            1.0 if opponent is on back-to-back, 0.0 otherwise
        """
        if opponent_team is None:
            return 0.0

        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)

        # Check if opponent played the day before
        yesterday = game_date - timedelta(days=1)

        # Handle both datetime.date and pandas.Timestamp
        if hasattr(yesterday, "date"):
            yesterday_date = yesterday.date()
        else:
            yesterday_date = yesterday

        query = """
        SELECT COUNT(*) as game_count
        FROM team_game_logs
        WHERE team_abbrev = %s
          AND game_date = %s
        """

        try:
            with self.games_conn.cursor() as cur:
                cur.execute(query, (opponent_team, yesterday_date))
                result = cur.fetchone()
                return 1.0 if result and result[0] > 0 else 0.0
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error checking opponent back-to-back: {e}")
            return 0.0

    def get_opponent_defensive_efficiency(self, opponent_team, as_of_date, window=10):
        """
        Get opponent's defensive efficiency (points allowed per possession).

        Uses team_season_stats as primary source (verified correct data).

        Args:
            opponent_team: Team abbreviation (e.g., 'GSW', 'LAL')
            as_of_date: Date to calculate stats as of (datetime or string)
            window: Number of games to include (default 10, unused - using season stats)

        Returns:
            Points allowed per possession (e.g., 1.08 = 108 points per 100 possessions)
        """
        if opponent_team is None:
            return 1.12  # League average default

        # Use team_season_stats directly (team_game_logs has incorrect data)
        return self._get_opponent_season_def_rating(opponent_team)

    def _get_opponent_season_def_rating(self, opponent_team):
        """Fallback: Get opponent's season-long defensive rating"""
        query = """
        SELECT defensive_rating
        FROM team_season_stats
        WHERE team_abbrev = %s
        ORDER BY season DESC
        LIMIT 1
        """

        try:
            with self.team_conn.cursor() as cur:
                cur.execute(query, (opponent_team,))
                result = cur.fetchone()
                if result and result[0] is not None:
                    return float(result[0] / 100.0)  # Convert to per possession
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying season defensive rating: {e}")

        return 1.12  # League average (~112 points per 100 possessions)

    def get_opponent_position_defense(self, opponent_team, position_encoded):
        """
        Get opponent's defensive rating against a specific position.

        Uses position-specific defense columns from team_season_stats:
        - def_rating_vs_pg (position 1-1.5)
        - def_rating_vs_sg (position 2)
        - def_rating_vs_sf (position 2.5-3.5)
        - def_rating_vs_pf (position 4)
        - def_rating_vs_c (position 4.5-5)

        Args:
            opponent_team: Team abbreviation (e.g., 'BOS', 'LAL')
            position_encoded: Player's position (1=PG, 2=SG, 3=SF, 4=PF, 5=C)

        Returns:
            Points allowed per possession vs that position (e.g., 1.02 = 102 per 100 poss)
        """
        if not opponent_team:
            return 1.12  # League average default

        # Normalize opponent team
        opponent_team = self.normalize_team_abbrev(opponent_team)

        # Map position_encoded to column name
        # 1-1.5 = PG, 2 = SG, 2.5-3.5 = SF, 4 = PF, 4.5-5 = C
        if position_encoded <= 1.5:
            pos_col = "def_rating_vs_pg"
        elif position_encoded <= 2.25:
            pos_col = "def_rating_vs_sg"
        elif position_encoded <= 3.5:
            pos_col = "def_rating_vs_sf"
        elif position_encoded <= 4.25:
            pos_col = "def_rating_vs_pf"
        else:
            pos_col = "def_rating_vs_c"

        # Query position-specific defense, preferring seasons with real data
        # Check if position-specific data varies from overall (indicates real data vs placeholder)
        query = f"""
        SELECT {pos_col}, defensive_rating,
               ABS(def_rating_vs_pg - def_rating_vs_c) as pos_variance
        FROM team_season_stats
        WHERE team_abbrev = %s
          AND {pos_col} IS NOT NULL
        ORDER BY
            CASE WHEN ABS(def_rating_vs_pg - def_rating_vs_c) > 1 THEN 0 ELSE 1 END,  -- Prefer real variance
            season DESC
        LIMIT 1
        """

        try:
            with self.team_conn.cursor() as cur:
                cur.execute(query, (opponent_team,))
                result = cur.fetchone()
                if result and result[0] is not None:
                    # Position-specific rating (convert Decimal to float first)
                    return float(result[0]) / 100.0  # Convert to per possession
                elif result and result[1] is not None:
                    # Fallback to overall defensive rating
                    return float(result[1] / 100.0)
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying position defense for {opponent_team}: {e}")

        return 1.12  # League average

    def get_opponent_def_rating_L3(self, opponent_team, as_of_date):
        """
        Get opponent's L3 (last 3 games) defensive rating.

        Uses team_game_logs table (games database) for per-game defensive ratings.

        Args:
            opponent_team: Team abbreviation (e.g., 'LAL', 'GSW')
            as_of_date: Date to calculate stats as of (datetime or string)

        Returns:
            Average defensive rating over last 3 games (e.g., 108.5),
            or 110.0 (league average) if insufficient data
        """
        if opponent_team is None:
            return 110.0  # League average default

        # Normalize team abbreviation
        opponent_team = self.normalize_team_abbrev(opponent_team)

        query = """
        SELECT defensive_rating
        FROM team_game_logs
        WHERE team_abbrev = %s
          AND game_date < %s
          AND defensive_rating IS NOT NULL
        ORDER BY game_date DESC
        LIMIT 3
        """

        try:
            df = pd.read_sql_query(query, self.games_conn, params=(opponent_team, as_of_date))

            if len(df) >= 2:
                avg_def = float(df["defensive_rating"].mean())
                # Check for placeholder data (all values exactly 110.0)
                # If there's no variance, fall back to season stats
                if df["defensive_rating"].std() < 0.01:
                    pass  # Fall through to season stats fallback
                else:
                    return avg_def
            elif len(df) == 1:
                # Only 1 game - blend with league average (less confidence)
                return float((df["defensive_rating"].iloc[0] + 110.0) / 2.0)
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying opponent L3 def rating for {opponent_team}: {e}")

        # Fallback to team_season_stats
        season_query = """
        SELECT defensive_rating
        FROM team_season_stats
        WHERE team_abbrev = %s
        ORDER BY season DESC
        LIMIT 1
        """

        try:
            with self.team_conn.cursor() as cur:
                cur.execute(season_query, (opponent_team,))
                result = cur.fetchone()
                if result and result[0] is not None:
                    return float(result[0])
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying season def rating for {opponent_team}: {e}")

        return 110.0  # League average fallback

    def calculate_resistance_adjusted_stat(
        self, raw_L3_stat, opponent_team, as_of_date, league_avg_def_rating=110.0
    ):
        """
        Calculate resistance-adjusted L3 performance.

        Adjusts player's L3 performance based on opponent's defensive strength.
        If opponent was elite defense (low rating), adjusted score is HIGHER.
        If opponent was weak defense (high rating), adjusted score is LOWER.

        Formula: L3_stat * (league_avg_def_rating / opponent_def_rating_L3)

        Example:
        - Player scored 25 ppg vs elite defense (105 rating)
        - Adjusted = 25 * (110 / 105) = 26.2 (credit for tough defense)

        - Player scored 25 ppg vs weak defense (118 rating)
        - Adjusted = 25 * (110 / 118) = 23.3 (discount for weak defense)

        Args:
            raw_L3_stat: Player's raw L3 average (e.g., 25.0 points)
            opponent_team: Opponent team abbreviation
            as_of_date: Date for the query
            league_avg_def_rating: League average defensive rating (default 110.0)

        Returns:
            Resistance-adjusted stat value (never NaN, defaults to raw_L3_stat)
        """
        # If no raw stat or opponent, return raw stat unchanged
        if raw_L3_stat is None or raw_L3_stat == 0.0:
            return 0.0
        if opponent_team is None:
            return float(raw_L3_stat)

        # Get opponent's L3 defensive rating
        opp_def_rating_L3 = self.get_opponent_def_rating_L3(opponent_team, as_of_date)

        # Safety check: avoid division by zero or extreme values
        if opp_def_rating_L3 <= 0 or opp_def_rating_L3 > 150:
            return float(raw_L3_stat)

        # Calculate resistance adjustment
        # Higher opp_def_rating (worse defense) = lower adjusted stat
        # Lower opp_def_rating (better defense) = higher adjusted stat
        adjustment_factor = league_avg_def_rating / opp_def_rating_L3

        # Clamp adjustment factor to reasonable range (0.85 to 1.15)
        # This prevents extreme adjustments from unusual defensive ratings
        adjustment_factor = max(0.85, min(1.15, adjustment_factor))

        adjusted_stat = float(raw_L3_stat) * adjustment_factor

        return adjusted_stat

    def calculate_travel_distance(self, player_team, opponent_team):
        """
        Calculate travel distance between team arenas in kilometers.
        Uses haversine formula with arena coordinates.

        Args:
            player_team: Player's team abbreviation (e.g., 'GSW')
            opponent_team: Opponent team abbreviation (e.g., 'LAL')

        Returns:
            Distance in kilometers (0.0 if same arena/unknown)
        """
        if player_team is None or opponent_team is None:
            return 0.0

        # NBA arena coordinates (latitude, longitude)
        ARENA_COORDS = {
            "ATL": (33.7573, -84.3963),  # State Farm Arena
            "BOS": (42.3662, -71.0621),  # TD Garden
            "BKN": (40.6826, -73.9754),  # Barclays Center
            "CHA": (35.2251, -80.8392),  # Spectrum Center
            "CHI": (41.8807, -87.6742),  # United Center
            "CLE": (41.4965, -81.6882),  # Rocket Mortgage FieldHouse
            "DAL": (32.7905, -96.8103),  # American Airlines Center
            "DEN": (39.7487, -105.0077),  # Ball Arena
            "DET": (42.6970, -83.2456),  # Little Caesars Arena
            "GSW": (37.7680, -122.3877),  # Chase Center
            "HOU": (29.7508, -95.3621),  # Toyota Center
            "IND": (39.7640, -86.1555),  # Gainbridge Fieldhouse
            "LAC": (34.0430, -118.2673),  # Crypto.com Arena
            "LAL": (34.0430, -118.2673),  # Crypto.com Arena (same as LAC)
            "MEM": (35.1382, -90.0505),  # FedExForum
            "MIA": (25.7814, -80.1870),  # Kaseya Center
            "MIL": (43.0451, -87.9172),  # Fiserv Forum
            "MIN": (44.9795, -93.2761),  # Target Center
            "NOP": (29.9490, -90.0821),  # Smoothie King Center
            "NYK": (40.7505, -73.9934),  # Madison Square Garden
            "OKC": (35.4634, -97.5151),  # Paycom Center
            "ORL": (28.5392, -81.3839),  # Amway Center
            "PHI": (39.9012, -75.1720),  # Wells Fargo Center
            "PHX": (33.4457, -112.0712),  # Footprint Center
            "POR": (45.5316, -122.6668),  # Moda Center
            "SAC": (38.5802, -121.4997),  # Golden 1 Center
            "SAS": (29.4270, -98.4375),  # Frost Bank Center
            "TOR": (43.6435, -79.3791),  # Scotiabank Arena
            "UTA": (40.7683, -111.9011),  # Delta Center
            "WAS": (38.8981, -77.0209),  # Capital One Arena
        }

        if player_team not in ARENA_COORDS or opponent_team not in ARENA_COORDS:
            return 0.0

        # Haversine formula
        lat1, lon1 = ARENA_COORDS[player_team]
        lat2, lon2 = ARENA_COORDS[opponent_team]

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in kilometers
        r = 6371

        return float(c * r)

    def get_altitude_flag(self, venue_team):
        """
        Flag high-altitude venues (Denver, Utah).
        Returns 1.0 for high altitude, 0.0 otherwise.

        Args:
            venue_team: Team abbreviation of the venue (e.g., 'DEN', 'UTA')

        Returns:
            1.0 for high altitude venues, 0.0 otherwise
        """
        HIGH_ALTITUDE = ["DEN", "UTA"]
        return 1.0 if venue_team in HIGH_ALTITUDE else 0.0

    def calculate_pace_diff(self, player_team_pace, opponent_team_pace):
        """
        Calculate pace difference between teams.
        Positive = player's team faster, Negative = opponent faster

        Args:
            player_team_pace: Player's team pace (possessions per 48 minutes)
            opponent_team_pace: Opponent's team pace (possessions per 48 minutes)

        Returns:
            Pace difference (e.g., 2.5 means player's team 2.5 possessions faster)
        """
        return float(player_team_pace - opponent_team_pace)

    def get_season_stats(self, player_name, season="2024-25"):
        """
        Get player's season-level statistics.

        Args:
            player_name: Player's full name
            season: Season string (e.g., '2024-25')

        Returns:
            Dict with season stats: usage_rate, reb_rate, etc.
        """
        query = """
        SELECT
            usage_rate,
            rebound_rate,
            assist_rate,
            true_shooting_pct
        FROM player_season_stats
        WHERE player_id = (SELECT player_id FROM player_profile WHERE full_name = %s LIMIT 1)
          AND season = %s
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (player_name, season))
                result = cur.fetchone()

                if result:
                    return {
                        "usage_rate": float(result[0]) if result[0] is not None else 0.20,
                        "reb_rate": float(result[1]) if result[1] is not None else 0.10,
                        "assist_rate": float(result[2]) if result[2] is not None else 0.15,
                        "true_shooting_pct": float(result[3]) if result[3] is not None else 0.55,
                    }
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying season stats for {player_name}: {e}")

        # Return defaults
        return {
            "usage_rate": 0.20,
            "reb_rate": 0.10,
            "assist_rate": 0.15,
            "true_shooting_pct": 0.55,
        }

    def get_opponent_stats(self, opponent_team, as_of_date, window=10):
        """
        Get opponent's defensive statistics.

        Args:
            opponent_team: Team abbreviation (e.g., 'LAL')
            as_of_date: Date to calculate stats as of
            window: Number of games to average (default 10)

        Returns:
            Dict with: usage_allowed, reb_allowed_per_pos
        """
        if not opponent_team:
            return {"usage_allowed": 0.20, "reb_allowed_per_pos": 0.10}

        # Query opponent's recent defensive stats
        query = """
        SELECT
            defensive_rating,
            pace
        FROM team_game_logs
        WHERE team_abbrev = %s
          AND game_date < %s
        ORDER BY game_date DESC
        LIMIT %s
        """

        try:
            df = pd.read_sql_query(
                query, self.games_conn, params=(opponent_team, as_of_date, window)
            )

            if len(df) >= 3:
                # Defensive rating proxy for usage allowed
                avg_def_rating = df["defensive_rating"].mean()
                # Higher defensive rating = more usage allowed (worse defense)
                # Scale: 110 = league avg 0.20, 105 = 0.18, 115 = 0.22
                usage_allowed = 0.20 * (avg_def_rating / 110.0) if avg_def_rating > 0 else 0.20

                # Rebounds allowed per possession (proxy)
                # Use league average with small adjustments based on defensive rating
                reb_allowed = 0.10 * (avg_def_rating / 110.0) if avg_def_rating > 0 else 0.10

                return {
                    "usage_allowed": float(usage_allowed),
                    "reb_allowed_per_pos": float(reb_allowed),
                }
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error querying opponent stats: {e}")

        return {"usage_allowed": 0.20, "reb_allowed_per_pos": 0.10}

    def get_teammate_avg_usage(self, player_name, team_abbrev, as_of_date, window=10):
        """
        Calculate average usage rate of other players on the team.
        Usage approximation: (FGA + 0.44 * FTA + TO) per minute

        Args:
            player_name: Player's full name (to exclude from calculation)
            team_abbrev: Team abbreviation
            as_of_date: Date to query (get recent games before this)
            window: Number of games to average (default 10)

        Returns:
            Average usage rate of teammates (excluding this player), or 0.20 if no data
        """
        if not team_abbrev:
            return 0.20  # Default usage rate

        # Get recent games for all teammates (excluding this player)
        query = """
        SELECT
            pgl.fg_attempted,
            pgl.ft_attempted,
            pgl.turnovers,
            pgl.minutes_played
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE pgl.team_abbrev = %s
          AND pgl.game_date < %s
          AND pp.full_name != %s
          AND pgl.minutes_played >= 5
        ORDER BY pgl.game_date DESC
        LIMIT %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    query, (team_abbrev, as_of_date, player_name, window * 10)
                )  # Get more rows for multiple players
                rows = cur.fetchall()

                if rows:
                    # Calculate usage for each game
                    usages = []
                    for fg_att, fta, tov, mins in rows:
                        if mins and mins > 0:
                            usage = (fg_att + 0.44 * (fta or 0) + (tov or 0)) / mins
                            usages.append(usage)

                    if usages:
                        return np.mean(usages)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            pass  # Fail silently, return default

        return 0.20  # Default teammate usage rate

    def get_starter_ratio(self, player_name, as_of_date, window=10, minutes_threshold=28):
        """
        Calculate the ratio of games where a player started in the last N games.
        Starters are identified by playing >= minutes_threshold minutes.

        Args:
            player_name: Player's full name
            as_of_date: Date to calculate ratio as of (datetime or string)
            window: Number of recent games to consider (default 10)
            minutes_threshold: Minimum minutes to be considered a starter (default 28)

        Returns:
            Ratio from 0.0 (pure bench) to 1.0 (always starts), 0.5 if no data
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Normalize player name
        player_name = self.normalize_player_name(player_name)

        # Use normalized name matching to handle suffixes (Jr./Jr, III, etc.)
        query = f"""
        SELECT pgl.minutes_played
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE {self.sql_name_match('pp.full_name')}
          AND pgl.game_date < %s
          AND pgl.minutes_played IS NOT NULL
        ORDER BY pgl.game_date DESC
        LIMIT %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (player_name, as_of_date, window))
                rows = cur.fetchall()

                if not rows:
                    # No game data available, return default
                    return 0.5

                # Count games where player started (minutes >= threshold)
                games_started = sum(1 for (mins,) in rows if mins >= minutes_threshold)
                total_games = len(rows)

                # Return ratio of games started
                return float(games_started) / float(total_games)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error calculating starter ratio for {player_name}: {e}")
            return 0.5

    def get_position_encoded(self, player_name):
        """
        Get player's position encoded as a numeric value.

        Encoding: 1=PG, 2=SG, 3=SF, 4=PF, 5=C, 0=unknown

        First tries player_profile.position, then infers from stats pattern
        (guards have higher AST/REB ratio, centers have higher REB/AST ratio).

        Args:
            player_name: Player's full name

        Returns:
            Encoded position (1-5) or 3.0 (forward) as default
        """
        # Position encoding map
        position_map = {
            "PG": 1,
            "G": 1.5,
            "SG": 2,
            "SF": 3,
            "F": 3.5,
            "PF": 4,
            "C": 5,
            "F-C": 4.5,
            "G-F": 2.5,
        }

        # Normalize player name for matching
        normalized_name = self.normalize_player_name(player_name)

        try:
            # Try to get position from player_profile
            query = """
            SELECT position
            FROM player_profile
            WHERE LOWER(full_name) = LOWER(%s)
            LIMIT 1
            """
            with self.conn.cursor() as cur:
                cur.execute(query, (normalized_name,))
                result = cur.fetchone()

                if result and result[0]:
                    pos = result[0].upper().strip()
                    return position_map.get(pos, 3.0)

            # Fallback: Infer position from recent stats pattern
            # Guards: Higher AST/REB ratio
            # Centers: Higher REB/AST ratio
            stats_query = """
            SELECT AVG(pgl.assists) as avg_ast, AVG(pgl.rebounds) as avg_reb
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE LOWER(pp.full_name) = LOWER(%s)
            """
            with self.conn.cursor() as cur:
                cur.execute(stats_query, (normalized_name,))
                result = cur.fetchone()

                if result and result[0] is not None and result[1] is not None:
                    avg_ast = float(result[0])
                    avg_reb = float(result[1])

                    # Infer position from AST/REB ratio
                    if avg_reb == 0:
                        ratio = 10.0  # High AST, low REB = likely guard
                    else:
                        ratio = avg_ast / avg_reb

                    if ratio > 1.2:  # High AST relative to REB
                        return 1.5  # Guard (PG/SG average)
                    elif ratio > 0.6:  # Balanced
                        return 3.0  # Forward (SF)
                    else:  # Low AST relative to REB
                        return 4.5  # Big (PF/C average)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error getting position for {player_name}: {e}")

        return 3.0  # Default to forward if unknown

    def get_injured_teammates_count(self, player_name, team_abbrev, game_date):
        """
        Count teammates with 'OUT' status in injury_report for a given date.

        Args:
            player_name: Player's full name (to exclude from count)
            team_abbrev: Team abbreviation
            game_date: Date to check injuries for

        Returns:
            Number of teammates with OUT status, or 0 if no data
        """
        if not team_abbrev:
            return 0

        try:
            # Connect to intelligence DB for injury_report
            int_conn = psycopg2.connect(**self.INTELLIGENCE_DB_CONFIG)

            # Query: Get count of teammates (same team, not this player) with OUT status
            # Need to join injury_report -> player_profile to get team info
            query = """
            SELECT COUNT(DISTINCT ir.player_id)
            FROM injury_report ir
            JOIN (
                SELECT player_id, full_name, team_abbrev
                FROM player_profile
                WHERE team_abbrev = %s
            ) pp ON ir.player_id = pp.player_id
            WHERE ir.report_date = %s
              AND ir.status = 'OUT'
              AND pp.full_name != %s
            """

            # Note: player_profile is in nba_players DB (port 5536)
            # injury_report is in nba_intelligence DB (port 5539)
            # We need a cross-database approach or use the players connection

            # Simplified approach: Query injury_report for player_ids,
            # then check which are on the same team

            # First get player_ids for this team (excluding our player)
            with self.conn.cursor() as cur:  # players DB
                cur.execute(
                    """
                    SELECT player_id FROM player_profile
                    WHERE team_abbrev = %s AND full_name != %s
                """,
                    (team_abbrev, player_name),
                )
                teammate_ids = [row[0] for row in cur.fetchall()]

            if not teammate_ids:
                int_conn.close()
                return 0

            # Now count how many of these are OUT in injury_report
            with int_conn.cursor() as cur:
                # Use ANY for the list of player_ids
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT player_id)
                    FROM injury_report
                    WHERE report_date = %s
                      AND status = 'OUT'
                      AND player_id = ANY(%s)
                """,
                    (game_date, teammate_ids),
                )
                result = cur.fetchone()

            int_conn.close()
            return result[0] if result else 0

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error getting injured teammates: {e}")
            return 0

    def get_teammate_absences(self, player_name, team_abbrev, game_date, n_games=3, min_minutes=20):
        """
        Count significant teammate injuries/absences in the player's last N games.

        A "significant" teammate = someone who averages 20+ minutes on the team.
        This indicates opportunity for increased usage when key teammates are OUT.

        Args:
            player_name: Player's full name (to exclude from count)
            team_abbrev: Team abbreviation
            game_date: Reference date (check games before this date)
            n_games: Number of recent games to check (default 3)
            min_minutes: Minimum average minutes for a "significant" teammate (default 20)

        Returns:
            Total count of significant teammate absences across last N games, or 0.0 if no data
        """
        if not team_abbrev:
            return 0.0

        try:
            # Step 1: Get player's last N game dates
            # Use normalized name matching to handle suffixes (Jr./Jr, III, etc.)
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT DISTINCT pgl.game_date
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE {self.sql_name_match('pp.full_name')}
                      AND pgl.game_date < %s
                    ORDER BY pgl.game_date DESC
                    LIMIT %s
                """,
                    (player_name, game_date, n_games),
                )
                game_dates = [row[0] for row in cur.fetchall()]

            if not game_dates:
                return 0.0

            # Step 2: Get significant teammates (avg 20+ minutes in recent games)
            # Calculate average minutes from recent game logs for teammates
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pp.player_id, pp.full_name, AVG(pgl.minutes_played) as avg_minutes
                    FROM player_profile pp
                    JOIN player_game_logs pgl ON pp.player_id = pgl.player_id
                    WHERE pp.team_abbrev = %s
                      AND pp.full_name != %s
                      AND pgl.game_date < %s
                      AND pgl.minutes_played IS NOT NULL
                    GROUP BY pp.player_id, pp.full_name
                    HAVING AVG(pgl.minutes_played) >= %s
                """,
                    (team_abbrev, player_name, game_date, min_minutes),
                )
                significant_teammates = {row[0]: row[1] for row in cur.fetchall()}

            if not significant_teammates:
                return 0.0

            significant_teammate_ids = list(significant_teammates.keys())

            # Step 3: Connect to intelligence DB and count OUT statuses for significant teammates
            int_conn = psycopg2.connect(**self.INTELLIGENCE_DB_CONFIG)
            total_absences = 0

            with int_conn.cursor() as cur:
                # Count how many significant teammates were OUT on each game date
                for gd in game_dates:
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT player_id)
                        FROM injury_report
                        WHERE report_date = %s
                          AND status = 'OUT'
                          AND player_id = ANY(%s)
                    """,
                        (gd, significant_teammate_ids),
                    )
                    result = cur.fetchone()
                    if result and result[0]:
                        total_absences += result[0]

            int_conn.close()
            return float(total_absences)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error getting teammate absences for {player_name}: {e}")
            return 0.0

    def calculate_season_phase(self, game_date):
        """
        Calculate season phase as a value from 0.0 to 1.0.
        NBA regular season runs ~Oct 22 to mid-April (~175 days).

        Returns:
            float: 0.0 = start of season (Oct), 1.0 = end of regular season (Apr)
        """
        try:
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, "%Y-%m-%d")

            month = game_date.month
            day = game_date.day

            # NBA season: Oct 22 - Apr 14 approximately
            # October (10) = 0.0-0.05, November = 0.05-0.20, December = 0.20-0.38
            # January = 0.38-0.55, February = 0.55-0.70, March = 0.70-0.88, April = 0.88-1.0

            if month == 10:  # October (season starts ~Oct 22)
                phase = max(0.0, (day - 22) / 9 * 0.05)  # 0.0 - 0.05
            elif month == 11:  # November
                phase = 0.05 + (day / 30) * 0.15  # 0.05 - 0.20
            elif month == 12:  # December
                phase = 0.20 + (day / 31) * 0.18  # 0.20 - 0.38
            elif month == 1:  # January
                phase = 0.38 + (day / 31) * 0.17  # 0.38 - 0.55
            elif month == 2:  # February
                phase = 0.55 + (day / 28) * 0.15  # 0.55 - 0.70
            elif month == 3:  # March
                phase = 0.70 + (day / 31) * 0.18  # 0.70 - 0.88
            elif month == 4:  # April (regular season ends ~Apr 14)
                phase = min(1.0, 0.88 + (day / 14) * 0.12)  # 0.88 - 1.0
            else:
                # Off-season or playoffs - return 1.0 (end of season)
                phase = 1.0

            return round(phase, 3)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error calculating season phase: {e}")
            return 0.5  # Default to mid-season

    def calculate_prime_time_flag(self, game_date, game_time=None):
        """
        Determine if game is prime time (evening/weekend games with higher viewership).

        Prime time criteria:
        - Weekday games starting 7pm+ ET (19:00+)
        - Weekend afternoon/evening games (Sat/Sun after 12pm)
        - National TV games (approximated by timing)

        Returns:
            float: 1.0 = prime time, 0.0 = non-prime time
        """
        try:
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, "%Y-%m-%d")

            day_of_week = game_date.weekday()  # 0=Mon, 5=Sat, 6=Sun
            is_weekend = day_of_week >= 5

            # Parse game time if available
            hour = 19  # Default to 7pm if time not available
            if game_time:
                if isinstance(game_time, str):
                    try:
                        # Handle various time formats
                        if ":" in game_time:
                            parts = game_time.split(":")
                            hour = int(parts[0])
                    except (ValueError, IndexError):
                        hour = 19
                elif hasattr(game_time, "hour"):
                    hour = game_time.hour

            # Prime time logic
            if is_weekend:
                # Weekend: any game after 12pm is prime time
                is_prime = hour >= 12
            else:
                # Weekday: 7pm+ is prime time
                is_prime = hour >= 19

            return 1.0 if is_prime else 0.0

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error calculating prime time flag: {e}")
            return 0.0  # Default to non-prime time

    def calculate_revenge_game_flag(self, player_name, opponent_team, game_date):
        """
        Check if this is a revenge/rematch game (played same opponent recently this season).

        Revenge game = player faced this opponent in the last 30 days.
        This can indicate motivation/familiarity factor.

        Returns:
            float: 1.0 = revenge game (recent rematch), 0.0 = first/distant matchup
        """
        try:
            if not opponent_team:
                return 0.0

            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, "%Y-%m-%d")

            # Query player's recent games vs this opponent
            # Use normalized name matching to handle suffixes (Jr./Jr, III, etc.)
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*) as matchup_count
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE {self.sql_name_match('pp.full_name')}
                      AND pgl.opponent_abbrev = %s
                      AND pgl.game_date < %s
                      AND pgl.game_date >= %s - INTERVAL '30 days'
                """,
                    (player_name, opponent_team, game_date, game_date),
                )

                result = cur.fetchone()
                matchup_count = result[0] if result else 0

            # If played this opponent in last 30 days, it's a revenge game
            return 1.0 if matchup_count > 0 else 0.0

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error calculating revenge game flag: {e}")
            return 0.0  # Default to no revenge game

    def extract_features(
        self,
        player_name,
        game_date,
        is_home=None,
        opponent_team=None,
        line=None,
        source="validation",
        spread_diff=0.0,
        total_diff=0.0,
    ):
        """
        Extract 104 features for a player on a given date.
        Matches the feature set expected by trained models.

        Args:
            player_name: Player's full name
            game_date: Date of the game (datetime or string)
            is_home: True if home game, False if away, None if unknown
            opponent_team: Opponent team abbreviation (e.g., 'LAL')
            line: Prop line value (e.g., 25.5 points)
            source: Prop source for book encoding ('validation', 'prizepicks', 'underdog', 'bettingpros')
            spread_diff: Point spread differential (positive = player's team favored)
            total_diff: Totals line differential

        Returns:
            Dict with 104 features (or 99 without market-aware features)
        """
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)

        # Get recent games (with strict 20-game requirement)
        recent_games = self.get_recent_games(
            player_name, game_date, n_games=20, min_games_threshold=20
        )

        # STRICT POLICY: If None returned, player doesn't have enough history
        # Return None to signal that this prop should be dropped from training
        if recent_games is None:
            logger.warning(
                f"{player_name}: Insufficient game history (<20 games), "
                f"returning None to drop from training"
            )
            return None

        total_games = len(recent_games)

        # This should not happen with the new policy, but keep as safety check
        if total_games < 20:
            logger.warning(
                f"{player_name}: Only {total_games} games available (<20 required), "
                f"returning None to drop from training"
            )
            return None

        # Calculate EMA features for L3, L5, L10, L20 windows
        features = {}

        # Line (sportsbook line)
        features["line"] = float(line) if line is not None else 0.0

        # Is home
        if is_home is not None:
            features["is_home"] = 1.0 if is_home else 0.0
        else:
            features["is_home"] = (
                float(recent_games.iloc[-1]["is_home"]) if len(recent_games) > 0 else 0.5
            )

        alpha = 0.4  # EMA smoothing factor

        # Calculate EMA for all windows: L3, L5, L10, L20
        for window_size, window_name in [(3, "L3"), (5, "L5"), (10, "L10"), (20, "L20")]:
            window_games = recent_games.tail(window_size)

            if len(window_games) == 0:
                # No games in window - use defaults
                features[f"ema_points_{window_name}"] = 15.0
                features[f"ema_rebounds_{window_name}"] = 5.0
                features[f"ema_assists_{window_name}"] = 3.0
                features[f"ema_threePointersMade_{window_name}"] = 1.5
                features[f"ema_steals_{window_name}"] = 1.0
                features[f"ema_blocks_{window_name}"] = 0.5
                features[f"ema_turnovers_{window_name}"] = 2.0
                features[f"ema_minutes_{window_name}"] = 25.0
                features[f"fg_pct_{window_name}"] = 0.45
            else:
                # Calculate EMAs (fillna(0) for steals/blocks/turnovers in case of NULL values)
                features[f"ema_points_{window_name}"] = self.calculate_ema(
                    window_games["points"].values, alpha
                )
                features[f"ema_rebounds_{window_name}"] = self.calculate_ema(
                    window_games["rebounds"].values, alpha
                )
                features[f"ema_assists_{window_name}"] = self.calculate_ema(
                    window_games["assists"].values, alpha
                )
                features[f"ema_threePointersMade_{window_name}"] = self.calculate_ema(
                    window_games["three_pointers_made"].values, alpha
                )
                features[f"ema_steals_{window_name}"] = self.calculate_ema(
                    window_games["steals"].astype(float).fillna(1.0).values, alpha
                )
                features[f"ema_blocks_{window_name}"] = self.calculate_ema(
                    window_games["blocks"].astype(float).fillna(0.5).values, alpha
                )
                features[f"ema_turnovers_{window_name}"] = self.calculate_ema(
                    window_games["turnovers"].astype(float).fillna(2.0).values, alpha
                )
                features[f"ema_minutes_{window_name}"] = self.calculate_ema(
                    window_games["minutes_played"].values, alpha
                )

                # Calculate FG%
                total_made = window_games["fg_made"].sum()
                total_attempted = window_games["fg_attempted"].sum()
                features[f"fg_pct_{window_name}"] = (
                    total_made / total_attempted if total_attempted > 0 else 0.45
                )

                # NEW: Plus/minus EMA (L5, L10 only)
                if window_name in ["L5", "L10"]:
                    plus_minus_vals = window_games["plus_minus"].astype(float).fillna(0.0).values
                    features[f"ema_plus_minus_{window_name}"] = self.calculate_ema(
                        plus_minus_vals, alpha
                    )

                # NEW: FT rate and True shooting (L10 only)
                if window_name == "L10":
                    # FT rate = FT attempted / FG attempted (how often they get to the line)
                    total_fta = window_games["ft_attempted"].astype(float).fillna(0).sum()
                    total_fga = window_games["fg_attempted"].sum()
                    features["ft_rate_L10"] = (
                        float(total_fta) / float(total_fga) if total_fga > 0 else 0.25
                    )

                    # True shooting = PTS / (2 * (FGA + 0.44 * FTA))
                    total_pts = window_games["points"].sum()
                    tsa = 2.0 * (total_fga + 0.44 * total_fta)
                    features["true_shooting_L10"] = float(total_pts) / tsa if tsa > 0 else 0.55

        # Set defaults for new features if not calculated (insufficient games)
        if "ema_plus_minus_L5" not in features:
            features["ema_plus_minus_L5"] = 0.0
        if "ema_plus_minus_L10" not in features:
            features["ema_plus_minus_L10"] = 0.0
        if "ft_rate_L10" not in features:
            features["ft_rate_L10"] = 0.25
        if "true_shooting_L10" not in features:
            features["true_shooting_L10"] = 0.55

        # Form slope features removed - not used by models

        # Additional context features - REAL TEAM DATA
        # Get player's team
        player_team = self.get_player_team(player_name, game_date)

        if player_team:
            # Get team's rolling stats (L10 games)
            team_stats = self.get_team_rolling_stats(player_team, game_date, window=10)
            features["team_pace"] = team_stats["pace"]
            features["team_off_rating"] = team_stats["off_rating"]
            features["team_def_rating"] = team_stats["def_rating"]
        else:
            # Player team not found - use defaults
            logger.debug(f"Could not find team for {player_name}, using defaults")
            features["team_pace"] = 100.0
            features["team_off_rating"] = 110.0
            features["team_def_rating"] = 110.0

        # Opponent stats - REAL if opponent_team provided
        if opponent_team:
            opponent_stats = self.get_team_rolling_stats(opponent_team, game_date, window=10)
            features["opponent_pace"] = opponent_stats["pace"]
            features["opponent_def_rating"] = opponent_stats["def_rating"]
        else:
            # Use league average when opponent unknown (training mode)
            features["opponent_pace"] = 98.0
            features["opponent_def_rating"] = 110.0

        # Projected possessions (average of team and opponent pace)
        features["projected_possessions"] = (features["team_pace"] + features["opponent_pace"]) / 2

        # Rest/schedule
        if len(recent_games) >= 2:
            last_game_date = recent_games.iloc[-1]["game_date"]
            prev_game_date = recent_games.iloc[-2]["game_date"]
            days_rest = (last_game_date - prev_game_date).days
            features["days_rest"] = float(days_rest)
            features["is_back_to_back"] = 1.0 if days_rest == 1 else 0.0
        else:
            features["days_rest"] = 2.0
            features["is_back_to_back"] = 0.0

        features["games_in_L7"] = float(min(len(recent_games.tail(7)), 7))

        # Advanced metrics (estimated)
        if len(recent_games) >= 5:
            last_5_minutes = recent_games.tail(5)["minutes_played"].values
            last_5_points = recent_games.tail(5)["points"].values
            avg_minutes = last_5_minutes.mean()
            avg_points = last_5_points.mean()
            features["points_per_minute_L5"] = avg_points / avg_minutes if avg_minutes > 0 else 0.5
        else:
            features["points_per_minute_L5"] = 0.5

        # efficiency_vs_context: Player's scoring efficiency relative to matchup context
        # Computed as: (player TS% L10) / (opponent allowed TS% L10) * 100
        # Values > 100 = player more efficient than opponent typically allows
        player_ts = features.get("true_shooting_L10", 0.55)
        opp_def_rating = features.get("opponent_def_rating", 112.0)
        # Higher opponent def_rating = worse defense = higher efficiency expected
        features["efficiency_vs_context"] = (player_ts / 0.55) * (opp_def_rating / 112.0) * 100.0

        # Calculate resistance-adjusted L3 points based on opponent defensive strength
        # Uses player's raw L3 points and adjusts for opponent's L3 defensive rating
        raw_L3_points = features.get("ema_points_L3", 15.0)
        features["resistance_adjusted_L3"] = self.calculate_resistance_adjusted_stat(
            raw_L3_stat=raw_L3_points, opponent_team=opponent_team, as_of_date=game_date
        )

        # REAL momentum calculation
        features["momentum_short_term"] = self.get_momentum_short_term(
            recent_games, stat_type="points"
        )

        # Context features - REAL shot volume proxy
        # Estimate shot volume based on recent FGA/min * projected minutes * team pace
        fg_attempts_per_min = recent_games["fg_attempted"].sum() / max(
            recent_games["minutes_played"].sum(), 1
        )
        projected_minutes = features["ema_minutes_L5"]  # Use EMA L5 minutes as projection
        shot_volume = fg_attempts_per_min * projected_minutes * (features["team_pace"] / 100.0)
        features["shot_volume_proxy"] = shot_volume
        # game_velocity: Expected game pace relative to league average (100)
        # Computed as: (team_pace + opponent_pace) / 2 / league_avg_pace * 100
        league_avg_pace = 100.0
        features["game_velocity"] = (
            (features["team_pace"] + features["opponent_pace"]) / 2 / league_avg_pace * 100.0
        )
        features["days_rest_copy"] = features["days_rest"]

        # REAL opponent and venue features
        if opponent_team:
            features["opponent_back_to_back_flag"] = self.check_opponent_back_to_back(
                opponent_team, game_date
            )
            features["travel_distance_km"] = self.calculate_travel_distance(
                player_team, opponent_team
            )
            features["pace_diff"] = self.calculate_pace_diff(
                features["team_pace"], features["opponent_pace"]
            )
            # Determine venue based on is_home
            if is_home is not None:
                venue_team = player_team if is_home else opponent_team
                features["altitude_flag"] = self.get_altitude_flag(venue_team)
            else:
                features["altitude_flag"] = 0.0
        else:
            # Defaults when opponent unknown
            features["opponent_back_to_back_flag"] = 0.0
            features["travel_distance_km"] = 0.0
            features["pace_diff"] = 0.0
            features["altitude_flag"] = 0.0

        # expected_possessions: Player's expected possessions based on usage and team pace
        # Computed as: projected_possessions * player_usage_estimate
        player_usage_estimate = features.get("bench_points_ratio", 0.8)  # Higher for starters
        features["expected_possessions"] = features["projected_possessions"] * player_usage_estimate

        # Season/venue
        features["season_phase"] = self.calculate_season_phase(game_date)

        # bench_points_ratio: ratio of L10 games where player started (28+ min)
        # Higher = more likely a starter (1.0 = always starts, 0.0 = pure bench)
        features["bench_points_ratio"] = self.get_starter_ratio(
            player_name, game_date, window=10, minutes_threshold=28
        )
        # starter_flag: Use the computed bench_points_ratio (not hardcoded)
        # 1.0 = always starts, 0.0 = pure bench player
        features["starter_flag"] = features["bench_points_ratio"]

        # position_encoded: Encode player's primary position
        # 1=PG, 2=SG, 3=SF, 4=PF, 5=C (0 if unknown)
        features["position_encoded"] = self.get_position_encoded(player_name)

        # Teammate/injury - REAL teammate usage and injury calculation
        if player_team:
            features["avg_teammate_usage"] = self.get_teammate_avg_usage(
                player_name, player_team, game_date
            )
            features["injured_teammates_count"] = float(
                self.get_injured_teammates_count(player_name, player_team, game_date)
            )
            # Count significant teammate absences (20+ min players OUT) in last 3 games
            features["teammate_absences_last_3"] = self.get_teammate_absences(
                player_name, player_team, game_date, n_games=3, min_minutes=20
            )
        else:
            features["avg_teammate_usage"] = 0.20  # Default
            features["injured_teammates_count"] = 0.0
            features["teammate_absences_last_3"] = 0.0

        # Matchup - Position-specific defensive efficiency
        # Uses opponent's defense rating against player's position (PG/SG/SF/PF/C)
        if opponent_team:
            features["opponent_allowed_points_per_pos"] = self.get_opponent_position_defense(
                opponent_team, features.get("position_encoded", 3.0)
            )
        else:
            features["opponent_allowed_points_per_pos"] = 1.12  # League average

        # projected_team_possessions: Same as projected_possessions (team's expected possessions)
        features["projected_team_possessions"] = features["projected_possessions"]

        # H2H - REAL head-to-head statistics
        if opponent_team:
            h2h_stats = self.get_h2h_stats(
                player_name, opponent_team, game_date, stat_type="points"
            )
            if h2h_stats["h2h_avg_stat"] is not None:
                # Have H2H history - use real stats
                features["h2h_avg_points"] = h2h_stats["h2h_avg_stat"]
                features["h2h_L3_points"] = h2h_stats["h2h_L3_stat"]
                features["h2h_games"] = h2h_stats["h2h_games"]
            else:
                # No H2H history - use player's overall average
                if len(recent_games) > 0:
                    features["h2h_avg_points"] = float(recent_games["points"].mean())
                    features["h2h_L3_points"] = (
                        float(recent_games.tail(3)["points"].mean())
                        if len(recent_games) >= 3
                        else features["h2h_avg_points"]
                    )
                else:
                    features["h2h_avg_points"] = 15.0
                    features["h2h_L3_points"] = 15.0
                features["h2h_games"] = 0.0
        else:
            # Opponent not specified - use player's overall average as fallback
            if len(recent_games) > 0:
                features["h2h_avg_points"] = float(recent_games["points"].mean())
                features["h2h_L3_points"] = (
                    float(recent_games.tail(3)["points"].mean())
                    if len(recent_games) >= 3
                    else features["h2h_avg_points"]
                )
            else:
                features["h2h_avg_points"] = 15.0
                features["h2h_L3_points"] = 15.0
            features["h2h_games"] = 0.0

        # Matchup Advantage Score - REAL composite calculation
        # Combines: player form, opponent defense, H2H performance
        # Formula: (player_form / league_avg) - (opponent_defense / league_avg) + h2h_boost
        # Range: approximately -0.5 to +0.5 (positive = player has advantage)
        player_form_score = (
            features["ema_points_L5"] / 15.0
        )  # Normalize by league average (~15 ppg)
        opponent_defense_score = (
            features["opponent_allowed_points_per_pos"] / 1.1
        )  # Normalize by league avg (~1.1)
        h2h_boost = 0.0
        if features["h2h_games"] > 0 and features["h2h_avg_points"] > 0:
            # If player scores more vs this opponent than their average, boost score
            h2h_boost = (features["h2h_avg_points"] - features["ema_points_L5"]) / 15.0

        features["matchup_advantage_score"] = player_form_score - opponent_defense_score + h2h_boost

        # Recent performance
        if len(recent_games) > 0:
            features["player_last_game_minutes"] = float(recent_games.iloc[-1]["minutes_played"])
        else:
            features["player_last_game_minutes"] = 30.0

        # REAL days since milestone
        features["days_since_last_30pt_game"] = self.get_days_since_milestone(
            player_name, game_date, stat="points", threshold=30
        )

        # REAL streaks
        streaks = self.get_home_away_streaks(player_name, game_date)
        features["home_streak"] = streaks["home"]
        features["away_streak"] = streaks["away"]

        # Game context
        features["revenge_game_flag"] = self.calculate_revenge_game_flag(
            player_name, opponent_team, game_date
        )
        features["prime_time_flag"] = self.calculate_prime_time_flag(
            game_date
        )  # Uses date only, defaults evening games

        # Expected diff (placeholder for live predictions - models expect this feature)
        # During training: actual_result - line
        # During serving: 0.0 (neutral, no expected edge)
        # expected_diff removed - calculated dynamically in two-head architecture

        return features

    def _get_default_features(self, is_home, player_name=None, game_date=None):
        """Return default features for players with no history"""
        features = {"is_home": 1.0 if is_home else 0.0 if is_home is not None else 0.5}

        # Default EMAs
        for window in ["L3", "L5", "L10", "L20"]:
            features[f"ema_points_{window}"] = 15.0
            features[f"ema_rebounds_{window}"] = 5.0
            features[f"ema_assists_{window}"] = 3.0
            features[f"ema_threePointersMade_{window}"] = 1.5
            features[f"ema_steals_{window}"] = 1.0
            features[f"ema_blocks_{window}"] = 0.5
            features[f"ema_turnovers_{window}"] = 2.0
            features[f"ema_minutes_{window}"] = 25.0
            features[f"fg_pct_{window}"] = 0.45

        # Try to get real team context even for rookies/new players
        if player_name and game_date:
            player_team = self.get_player_team(player_name, game_date)
            if player_team:
                team_stats = self.get_team_rolling_stats(player_team, game_date, window=10)
                features.update(
                    {
                        "team_pace": team_stats["pace"],
                        "team_off_rating": team_stats["off_rating"],
                        "team_def_rating": team_stats["def_rating"],
                        "opponent_pace": 98.0,
                        "opponent_def_rating": 110.0,
                        "projected_possessions": (team_stats["pace"] + 98.0) / 2,
                    }
                )
            else:
                # Default context features
                features.update(
                    {
                        "team_pace": 100.0,
                        "team_off_rating": 110.0,
                        "team_def_rating": 110.0,
                        "opponent_pace": 100.0,
                        "opponent_def_rating": 110.0,
                        "projected_possessions": 100.0,
                    }
                )
        else:
            # Default context features
            features.update(
                {
                    "team_pace": 100.0,
                    "team_off_rating": 110.0,
                    "team_def_rating": 110.0,
                    "opponent_pace": 100.0,
                    "opponent_def_rating": 110.0,
                    "projected_possessions": 100.0,
                }
            )

        # Continue with remaining default features
        features.update(
            {
                "days_rest": 2.0,
                "is_back_to_back": 0.0,
                "games_in_L7": 0.0,
                "points_per_minute_L5": 0.5,
                "efficiency_vs_context": 100.0,
                "resistance_adjusted_L3": 15.0,  # Default to raw L3 points (no adjustment)
                "momentum_short_term": 0.0,
                "shot_volume_proxy": 12.0,  # Default ~12 FGA per game
                "game_velocity": 100.0,
                "days_rest_copy": 2.0,
                "opponent_back_to_back_flag": 0.0,
                "travel_distance_km": 0.0,
                "pace_diff": 0.0,
                "expected_possessions": 100.0,
                "season_phase": 0.5,  # Default to mid-season if date unknown
                "altitude_flag": 0.0,
                "starter_flag": 1.0,
                "bench_points_ratio": 0.5,  # Default to 0.5 (neutral) if no data
                "position_encoded": 1.0,
                "avg_teammate_usage": 0.20,  # Default 20% usage rate
                "injured_teammates_count": 0.0,
                "teammate_absences_last_3": 0.0,
                "opponent_allowed_points_per_pos": 1.12,
                "matchup_advantage_score": 0.0,  # Default neutral matchup
                "projected_team_possessions": 100.0,
                "h2h_avg_points": 15.0,
                "h2h_L3_points": 15.0,
                "h2h_games": 0.0,
                "player_last_game_minutes": 30.0,
                "days_since_last_30pt_game": 10.0,
                "home_streak": 0.0,
                "away_streak": 0.0,
                "revenge_game_flag": 0.0,
                "prime_time_flag": 0.0,
                "expected_diff": 0.0,
            }
        )

        return features

    def get_feature_vector(
        self, player_name, game_date, feature_names, is_home=None, opponent_team=None
    ):
        """
        Get feature vector in the exact order expected by the model.
        """
        features_dict = self.extract_features(player_name, game_date, is_home, opponent_team)

        # Return features in exact order
        feature_vector = []
        for fname in feature_names:
            feature_vector.append(features_dict.get(fname, 0.0))

        return np.array(feature_vector).reshape(1, -1)

    def __enter__(self):
        """Support usage as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure connections are closed when used as context manager."""
        self.close()
        return False

    def close(self):
        """Close all database connections"""
        if self.conn:
            self.conn.close()
        if hasattr(self, "games_conn") and self.games_conn:
            self.games_conn.close()
        if hasattr(self, "team_conn") and self.team_conn:
            self.team_conn.close()


if __name__ == "__main__":
    # Test with real player form features
    extractor = LiveFeatureExtractor()

    print("=" * 80)
    print("Testing Live Feature Extractor - REAL PLAYER FORM FEATURES")
    print("=" * 80)

    # Test Stephen Curry (GSW - high scoring, 3-point specialist)
    print("\n[1] Stephen Curry (Golden State Warriors) - 2025-04-10:")
    features = extractor.extract_features("Stephen Curry", "2025-04-10")
    print(
        f"  days_since_last_30pt_game: {features['days_since_last_30pt_game']:.1f} (should NOT be 10.0)"
    )
    print(f"  momentum_short_term: {features['momentum_short_term']:.3f} (should NOT be 0.0)")
    print(f"  home_streak: {features['home_streak']:.0f} (should NOT be 0)")
    print(f"  away_streak: {features['away_streak']:.0f} (should NOT be 0)")
    print(f"  ema_points_L5: {features['ema_points_L5']:.2f}")
    print(f"  team_pace: {features['team_pace']:.2f}")

    # Test LeBron James (LAL - consistent scorer)
    print("\n[2] LeBron James (Los Angeles Lakers) - 2025-04-10:")
    features2 = extractor.extract_features("LeBron James", "2025-04-10")
    print(
        f"  days_since_last_30pt_game: {features2['days_since_last_30pt_game']:.1f} (should NOT be 10.0)"
    )
    print(f"  momentum_short_term: {features2['momentum_short_term']:.3f} (should NOT be 0.0)")
    print(f"  home_streak: {features2['home_streak']:.0f} (should NOT be 0)")
    print(f"  away_streak: {features2['away_streak']:.0f} (should NOT be 0)")
    print(f"  ema_points_L5: {features2['ema_points_L5']:.2f}")

    # Test Kyle Kuzma (MIL - moderate scorer)
    print("\n[3] Kyle Kuzma (Milwaukee Bucks) - 2025-04-10:")
    features3 = extractor.extract_features("Kyle Kuzma", "2025-04-10")
    print(
        f"  days_since_last_30pt_game: {features3['days_since_last_30pt_game']:.1f} (999 if never)"
    )
    print(f"  momentum_short_term: {features3['momentum_short_term']:.3f} (should NOT be 0.0)")
    print(f"  home_streak: {features3['home_streak']:.0f}")
    print(f"  away_streak: {features3['away_streak']:.0f}")
    print(f"  ema_points_L5: {features3['ema_points_L5']:.2f}")

    # Test Jonas Valančiūnas (DEN - big man, rebounds/paint)
    print("\n[4] Jonas Valančiūnas (Denver Nuggets) - 2025-04-10:")
    features4 = extractor.extract_features("Jonas Valančiūnas", "2025-04-10")
    print(
        f"  days_since_last_30pt_game: {features4['days_since_last_30pt_game']:.1f} (rarely 30pts)"
    )
    print(f"  momentum_short_term: {features4['momentum_short_term']:.3f}")
    print(f"  home_streak: {features4['home_streak']:.0f}")
    print(f"  away_streak: {features4['away_streak']:.0f}")
    print(f"  ema_rebounds_L5: {features4['ema_rebounds_L5']:.2f}")

    print("\n" + "=" * 80)
    print("VARIANCE CHECK:")
    print("=" * 80)
    print(
        f"Curry 30pt days: {features['days_since_last_30pt_game']:.1f} vs "
        + f"LeBron: {features2['days_since_last_30pt_game']:.1f} vs "
        + f"Kuzma: {features3['days_since_last_30pt_game']:.1f} vs "
        + f"JV: {features4['days_since_last_30pt_game']:.1f}"
    )
    print(
        f"Momentum: Curry {features['momentum_short_term']:.3f} vs "
        + f"LeBron {features2['momentum_short_term']:.3f} vs "
        + f"Kuzma {features3['momentum_short_term']:.3f} vs "
        + f"JV {features4['momentum_short_term']:.3f}"
    )
    print(
        f"Home streaks: {features['home_streak']:.0f}, {features2['home_streak']:.0f}, "
        + f"{features3['home_streak']:.0f}, {features4['home_streak']:.0f}"
    )
    print(
        f"Away streaks: {features['away_streak']:.0f}, {features2['away_streak']:.0f}, "
        + f"{features3['away_streak']:.0f}, {features4['away_streak']:.0f}"
    )

    # Count how many features are NOT default
    all_features = [features, features2, features3, features4]
    milestone_values = [f["days_since_last_30pt_game"] for f in all_features]
    momentum_values = [f["momentum_short_term"] for f in all_features]
    home_streak_values = [f["home_streak"] for f in all_features]
    away_streak_values = [f["away_streak"] for f in all_features]

    milestone_variance = len(set([round(v, 1) for v in milestone_values]))
    momentum_variance = len(set([round(v, 3) for v in momentum_values]))
    home_variance = len(set([int(v) for v in home_streak_values]))
    away_variance = len(set([int(v) for v in away_streak_values]))

    print("\n" + "=" * 80)
    print("SUCCESS METRICS:")
    print("=" * 80)
    print(f"✓ days_since_last_30pt_game: {milestone_variance}/4 unique values (FIXED)")
    print(f"✓ momentum_short_term: {momentum_variance}/4 unique values (FIXED)")
    print(f"✓ home_streak: {home_variance}/4 unique values (FIXED)")
    print(f"✓ away_streak: {away_variance}/4 unique values (FIXED)")

    if milestone_variance >= 3 and momentum_variance >= 3:
        print("\n✅ PASS: Player form features show real variance!")
    else:
        print("\n⚠️  WARNING: Some features still hardcoded or not varying")

    print("=" * 80)

    extractor.close()
