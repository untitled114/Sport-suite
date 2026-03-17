"""
Direct Line Feature Extractor (V4)
====================================
Extracts 19 features from direct sportsbook data and line snapshots.

These features capture information only available when we fetch lines
directly from sportsbooks (not through BettingPros aggregation), including
line movement velocity, BettingPros discrepancy, cross-platform agreement,
and freshness signals.

NOT wired into XL/V3 models -- these are stubs for V4 training dataset
building. They require the nba_line_snapshots table and direct fetcher
data in nba_props_xl (fetch_source='direct').

Database Requirements:
    - nba_props_xl: with fetch_source, bp_reported_line, bp_discrepancy columns
    - nba_line_snapshots: append-only line movement history
    Both on nba_intelligence (port 5539).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)

EST = ZoneInfo("America/New_York")

# Encoding for direct sportsbook sources (distinct from BP book IDs).
# These IDs are used for categorical features (softest/hardest book).
BOOK_ENCODING = {
    "draftkings_direct": 1,
    "fanduel_direct": 2,
    "espnbet_direct": 3,
    "betmgm_direct": 4,
    "caesars_direct": 5,
    "betrivers_direct": 6,
    "fanatics_direct": 7,
    "bet365_direct": 8,
    "hardrock_direct": 9,
    "underdog_direct": 10,
    "prizepicks": 11,
}

# Map book_name values in nba_props_xl to their encoding keys.
# Direct fetchers write book_name with a "_direct" suffix; DFS platforms don't.
BOOK_NAME_TO_ENCODING_KEY = {
    "draftkings_direct": "draftkings_direct",
    "fanduel_direct": "fanduel_direct",
    "espnbet_direct": "espnbet_direct",
    "betmgm_direct": "betmgm_direct",
    "caesars_direct": "caesars_direct",
    "betrivers_direct": "betrivers_direct",
    "fanatics_direct": "fanatics_direct",
    "bet365_direct": "bet365_direct",
    "hardrock_direct": "hardrock_direct",
    "underdog_direct": "underdog_direct",
    "prizepicks": "prizepicks",
    # Fallbacks for BP-style names appearing with fetch_source='direct'
    "DraftKings": "draftkings_direct",
    "FanDuel": "fanduel_direct",
    "ESPNBet": "espnbet_direct",
    "BetMGM": "betmgm_direct",
    "Caesars": "caesars_direct",
    "BetRivers": "betrivers_direct",
    "Fanatics": "fanatics_direct",
    "Bet365": "bet365_direct",
    "HardRock": "hardrock_direct",
    "Underdog": "underdog_direct",
    "PrizePicks": "prizepicks",
}

# Corresponding BP book names (without _direct suffix) for coverage ratio.
DIRECT_TO_BP_BOOK = {
    "draftkings_direct": "DraftKings",
    "fanduel_direct": "FanDuel",
    "espnbet_direct": "ESPNBet",
    "betmgm_direct": "BetMGM",
    "caesars_direct": "Caesars",
    "betrivers_direct": "BetRivers",
    "fanatics_direct": "Fanatics",
    "bet365_direct": "Bet365",
    "hardrock_direct": "HardRock",
}


class DirectLineFeatureExtractor(BaseFeatureExtractor):
    """V4 features from direct sportsbook line data.

    These features are NOT wired into XL/V3 models -- they're stubs for
    V4 training dataset building. They require the nba_line_snapshots
    table and direct fetcher data in nba_props_xl.

    Features (19 total):
    - num_direct_sources: how many books we have direct lines from
    - direct_consensus: consensus from direct lines only (no BP)
    - direct_spread: line spread from direct sources
    - bp_line_latency_avg: avg timing gap between direct and BP update (seconds)
    - bp_discrepancy_avg: avg line difference direct vs BP
    - bp_discrepancy_max: max line difference direct vs BP
    - line_movement_velocity: rate of change from snapshots table (points/hour)
    - opening_vs_current: opening line drift (current - earliest snapshot)
    - odds_vig_avg: average vig/juice from direct odds data
    - cross_platform_agreement: sportsbook consensus vs DFS consensus score
    - snapshot_count: number of line snapshots for this prop
    - hours_tracked: time span of snapshots (hours)
    - direct_softest_book: encoded ID of softest direct book
    - direct_hardest_book: encoded ID of hardest direct book
    - direct_line_std: standard deviation across direct lines
    - direct_odds_spread: max odds - min odds across books
    - bp_coverage_ratio: fraction of direct books also in BP
    - line_convergence: are lines converging or diverging over time
    - freshness_score: how recently were direct lines updated (minutes ago)
    """

    FEATURE_NAMES = (
        "num_direct_sources",
        "direct_consensus",
        "direct_spread",
        "bp_line_latency_avg",
        "bp_discrepancy_avg",
        "bp_discrepancy_max",
        "line_movement_velocity",
        "opening_vs_current",
        "odds_vig_avg",
        "cross_platform_agreement",
        "snapshot_count",
        "hours_tracked",
        "direct_softest_book",
        "direct_hardest_book",
        "direct_line_std",
        "direct_odds_spread",
        "bp_coverage_ratio",
        "line_convergence",
        "freshness_score",
    )

    def __init__(self, conn: Any):
        """
        Initialize direct line feature extractor.

        Args:
            conn: Database connection to nba_intelligence database (port 5539)
        """
        super().__init__(conn, name="DirectLineFeatures")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for all 19 direct line features."""
        return {
            "num_direct_sources": 0.0,
            "direct_consensus": 0.0,
            "direct_spread": 0.0,
            "bp_line_latency_avg": 0.0,
            "bp_discrepancy_avg": 0.0,
            "bp_discrepancy_max": 0.0,
            "line_movement_velocity": 0.0,
            "opening_vs_current": 0.0,
            "odds_vig_avg": 0.0,
            "cross_platform_agreement": 0.0,
            "snapshot_count": 0.0,
            "hours_tracked": 0.0,
            "direct_softest_book": 0.0,
            "direct_hardest_book": 0.0,
            "direct_line_std": 0.0,
            "direct_odds_spread": 0.0,
            "bp_coverage_ratio": 0.0,
            "line_convergence": 0.0,
            "freshness_score": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract 19 direct line features for a player/game/stat combination.

        Args:
            player_name: Player's full name
            game_date: Game date (datetime or string)
            stat_type: Stat type (POINTS, REBOUNDS, etc.)
            **kwargs: Additional context (unused for now)

        Returns:
            Dict mapping feature names to float values
        """
        game_date_str = self._normalize_date(game_date)

        # Fetch direct props and snapshots in two queries
        direct_df = self._fetch_direct_props(player_name, game_date_str, stat_type)
        snapshot_df = self._fetch_snapshots(player_name, game_date_str, stat_type)
        bp_df = self._fetch_bp_props(player_name, game_date_str, stat_type)

        if direct_df is None and snapshot_df is None:
            return self.get_defaults()

        features = self.get_defaults()

        # Compute feature groups
        if direct_df is not None and len(direct_df) > 0:
            self._compute_direct_line_features(direct_df, features)
            self._compute_bp_discrepancy_features(direct_df, features)
            self._compute_odds_features(direct_df, features)
            self._compute_cross_platform_features(direct_df, features)

            if bp_df is not None and len(bp_df) > 0:
                self._compute_bp_coverage(direct_df, bp_df, features)

        if snapshot_df is not None and len(snapshot_df) > 0:
            self._compute_snapshot_features(snapshot_df, features)
            self._compute_line_movement_features(snapshot_df, features)
            self._compute_freshness(snapshot_df, features)

        return self.validate_features(features)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_direct_props(
        self, player_name: str, game_date_str: str, stat_type: str
    ) -> Optional[pd.DataFrame]:
        """Fetch props from nba_props_xl where fetch_source='direct'."""
        query = """
            SELECT book_name, over_line, over_odds, under_line, under_odds,
                   bp_reported_line, bp_discrepancy, fetch_timestamp
            FROM nba_props_xl
            WHERE player_name = %s
              AND game_date = %s
              AND stat_type = %s
              AND fetch_source = 'direct'
              AND is_active = true
              AND over_line IS NOT NULL
        """
        return self._safe_query(query, (player_name, game_date_str, stat_type.upper()))

    def _fetch_bp_props(
        self, player_name: str, game_date_str: str, stat_type: str
    ) -> Optional[pd.DataFrame]:
        """Fetch BettingPros-sourced props for coverage comparison."""
        query = """
            SELECT DISTINCT book_name
            FROM nba_props_xl
            WHERE player_name = %s
              AND game_date = %s
              AND stat_type = %s
              AND fetch_source = 'bettingpros'
              AND is_active = true
        """
        return self._safe_query(query, (player_name, game_date_str, stat_type.upper()))

    def _fetch_snapshots(
        self, player_name: str, game_date_str: str, stat_type: str
    ) -> Optional[pd.DataFrame]:
        """Fetch line snapshots from nba_line_snapshots."""
        query = """
            SELECT book_name, over_line, over_odds, under_odds,
                   fetch_source, snapshot_at
            FROM nba_line_snapshots
            WHERE player_name = %s
              AND game_date = %s
              AND stat_type = %s
            ORDER BY snapshot_at ASC
        """
        return self._safe_query(query, (player_name, game_date_str, stat_type.upper()))

    # ------------------------------------------------------------------
    # Feature computation: direct lines
    # ------------------------------------------------------------------

    def _compute_direct_line_features(self, df: pd.DataFrame, features: Dict[str, float]) -> None:
        """Compute line spread, consensus, softest/hardest from direct sources."""
        lines = df["over_line"].astype(float)

        features["num_direct_sources"] = float(len(df))
        features["direct_consensus"] = float(lines.mean())
        features["direct_spread"] = float(lines.max() - lines.min())
        features["direct_line_std"] = float(lines.std()) if len(lines) > 1 else 0.0

        # Softest book = lowest over_line (easiest to go over)
        softest_idx = lines.idxmin()
        hardest_idx = lines.idxmax()
        softest_book = df.loc[softest_idx, "book_name"]
        hardest_book = df.loc[hardest_idx, "book_name"]

        features["direct_softest_book"] = float(self._encode_direct_book(softest_book))
        features["direct_hardest_book"] = float(self._encode_direct_book(hardest_book))

    def _compute_bp_discrepancy_features(
        self, df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Compute BettingPros discrepancy features from direct vs BP lines."""
        if "bp_discrepancy" not in df.columns:
            return

        discrepancies = df["bp_discrepancy"].dropna().astype(float)
        if len(discrepancies) == 0:
            return

        features["bp_discrepancy_avg"] = float(discrepancies.mean())
        features["bp_discrepancy_max"] = float(discrepancies.abs().max())

        # Latency: estimate from fetch_timestamp differences between direct
        # and BP rows. For now, use discrepancy magnitude as a proxy -- true
        # latency requires matching timestamps across fetch_source values,
        # which we approximate by looking at bp_reported_line staleness.
        if "fetch_timestamp" in df.columns and "bp_reported_line" in df.columns:
            has_bp = df["bp_reported_line"].notna()
            if has_bp.any():
                # Rows where we have both direct and BP data -- use the
                # discrepancy magnitude scaled to a rough seconds estimate.
                # A 0.5 point discrepancy ~ 300 seconds of latency (heuristic).
                avg_abs_disc = float(discrepancies.abs().mean())
                features["bp_line_latency_avg"] = avg_abs_disc * 600.0
            else:
                features["bp_line_latency_avg"] = 0.0

    def _compute_odds_features(self, df: pd.DataFrame, features: Dict[str, float]) -> None:
        """Compute vig and odds spread from direct odds data."""
        over_odds = df["over_odds"].dropna().astype(float)
        under_odds = df["under_odds"].dropna().astype(float)

        if len(over_odds) == 0 or len(under_odds) == 0:
            return

        # Compute vig for each book: implied_over + implied_under - 1
        vigs: List[float] = []
        for _, row in df.iterrows():
            o_odds = row.get("over_odds")
            u_odds = row.get("under_odds")
            if pd.notna(o_odds) and pd.notna(u_odds):
                imp_over = self._american_to_implied(float(o_odds))
                imp_under = self._american_to_implied(float(u_odds))
                if imp_over > 0 and imp_under > 0:
                    vigs.append(imp_over + imp_under - 1.0)

        if vigs:
            features["odds_vig_avg"] = float(np.mean(vigs))

        # Odds spread: range of over_odds across books
        features["direct_odds_spread"] = float(over_odds.max() - over_odds.min())

    def _compute_cross_platform_features(
        self, df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Score sportsbook vs DFS consensus agreement.

        Sportsbooks: draftkings, fanduel, betmgm, caesars, betrivers, espnbet, etc.
        DFS: underdog_direct, prizepicks

        A score of 1.0 means perfect agreement; 0.0 means no overlap or
        insufficient data.
        """
        dfs_books = {"underdog_direct", "prizepicks", "Underdog", "PrizePicks"}
        sportsbook_rows = df[~df["book_name"].isin(dfs_books)]
        dfs_rows = df[df["book_name"].isin(dfs_books)]

        if len(sportsbook_rows) == 0 or len(dfs_rows) == 0:
            features["cross_platform_agreement"] = 0.0
            return

        sb_consensus = float(sportsbook_rows["over_line"].astype(float).mean())
        dfs_consensus = float(dfs_rows["over_line"].astype(float).mean())

        # Agreement = 1 - normalized distance.  A 2-point gap = 0; perfect = 1.
        gap = abs(sb_consensus - dfs_consensus)
        agreement = max(0.0, 1.0 - gap / 2.0)
        features["cross_platform_agreement"] = agreement

    def _compute_bp_coverage(
        self,
        direct_df: pd.DataFrame,
        bp_df: pd.DataFrame,
        features: Dict[str, float],
    ) -> None:
        """Fraction of direct books that also appear in BettingPros data."""
        direct_books = set(direct_df["book_name"].unique())
        bp_books = set(bp_df["book_name"].unique())

        # Map direct book names to their BP equivalents
        mapped_direct: set = set()
        for book in direct_books:
            enc_key = BOOK_NAME_TO_ENCODING_KEY.get(book, book)
            bp_name = DIRECT_TO_BP_BOOK.get(enc_key, "")
            if bp_name:
                mapped_direct.add(bp_name)

        if len(mapped_direct) == 0:
            features["bp_coverage_ratio"] = 0.0
            return

        overlap = mapped_direct & bp_books
        features["bp_coverage_ratio"] = float(len(overlap)) / float(len(mapped_direct))

    # ------------------------------------------------------------------
    # Feature computation: snapshots (line movement)
    # ------------------------------------------------------------------

    def _compute_snapshot_features(self, df: pd.DataFrame, features: Dict[str, float]) -> None:
        """Compute snapshot count and hours tracked."""
        features["snapshot_count"] = float(len(df))

        if "snapshot_at" in df.columns and len(df) >= 2:
            timestamps = pd.to_datetime(df["snapshot_at"])
            span = timestamps.max() - timestamps.min()
            features["hours_tracked"] = max(span.total_seconds() / 3600.0, 0.0)

    def _compute_line_movement_features(self, df: pd.DataFrame, features: Dict[str, float]) -> None:
        """Compute velocity, opening drift, and convergence from snapshots."""
        if "snapshot_at" not in df.columns or "over_line" not in df.columns:
            return

        df = df.copy()
        df["over_line"] = df["over_line"].astype(float)
        df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
        df = df.sort_values("snapshot_at")

        # ------ Line movement velocity (points per hour) ------
        # Use the consensus (mean across books) at each snapshot time.
        # Group by snapshot_at (rounded to minute) to get consensus per tick.
        df["snap_minute"] = df["snapshot_at"].dt.floor("min")
        consensus_by_time = df.groupby("snap_minute")["over_line"].mean().reset_index()
        consensus_by_time.columns = ["snap_minute", "consensus"]

        if len(consensus_by_time) >= 2:
            first = consensus_by_time.iloc[0]
            last = consensus_by_time.iloc[-1]
            time_delta_hours = (last["snap_minute"] - first["snap_minute"]).total_seconds() / 3600.0

            if time_delta_hours > 0:
                line_change = float(last["consensus"] - first["consensus"])
                features["line_movement_velocity"] = line_change / time_delta_hours
            else:
                features["line_movement_velocity"] = 0.0

            # ------ Opening vs current ------
            features["opening_vs_current"] = float(last["consensus"] - first["consensus"])

            # ------ Line convergence ------
            # Split snapshots into first half and second half.
            # If std decreases, lines are converging (positive value).
            # If std increases, lines are diverging (negative value).
            midpoint = len(df) // 2
            if midpoint > 1 and len(df) - midpoint > 1:
                first_half_std = df.iloc[:midpoint]["over_line"].std()
                second_half_std = df.iloc[midpoint:]["over_line"].std()
                # Positive = converging, negative = diverging
                features["line_convergence"] = float(first_half_std - second_half_std)

    def _compute_freshness(self, df: pd.DataFrame, features: Dict[str, float]) -> None:
        """How many minutes ago was the most recent snapshot taken.

        A lower value means fresher data. Uses EST reference time.
        """
        if "snapshot_at" not in df.columns or len(df) == 0:
            return

        timestamps = pd.to_datetime(df["snapshot_at"])
        most_recent = timestamps.max()

        # If timestamp is naive, assume EST
        now = datetime.now(EST)
        if most_recent.tzinfo is None:
            most_recent = most_recent.replace(tzinfo=EST)

        delta = now - most_recent
        minutes_ago = max(delta.total_seconds() / 60.0, 0.0)

        # Cap at 1440 (24 hours) to avoid extreme values from old data
        features["freshness_score"] = min(minutes_ago, 1440.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_direct_book(self, book_name: str) -> int:
        """Encode a direct book name to its numeric ID.

        Args:
            book_name: Raw book name from nba_props_xl

        Returns:
            Integer encoding (0 if unknown)
        """
        if not book_name:
            return 0
        enc_key = BOOK_NAME_TO_ENCODING_KEY.get(book_name, "")
        return BOOK_ENCODING.get(enc_key, 0)

    @staticmethod
    def _american_to_implied(odds: float) -> float:
        """Convert American odds to implied probability.

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0.0 to 1.0), or 0.0 if invalid
        """
        if odds == 0:
            return 0.0
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    def extract_from_dataframes(
        self,
        direct_df: Optional[pd.DataFrame],
        snapshot_df: Optional[pd.DataFrame],
        bp_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Extract features from pre-queried DataFrames.

        Useful when data is already available from the pipeline (avoids
        redundant database queries).

        Args:
            direct_df: DataFrame from nba_props_xl (fetch_source='direct')
                       Expected columns: book_name, over_line, over_odds,
                       under_line, under_odds, bp_reported_line,
                       bp_discrepancy, fetch_timestamp
            snapshot_df: DataFrame from nba_line_snapshots
                         Expected columns: book_name, over_line, over_odds,
                         under_odds, fetch_source, snapshot_at
            bp_df: Optional DataFrame of BettingPros book_name values
                   Expected columns: book_name

        Returns:
            Dict with 19 direct line features
        """
        if (direct_df is None or len(direct_df) == 0) and (
            snapshot_df is None or len(snapshot_df) == 0
        ):
            return self.get_defaults()

        features = self.get_defaults()

        if direct_df is not None and len(direct_df) > 0:
            self._compute_direct_line_features(direct_df, features)
            self._compute_bp_discrepancy_features(direct_df, features)
            self._compute_odds_features(direct_df, features)
            self._compute_cross_platform_features(direct_df, features)

            if bp_df is not None and len(bp_df) > 0:
                self._compute_bp_coverage(direct_df, bp_df, features)

        if snapshot_df is not None and len(snapshot_df) > 0:
            self._compute_snapshot_features(snapshot_df, features)
            self._compute_line_movement_features(snapshot_df, features)
            self._compute_freshness(snapshot_df, features)

        return self.validate_features(features)
