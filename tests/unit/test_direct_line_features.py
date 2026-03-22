"""
Unit tests for nba.features.extractors.direct_line_features

Tests cover:
- BOOK_ENCODING, BOOK_NAME_TO_ENCODING_KEY, DIRECT_TO_BP_BOOK constants
- DirectLineFeatureExtractor.get_defaults()
- DirectLineFeatureExtractor._encode_direct_book()
- DirectLineFeatureExtractor._american_to_implied()
- DirectLineFeatureExtractor.extract() with mocked DB
- DirectLineFeatureExtractor.extract_from_dataframes() with real DataFrames
- _compute_direct_line_features
- _compute_bp_discrepancy_features
- _compute_odds_features
- _compute_cross_platform_features
- _compute_bp_coverage
- _compute_snapshot_features
- _compute_line_movement_features
- _compute_freshness
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nba.features.extractors.direct_line_features import (
    BOOK_ENCODING,
    BOOK_NAME_TO_ENCODING_KEY,
    DIRECT_TO_BP_BOOK,
    DirectLineFeatureExtractor,
)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────


class TestConstants:
    def test_book_encoding_has_entries(self):
        assert len(BOOK_ENCODING) >= 10
        assert BOOK_ENCODING["draftkings_direct"] == 1
        assert BOOK_ENCODING["prizepicks"] == 11

    def test_book_name_to_encoding_key_has_fallbacks(self):
        assert BOOK_NAME_TO_ENCODING_KEY["DraftKings"] == "draftkings_direct"
        assert BOOK_NAME_TO_ENCODING_KEY["PrizePicks"] == "prizepicks"

    def test_direct_to_bp_book_mapping(self):
        assert DIRECT_TO_BP_BOOK["draftkings_direct"] == "DraftKings"
        assert DIRECT_TO_BP_BOOK["fanduel_direct"] == "FanDuel"


# ─────────────────────────────────────────────────────────────────
# get_defaults
# ─────────────────────────────────────────────────────────────────


class TestGetDefaults:
    def test_returns_19_features(self):
        defaults = DirectLineFeatureExtractor.get_defaults()
        assert len(defaults) == 19

    def test_all_values_are_floats(self):
        defaults = DirectLineFeatureExtractor.get_defaults()
        for key, val in defaults.items():
            assert isinstance(val, float), f"{key} should be float"

    def test_all_default_to_zero(self):
        defaults = DirectLineFeatureExtractor.get_defaults()
        for key, val in defaults.items():
            assert val == 0.0, f"{key} should default to 0.0"

    def test_feature_names_match_defaults(self):
        defaults = DirectLineFeatureExtractor.get_defaults()
        for name in DirectLineFeatureExtractor.FEATURE_NAMES:
            assert name in defaults, f"FEATURE_NAMES entry '{name}' missing from get_defaults()"


# ─────────────────────────────────────────────────────────────────
# _encode_direct_book
# ─────────────────────────────────────────────────────────────────


class TestEncodeDirectBook:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_known_direct_book(self):
        assert self.extractor._encode_direct_book("draftkings_direct") == 1

    def test_known_bp_style_book(self):
        assert self.extractor._encode_direct_book("DraftKings") == 1

    def test_unknown_book_returns_zero(self):
        assert self.extractor._encode_direct_book("random_book") == 0

    def test_empty_string_returns_zero(self):
        assert self.extractor._encode_direct_book("") == 0

    def test_none_returns_zero(self):
        assert self.extractor._encode_direct_book(None) == 0

    def test_prizepicks(self):
        assert self.extractor._encode_direct_book("prizepicks") == 11


# ─────────────────────────────────────────────────────────────────
# _american_to_implied
# ─────────────────────────────────────────────────────────────────


class TestAmericanToImplied:
    def test_minus_110(self):
        # -110 → 110/(110+100) = 0.5238
        assert DirectLineFeatureExtractor._american_to_implied(-110) == pytest.approx(
            0.5238, abs=0.001
        )

    def test_plus_150(self):
        # +150 → 100/(150+100) = 0.4
        assert DirectLineFeatureExtractor._american_to_implied(150) == pytest.approx(0.4, abs=0.001)

    def test_zero_returns_zero(self):
        assert DirectLineFeatureExtractor._american_to_implied(0) == 0.0

    def test_minus_200(self):
        # -200 → 200/300 = 0.6667
        assert DirectLineFeatureExtractor._american_to_implied(-200) == pytest.approx(
            0.6667, abs=0.001
        )

    def test_plus_100(self):
        # +100 → 100/200 = 0.5
        assert DirectLineFeatureExtractor._american_to_implied(100) == pytest.approx(0.5, abs=0.001)

    def test_large_favorite(self):
        # -500 → 500/600 = 0.833
        result = DirectLineFeatureExtractor._american_to_implied(-500)
        assert 0.8 < result < 0.9

    def test_large_underdog(self):
        # +500 → 100/600 = 0.167
        result = DirectLineFeatureExtractor._american_to_implied(500)
        assert 0.1 < result < 0.2


# ─────────────────────────────────────────────────────────────────
# extract (with mocked DB)
# ─────────────────────────────────────────────────────────────────


class TestExtract:
    def test_all_none_returns_defaults(self):
        """When no direct, snapshot, or BP data → defaults."""
        extractor = DirectLineFeatureExtractor(conn=MagicMock())
        with (
            patch.object(extractor, "_fetch_direct_props", return_value=None),
            patch.object(extractor, "_fetch_snapshots", return_value=None),
            patch.object(extractor, "_fetch_bp_props", return_value=None),
        ):
            result = extractor.extract("LeBron James", "2026-03-16", "POINTS")
        defaults = DirectLineFeatureExtractor.get_defaults()
        assert result == defaults

    def test_with_direct_data_computes_features(self):
        extractor = DirectLineFeatureExtractor(conn=MagicMock())
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct"],
                "over_line": [25.5, 26.0],
                "over_odds": [-110.0, -115.0],
                "under_line": [25.5, 26.0],
                "under_odds": [-110.0, -105.0],
                "bp_reported_line": [25.5, 26.0],
                "bp_discrepancy": [0.0, 0.5],
                "fetch_timestamp": [datetime(2026, 3, 16, 12), datetime(2026, 3, 16, 12)],
            }
        )
        with (
            patch.object(extractor, "_fetch_direct_props", return_value=direct_df),
            patch.object(extractor, "_fetch_snapshots", return_value=None),
            patch.object(extractor, "_fetch_bp_props", return_value=None),
        ):
            result = extractor.extract("LeBron James", "2026-03-16", "POINTS")

        assert result["num_direct_sources"] == 2.0
        assert result["direct_consensus"] == pytest.approx(25.75, abs=0.01)
        assert result["direct_spread"] == pytest.approx(0.5, abs=0.01)

    def test_with_snapshot_data(self):
        extractor = DirectLineFeatureExtractor(conn=MagicMock())
        ts1 = datetime(2026, 3, 16, 10, 0)
        ts2 = datetime(2026, 3, 16, 14, 0)
        snapshot_df = pd.DataFrame(
            {
                "book_name": ["dk_direct", "dk_direct"],
                "over_line": [25.0, 26.0],
                "over_odds": [-110, -110],
                "under_odds": [-110, -110],
                "fetch_source": ["direct", "direct"],
                "snapshot_at": [ts1, ts2],
            }
        )
        with (
            patch.object(extractor, "_fetch_direct_props", return_value=None),
            patch.object(extractor, "_fetch_snapshots", return_value=snapshot_df),
            patch.object(extractor, "_fetch_bp_props", return_value=None),
        ):
            result = extractor.extract("LeBron James", "2026-03-16", "POINTS")

        assert result["snapshot_count"] == 2.0
        assert result["hours_tracked"] == pytest.approx(4.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────
# extract_from_dataframes
# ─────────────────────────────────────────────────────────────────


class TestExtractFromDataframes:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_all_none_returns_defaults(self):
        result = self.extractor.extract_from_dataframes(None, None, None)
        assert result == DirectLineFeatureExtractor.get_defaults()

    def test_empty_dfs_return_defaults(self):
        result = self.extractor.extract_from_dataframes(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert result == DirectLineFeatureExtractor.get_defaults()

    def test_with_direct_df_only(self):
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct", "betmgm_direct"],
                "over_line": [24.5, 25.0, 25.5],
                "over_odds": [-110.0, -115.0, -108.0],
                "under_line": [24.5, 25.0, 25.5],
                "under_odds": [-110.0, -105.0, -112.0],
                "bp_reported_line": [None, None, None],
                "bp_discrepancy": [None, None, None],
                "fetch_timestamp": [datetime(2026, 3, 16)] * 3,
            }
        )
        result = self.extractor.extract_from_dataframes(direct_df, None, None)
        assert result["num_direct_sources"] == 3.0
        assert result["direct_consensus"] == pytest.approx(25.0, abs=0.01)
        assert result["direct_spread"] == 1.0
        assert result["direct_softest_book"] == 1.0  # draftkings
        assert result["direct_hardest_book"] == 4.0  # betmgm

    def test_with_bp_coverage(self):
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct"],
                "over_line": [25.0, 25.5],
                "over_odds": [-110.0, -110.0],
                "under_line": [25.0, 25.5],
                "under_odds": [-110.0, -110.0],
                "bp_reported_line": [None, None],
                "bp_discrepancy": [None, None],
                "fetch_timestamp": [datetime(2026, 3, 16)] * 2,
            }
        )
        bp_df = pd.DataFrame(
            {
                "book_name": ["DraftKings", "BetMGM"],
            }
        )
        result = self.extractor.extract_from_dataframes(direct_df, None, bp_df)
        # DraftKings is in both → 1/2 coverage
        assert result["bp_coverage_ratio"] == pytest.approx(0.5, abs=0.01)


# ─────────────────────────────────────────────────────────────────
# _compute_direct_line_features
# ─────────────────────────────────────────────────────────────────


class TestComputeDirectLineFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_single_book(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct"],
                "over_line": [25.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_direct_line_features(df, features)
        assert features["num_direct_sources"] == 1.0
        assert features["direct_consensus"] == 25.5
        assert features["direct_spread"] == 0.0
        assert features["direct_line_std"] == 0.0

    def test_multiple_books(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct", "espnbet_direct"],
                "over_line": [24.0, 25.0, 26.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_direct_line_features(df, features)
        assert features["num_direct_sources"] == 3.0
        assert features["direct_spread"] == 2.0
        assert features["direct_line_std"] > 0


# ─────────────────────────────────────────────────────────────────
# _compute_bp_discrepancy_features
# ─────────────────────────────────────────────────────────────────


class TestComputeBPDiscrepancyFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_no_discrepancy_column(self):
        df = pd.DataFrame({"book_name": ["dk"], "over_line": [25.0]})
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_discrepancy_features(df, features)
        # Should not modify discrepancy features
        assert features["bp_discrepancy_avg"] == 0.0

    def test_all_null_discrepancies(self):
        df = pd.DataFrame(
            {
                "book_name": ["dk"],
                "over_line": [25.0],
                "bp_discrepancy": [None],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_discrepancy_features(df, features)
        assert features["bp_discrepancy_avg"] == 0.0

    def test_computes_avg_and_max(self):
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd"],
                "over_line": [25.0, 25.5],
                "bp_discrepancy": [0.5, -1.0],
                "fetch_timestamp": [datetime(2026, 3, 16), datetime(2026, 3, 16)],
                "bp_reported_line": [24.5, 26.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_discrepancy_features(df, features)
        assert features["bp_discrepancy_avg"] == pytest.approx(-0.25, abs=0.01)
        assert features["bp_discrepancy_max"] == pytest.approx(1.0, abs=0.01)
        # bp_line_latency_avg should be set (heuristic = abs_disc * 600)
        assert features["bp_line_latency_avg"] > 0


# ─────────────────────────────────────────────────────────────────
# _compute_odds_features
# ─────────────────────────────────────────────────────────────────


class TestComputeOddsFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_no_odds_data(self):
        df = pd.DataFrame(
            {
                "book_name": ["dk"],
                "over_line": [25.0],
                "over_odds": [None],
                "under_odds": [None],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        assert features["odds_vig_avg"] == 0.0

    def test_standard_juice(self):
        """Standard -110/-110 should have ~4.5% vig."""
        df = pd.DataFrame(
            {
                "book_name": ["dk"],
                "over_line": [25.0],
                "over_odds": [-110.0],
                "under_odds": [-110.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        # -110 → implied = 110/210 ≈ 0.5238. Vig = 2*0.5238 - 1 ≈ 0.0476
        assert features["odds_vig_avg"] == pytest.approx(0.0476, abs=0.005)

    def test_odds_spread(self):
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd"],
                "over_line": [25.0, 25.0],
                "over_odds": [-120.0, -105.0],
                "under_odds": [-100.0, -115.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        assert features["direct_odds_spread"] == pytest.approx(15.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────
# _compute_cross_platform_features
# ─────────────────────────────────────────────────────────────────


class TestComputeCrossPlatformFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_no_dfs_books_zero(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct"],
                "over_line": [25.0, 25.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        assert features["cross_platform_agreement"] == 0.0

    def test_no_sportsbooks_zero(self):
        df = pd.DataFrame(
            {
                "book_name": ["underdog_direct", "prizepicks"],
                "over_line": [25.0, 25.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        assert features["cross_platform_agreement"] == 0.0

    def test_perfect_agreement(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "underdog_direct"],
                "over_line": [25.0, 25.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        assert features["cross_platform_agreement"] == pytest.approx(1.0, abs=0.01)

    def test_moderate_disagreement(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "underdog_direct"],
                "over_line": [25.0, 26.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        # Gap of 1.0 → agreement = 1 - 1/2 = 0.5
        assert features["cross_platform_agreement"] == pytest.approx(0.5, abs=0.01)

    def test_large_disagreement_capped_at_zero(self):
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "underdog_direct"],
                "over_line": [25.0, 28.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        assert features["cross_platform_agreement"] == 0.0


# ─────────────────────────────────────────────────────────────────
# _compute_snapshot_features
# ─────────────────────────────────────────────────────────────────


class TestComputeSnapshotFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_count_and_hours_tracked(self):
        ts1 = datetime(2026, 3, 16, 10, 0)
        ts2 = datetime(2026, 3, 16, 12, 0)
        ts3 = datetime(2026, 3, 16, 14, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts1, ts2, ts3],
                "over_line": [25.0, 25.5, 25.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_snapshot_features(df, features)
        assert features["snapshot_count"] == 3.0
        assert features["hours_tracked"] == pytest.approx(4.0, abs=0.01)

    def test_single_snapshot_no_hours(self):
        df = pd.DataFrame(
            {
                "snapshot_at": [datetime(2026, 3, 16, 10, 0)],
                "over_line": [25.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_snapshot_features(df, features)
        assert features["snapshot_count"] == 1.0
        assert features["hours_tracked"] == 0.0  # Can't compute span with 1 row


# ─────────────────────────────────────────────────────────────────
# _compute_line_movement_features
# ─────────────────────────────────────────────────────────────────


class TestComputeLineMovementFeatures:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_upward_movement(self):
        ts1 = datetime(2026, 3, 16, 10, 0)
        ts2 = datetime(2026, 3, 16, 12, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts1, ts2],
                "over_line": [25.0, 27.0],
                "book_name": ["dk", "dk"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        # 2 points / 2 hours = 1.0 pts/hr
        assert features["line_movement_velocity"] == pytest.approx(1.0, abs=0.01)
        assert features["opening_vs_current"] == pytest.approx(2.0, abs=0.01)

    def test_downward_movement(self):
        ts1 = datetime(2026, 3, 16, 10, 0)
        ts2 = datetime(2026, 3, 16, 14, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts1, ts2],
                "over_line": [28.0, 26.0],
                "book_name": ["dk", "dk"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_movement_velocity"] < 0
        assert features["opening_vs_current"] == pytest.approx(-2.0, abs=0.01)

    def test_convergence_with_enough_snapshots(self):
        """Test that convergence is computed when midpoint > 1."""
        ts_base = datetime(2026, 3, 16, 8, 0)
        # First half: wider spread (24, 27) → std ≈ 2.12
        # Second half: converged (25.5, 25.5) → std = 0
        df = pd.DataFrame(
            {
                "snapshot_at": [
                    ts_base,
                    ts_base + timedelta(hours=1),
                    ts_base + timedelta(hours=2),
                    ts_base + timedelta(hours=3),
                ],
                "over_line": [24.0, 27.0, 25.5, 25.5],
                "book_name": ["dk", "fd", "dk", "fd"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        # first_half_std > second_half_std → positive convergence
        assert features["line_convergence"] > 0

    def test_no_snapshot_at_column(self):
        df = pd.DataFrame({"over_line": [25.0, 26.0]})
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_movement_velocity"] == 0.0


# ─────────────────────────────────────────────────────────────────
# _compute_freshness
# ─────────────────────────────────────────────────────────────────


class TestComputeFreshness:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_recent_snapshot_low_freshness(self):
        """A snapshot from ~1 minute ago should have freshness ~1."""
        from zoneinfo import ZoneInfo

        EST = ZoneInfo("America/New_York")
        recent = datetime.now(EST) - timedelta(minutes=1)
        df = pd.DataFrame(
            {
                "snapshot_at": [recent],
                "over_line": [25.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_freshness(df, features)
        assert 0 <= features["freshness_score"] <= 5

    def test_old_snapshot_capped(self):
        """A snapshot from 48 hours ago should be capped at 1440."""
        from zoneinfo import ZoneInfo

        EST = ZoneInfo("America/New_York")
        old = datetime.now(EST) - timedelta(hours=48)
        df = pd.DataFrame(
            {
                "snapshot_at": [old],
                "over_line": [25.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_freshness(df, features)
        assert features["freshness_score"] == 1440.0

    def test_no_snapshot_at_column(self):
        df = pd.DataFrame({"over_line": [25.0]})
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_freshness(df, features)
        assert features["freshness_score"] == 0.0

    def test_empty_df(self):
        df = pd.DataFrame({"snapshot_at": [], "over_line": []})
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_freshness(df, features)
        assert features["freshness_score"] == 0.0


# ─────────────────────────────────────────────────────────────────
# _compute_bp_coverage
# ─────────────────────────────────────────────────────────────────


class TestComputeBpCoverage:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_full_coverage(self):
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct"],
            }
        )
        bp_df = pd.DataFrame(
            {
                "book_name": ["DraftKings", "FanDuel"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_coverage(direct_df, bp_df, features)
        assert features["bp_coverage_ratio"] == pytest.approx(1.0)

    def test_partial_coverage(self):
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct", "espnbet_direct"],
            }
        )
        bp_df = pd.DataFrame(
            {
                "book_name": ["DraftKings"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_coverage(direct_df, bp_df, features)
        assert features["bp_coverage_ratio"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_no_mappable_books(self):
        """DFS books like prizepicks have no BP mapping → 0 coverage."""
        direct_df = pd.DataFrame(
            {
                "book_name": ["prizepicks"],
            }
        )
        bp_df = pd.DataFrame(
            {
                "book_name": ["DraftKings"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_coverage(direct_df, bp_df, features)
        assert features["bp_coverage_ratio"] == 0.0


# ─────────────────────────────────────────────────────────────────
# Coverage: _compute_line_movement_features with actual snapshot data
# Lines 217, 234-245, 251-260, 266-275, 326, 444, 559-561
# ─────────────────────────────────────────────────────────────────


class TestLineMovementFeaturesAdvanced:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_zero_time_delta_zero_velocity(self):
        """Line 444: time_delta_hours == 0 → velocity = 0.0."""
        ts = datetime(2026, 3, 16, 10, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts, ts],
                "over_line": [25.0, 26.0],
                "book_name": ["dk", "fd"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_movement_velocity"] == 0.0

    def test_single_consensus_no_velocity(self):
        """When only one consensus time point, skip velocity."""
        ts = datetime(2026, 3, 16, 10, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts],
                "over_line": [25.0],
                "book_name": ["dk"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_movement_velocity"] == 0.0

    def test_no_over_line_column(self):
        """Branch: missing over_line column returns early."""
        ts = datetime(2026, 3, 16, 10, 0)
        df = pd.DataFrame({"snapshot_at": [ts], "book_name": ["dk"]})
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_movement_velocity"] == 0.0

    def test_convergence_not_computed_with_too_few_snapshots(self):
        """Convergence needs midpoint > 1, so 2 snapshots won't compute it."""
        ts1 = datetime(2026, 3, 16, 10, 0)
        ts2 = datetime(2026, 3, 16, 12, 0)
        df = pd.DataFrame(
            {
                "snapshot_at": [ts1, ts2],
                "over_line": [25.0, 26.0],
                "book_name": ["dk", "dk"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        assert features["line_convergence"] == 0.0  # Default, not computed

    def test_divergence_negative_convergence(self):
        """Lines converging = positive, diverging = negative."""
        ts_base = datetime(2026, 3, 16, 8, 0)
        # First half: tight (25.0, 25.0)
        # Second half: wide (23.0, 27.0)
        df = pd.DataFrame(
            {
                "snapshot_at": [
                    ts_base,
                    ts_base + timedelta(hours=1),
                    ts_base + timedelta(hours=2),
                    ts_base + timedelta(hours=3),
                ],
                "over_line": [25.0, 25.0, 23.0, 27.0],
                "book_name": ["dk", "fd", "dk", "fd"],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_line_movement_features(df, features)
        # first_half_std < second_half_std → negative convergence (diverging)
        assert features["line_convergence"] < 0


class TestCrossPlatformFeaturesAdvanced:
    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_with_underdog_capital_name(self):
        """Cross-platform uses 'Underdog' in DFS set."""
        df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "Underdog"],
                "over_line": [25.0, 25.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        # Gap = 0.5 → agreement = 1 - 0.5/2 = 0.75
        assert features["cross_platform_agreement"] == pytest.approx(0.75, abs=0.01)

    def test_with_prizepicks_capital_name(self):
        """Cross-platform uses 'PrizePicks' in DFS set."""
        df = pd.DataFrame(
            {
                "book_name": ["fanduel_direct", "PrizePicks"],
                "over_line": [25.0, 25.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_cross_platform_features(df, features)
        assert features["cross_platform_agreement"] == pytest.approx(1.0, abs=0.01)


class TestExtractWithMultipleDataSources:
    """Test extract() method with both direct and snapshot data present."""

    def test_full_data_pipeline(self):
        """Cover the full extract path with direct + snapshot + bp data."""
        extractor = DirectLineFeatureExtractor(conn=MagicMock())

        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct", "underdog_direct"],
                "over_line": [25.0, 25.5, 26.0],
                "over_odds": [-110.0, -115.0, -108.0],
                "under_line": [25.0, 25.5, 26.0],
                "under_odds": [-110.0, -105.0, -112.0],
                "bp_reported_line": [25.0, 25.5, None],
                "bp_discrepancy": [0.0, 0.5, None],
                "fetch_timestamp": [datetime(2026, 3, 16, 12)] * 3,
            }
        )

        from zoneinfo import ZoneInfo

        EST = ZoneInfo("America/New_York")
        ts1 = datetime(2026, 3, 16, 10, 0, tzinfo=EST)
        ts2 = datetime(2026, 3, 16, 11, 0, tzinfo=EST)
        ts3 = datetime(2026, 3, 16, 12, 0, tzinfo=EST)
        ts4 = datetime(2026, 3, 16, 13, 0, tzinfo=EST)
        snapshot_df = pd.DataFrame(
            {
                "book_name": ["dk_direct", "dk_direct", "fd_direct", "fd_direct"],
                "over_line": [25.0, 25.5, 25.5, 25.5],
                "over_odds": [-110, -110, -115, -115],
                "under_odds": [-110, -110, -105, -105],
                "fetch_source": ["direct"] * 4,
                "snapshot_at": [ts1, ts2, ts3, ts4],
            }
        )

        bp_df = pd.DataFrame({"book_name": ["DraftKings", "FanDuel"]})

        with (
            patch.object(extractor, "_fetch_direct_props", return_value=direct_df),
            patch.object(extractor, "_fetch_snapshots", return_value=snapshot_df),
            patch.object(extractor, "_fetch_bp_props", return_value=bp_df),
        ):
            result = extractor.extract("LeBron James", "2026-03-16", "POINTS")

        assert result["num_direct_sources"] == 3.0
        assert result["direct_consensus"] > 0
        assert result["snapshot_count"] == 4.0
        assert result["hours_tracked"] > 0
        assert result["bp_coverage_ratio"] > 0
        assert result["cross_platform_agreement"] > 0  # Has underdog_direct
        assert result["freshness_score"] >= 0


class TestExtractFromDataframesWithSnapshots:
    """Cover lines 559-561: extract_from_dataframes with snapshot data."""

    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_snapshot_only(self):
        """Lines 559-561: snapshot_df provided without direct_df."""
        from zoneinfo import ZoneInfo

        EST = ZoneInfo("America/New_York")
        ts1 = datetime(2026, 3, 16, 10, 0, tzinfo=EST)
        ts2 = datetime(2026, 3, 16, 12, 0, tzinfo=EST)
        snapshot_df = pd.DataFrame(
            {
                "book_name": ["dk_direct", "dk_direct"],
                "over_line": [25.0, 26.0],
                "over_odds": [-110, -110],
                "under_odds": [-110, -110],
                "fetch_source": ["direct", "direct"],
                "snapshot_at": [ts1, ts2],
            }
        )
        result = self.extractor.extract_from_dataframes(None, snapshot_df, None)
        assert result["snapshot_count"] == 2.0
        assert result["hours_tracked"] == pytest.approx(2.0, abs=0.01)
        assert result["freshness_score"] >= 0

    def test_direct_and_snapshot_together(self):
        """Full path: direct_df + snapshot_df + bp_df all provided."""
        from zoneinfo import ZoneInfo

        EST = ZoneInfo("America/New_York")
        direct_df = pd.DataFrame(
            {
                "book_name": ["draftkings_direct", "fanduel_direct"],
                "over_line": [25.0, 25.5],
                "over_odds": [-110.0, -115.0],
                "under_line": [25.0, 25.5],
                "under_odds": [-110.0, -105.0],
                "bp_reported_line": [25.0, None],
                "bp_discrepancy": [0.5, None],
                "fetch_timestamp": [datetime(2026, 3, 16)] * 2,
            }
        )
        ts1 = datetime(2026, 3, 16, 10, 0, tzinfo=EST)
        ts2 = datetime(2026, 3, 16, 14, 0, tzinfo=EST)
        snapshot_df = pd.DataFrame(
            {
                "book_name": ["dk_direct", "dk_direct"],
                "over_line": [24.5, 25.5],
                "over_odds": [-110, -110],
                "under_odds": [-110, -110],
                "fetch_source": ["direct", "direct"],
                "snapshot_at": [ts1, ts2],
            }
        )
        bp_df = pd.DataFrame({"book_name": ["DraftKings"]})

        result = self.extractor.extract_from_dataframes(direct_df, snapshot_df, bp_df)
        assert result["num_direct_sources"] == 2.0
        assert result["snapshot_count"] == 2.0
        assert result["hours_tracked"] > 0
        assert result["bp_coverage_ratio"] > 0


class TestFetchMethodsViaExtract:
    """Cover lines 234-245, 251-260, 266-275 by letting fetch methods execute."""

    def test_fetch_direct_props_builds_correct_query(self):
        """Lines 234-245: _fetch_direct_props builds and executes query."""
        mock_conn = MagicMock()
        extractor = DirectLineFeatureExtractor(conn=mock_conn)

        with patch.object(extractor, "_safe_query", return_value=None) as mock_sq:
            result = extractor._fetch_direct_props("LeBron", "2026-03-16", "points")
        assert result is None
        call_args = mock_sq.call_args
        assert "fetch_source = 'direct'" in call_args[0][0]
        assert call_args[0][1] == ("LeBron", "2026-03-16", "POINTS")

    def test_fetch_bp_props_builds_correct_query(self):
        """Lines 251-260: _fetch_bp_props builds and executes query."""
        mock_conn = MagicMock()
        extractor = DirectLineFeatureExtractor(conn=mock_conn)

        with patch.object(extractor, "_safe_query", return_value=None) as mock_sq:
            result = extractor._fetch_bp_props("LeBron", "2026-03-16", "rebounds")
        assert result is None
        call_args = mock_sq.call_args
        assert "fetch_source = 'bettingpros'" in call_args[0][0]
        assert call_args[0][1] == ("LeBron", "2026-03-16", "REBOUNDS")

    def test_fetch_snapshots_builds_correct_query(self):
        """Lines 266-275: _fetch_snapshots builds and executes query."""
        mock_conn = MagicMock()
        extractor = DirectLineFeatureExtractor(conn=mock_conn)

        with patch.object(extractor, "_safe_query", return_value=None) as mock_sq:
            result = extractor._fetch_snapshots("LeBron", "2026-03-16", "points")
        assert result is None
        call_args = mock_sq.call_args
        assert "nba_line_snapshots" in call_args[0][0]
        assert call_args[0][1] == ("LeBron", "2026-03-16", "POINTS")


class TestBpDiscrepancyLatencyNoReportedLine:
    """Test bp_discrepancy with bp_reported_line all None (line 326)."""

    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_no_bp_reported_line_latency_zero(self):
        """Line 326: bp_reported_line all None → latency = 0."""
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd"],
                "over_line": [25.0, 25.5],
                "bp_discrepancy": [0.5, -1.0],
                "fetch_timestamp": [datetime(2026, 3, 16), datetime(2026, 3, 16)],
                "bp_reported_line": [None, None],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_discrepancy_features(df, features)
        assert features["bp_line_latency_avg"] == 0.0
        assert features["bp_discrepancy_avg"] != 0.0  # Discrepancy still computed

    def test_discrepancy_no_fetch_timestamp_column(self):
        """Branch 317->exit: bp_discrepancy exists but no fetch_timestamp column."""
        df = pd.DataFrame(
            {
                "book_name": ["dk"],
                "over_line": [25.0],
                "bp_discrepancy": [0.5],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_bp_discrepancy_features(df, features)
        assert features["bp_discrepancy_avg"] == 0.5
        # Latency should stay at default since no fetch_timestamp column
        assert features["bp_line_latency_avg"] == 0.0


class TestOddsEdgeCases:
    """Cover remaining odds computation branch misses."""

    def setup_method(self):
        self.extractor = DirectLineFeatureExtractor(conn=MagicMock())

    def test_zero_odds_skips_vig(self):
        """Branch 344->338: odds of 0 → implied = 0 → skip vig append."""
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd"],
                "over_line": [25.0, 25.0],
                "over_odds": [0.0, -110.0],
                "under_odds": [0.0, -110.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        # Only the second row should contribute to vig (dk has odds=0)
        assert features["odds_vig_avg"] > 0

    def test_mixed_nan_odds_in_rows(self):
        """Branch 341->338: individual row has NaN odds while column has non-null values."""
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd", "bg"],
                "over_line": [25.0, 25.0, 25.0],
                "over_odds": [-110.0, None, -108.0],
                "under_odds": [-110.0, -105.0, None],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        # Only dk has both non-null → 1 vig entry
        assert features["odds_vig_avg"] > 0

    def test_vigs_empty_when_no_row_has_both_odds(self):
        """Branch 347->351: vigs list empty → odds_vig_avg stays default."""
        df = pd.DataFrame(
            {
                "book_name": ["dk", "fd"],
                "over_line": [25.0, 25.0],
                "over_odds": [-110.0, None],
                "under_odds": [None, -110.0],
            }
        )
        features = DirectLineFeatureExtractor.get_defaults()
        self.extractor._compute_odds_features(df, features)
        # No row has both over_odds AND under_odds non-null, so vigs is empty
        assert features["odds_vig_avg"] == 0.0
        # But odds_spread should still be computed from the non-null over_odds
        assert features["direct_odds_spread"] == 0.0  # Only 1 non-null over_odds
