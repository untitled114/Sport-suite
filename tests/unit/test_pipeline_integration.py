"""Tests for pipeline integration — verifies new modules work together."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestModuleImports:
    """Verify all new modules are importable and have correct interfaces."""

    def test_edge_calculator(self):
        from nba.core.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()
        assert hasattr(calc, "calculate_edge")
        assert hasattr(calc, "calculate_edge_from_probs")

    def test_probability_engine(self):
        from nba.core.probability_engine import ProbabilityEngine

        engine = ProbabilityEngine()
        assert hasattr(engine, "calculate_probability")
        assert hasattr(engine, "implied_probability")

    def test_kelly_sizer(self):
        from nba.core.bet_sizing import KellySizer

        sizer = KellySizer(bankroll=1000)
        assert hasattr(sizer, "size_bet")

    def test_feature_store(self):
        from nba.features.feature_store import FeatureStore

        store = FeatureStore()
        assert hasattr(store, "write_features")
        assert hasattr(store, "read_player_features")
        assert hasattr(store, "write_batch")

    def test_consolidated_db(self):
        from nba.config.database import SCHEMAS, get_connection, get_schema_config

        assert callable(get_connection)
        assert callable(get_schema_config)
        assert len(SCHEMAS) == 6


class TestEdgeCalculation:

    def test_positive_edge(self):
        from nba.core.edge_calculator import EdgeCalculator

        calc = EdgeCalculator(min_edge=0.05)
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert result["has_edge"] is True
        assert result["edge"] > 0.05
        assert result["kelly_fraction"] > 0
        assert result["expected_value"] > 0

    def test_no_edge(self):
        from nba.core.edge_calculator import EdgeCalculator

        calc = EdgeCalculator(min_edge=0.05)
        result = calc.calculate_edge(model_prob=0.50, book_odds=-110)
        assert result["has_edge"] is False

    def test_suspicious_edge(self):
        from nba.core.edge_calculator import EdgeCalculator

        calc = EdgeCalculator(min_edge=0.05, max_edge=0.25)
        result = calc.calculate_edge(model_prob=0.85, book_odds=-110)
        assert result["is_suspicious"] is True

    def test_underdog_odds(self):
        from nba.core.edge_calculator import EdgeCalculator

        calc = EdgeCalculator(min_edge=0.05)
        result = calc.calculate_edge(model_prob=0.55, book_odds=+130)
        assert result["implied_prob"] < 0.5


class TestProbabilityDistributions:

    def test_normal_above_line(self):
        from nba.core.probability_engine import ProbabilityEngine

        p = ProbabilityEngine().calculate_probability(25.0, 5.0, 22.5, "POINTS")
        assert 0.5 < p < 1.0

    def test_normal_below_line(self):
        from nba.core.probability_engine import ProbabilityEngine

        p = ProbabilityEngine().calculate_probability(20.0, 5.0, 25.5, "POINTS")
        assert 0.0 < p < 0.5

    def test_poisson_threes(self):
        from nba.core.probability_engine import ProbabilityEngine

        p = ProbabilityEngine().calculate_probability(3.0, 1.5, 2.5, "THREES")
        assert 0.0 < p < 1.0

    def test_implied_probability(self):
        from nba.core.probability_engine import ProbabilityEngine

        assert abs(ProbabilityEngine.implied_probability(-110) - 0.5238) < 0.01
        assert abs(ProbabilityEngine.implied_probability(+200) - 0.3333) < 0.01


class TestKellySizing:

    def test_basic_sizing(self):
        from nba.core.bet_sizing import KellySizer

        result = KellySizer(bankroll=1000, fraction=0.25, max_bet_pct=0.03).size_bet(0.62, -110)
        assert result["bet_amount"] > 0
        assert result["bet_amount"] <= 30.0

    def test_no_edge_no_bet(self):
        from nba.core.bet_sizing import KellySizer

        result = KellySizer(bankroll=1000).size_bet(model_prob=0.48, odds=-110)
        assert result["has_edge"] is False

    def test_cap_applied(self):
        from nba.core.bet_sizing import KellySizer

        result = KellySizer(bankroll=1000, fraction=0.25, max_bet_pct=0.03).size_bet(0.90, -110)
        assert result["bet_amount"] <= 30.0


class TestFeatureStoreIntegration:

    def test_write(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        conn.closed = False
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        store = FeatureStore(conn=conn)
        assert store.write_features("LeBron", "2026-03-20", "POINTS", "xl_v1", {"a": 1}) is True

    def test_batch_write(self):
        from nba.features.feature_store import FeatureStore

        conn = MagicMock()
        conn.closed = False
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
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
                "stat_type": "REBOUNDS",
                "feature_values": {"b": 2},
            },
        ]
        assert store.write_batch(rows, "xl_v1") == 2


class TestPickEnrichmentWorkflow:
    """Test the full enrichment workflow."""

    def test_full_enrichment(self):
        from nba.core.bet_sizing import KellySizer
        from nba.core.edge_calculator import EdgeCalculator
        from nba.core.probability_engine import ProbabilityEngine

        pick = {"p_over": 0.72, "best_line": 25.5}

        dist_prob = ProbabilityEngine().calculate_probability(27.0, 5.5, 25.5, "POINTS")
        pick["distribution_prob"] = round(dist_prob, 4)
        pick["prob_agreement"] = round(abs(pick["p_over"] - dist_prob), 4)

        edge = EdgeCalculator(min_edge=0.05).calculate_edge(0.72, -110)
        pick.update(
            {
                k: edge[k]
                for k in ["implied_prob", "edge", "has_edge", "expected_value", "kelly_fraction"]
            }
        )

        kelly = KellySizer(bankroll=1000).size_bet(0.72, -110)
        pick["bet_amount"] = kelly["bet_amount"]

        assert pick["has_edge"] is True
        assert pick["distribution_prob"] > 0.5
        assert pick["bet_amount"] > 0


class TestConsolidatedDB:

    @pytest.mark.parametrize(
        "schema", ["players", "games", "teams", "intelligence", "axiom", "features"]
    )
    def test_schema_config(self, schema):
        from nba.config.database import get_schema_config

        config = get_schema_config(schema)
        assert config["port"] == 5500
        assert f"search_path={schema}" in config["options"]

    @patch("psycopg2.connect")
    def test_get_connection_all_schemas(self, mock_connect):
        from nba.config.database import get_connection

        mock_connect.return_value = MagicMock()
        for schema in ["players", "games", "teams", "intelligence", "axiom", "features"]:
            get_connection(schema)
        assert mock_connect.call_count == 6
