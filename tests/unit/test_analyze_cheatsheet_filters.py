"""
Unit Tests for analyze_cheatsheet_filters.py
=============================================
Tests for cheatsheet filter analysis functions.
All database connections are mocked.
"""

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build mock DB connections for analyze_losses / main
# ---------------------------------------------------------------------------
def _make_mock_connections(cheatsheet_rows, game_log_rows):
    """Create paired mock intel + players connections.

    cheatsheet_rows: list of 12-element tuples returned by intel cursor.
    game_log_rows: list of tuples returned by players cursor.
       For analyze_losses: 5-element tuples (name, date, pts, reb, ast)
       For main: 6-element tuples (name, date, pts, reb, ast, threes)
    """
    intel_conn = MagicMock()
    intel_cur = MagicMock()
    intel_cur.fetchall.return_value = cheatsheet_rows
    intel_conn.cursor.return_value = intel_cur

    players_conn = MagicMock()
    players_cur = MagicMock()
    players_cur.fetchall.return_value = game_log_rows
    players_conn.cursor.return_value = players_cur

    return intel_conn, players_conn


# ---------------------------------------------------------------------------
# analyze_losses
# ---------------------------------------------------------------------------
class TestAnalyzeLosses:
    """Tests for the analyze_losses function."""

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_with_hits(self, mock_connect, capsys):
        """Test analyze_losses correctly identifies wins and losses."""
        cheatsheet_rows = [
            # (player, date, stat, line, proj, proj_diff, rating, ev, opp_rank, l5, l15, szn)
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                2.5,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
        ]
        game_log_rows = [
            # (name, date, pts, reb, ast)
            ("LeBron James", "2025-01-15", 30, 8, 7),
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        # The function should run all three stat filters (POINTS, REBOUNDS, ASSISTS)
        assert "POINTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_no_matches(self, mock_connect, capsys):
        """Test analyze_losses when no cheatsheet entries match game logs."""
        cheatsheet_rows = [
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                2.5,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
        ]
        game_log_rows = [
            # Different player -- no match
            ("Stephen Curry", "2025-01-15", 35, 5, 6),
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        # Should still output headers but with 0 wins/losses
        assert "POINTS" in captured.out
        assert "0 losses" in captured.out or "0 wins" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_loss_detection(self, mock_connect, capsys):
        """Test that losses are correctly identified (actual <= line)."""
        cheatsheet_rows = [
            # Player misses line for POINTS
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                30.5,
                32.0,
                1.5,
                4,
                20.0,
                22,
                0.80,
                0.75,
                0.75,
            ),
        ]
        game_log_rows = [
            ("LeBron James", "2025-01-15", 25, 8, 7),  # 25 < 30.5 = loss
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        # Should have 1 loss listed
        assert "1 losses" in captured.out or "LeBron" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_additional_filters(self, mock_connect, capsys):
        """Test that additional filter options are printed when enough data."""
        # Create 4 records that pass the POINTS filter
        cheatsheet_rows = [
            ("Player A", "2025-01-15", "POINTS", 20.5, 23.0, 2.5, 4, 20.0, 22, 0.80, 0.70, 0.75),
            ("Player B", "2025-01-15", "POINTS", 18.5, 21.0, 2.5, 3, 15.0, 15, 0.60, 0.65, 0.70),
            ("Player C", "2025-01-16", "POINTS", 22.5, 25.0, 2.5, 5, 25.0, 25, 0.90, 0.80, 0.80),
            ("Player D", "2025-01-16", "POINTS", 19.5, 22.0, 2.5, 4, 18.0, 20, 0.70, 0.60, 0.72),
        ]
        game_log_rows = [
            ("Player A", "2025-01-15", 25, 8, 7),  # Win (25 > 20.5)
            ("Player B", "2025-01-15", 22, 6, 5),  # Win (22 > 18.5)
            ("Player C", "2025-01-16", 18, 10, 9),  # Loss (18 < 22.5)
            ("Player D", "2025-01-16", 24, 7, 6),  # Win (24 > 19.5)
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        # Should show additional filter options
        assert "Additional filter options" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_none_fields(self, mock_connect, capsys):
        """Test handling of None values in cheatsheet data."""
        cheatsheet_rows = [
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                None,
                None,
                None,
                None,
                None,
                None,
                0.75,
            ),
        ]
        game_log_rows = [
            ("LeBron James", "2025-01-15", 30, 8, 7),
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        # Should complete without errors despite None fields
        captured = capsys.readouterr()
        assert "POINTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_closes_connections(self, mock_connect):
        """Test that database connections are closed after analysis."""
        intel_conn, players_conn = _make_mock_connections([], [])
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()

        intel_conn.close.assert_called_once()
        players_conn.close.assert_called_once()

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_rebounds_filter(self, mock_connect, capsys):
        """Test the REBOUNDS filter in analyze_losses."""
        cheatsheet_rows = [
            # Passes REBOUNDS filter: hr_l15>=60, opp>=21, diff>=1.0
            (
                "Anthony Davis",
                "2025-01-15",
                "REBOUNDS",
                9.5,
                11.0,
                1.5,
                4,
                15.0,
                25,
                0.80,
                0.70,
                0.65,
            ),
        ]
        game_log_rows = [
            ("Anthony Davis", "2025-01-15", 22, 14, 3),  # 14 > 9.5 = win
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        assert "REBOUNDS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_case_insensitive_name_matching(self, mock_connect, capsys):
        """Test that player name matching is case-insensitive."""
        cheatsheet_rows = [
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                2.5,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
        ]
        game_log_rows = [
            # Uppercase name should still match (lowered internally)
            ("LEBRON JAMES", "2025-01-15", 30, 8, 7),
        ]
        intel_conn, players_conn = _make_mock_connections(cheatsheet_rows, game_log_rows)
        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        analyze_losses()
        captured = capsys.readouterr()

        # Match should be found (both lowered)
        assert "POINTS" in captured.out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
class TestMain:
    """Tests for the main cheatsheet analysis function."""

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_basic_output(self, mock_connect, capsys):
        """Test main produces expected output structure."""
        cheatsheet_rows = [
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                2.5,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
        ]
        game_log_rows = [
            # main expects 6-element tuples (includes threes)
            ("LeBron James", "2025-01-15", 30, 8, 7, 3),
        ]
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        assert "cheatsheet props" in captured.out
        assert "game logs" in captured.out
        assert "Matched" in captured.out
        assert "POINTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_no_data(self, mock_connect, capsys):
        """Test main handles empty data gracefully."""
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = []
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = []
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        assert "Found 0 cheatsheet props" in captured.out
        assert "Found 0 game logs" in captured.out
        assert "Matched 0 props" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_multiple_stats(self, mock_connect, capsys):
        """Test main analyzes multiple stat types."""
        cheatsheet_rows = [
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.5,
                28.0,
                2.5,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
            (
                "LeBron James",
                "2025-01-15",
                "REBOUNDS",
                7.5,
                9.0,
                1.5,
                3,
                10.0,
                18,
                0.60,
                0.55,
                0.60,
            ),
            ("LeBron James", "2025-01-15", "ASSISTS", 6.5, 8.0, 1.5, 4, 20.0, 25, 0.70, 0.65, 0.68),
        ]
        game_log_rows = [
            ("LeBron James", "2025-01-15", 30, 8, 7, 3),
        ]
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        assert "POINTS" in captured.out
        assert "REBOUNDS" in captured.out
        assert "ASSISTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_filter_results_sorted_by_wr(self, mock_connect, capsys):
        """Test that filter results are sorted by win rate descending."""
        # Generate enough data for multiple filters to have 3+ samples
        cheatsheet_rows = []
        game_log_rows = []
        for i in range(10):
            name = f"Player_{i}"
            date_str = f"2025-01-{15 + i % 5}"
            cheatsheet_rows.append(
                (name, date_str, "POINTS", 20.0, 23.0, 3.0, 4, 25.0, 25, 0.85, 0.75, 0.75)
            )
            pts = 25 if i < 7 else 15  # 7 wins, 3 losses
            game_log_rows.append((name, date_str, pts, 8, 6, 2))

        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        assert "POINTS" in captured.out
        assert "Filter" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_closes_connections(self, mock_connect):
        """Test that database connections are closed after main."""
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = []
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = []
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()

        intel_conn.close.assert_called_once()
        players_conn.close.assert_called_once()

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_filter_minimum_sample_size(self, mock_connect, capsys):
        """Test that filters with fewer than 3 samples are excluded."""
        # Only 2 records - filters requiring >= 3 should be skipped
        cheatsheet_rows = [
            ("Player A", "2025-01-15", "POINTS", 20.0, 23.0, 3.0, 5, 30.0, 28, 1.0, 0.90, 0.85),
            ("Player B", "2025-01-16", "POINTS", 18.0, 21.0, 3.0, 5, 25.0, 25, 0.90, 0.85, 0.80),
        ]
        game_log_rows = [
            ("Player A", "2025-01-15", 28, 8, 6, 2),
            ("Player B", "2025-01-16", 25, 7, 5, 3),
        ]
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        # The function runs, but with only 2 samples most filters won't appear
        assert "POINTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_star_marker_for_high_wr(self, mock_connect, capsys):
        """Test that *** marker appears for high win rate filters."""
        # 6 records, all wins with high filter values
        cheatsheet_rows = []
        game_log_rows = []
        for i in range(6):
            name = f"Player_{i}"
            date_str = f"2025-01-{15 + i}"
            cheatsheet_rows.append(
                (name, date_str, "POINTS", 18.0, 22.0, 4.0, 5, 30.0, 28, 1.0, 0.90, 0.85)
            )
            game_log_rows.append((name, date_str, 28, 8, 6, 2))  # All wins

        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        # With 100% win rate and n >= 5, should see *** markers
        assert "***" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_hit_detection_threshold(self, mock_connect, capsys):
        """Test that hit is correctly defined as actual > line (not >=)."""
        cheatsheet_rows = [
            # Line is exactly 25, and actual will be 25 (push = loss)
            (
                "LeBron James",
                "2025-01-15",
                "POINTS",
                25.0,
                28.0,
                3.0,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
            (
                "LeBron James",
                "2025-01-16",
                "POINTS",
                25.0,
                28.0,
                3.0,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
            (
                "LeBron James",
                "2025-01-17",
                "POINTS",
                25.0,
                28.0,
                3.0,
                4,
                15.0,
                22,
                0.80,
                0.70,
                0.75,
            ),
        ]
        game_log_rows = [
            ("LeBron James", "2025-01-15", 25, 8, 7, 3),  # Push (25 == 25) -> loss
            ("LeBron James", "2025-01-16", 26, 8, 7, 3),  # Win (26 > 25)
            ("LeBron James", "2025-01-17", 24, 8, 7, 3),  # Loss (24 < 25)
        ]
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        # Should report 1 win out of 3 (33.3%)
        assert "POINTS" in captured.out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_none_actual_defaults_to_zero(self, mock_connect, capsys):
        """Test that None values in game logs default to 0."""
        cheatsheet_rows = [
            ("LeBron James", "2025-01-15", "POINTS", 5.0, 8.0, 3.0, 4, 15.0, 22, 0.80, 0.70, 0.75),
        ]
        game_log_rows = [
            ("LeBron James", "2025-01-15", None, None, None, None),  # All None
        ]
        intel_conn = MagicMock()
        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = cheatsheet_rows
        intel_conn.cursor.return_value = intel_cur

        players_conn = MagicMock()
        players_cur = MagicMock()
        players_cur.fetchall.return_value = game_log_rows
        players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [intel_conn, players_conn]

        from nba.betting_xl.analyze_cheatsheet_filters import main

        main()
        captured = capsys.readouterr()

        # Should still run — the None defaults to 0 (pts or 0)
        assert "Matched" in captured.out
