"""
Tests for nba.betting_xl.quick_refresh — covers the Windows compatibility
try/except block for time.tzset().
"""

from unittest.mock import patch


class TestTzsetWindowsCompat:
    def test_tzset_attribute_error_is_handled(self):
        """Lines 30-31: time.tzset raises AttributeError on Windows — must not crash."""
        import importlib

        with patch("time.tzset", side_effect=AttributeError("tzset not available")):
            # Re-import the module to re-execute the module-level code
            import nba.betting_xl.quick_refresh as qr

            importlib.reload(qr)
        # If we get here without exception, the try/except worked
        assert True
