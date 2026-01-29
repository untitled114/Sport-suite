"""
Season helper utilities for NBA date/season conversions.

NBA seasons use END year convention:
- 2024-25 season = 2025
- 2023-24 season = 2024

Season typically runs October to June.
"""

from datetime import datetime, date
from typing import Union


def date_to_season(game_date: Union[str, date, datetime]) -> int:
    """
    Convert a game date to NBA season identifier (END year convention).

    Args:
        game_date: Date as string (YYYY-MM-DD), date, or datetime object

    Returns:
        Season year (e.g., 2025 for the 2024-25 season)

    Examples:
        date_to_season('2024-11-15') -> 2025  (Nov 2024 = 2024-25 season)
        date_to_season('2025-03-10') -> 2025  (Mar 2025 = 2024-25 season)
        date_to_season('2025-10-22') -> 2026  (Oct 2025 = 2025-26 season)
    """
    # Parse string dates
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date[:10], '%Y-%m-%d').date()
    elif isinstance(game_date, datetime):
        game_date = game_date.date()

    year = game_date.year
    month = game_date.month

    # NBA season starts in October
    # If month >= October (10), we're in the NEW season (year + 1)
    # If month < October, we're still in the PREVIOUS season (year)
    if month >= 10:
        return year + 1
    else:
        return year


def season_to_date_range(season: int) -> tuple:
    """
    Get the approximate date range for a season.

    Args:
        season: Season year (END year convention)

    Returns:
        Tuple of (start_date, end_date)
    """
    start_year = season - 1
    return (
        date(start_year, 10, 1),  # Season starts ~October
        date(season, 6, 30)        # Season ends ~June
    )
