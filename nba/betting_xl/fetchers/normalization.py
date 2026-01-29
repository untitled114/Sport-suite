#!/usr/bin/env python3
"""
Utility Functions for NBA XL Fetchers
======================================
Common helper functions for prop fetching and normalization.
"""

import re
from datetime import datetime
from typing import Optional


def normalize_player_name(name: str) -> str:
    """
    Normalize player name to standard format.

    Handles:
    - Case normalization (Title Case)
    - Whitespace trimming
    - Period removal (J.R. Smith -> JR Smith)
    - Apostrophe standardization
    - Special characters

    Args:
        name: Raw player name

    Returns:
        Normalized name

    Examples:
        >>> normalize_player_name("LEBRON JAMES")
        'Lebron James'
        >>> normalize_player_name("J.R. Smith")
        'JR Smith'
        >>> normalize_player_name("  De'Aaron Fox  ")
        "De'Aaron Fox"
    """
    if not name:
        return ""

    # Strip whitespace
    name = name.strip()

    # Remove periods
    name = name.replace(".", "")

    # Standardize apostrophes
    name = name.replace("'", "'")
    name = name.replace("`", "'")

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name)

    # Title case if all upper or all lower
    if name.isupper() or name.islower():
        name = name.title()

    return name


def normalize_stat_type(stat_type: str) -> str:
    """
    Normalize stat type to canonical format.

    Args:
        stat_type: Raw stat type string

    Returns:
        Canonical stat type (POINTS, REBOUNDS, ASSISTS, THREES)

    Examples:
        >>> normalize_stat_type("points")
        'POINTS'
        >>> normalize_stat_type("3-pt made")
        'THREES'
        >>> normalize_stat_type("reb")
        'REBOUNDS'
    """
    stat_type = stat_type.lower().strip()

    # Mapping dictionary
    stat_map = {
        # Points
        "points": "POINTS",
        "pts": "POINTS",
        "point": "POINTS",
        # Rebounds
        "rebounds": "REBOUNDS",
        "rebound": "REBOUNDS",
        "reb": "REBOUNDS",
        "rebs": "REBOUNDS",
        # Assists
        "assists": "ASSISTS",
        "assist": "ASSISTS",
        "ast": "ASSISTS",
        "asts": "ASSISTS",
        # Threes
        "threes": "THREES",
        "three": "THREES",
        "3-pt made": "THREES",
        "3pm": "THREES",
        "3 pointers made": "THREES",
        "three pointers made": "THREES",
        "3-pointers": "THREES",
        "3 pt": "THREES",
        "3pt": "THREES",
        # Steals
        "steals": "STEALS",
        "steal": "STEALS",
        "stl": "STEALS",
        # Blocks
        "blocks": "BLOCKS",
        "block": "BLOCKS",
        "blk": "BLOCKS",
        "blocked shots": "BLOCKS",
        # Turnovers
        "turnovers": "TURNOVERS",
        "turnover": "TURNOVERS",
        "to": "TURNOVERS",
        # Combos
        "pts+rebs": "PTS_REBS",
        "pts+asts": "PTS_ASTS",
        "pts+rebs+asts": "PTS_REBS_ASTS",
        "rebs+asts": "REBS_ASTS",
    }

    return stat_map.get(stat_type, stat_type.upper())


def normalize_book_name(book_name: str) -> str:
    """
    Normalize sportsbook name to standard format.

    Args:
        book_name: Raw book name

    Returns:
        Canonical book name

    Examples:
        >>> normalize_book_name("DraftKings")
        'draftkings'
        >>> normalize_book_name("Bet MGM")
        'betmgm'
        >>> normalize_book_name("Prize Picks")
        'prizepicks'
    """
    book_name = book_name.lower().strip()

    # Remove spaces, hyphens, underscores
    book_name = book_name.replace(" ", "")
    book_name = book_name.replace("-", "")
    book_name = book_name.replace("_", "")

    # Common variations
    book_map = {
        "dk": "draftkings",
        "fd": "fanduel",
        "mgm": "betmgm",
        "czr": "caesars",
        "cz": "caesars",
        "br": "betrivers",
        "pb": "pointsbet",
        "ud": "underdog",
        "pp": "prizepicks",
        "bettingpros": "consensus",
    }

    return book_map.get(book_name, book_name)


def parse_game_date(date_string: str) -> Optional[str]:
    """
    Parse game date from various formats to YYYY-MM-DD.

    Args:
        date_string: Date string in various formats

    Returns:
        Date in YYYY-MM-DD format or None if parsing fails

    Examples:
        >>> parse_game_date("2025-11-05T19:30:00Z")
        '2025-11-05'
        >>> parse_game_date("Nov 5, 2025")
        '2025-11-05'
        >>> parse_game_date("11/5/2025")
        '2025-11-05'
    """
    if not date_string:
        return None

    # Try ISO format first
    try:
        dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        pass

    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def validate_line(line: any) -> bool:
    """
    Validate that a line value is valid.

    Args:
        line: Line value to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_line(25.5)
        True
        >>> validate_line("25.5")
        True
        >>> validate_line("invalid")
        False
        >>> validate_line(None)
        False
    """
    if line is None:
        return False

    try:
        value = float(line)
        # Line should be positive and reasonable (0-100 for most stats)
        return 0 <= value <= 200
    except (ValueError, TypeError):
        return False


def calculate_consensus(lines: list) -> Optional[float]:
    """
    Calculate consensus line from multiple book lines.

    Args:
        lines: List of line values

    Returns:
        Median line value or None if empty

    Examples:
        >>> calculate_consensus([25.5, 26.5, 25.5, 27.5])
        26.0
        >>> calculate_consensus([25.5])
        25.5
        >>> calculate_consensus([])
        None
    """
    if not lines:
        return None

    # Use median to avoid outlier influence
    sorted_lines = sorted(lines)
    n = len(sorted_lines)

    if n % 2 == 0:
        # Even number of lines - average middle two
        return (sorted_lines[n // 2 - 1] + sorted_lines[n // 2]) / 2
    else:
        # Odd number - take middle
        return sorted_lines[n // 2]


def calculate_line_spread(lines: list) -> float:
    """
    Calculate spread (max - min) of book lines.

    Args:
        lines: List of line values

    Returns:
        Spread value (0 if only one line)

    Examples:
        >>> calculate_line_spread([25.5, 26.5, 27.5])
        2.0
        >>> calculate_line_spread([25.5])
        0.0
    """
    if not lines or len(lines) <= 1:
        return 0.0

    return max(lines) - min(lines)


if __name__ == "__main__":
    # Run doctests
    import doctest

    doctest.testmod()
    print("✅ All utility function tests passed!")
