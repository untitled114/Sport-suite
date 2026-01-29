"""
Shared Constants for NBA Suite
==============================
Centralized constants to eliminate duplication across files.

This module contains:
- STAT_COLUMN_MAP: Maps stat types to database column names
- VALID_STAT_TYPES: Core betting markets (POINTS, REBOUNDS, ASSISTS, THREES)
- PRIORITY_BOOKS: Sportsbook IDs for line shopping
- TEAM_ABBREV_MAP: Normalizes team abbreviations across sources
- ACTIVE_MARKETS: Currently enabled prediction markets

Usage:
    from nba.config.constants import STAT_COLUMN_MAP, VALID_STAT_TYPES

Example:
    >>> from nba.config.constants import STAT_COLUMN_MAP
    >>> column = STAT_COLUMN_MAP['POINTS']  # Returns 'points'
    >>> column = STAT_COLUMN_MAP['THREES']  # Returns 'three_pointers_made'
"""

from typing import Dict, FrozenSet, Set, Tuple

# =============================================================================
# STAT TYPE MAPPINGS
# =============================================================================

# Maps stat types to player_game_logs column names
STAT_COLUMN_MAP: Dict[str, str] = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "three_pointers_made",
    "STEALS": "steals",
    "BLOCKS": "blocks",
}

# Valid stat types for props (core betting markets)
VALID_STAT_TYPES: FrozenSet[str] = frozenset({"POINTS", "REBOUNDS", "ASSISTS", "THREES"})

# Extended stat types (includes all trackable stats)
EXTENDED_STAT_TYPES: FrozenSet[str] = frozenset(
    {"POINTS", "REBOUNDS", "ASSISTS", "THREES", "STEALS", "BLOCKS"}
)

# Combo stat mappings (for parlay validation)
COMBO_STAT_MAP: Dict[str, Tuple[str, ...]] = {
    "PA": ("points", "assists"),
    "PR": ("points", "rebounds"),
    "RA": ("rebounds", "assists"),
    "PRA": ("points", "rebounds", "assists"),
    "POINTS_ASSISTS": ("points", "assists"),
    "POINTS_REBOUNDS": ("points", "rebounds"),
    "REBOUNDS_ASSISTS": ("rebounds", "assists"),
    "POINTS_REBOUNDS_ASSISTS": ("points", "rebounds", "assists"),
}

# =============================================================================
# SPORTSBOOK MAPPINGS
# =============================================================================

# Priority sportsbooks for line shopping (BettingPros book IDs)
# Updated Jan 2026: Removed bet365 (binary responses), fanatics (API errors)
PRIORITY_BOOKS: Dict[int, str] = {
    12: "draftkings",
    10: "fanduel",
    19: "betmgm",
    13: "caesars",
    18: "betrivers",
    33: "espnbet",
    36: "underdog",  # DFS site - typically softer lines
}

# All supported sportsbooks (including lower priority)
ALL_BOOKS: Dict[int, str] = {
    **PRIORITY_BOOKS,
    15: "sugarhouse",
    27: "partycasino",
    37: "unibet",
    38: "sisportsbook",
    39: "fliff",
    49: "betway",
    60: "superbook",
    63: "action247",
}

# Books that don't support certain markets
BOOK_MARKET_EXCLUSIONS: Dict[int, Set[str]] = {
    33: {"threes"},  # espnbet doesn't offer THREES
    36: {"threes"},  # underdog doesn't offer THREES
}

# =============================================================================
# BETTINGPROS API MAPPINGS
# =============================================================================

# Market name to BettingPros market ID
BETTINGPROS_MARKET_IDS: Dict[str, int] = {
    "points": 156,
    "rebounds": 157,
    "assists": 151,
    "threes": 162,
}

# =============================================================================
# TEAM ABBREVIATION MAPPINGS
# =============================================================================

# Normalize team abbreviations from various sources to database format
TEAM_ABBREV_MAP: Dict[str, str] = {
    "NO": "NOP",  # New Orleans Pelicans
    "SA": "SAS",  # San Antonio Spurs
    "UTAH": "UTA",  # Utah Jazz
    "GS": "GSW",  # Golden State Warriors
    "NY": "NYK",  # New York Knicks
    "BKN": "BKN",  # Brooklyn Nets
    "BRK": "BKN",  # Brooklyn Nets (alternate)
    "WSH": "WAS",  # Washington Wizards (ESPN format)
    "PHX": "PHO",  # Phoenix Suns (some sources)
    "PHO": "PHX",  # Phoenix Suns (normalize to PHX)
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Active prediction markets (disabled markets are excluded)
ACTIVE_MARKETS: FrozenSet[str] = frozenset({"POINTS", "REBOUNDS"})

# Disabled markets with reasons
DISABLED_MARKETS: Dict[str, str] = {
    "ASSISTS": "14.6% WR, -72.05% ROI (severe underperformance)",
    "THREES": "46.5% WR, -11.23% ROI (losing strategy)",
}
