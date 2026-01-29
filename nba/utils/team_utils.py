#!/usr/bin/env python3
"""
Team Abbreviation Normalization Utilities
==========================================
Centralized team abbreviation handling for NBA XL system.

All team abbreviations are normalized to NBA.com official 3-letter codes.
This eliminates inconsistencies between ESPN API, BettingPros, and database sources.

Usage:
    from nba.utils.team_utils import normalize_team_abbrev, validate_team_abbrev

    team = normalize_team_abbrev('GS')  # Returns 'GSW'
    is_valid = validate_team_abbrev('GSW')  # Returns True
"""

from typing import Optional

# Non-standard → Standard team abbreviation mapping
TEAM_NORMALIZATION_MAP = {
    # Golden State Warriors
    "GS": "GSW",
    "GOLDEN STATE": "GSW",
    # New Orleans Pelicans
    "NO": "NOP",
    "NOR": "NOP",
    "NEW ORLEANS": "NOP",
    # San Antonio Spurs
    "SA": "SAS",
    "SAN ANTONIO": "SAS",
    # Washington Wizards
    "WSH": "WAS",
    "WASHINGTON": "WAS",
    # New York Knicks
    "NY": "NYK",
    "NEW YORK": "NYK",
    # Utah Jazz
    "UTAH": "UTA",
    "UTH": "UTA",
    # Empty string → NULL
    "": None,
}


# Full team name → Abbreviation mapping (for scraping sources like StatMuse)
TEAM_NAME_TO_ABBREV = {
    # Full names (lowercase for case-insensitive matching)
    "hawks": "ATL",
    "atlanta hawks": "ATL",
    "atlanta": "ATL",
    "nets": "BKN",
    "brooklyn nets": "BKN",
    "brooklyn": "BKN",
    "celtics": "BOS",
    "boston celtics": "BOS",
    "boston": "BOS",
    "hornets": "CHA",
    "charlotte hornets": "CHA",
    "charlotte": "CHA",
    "bulls": "CHI",
    "chicago bulls": "CHI",
    "chicago": "CHI",
    "cavaliers": "CLE",
    "cleveland cavaliers": "CLE",
    "cleveland": "CLE",
    "cavs": "CLE",
    "mavericks": "DAL",
    "dallas mavericks": "DAL",
    "dallas": "DAL",
    "mavs": "DAL",
    "nuggets": "DEN",
    "denver nuggets": "DEN",
    "denver": "DEN",
    "pistons": "DET",
    "detroit pistons": "DET",
    "detroit": "DET",
    "warriors": "GSW",
    "golden state warriors": "GSW",
    "golden state": "GSW",
    "rockets": "HOU",
    "houston rockets": "HOU",
    "houston": "HOU",
    "pacers": "IND",
    "indiana pacers": "IND",
    "indiana": "IND",
    "clippers": "LAC",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "lakers": "LAL",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",
    "grizzlies": "MEM",
    "memphis grizzlies": "MEM",
    "memphis": "MEM",
    "heat": "MIA",
    "miami heat": "MIA",
    "miami": "MIA",
    "bucks": "MIL",
    "milwaukee bucks": "MIL",
    "milwaukee": "MIL",
    "timberwolves": "MIN",
    "minnesota timberwolves": "MIN",
    "minnesota": "MIN",
    "wolves": "MIN",
    "pelicans": "NOP",
    "new orleans pelicans": "NOP",
    "new orleans": "NOP",
    "knicks": "NYK",
    "new york knicks": "NYK",
    "new york": "NYK",
    "thunder": "OKC",
    "oklahoma city thunder": "OKC",
    "oklahoma city": "OKC",
    "magic": "ORL",
    "orlando magic": "ORL",
    "orlando": "ORL",
    "76ers": "PHI",
    "philadelphia 76ers": "PHI",
    "philadelphia": "PHI",
    "sixers": "PHI",
    "suns": "PHX",
    "phoenix suns": "PHX",
    "phoenix": "PHX",
    "trail blazers": "POR",
    "portland trail blazers": "POR",
    "portland": "POR",
    "blazers": "POR",
    "kings": "SAC",
    "sacramento kings": "SAC",
    "sacramento": "SAC",
    "spurs": "SAS",
    "san antonio spurs": "SAS",
    "san antonio": "SAS",
    "raptors": "TOR",
    "toronto raptors": "TOR",
    "toronto": "TOR",
    "jazz": "UTA",
    "utah jazz": "UTA",
    "utah": "UTA",
    "wizards": "WAS",
    "washington wizards": "WAS",
    "washington": "WAS",
}


# Valid NBA team abbreviations (NBA.com official format)
VALID_NBA_TEAMS = {
    "ATL",  # Atlanta Hawks
    "BKN",  # Brooklyn Nets
    "BOS",  # Boston Celtics
    "CHA",  # Charlotte Hornets
    "CHI",  # Chicago Bulls
    "CLE",  # Cleveland Cavaliers
    "DAL",  # Dallas Mavericks
    "DEN",  # Denver Nuggets
    "DET",  # Detroit Pistons
    "GSW",  # Golden State Warriors
    "HOU",  # Houston Rockets
    "IND",  # Indiana Pacers
    "LAC",  # LA Clippers
    "LAL",  # Los Angeles Lakers
    "MEM",  # Memphis Grizzlies
    "MIA",  # Miami Heat
    "MIL",  # Milwaukee Bucks
    "MIN",  # Minnesota Timberwolves
    "NOP",  # New Orleans Pelicans
    "NYK",  # New York Knicks
    "OKC",  # Oklahoma City Thunder
    "ORL",  # Orlando Magic
    "PHI",  # Philadelphia 76ers
    "PHX",  # Phoenix Suns
    "POR",  # Portland Trail Blazers
    "SAC",  # Sacramento Kings
    "SAS",  # San Antonio Spurs
    "TOR",  # Toronto Raptors
    "UTA",  # Utah Jazz
    "WAS",  # Washington Wizards
}


def normalize_team_abbrev(abbrev: Optional[str]) -> Optional[str]:
    """
    Normalize team abbreviation to canonical NBA.com format.

    Handles common variations like:
    - GS → GSW
    - NO/NOR → NOP
    - SA → SAS
    - WSH → WAS
    - NY → NYK
    - UTAH → UTA
    - Empty string → None

    Args:
        abbrev: Team abbreviation (may be non-standard or None)

    Returns:
        Canonical 3-letter team abbreviation or None

    Examples:
        >>> normalize_team_abbrev('GS')
        'GSW'
        >>> normalize_team_abbrev('WSH')
        'WAS'
        >>> normalize_team_abbrev('LAL')
        'LAL'
        >>> normalize_team_abbrev('')
        None
        >>> normalize_team_abbrev(None)
        None
    """
    if not abbrev or abbrev == "":
        return None

    # Convert to uppercase for case-insensitive matching
    abbrev = abbrev.upper().strip()

    # Apply normalization map
    normalized = TEAM_NORMALIZATION_MAP.get(abbrev, abbrev)

    # Return normalized value (may be None if empty string was mapped)
    return normalized


def validate_team_abbrev(abbrev: Optional[str]) -> bool:
    """
    Check if team abbreviation is valid (canonical format).

    Args:
        abbrev: Team abbreviation to validate

    Returns:
        True if abbreviation is in canonical format, False otherwise

    Examples:
        >>> validate_team_abbrev('GSW')
        True
        >>> validate_team_abbrev('GS')
        False
        >>> validate_team_abbrev('LAL')
        True
        >>> validate_team_abbrev('LAKERS')
        False
        >>> validate_team_abbrev(None)
        False
    """
    if not abbrev:
        return False

    return abbrev.upper().strip() in VALID_NBA_TEAMS


def get_all_valid_teams() -> set:
    """
    Get set of all valid NBA team abbreviations.

    Returns:
        Set of 30 NBA team abbreviations in canonical format
    """
    return VALID_NBA_TEAMS.copy()


def team_name_to_abbrev(name: Optional[str]) -> Optional[str]:
    """
    Convert team name (full or partial) to canonical abbreviation.

    Handles:
    - Full names: "Boston Celtics" → "BOS"
    - Short names: "Celtics" → "BOS"
    - City names: "Boston" → "BOS"
    - Already abbreviations: "BOS" → "BOS"

    Args:
        name: Team name or abbreviation

    Returns:
        Canonical 3-letter team abbreviation or None if not recognized

    Examples:
        >>> team_name_to_abbrev('Celtics')
        'BOS'
        >>> team_name_to_abbrev('Boston Celtics')
        'BOS'
        >>> team_name_to_abbrev('BOS')
        'BOS'
        >>> team_name_to_abbrev('Unknown Team')
        None
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # Check if it's already a valid abbreviation
    if name.upper() in VALID_NBA_TEAMS:
        return name.upper()

    # Check abbreviation normalization map (GS→GSW, etc.)
    normalized = TEAM_NORMALIZATION_MAP.get(name.upper())
    if normalized and normalized in VALID_NBA_TEAMS:
        return normalized

    # Check full name mapping
    if name_lower in TEAM_NAME_TO_ABBREV:
        return TEAM_NAME_TO_ABBREV[name_lower]

    # Partial match - check if any key is contained in the name
    for key, abbrev in TEAM_NAME_TO_ABBREV.items():
        if key in name_lower:
            return abbrev

    return None


# For backward compatibility with existing code
def get_team_mapping() -> dict:
    """Get the team normalization mapping dictionary."""
    return TEAM_NORMALIZATION_MAP.copy()
