#!/usr/bin/env python3
"""
Universal Player Name Normalizer - ONE SOURCE OF TRUTH
=======================================================
Import this module EVERYWHERE to ensure consistent name handling across:
- MongoDB migrations
- PostgreSQL queries
- Feature extractors
- Dataset builders
- Training pipelines
- Production inference

Usage:
    from nba.utils.name_normalizer import NameNormalizer

    normalizer = NameNormalizer()
    player_id = normalizer.get_canonical_player_id("Nikola Jokić")
    normalized = normalizer.normalize_name("P.J. Washington Jr.")

Author: Universal Name Normalization System
Date: 2025-11-09
"""

import os
import pickle
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2


class NameNormalizer:
    """
    Universal player name normalizer with caching.

    Handles:
    - Accented characters (Jokić → Jokic)
    - Suffixes (Jr., Sr., II, III, IV)
    - Nicknames (Nicolas -> Nic, Carlton -> Bub)
    - Duplicate player_ids (picks one with most game logs)
    - Caching for performance
    """

    # Nickname mappings: full name -> common name used by BettingPros
    # Maps AFTER normalization is applied
    NICKNAME_MAP = {
        "Nicolas Claxton": "Nic Claxton",
        "Carlton Carrington": "Bub Carrington",
        "Moritz Wagner": "Moe Wagner",
        "Mohamed Bamba": "Mo Bamba",
        "Luguentz Dort": "Lu Dort",
        "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
    }

    # Database config (connects to player_profile database)
    DB_CONFIG = {
        "host": "localhost",
        "port": 5536,  # nba_players database (where player_profile is located)
        "database": "nba_players",
        "user": os.getenv("DB_USER", "mlb_user"),
        "password": os.getenv("DB_PASSWORD"),
    }

    # Cache file location
    CACHE_FILE = Path(__file__).parent / ".name_normalizer_cache.pkl"
    CACHE_MAX_AGE_HOURS = 24  # Rebuild cache daily

    def __init__(self, use_cache: bool = True):
        """
        Initialize name normalizer.

        Args:
            use_cache: Use cached mapping if available (default: True)
        """
        self.use_cache = use_cache
        self.mapping: Dict[str, Dict] = (
            {}
        )  # normalized_name → {player_id, game_count, original_names}
        self.reverse_mapping: Dict[str, str] = {}  # original_name → normalized_name

        # Load mapping
        self._load_mapping()

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize player name to canonical form.

        Transformations:
        1. Convert to title case (nikola jokic → Nikola Jokic)
        2. Remove accents (Jokić → Jokic)
        3. Remove suffixes (Jr., Sr., II, III, IV)
        4. Remove periods from initials (P.J. → PJ)
        5. Normalize whitespace

        Examples:
            >>> NameNormalizer.normalize_name("Nikola Jokić")
            'Nikola Jokic'
            >>> NameNormalizer.normalize_name("nikola jokic")
            'Nikola Jokic'
            >>> NameNormalizer.normalize_name("P.J. Washington Jr.")
            'PJ Washington'
            >>> NameNormalizer.normalize_name("Russell Westbrook III")
            'Russell Westbrook'

        Args:
            name: Player name

        Returns:
            Normalized name
        """
        if not name:
            return ""

        # Remove periods first (before title casing to preserve initials like PJ, TJ)
        name = name.strip().replace(".", "")

        # Convert to title case (now "PJ Washington" stays "Pj Washington" temporarily)
        name = name.title()

        # Fix common initials that .title() breaks (PJ, TJ, AJ, etc.)
        # These should be all-caps
        initials = ["Pj ", "Tj ", "Cj ", "Aj ", "Rj ", "Jj ", "Dj ", "Og ", "Vj "]
        for initial in initials:
            if name.startswith(initial):
                name = initial.upper() + name[3:]  # Replace 'Pj ' with 'PJ '

        # Fix mixed-case names that .title() breaks (DeMar, DeRozan, LaMelo, LaRavia, etc.)
        mixed_case_fixes = {
            "Demar": "DeMar",
            "Derozan": "DeRozan",
            "Deandre": "DeAndre",
            "Deaaron": "De'Aaron",
            "Lamelo": "LaMelo",
            "Lonnie": "Lonnie",
            "Laravia": "LaRavia",
            "Lavine": "LaVine",
            "Lebron": "LeBron",
            "Mcconnell": "McConnell",
            "Mccollum": "McCollum",
            "Mcdaniels": "McDaniels",
            "Mcgee": "McGee",
            "Mckinnie": "McKinnie",
            "Day'Ron": "Day'Ron",
            "Dayron": "Day'Ron",
            "O'Neale": "O'Neale",
            "Oneale": "O'Neale",
        }
        for wrong, correct in mixed_case_fixes.items():
            if wrong in name:
                name = name.replace(wrong, correct)

        # Remove accents (ş → s, ć → c, etc.)
        nfd = unicodedata.normalize("NFD", name)
        name = "".join(char for char in nfd if unicodedata.category(char) != "Mn")

        # Remove suffixes (Jr, Sr, II, III, IV, V - with or without periods)
        # Check both with and without periods since we already removed them
        suffixes = [" Jr", " Sr", " Ii", " Iii", " Iv", " V"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Normalize whitespace
        name = " ".join(name.split())

        name = name.strip()

        # Apply nickname mappings
        if name in NameNormalizer.NICKNAME_MAP:
            name = NameNormalizer.NICKNAME_MAP[name]

        return name

    def _load_mapping(self):
        """Load player name mapping from cache or rebuild from database."""
        # Try to load from cache
        if self.use_cache and self._load_from_cache():
            return

        # Build from database
        print("Building name normalization mapping from database...")
        self._build_from_database()

        # Save cache
        if self.use_cache:
            self._save_to_cache()

    def _load_from_cache(self) -> bool:
        """Load mapping from cache file if fresh enough."""
        if not self.CACHE_FILE.exists():
            return False

        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(self.CACHE_FILE.stat().st_mtime)
            if cache_age > timedelta(hours=self.CACHE_MAX_AGE_HOURS):
                print(f"Cache is {cache_age.total_seconds()/3600:.1f} hours old, rebuilding...")
                return False

            # Load cache
            with open(self.CACHE_FILE, "rb") as f:
                data = pickle.load(f)

            self.mapping = data["mapping"]
            self.reverse_mapping = data["reverse_mapping"]

            print(f"✓ Loaded name mapping from cache ({len(self.mapping):,} normalized names)")
            return True

        except Exception as e:
            print(f"Error loading cache: {e}, rebuilding...")
            return False

    def _save_to_cache(self):
        """Save mapping to cache file."""
        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "mapping": self.mapping,
                        "reverse_mapping": self.reverse_mapping,
                        "created_at": datetime.now().isoformat(),
                    },
                    f,
                )
            print(f"✓ Saved name mapping to cache")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _build_from_database(self):
        """
        Build name mapping from PostgreSQL.

        For each normalized name:
        - Find all player_ids with that name
        - Pick the one with MOST game logs as canonical
        """
        conn = psycopg2.connect(**self.DB_CONFIG)
        cursor = conn.cursor()

        try:
            # Query all players with game log counts
            query = """
            SELECT
                pp.full_name,
                pp.player_id,
                COALESCE(game_counts.game_count, 0) as game_count
            FROM player_profile pp
            LEFT JOIN (
                SELECT player_id, COUNT(*) as game_count
                FROM player_game_logs
                GROUP BY player_id
            ) game_counts ON pp.player_id = game_counts.player_id
            ORDER BY pp.full_name, game_count DESC
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            # Group by normalized name
            from collections import defaultdict

            name_groups = defaultdict(list)

            for full_name, player_id, game_count in rows:
                normalized = self.normalize_name(full_name)
                name_groups[normalized].append(
                    {"original_name": full_name, "player_id": player_id, "game_count": game_count}
                )

            # Build mapping - pick player_id with most game logs
            for normalized_name, players in name_groups.items():
                # Sort by game_count descending
                players_sorted = sorted(players, key=lambda p: p["game_count"], reverse=True)
                canonical = players_sorted[0]

                # Store mapping
                self.mapping[normalized_name] = {
                    "player_id": canonical["player_id"],
                    "game_count": canonical["game_count"],
                    "original_names": [p["original_name"] for p in players],
                    "all_player_ids": [p["player_id"] for p in players],
                    "has_duplicates": len(players) > 1,
                }

                # Build reverse mapping (all original names point to normalized)
                for player in players:
                    self.reverse_mapping[player["original_name"]] = normalized_name

            print(
                f"✓ Built mapping for {len(self.mapping):,} normalized names from {len(rows):,} database entries"
            )

            # Show some statistics
            duplicates = sum(1 for m in self.mapping.values() if m["has_duplicates"])
            if duplicates > 0:
                print(f"  → Resolved {duplicates:,} names with duplicate player_ids")

        finally:
            cursor.close()
            conn.close()

    def get_canonical_player_id(self, player_name: str) -> Optional[int]:
        """
        Get the canonical player_id for a player name.

        Handles all variations automatically:
        - "Nikola Jokić" → 203999 (most game logs)
        - "Nikola Jokic" → 203999 (normalized)
        - "P.J. Washington Jr." → correct player_id

        Args:
            player_name: Player name (any variation)

        Returns:
            Canonical player_id or None if not found
        """
        normalized = self.normalize_name(player_name)

        if normalized in self.mapping:
            return self.mapping[normalized]["player_id"]

        return None

    def get_player_info(self, player_name: str) -> Optional[Dict]:
        """
        Get full player information.

        Args:
            player_name: Player name

        Returns:
            Dict with player_id, game_count, original_names, etc. or None
        """
        normalized = self.normalize_name(player_name)
        return self.mapping.get(normalized)

    def has_duplicates(self, player_name: str) -> bool:
        """Check if a player name has duplicate player_ids in database."""
        info = self.get_player_info(player_name)
        return info["has_duplicates"] if info else False

    def get_all_variations(self, player_name: str) -> List[str]:
        """Get all name variations for a player."""
        info = self.get_player_info(player_name)
        return info["original_names"] if info else []

    def rebuild_cache(self):
        """Force rebuild of cache from database."""
        print("Forcing cache rebuild...")
        self._build_from_database()
        if self.use_cache:
            self._save_to_cache()

    def print_stats(self):
        """Print mapping statistics."""
        print("\n" + "=" * 80)
        print("NAME NORMALIZER STATISTICS")
        print("=" * 80)
        print(f"Total normalized names: {len(self.mapping):,}")
        print(f"Total original names: {len(self.reverse_mapping):,}")

        duplicates = sum(1 for m in self.mapping.values() if m["has_duplicates"])
        print(f"Names with duplicates: {duplicates:,}")

        # Show some examples
        print("\nDuplicate Examples (canonical player_id selected):")
        dup_examples = [(k, v) for k, v in self.mapping.items() if v["has_duplicates"]]
        dup_examples.sort(key=lambda x: x[1]["game_count"], reverse=True)

        for norm_name, info in dup_examples[:10]:
            print(
                f"  {norm_name:30s} → player_id={info['player_id']:8d} "
                f"({info['game_count']:4d} games, {len(info['all_player_ids'])} IDs)"
            )

        print("=" * 80 + "\n")


# Global singleton instance
_normalizer_instance = None


def get_normalizer(use_cache: bool = True, rebuild: bool = False) -> NameNormalizer:
    """
    Get global NameNormalizer singleton instance.

    Args:
        use_cache: Use cached mapping (default: True)
        rebuild: Force rebuild from database (default: False)

    Returns:
        NameNormalizer instance
    """
    global _normalizer_instance

    if _normalizer_instance is None or rebuild:
        _normalizer_instance = NameNormalizer(use_cache=use_cache)
        if rebuild:
            _normalizer_instance.rebuild_cache()

    return _normalizer_instance


# Convenience functions for direct use
def normalize_name(name: str) -> str:
    """Normalize a player name (static method, no database required)."""
    return NameNormalizer.normalize_name(name)


def get_canonical_player_id(player_name: str) -> Optional[int]:
    """Get canonical player_id for a player name."""
    return get_normalizer().get_canonical_player_id(player_name)


def get_player_info(player_name: str) -> Optional[Dict]:
    """Get full player information."""
    return get_normalizer().get_player_info(player_name)


if __name__ == "__main__":
    # Test the normalizer
    normalizer = NameNormalizer()
    normalizer.print_stats()

    # Test some examples
    print("Testing name normalization:")
    test_names = [
        "Nikola Jokić",
        "Nikola Jokic",
        "Luka Dončić",
        "P.J. Washington Jr.",
        "Russell Westbrook III",
        "Mikal Bridges",
    ]

    for name in test_names:
        normalized = normalize_name(name)
        player_id = get_canonical_player_id(name)
        info = get_player_info(name)

        if info:
            print(f"\n'{name}':")
            print(f"  Normalized: {normalized}")
            print(f"  Canonical player_id: {player_id}")
            print(f"  Game logs: {info['game_count']}")
            print(f"  Has duplicates: {info['has_duplicates']}")
            if info["has_duplicates"]:
                print(f"  All player_ids: {info['all_player_ids']}")
        else:
            print(f"\n'{name}': NOT FOUND")
