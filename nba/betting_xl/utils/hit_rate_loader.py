"""Helper to load BettingPros hit-rate data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class HitRateCache:
    """Loads hit-rate data saved by fetch_bettingpros_hit_rates."""

    def __init__(self, game_date: str, base_dir: Optional[Path] = None):
        self.game_date = game_date
        self.base_dir = base_dir or Path(__file__).parent.parent / "lines"
        self.records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._loaded = False

    def _normalize_key(self, player_name: str, stat_type: str) -> Tuple[str, str]:
        return (player_name.strip().lower(), stat_type.strip().upper())

    def load(self) -> None:
        if self._loaded:
            return

        file_path = self.base_dir / f"bettingpros_hit_rates_{self.game_date}.json"
        if not file_path.exists():
            self._loaded = True
            return

        try:
            with open(file_path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._loaded = True
            return

        for record in payload.get("records", []):
            player = record.get("player_name")
            stat_type = record.get("stat_type")
            if not player or not stat_type:
                continue
            key = self._normalize_key(player, stat_type)
            self.records[key] = record

        self._loaded = True

    def get(self, player_name: str, stat_type: str) -> Optional[Dict[str, Any]]:
        self.load()
        key = self._normalize_key(player_name, stat_type)
        return self.records.get(key)
