#!/usr/bin/env python3
"""
Base Fetcher for NBA XL System
================================
Abstract base class for all multi-source prop fetchers.

Features:
- Rate limiting (configurable delay between requests)
- Retry logic with exponential backoff
- User agent rotation
- Error handling and logging
- Standard output format (JSON)
- Response caching

All XL fetchers inherit from this class.
"""

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

EST = ZoneInfo("America/New_York")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for all XL prop fetchers"""

    # User agents to rotate
    USER_AGENTS = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    def __init__(
        self,
        source_name: str,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        verbose: bool = True,
        proxy_profile: Optional[str] = None,
    ):
        """
        Initialize base fetcher.

        Args:
            source_name: Name of the source (e.g., 'bettingpros_all_books')
            rate_limit: Minimum delay between requests in seconds
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            proxy_profile: Proxy profile name ('sportsbooks' for Colorado,
                          'prizepicks' for Florida). None = no proxy.
        """
        self.source_name = source_name
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.proxy_profile = proxy_profile

        # Track request timing for rate limiting
        self.last_request_time = 0

        # Session for connection pooling
        self.session = requests.Session()

        # Configure proxy on session if profile specified
        if proxy_profile:
            from nba.betting_xl.fetchers.proxy_manager import get_proxy_manager

            pm = get_proxy_manager()
            proxies = pm.get_proxies_dict(proxy_profile)
            if proxies:
                self.session.proxies.update(proxies)
                if verbose:
                    state = pm.PROFILE_STATES.get(proxy_profile, proxy_profile)
                    logger.info(f"[{source_name}] Using {state} proxy")

        # Output directory
        self.output_dir = Path(__file__).parent.parent / "lines"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atlas data registry — tracks API calls and bytes for ingestion auditing
        self._api_calls = 0
        self._bytes_transferred = 0
        self._error_count = 0
        self._last_error = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with random user agent"""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

    def _enforce_rate_limit(self):
        """Enforce rate limit between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            if self.verbose:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and exponential backoff.

        Args:
            url: URL to fetch
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: POST data
            headers: Additional headers (merged with default headers)

        Returns:
            Response object or None if all retries failed
        """
        # Merge headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        for attempt in range(self.max_retries):
            try:
                # Enforce rate limit
                self._enforce_rate_limit()

                # Make request
                if self.verbose:
                    logger.info(
                        f"[{self.source_name}] Request: {method} {url} (attempt {attempt + 1}/{self.max_retries})"
                    )

                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=request_headers,
                    timeout=self.timeout,
                )

                # Check status
                response.raise_for_status()

                # Track for Atlas data registry
                self._api_calls += 1
                self._bytes_transferred += len(response.content)

                if self.verbose:
                    logger.info(
                        f"[{self.source_name}] Success: {response.status_code} ({len(response.content)} bytes)"
                    )

                return response

            except requests.exceptions.HTTPError as e:
                self._api_calls += 1
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(f"[{self.source_name}] HTTP Error on attempt {attempt + 1}: {e}")

                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f"[{self.source_name}] Client error, not retrying: {e.response.status_code}"
                    )
                    return None

            except requests.exceptions.Timeout as e:
                self._api_calls += 1
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(f"[{self.source_name}] Timeout on attempt {attempt + 1}: {e}")

            except requests.exceptions.RequestException as e:
                self._api_calls += 1
                self._error_count += 1
                self._last_error = str(e)
                logger.warning(f"[{self.source_name}] Request error on attempt {attempt + 1}: {e}")

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff_time = 2**attempt + random.uniform(0, 1)
                logger.info(f"[{self.source_name}] Retrying in {backoff_time:.2f}s...")
                time.sleep(backoff_time)

        logger.error(f"[{self.source_name}] All retry attempts failed for {url}")
        return None

    @abstractmethod
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch props from source.

        Must be implemented by subclasses.

        Returns:
            List of prop dictionaries in standardized format
        """
        pass

    def normalize_player_name(self, name: str) -> str:
        """
        Normalize player name to standard format.

        Args:
            name: Raw player name

        Returns:
            Normalized name (title case, stripped)
        """
        if not name:
            return ""

        # Strip whitespace
        name = name.strip()

        # Handle common variations
        name = name.replace(".", "")  # Remove periods (J.R. Smith -> JR Smith)
        name = name.replace("'", "'")  # Standardize apostrophes

        # Title case (handles "LEBRON JAMES" -> "Lebron James")
        # But preserves "DeRozan", "McCollum" if properly formatted
        if name.isupper() or name.islower():
            name = name.title()

        return name

    def normalize_stat_type(self, stat_type: str) -> str:
        """
        Normalize stat type to canonical format.

        Args:
            stat_type: Raw stat type string

        Returns:
            Canonical stat type (POINTS, REBOUNDS, ASSISTS, THREES)
        """
        stat_type = stat_type.lower().strip()

        # Mapping dictionary
        stat_map = {
            "points": "POINTS",
            "pts": "POINTS",
            "point": "POINTS",
            "rebounds": "REBOUNDS",
            "rebound": "REBOUNDS",
            "reb": "REBOUNDS",
            "rebs": "REBOUNDS",
            "assists": "ASSISTS",
            "assist": "ASSISTS",
            "ast": "ASSISTS",
            "threes": "THREES",
            "three": "THREES",
            "3-pt made": "THREES",
            "3pm": "THREES",
            "3 pointers made": "THREES",
            "three pointers made": "THREES",
            "steals": "STEALS",
            "steal": "STEALS",
            "stl": "STEALS",
            "blocks": "BLOCKS",
            "block": "BLOCKS",
            "blk": "BLOCKS",
            "blocked shots": "BLOCKS",
            "turnovers": "TURNOVERS",
            "turnover": "TURNOVERS",
            "to": "TURNOVERS",
        }

        return stat_map.get(stat_type, stat_type.upper())

    def save_to_json(self, props: List[Dict[str, Any]], suffix: str = "") -> Path:
        """
        Save props to timestamped JSON file.

        Args:
            props: List of prop dictionaries
            suffix: Optional suffix for filename (e.g., "all_books")

        Returns:
            Path to saved file
        """
        timestamp = datetime.now(EST).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.source_name}_{timestamp}"
        if suffix:
            filename = f"{filename}_{suffix}"
        filename = f"{filename}.json"

        output_file = self.output_dir / filename

        output_data = {
            "source": self.source_name,
            "fetch_timestamp": datetime.now(EST).isoformat(),
            "total_props": len(props),
            "props": props,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.verbose:
            logger.info(f"[{self.source_name}] Saved {len(props)} props to: {output_file}")

        return output_file

    def validate_prop(self, prop: Dict[str, Any]) -> bool:
        """
        Validate that a prop has required fields.

        Args:
            prop: Prop dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["player_name", "stat_type", "line", "book_name"]

        for field in required_fields:
            if field not in prop or prop[field] is None:
                if self.verbose:
                    logger.warning(f"[{self.source_name}] Invalid prop: missing {field}")
                return False

        # Validate line is numeric and positive
        try:
            line_val = float(prop["line"])
        except (ValueError, TypeError):
            if self.verbose:
                logger.warning(
                    f"[{self.source_name}] Invalid prop: line not numeric: {prop.get('line')}"
                )
            return False

        if line_val <= 0:
            return False

        return True

    def deduplicate_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate props (same player, stat, book, line).

        Args:
            props: List of prop dictionaries

        Returns:
            Deduplicated list
        """
        seen = set()
        deduped = []

        for prop in props:
            key = (
                prop.get("player_name", "").lower(),
                prop.get("stat_type", "").upper(),
                prop.get("book_name", "").lower(),
                float(prop.get("line", 0)),
            )

            if key not in seen:
                seen.add(key)
                deduped.append(prop)

        if len(props) != len(deduped):
            logger.info(f"[{self.source_name}] Removed {len(props) - len(deduped)} duplicate props")

        return deduped

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get accumulated stats for Atlas data registry logging.

        Returns dict compatible with IngestionTracker / log_ingestion().
        """
        return {
            "api_calls_made": self._api_calls,
            "bytes_transferred": self._bytes_transferred,
            "error_count": self._error_count,
            "error_message": self._last_error,
        }

    def reset_registry_stats(self):
        """Reset accumulated stats (call between logical operations)."""
        self._api_calls = 0
        self._bytes_transferred = 0
        self._error_count = 0
        self._last_error = None

    def close(self):
        """Close session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
