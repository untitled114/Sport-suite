#!/usr/bin/env python3
"""
Proxy Manager for Direct Sportsbook Fetching
==============================================
Routes requests through state-specific residential proxies via IPRoyal.

Two proxy profiles:
- prizepicks: Florida IP (PrizePicks geo-compliance)
- sportsbooks: Colorado IP (all Colorado-legal sportsbooks + DFS)

Usage:
    from nba.betting_xl.fetchers.proxy_manager import ProxyManager

    pm = ProxyManager()
    proxies = pm.get_proxies_dict("sportsbooks")
    response = requests.get(url, proxies=proxies)
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProxyManager:
    """Multi-state proxy routing for sportsbook API access."""

    # Maps profile name -> env var holding the proxy URL
    PROFILES = {
        "prizepicks": "PRIZEPICKS_PROXY_URL",
        "sportsbooks": "SPORTSBOOK_PROXY_URL",
    }

    # Default states for documentation / logging
    PROFILE_STATES = {
        "prizepicks": "Florida",
        "sportsbooks": "Colorado",
    }

    def __init__(self):
        """Initialize proxy manager. Reads proxy URLs from environment."""
        self._cache: Dict[str, Optional[str]] = {}

    def get_proxy(self, profile: str) -> Optional[str]:
        """
        Get proxy URL for a given profile.

        Args:
            profile: One of 'prizepicks' or 'sportsbooks'

        Returns:
            Proxy URL string, or None if not configured
        """
        if profile not in self.PROFILES:
            logger.warning(
                f"Unknown proxy profile: {profile}. Available: {list(self.PROFILES.keys())}"
            )
            return None

        if profile not in self._cache:
            env_var = self.PROFILES[profile]
            url = os.getenv(env_var, "")
            self._cache[profile] = url if url else None

            if url:
                state = self.PROFILE_STATES.get(profile, "unknown")
                logger.debug(f"Proxy loaded for {profile} ({state})")
            else:
                logger.debug(f"No proxy configured for {profile} (env: {env_var})")

        return self._cache[profile]

    def get_proxies_dict(self, profile: str) -> Dict[str, str]:
        """
        Get proxies dict for requests/curl_cffi (both http and https).

        Args:
            profile: One of 'prizepicks' or 'sportsbooks'

        Returns:
            Dict like {"http": url, "https": url}, or empty dict if not configured
        """
        url = self.get_proxy(profile)
        if not url:
            return {}
        return {"http": url, "https": url}

    def is_configured(self, profile: str) -> bool:
        """Check if a proxy profile has a URL configured."""
        return bool(self.get_proxy(profile))

    def get_status(self) -> Dict[str, bool]:
        """Get configuration status of all proxy profiles."""
        return {profile: self.is_configured(profile) for profile in self.PROFILES}


# Module-level singleton for convenience
_manager: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """Get or create the singleton ProxyManager instance."""
    global _manager
    if _manager is None:
        _manager = ProxyManager()
    return _manager
