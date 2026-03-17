"""
NBA XL System - Multi-Source Prop Fetchers
============================================
Fetchers for props from multiple sportsbooks and DFS platforms.

Available Fetchers:
- BettingProsFetcher: BettingPros API (7+ sportsbooks via book_id)
- PrizePicksDirectFetcher: PrizePicks direct API (Florida proxy)
- DraftKingsDirectFetcher: DraftKings direct API (Colorado proxy)
- FanDuelDirectFetcher: FanDuel direct API (Colorado proxy)
- ESPNBetDirectFetcher: ESPNBet direct API (Colorado proxy)
- BetMGMDirectFetcher: BetMGM direct API (Colorado proxy)
- CaesarsDirectFetcher: Caesars direct API (Colorado proxy)
- BetRiversDirectFetcher: BetRivers direct API (Colorado proxy)
- FanaticsDirectFetcher: Fanatics direct API (Colorado proxy)
- Bet365DirectFetcher: bet365 direct API (Colorado proxy)
- HardRockDirectFetcher: Hard Rock Bet direct API (Colorado proxy)
- UnderdogDirectFetcher: Underdog Fantasy direct API (Colorado proxy)

Analytics:
- CheatSheetFetcher: BettingPros cheatsheet (projections, EV, hit rates)
- BettingProsHitRateFetcher: Consensus hit rates + streaks
- BettingProsTrendsFetcher: Line trends + streaks (analytics hub)

Usage:
    from nba.betting_xl.fetchers import BettingProsFetcher

    # Fetch today's props from BettingPros
    with BettingProsFetcher() as fetcher:
        props = fetcher.fetch()

    # Or use the orchestrator for all sources (BP + direct)
    from nba.betting_xl.fetchers.fetch_all import FetchOrchestrator

    orchestrator = FetchOrchestrator()
    result = orchestrator.run()
"""

from .base_fetcher import BaseFetcher
from .fetch_all import FetchOrchestrator
from .fetch_bettingpros import BettingProsFetcher
from .proxy_manager import ProxyManager, get_proxy_manager

__all__ = [
    "BaseFetcher",
    "BettingProsFetcher",
    "FetchOrchestrator",
    "ProxyManager",
    "get_proxy_manager",
]
