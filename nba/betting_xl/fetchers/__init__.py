"""
NBA XL System - Multi-Source Prop Fetchers
============================================
Fetchers for props from multiple sportsbooks.

Available Fetchers:
- BettingProsFetcher: Fetch from BettingPros API (consensus lines)
- [TODO] AllBooksFetcher: Scrape all books from BettingPros website
- [TODO] UnderdogFetcher: Scrape Underdog cheat sheet
- [TODO] PrizePicksFetcher: Scrape PrizePicks cheat sheet

Usage:
    from nba.betting_xl.fetchers import BettingProsFetcher

    # Fetch today's props
    with BettingProsFetcher() as fetcher:
        props = fetcher.fetch()

    # Or use the orchestrator to fetch from all sources
    from nba.betting_xl.fetchers.fetch_all import FetchOrchestrator

    orchestrator = FetchOrchestrator()
    result = orchestrator.run()
"""

from .base_fetcher import BaseFetcher
from .fetch_all import FetchOrchestrator
from .fetch_bettingpros import BettingProsFetcher

__all__ = [
    "BaseFetcher",
    "BettingProsFetcher",
    "FetchOrchestrator",
]
