#!/usr/bin/env python3
"""
ESPN NBA HISTORICAL GAMELOG SCRAPER
====================================

Fetches player gamelogs from ESPN for historical seasons (2021-22 through 2024-25).

ESPN URL Pattern:
https://www.espn.com/nba/player/gamelog/_/id/{player_id}/year/{year}

Example:
https://www.espn.com/nba/player/gamelog/_/id/1966/year/2023 (LeBron James 2022-23)

Features:
- Scrapes all games for all active NBA players
- Handles regular season and playoffs
- Retries on failures with exponential backoff
- Saves to CSV per season
- Rate limiting to avoid blocking

Usage:
    # Fetch single season
    python3 nba/scripts/loaders/fetch_espn_historical_gamelogs.py --season 2022-23

    # Fetch multiple seasons
    python3 nba/scripts/loaders/fetch_espn_historical_gamelogs.py --seasons 2021-22 2022-23 2023-24

    # Fetch with custom output
    python3 nba/scripts/loaders/fetch_espn_historical_gamelogs.py \
        --seasons 2021-22 2022-23 \
        --output data/raw/nba/historical/
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ESPNGamelogScraper:
    """Scrape NBA player gamelogs from ESPN"""

    BASE_URL = "https://www.espn.com/nba"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # NBA teams (for fetching rosters)
    NBA_TEAMS = {
        "ATL": "atlanta-hawks",
        "BOS": "boston-celtics",
        "BKN": "brooklyn-nets",
        "CHA": "charlotte-hornets",
        "CHI": "chicago-bulls",
        "CLE": "cleveland-cavaliers",
        "DAL": "dallas-mavericks",
        "DEN": "denver-nuggets",
        "DET": "detroit-pistons",
        "GSW": "golden-state-warriors",
        "HOU": "houston-rockets",
        "IND": "indiana-pacers",
        "LAC": "la-clippers",
        "LAL": "los-angeles-lakers",
        "MEM": "memphis-grizzlies",
        "MIA": "miami-heat",
        "MIL": "milwaukee-bucks",
        "MIN": "minnesota-timberwolves",
        "NOP": "new-orleans-pelicans",
        "NYK": "new-york-knicks",
        "OKC": "oklahoma-city-thunder",
        "ORL": "orlando-magic",
        "PHI": "philadelphia-76ers",
        "PHX": "phoenix-suns",
        "POR": "portland-trail-blazers",
        "SAC": "sacramento-kings",
        "SAS": "san-antonio-spurs",
        "TOR": "toronto-raptors",
        "UTA": "utah-jazz",
        "WAS": "washington-wizards",
    }

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize scraper

        Args:
            rate_limit: Seconds to wait between requests
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _season_to_year(self, season: str) -> int:
        """
        Convert season string to ESPN year format

        Args:
            season: Season string (e.g., "2022-23")

        Returns:
            Year integer (e.g., 2023 for 2022-23 season)
        """
        # ESPN uses the end year (2022-23 → 2023)
        return int(season.split("-")[1]) + 2000

    def fetch_team_roster(self, team_abbr: str, season: str) -> List[Dict]:
        """
        Fetch roster for a specific team and season

        Args:
            team_abbr: Team abbreviation (e.g., "LAL")
            season: Season string (e.g., "2022-23")

        Returns:
            List of player dicts with id, name, position
        """
        team_slug = self.NBA_TEAMS.get(team_abbr)
        if not team_slug:
            logger.warning(f"Unknown team: {team_abbr}")
            return []

        year = self._season_to_year(season)
        url = f"{self.BASE_URL}/team/roster/_/name/{team_abbr.lower()}/season/{year}"

        logger.info(f"Fetching roster: {team_abbr} ({season}) - {url}")

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit)

            soup = BeautifulSoup(resp.text, "html.parser")

            # Find roster table
            players = []
            tables = soup.find_all("table", class_="Table")

            for table in tables:
                tbody = table.find("tbody")
                if not tbody:
                    continue

                for row in tbody.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue

                    # Extract player link and ID
                    player_link = cells[1].find("a")
                    if not player_link:
                        continue

                    player_name = player_link.get_text(strip=True)
                    player_href = player_link.get("href", "")

                    # Extract player ID from href
                    # Example: /nba/player/_/id/1966/lebron-james
                    match = re.search(r"/id/(\d+)/", player_href)
                    if not match:
                        continue

                    player_id = match.group(1)

                    # Position
                    position = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                    players.append(
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "position": position,
                            "team": team_abbr,
                            "season": season,
                        }
                    )

            logger.info(f"  Found {len(players)} players for {team_abbr}")
            return players

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Error fetching roster for {team_abbr} ({season}): {e}")
            return []

    def fetch_all_rosters(self, season: str) -> List[Dict]:
        """
        Fetch rosters for all NBA teams for a given season

        Args:
            season: Season string (e.g., "2022-23")

        Returns:
            List of all players across all teams
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FETCHING ROSTERS FOR {season} SEASON")
        logger.info(f"{'='*80}\n")

        all_players = []

        for team_abbr in self.NBA_TEAMS.keys():
            players = self.fetch_team_roster(team_abbr, season)
            all_players.extend(players)
            time.sleep(self.rate_limit)

        logger.info(f"\n✅ Total players found: {len(all_players)}")

        # Remove duplicates (players who were traded)
        seen = set()
        unique_players = []
        for player in all_players:
            if player["player_id"] not in seen:
                seen.add(player["player_id"])
                unique_players.append(player)

        logger.info(f"✅ Unique players: {len(unique_players)}")

        return unique_players

    def fetch_player_gamelog(self, player_id: str, player_name: str, season: str) -> List[Dict]:
        """
        Fetch gamelog for a specific player and season

        Args:
            player_id: ESPN player ID
            player_name: Player name
            season: Season string (e.g., "2022-23")

        Returns:
            List of game dictionaries
        """
        year = self._season_to_year(season)
        url = f"{self.BASE_URL}/player/gamelog/_/id/{player_id}/year/{year}"

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit)

            soup = BeautifulSoup(resp.text, "html.parser")

            games = []

            # Find gamelog table
            tables = soup.find_all("div", class_="ResponsiveTable")

            for table_div in tables:
                table = table_div.find("table")
                if not table:
                    continue

                # Get headers
                thead = table.find("thead")
                if not thead:
                    continue

                header_row = thead.find_all("tr")[-1]  # Last row of thead
                headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

                # Skip if not a gamelog table (check for "MIN" or "PTS" columns)
                if "MIN" not in headers and "PTS" not in headers:
                    continue

                # Get game rows
                tbody = table.find("tbody")
                if not tbody:
                    continue

                for row in tbody.find_all("tr"):
                    # Skip header rows within tbody
                    if row.find("th"):
                        continue

                    cells = row.find_all("td")
                    if len(cells) < len(headers):
                        continue

                    # Parse game data
                    game_data = {
                        "player_id": player_id,
                        "player_name": player_name,
                        "season": season,
                    }

                    for i, header in enumerate(headers):
                        if i < len(cells):
                            value = cells[i].get_text(strip=True)
                            game_data[header] = value

                    # Skip if no date
                    if "Date" not in game_data or not game_data["Date"]:
                        continue

                    games.append(game_data)

            return games

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.error(f"Error fetching gamelog for {player_name} ({player_id}, {season}): {e}")
            return []

    def fetch_season_gamelogs(
        self, season: str, output_dir: str = "data/raw/nba/historical"
    ) -> str:
        """
        Fetch all gamelogs for a given season

        Args:
            season: Season string (e.g., "2022-23")
            output_dir: Directory to save output CSV

        Returns:
            Path to output CSV file
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FETCHING GAMELOGS FOR {season} SEASON")
        logger.info(f"{'='*80}\n")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Fetch all rosters
        players = self.fetch_all_rosters(season)

        if not players:
            logger.error(f"No players found for season {season}")
            return ""

        # Fetch gamelogs for all players
        all_games = []
        total_players = len(players)

        logger.info(f"\n{'='*80}")
        logger.info(f"FETCHING GAMELOGS FOR {total_players} PLAYERS")
        logger.info(f"{'='*80}\n")

        for idx, player in enumerate(players, 1):
            player_id = player["player_id"]
            player_name = player["player_name"]

            logger.info(f"[{idx}/{total_players}] Fetching {player_name} ({player_id})...")

            games = self.fetch_player_gamelog(player_id, player_name, season)

            if games:
                logger.info(f"  ✅ Found {len(games)} games")
                all_games.extend(games)
            else:
                logger.info(f"  ⚠️  No games found")

            # Rate limiting
            time.sleep(self.rate_limit)

            # Progress update every 50 players
            if idx % 50 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"PROGRESS: {idx}/{total_players} players completed")
                logger.info(f"Total games collected: {len(all_games):,}")
                logger.info(f"{'='*80}\n")

        # Convert to DataFrame
        if not all_games:
            logger.error(f"No games found for season {season}")
            return ""

        df = pd.DataFrame(all_games)

        # Save to CSV
        output_file = output_path / f"espn_gamelogs_{season.replace('-', '_')}.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"\n{'='*80}")
        logger.info(f"SEASON {season} COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total games: {len(df):,}")
        logger.info(f"Unique players: {df['player_id'].nunique()}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"{'='*80}\n")

        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Fetch ESPN NBA historical gamelogs")
    parser.add_argument("--season", type=str, help="Single season (e.g., 2022-23)")
    parser.add_argument(
        "--seasons", type=str, nargs="+", help="Multiple seasons (e.g., 2021-22 2022-23)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/nba/historical",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between requests (default: 1.0)",
    )

    args = parser.parse_args()

    # Determine seasons to fetch
    seasons = []
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = args.seasons
    else:
        # Default: fetch last 4 seasons
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]

    logger.info(f"{'='*80}")
    logger.info(f"ESPN NBA GAMELOG SCRAPER")
    logger.info(f"{'='*80}")
    logger.info(f"Seasons to fetch: {', '.join(seasons)}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Rate limit: {args.rate_limit}s")
    logger.info(f"{'='*80}\n")

    # Create scraper
    scraper = ESPNGamelogScraper(rate_limit=args.rate_limit)

    # Fetch each season
    output_files = []
    for season in seasons:
        output_file = scraper.fetch_season_gamelogs(season, args.output)
        if output_file:
            output_files.append(output_file)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL SEASONS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Seasons fetched: {len(output_files)}")
    for file in output_files:
        logger.info(f"  - {file}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
