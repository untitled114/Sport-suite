#!/usr/bin/env python3
"""
NBA HISTORICAL PLAYER PROPS FETCHER

Fetches historical NBA player prop lines from Sports Game Odds API
Target props: Points, Rebounds, Assists, 3PM, PRA, P+R, P+A, R+A

Data available: 2024-25 season only (Oct 22, 2024 - present)
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("FANTASYPROS_API_KEY")
BASE_URL = "https://api.sportsgameodds.com/v2/events"


class NBAPropsFetcher:
    """Fetch historical NBA player props from Sports Game Odds API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"x-api-key": api_key}

    def fetch_date(self, date: str, max_retries: int = 5) -> pd.DataFrame:
        """
        Fetch all NBA player props for a specific date using cursor pagination

        Args:
            date: Date in YYYY-MM-DD format
            max_retries: Maximum number of retries for rate limit errors

        Returns:
            DataFrame with filtered props
        """
        # Calculate date range (start of day to end of day)
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        starts_after = date_obj.strftime("%Y-%m-%dT00:00:00Z")
        starts_before = (date_obj + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        all_props = []
        next_cursor = None
        retry_count = 0
        base_delay = 2.0  # Start with 2 second delay

        try:
            # Use cursor pagination to fetch all events for the date
            while True:
                params = {
                    "leagueID": "NBA",
                    "startsAfter": starts_after,
                    "startsBefore": starts_before,
                    "finalized": "true",
                }

                if next_cursor:
                    params["cursor"] = next_cursor

                response = requests.get(BASE_URL, headers=self.headers, params=params, timeout=30)

                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if retry_count >= max_retries:
                        print(f"  {date}: Max retries exceeded (429)")
                        break

                    # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                    wait_time = base_delay * (2**retry_count)
                    retry_count += 1
                    print(
                        f"  {date}: Rate limited (429), waiting {wait_time:.0f}s (retry {retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue  # Retry the same request

                # Reset retry count on successful request
                if response.status_code == 200:
                    retry_count = 0

                if response.status_code != 200:
                    if response.status_code == 404:
                        print(f"  {date}: No data (404)")
                    else:
                        print(f"  {date}: Error {response.status_code}")
                    break

                result = response.json()

                if not result.get("data"):
                    if not all_props:  # Only print if we haven't found any data yet
                        print(f"  {date}: No data")
                    break

                # Parse events and their odds
                for event in result["data"]:
                    event_id = event.get("eventID")
                    teams = event.get("teams", {})
                    home_team = teams.get("home", {}).get("names", {}).get("short", "")
                    away_team = teams.get("away", {}).get("names", {}).get("short", "")

                    odds = event.get("odds", {})

                    # Process each odd
                    for _odd_id, odd in odds.items():
                        stat_id = odd.get("statID", "")
                        bet_type_id = odd.get("betTypeID", "")
                        stat_entity_id = odd.get("statEntityID", "")
                        period_id = odd.get("periodID", "")
                        side_id = odd.get("sideID", "")

                        # Only process over/under player props for full game
                        if bet_type_id != "ou" or period_id != "game":
                            continue

                        # Only process 'over' side to avoid duplicates
                        # (each prop has both 'over' and 'under' entries with same line)
                        if side_id != "over":
                            continue

                        # Skip team-level props (only want player props)
                        if stat_entity_id.upper() in ["HOME", "AWAY", "ALL", ""]:
                            continue

                        # Only process NBA player props
                        if "_NBA" not in stat_entity_id:
                            continue

                        # Parse the prop type from statID
                        prop_type = self._parse_prop_type(stat_id)

                        if prop_type:
                            # Extract player name from marketName
                            player_name = ""
                            market_name = odd.get("marketName", "")
                            if market_name:
                                # Remove " Over/Under" suffix
                                player_name = market_name.replace(" Over/Under", "").strip()
                                # Remove the stat type from the end
                                for suffix in [
                                    " Points",
                                    " Rebounds",
                                    " Assists",
                                    " Three Pointers Made",
                                    " Points + Rebounds + Assists",
                                    " Points + Rebounds",
                                    " Points + Assists",
                                    " Rebounds + Assists",
                                ]:
                                    if player_name.endswith(suffix):
                                        player_name = player_name[: -len(suffix)].strip()
                                        break

                            # Fallback to parsing from statEntityID if needed
                            if not player_name:
                                player_name = self._parse_player_name(stat_entity_id)

                            # Get team info (not always present in NBA data)
                            team = odd.get("team", "")

                            # Get line from fairOverUnder or bookOverUnder
                            line = odd.get("fairOverUnder") or odd.get("bookOverUnder")

                            # Get odds for over and under
                            odds_over_str = odd.get("fairOdds") or odd.get("bookOdds")

                            # Try to find the opposing odd for under odds
                            opposing_odd_id = odd.get("opposingOddID")
                            odds_under_str = None
                            if opposing_odd_id and opposing_odd_id in odds:
                                opposing_odd = odds[opposing_odd_id]
                                odds_under_str = opposing_odd.get("fairOdds") or opposing_odd.get(
                                    "bookOdds"
                                )

                            # Convert American odds strings to integers
                            odds_over = self._parse_american_odds(odds_over_str)
                            odds_under = self._parse_american_odds(odds_under_str)

                            # Get actual result if available
                            actual_result = odd.get("score")

                            all_props.append(
                                {
                                    "date": date,
                                    "event_id": event_id,
                                    "player": player_name,
                                    "team": team,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "prop_type": prop_type,
                                    "line": float(line) if line else 0.0,
                                    "odds_over": odds_over,
                                    "odds_under": odds_under,
                                    "actual_result": (
                                        actual_result if actual_result is not None else ""
                                    ),
                                    "stat_id": stat_id,
                                    "stat_entity_id": stat_entity_id,
                                }
                            )

                # Check for next page
                next_cursor = result.get("nextCursor")
                if not next_cursor:
                    break

                # Small delay between pagination requests
                time.sleep(0.2)

            # Create DataFrame from all props
            if all_props:
                df = pd.DataFrame(all_props)
                print(
                    f"  {date}: {len(df)} props (PTS:{(df['prop_type']=='points').sum()}, "
                    f"REB:{(df['prop_type']=='rebounds').sum()}, "
                    f"AST:{(df['prop_type']=='assists').sum()}, "
                    f"PRA:{(df['prop_type']=='points+rebounds+assists').sum()})"
                )
                return df
            else:
                print(f"  {date}: 0 relevant props")
                return pd.DataFrame()

        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            print(f"  {date}: Error - {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def _parse_prop_type(self, stat_id: str) -> str:
        """
        Parse statID to determine prop type

        NBA statID examples from API:
            - points
            - rebounds
            - assists
            - threePointersMade
            - points+rebounds+assists
            - points+rebounds
            - points+assists
            - rebounds+assists

        Args:
            stat_id: The statID string from the API

        Returns:
            Normalized prop type or None if not relevant
        """
        if not stat_id:
            return None

        # Map of API statIDs to our prop types (keep same naming)
        target_props = [
            "points",
            "rebounds",
            "assists",
            "threePointersMade",
            "points+rebounds+assists",
            "points+rebounds",
            "points+assists",
            "rebounds+assists",
        ]

        return stat_id if stat_id in target_props else None

    def _parse_player_name(self, stat_entity_id: str) -> str:
        """
        Parse player name from statEntityID

        Format: "FIRSTNAME_LASTNAME_#_NBA"
        Example: "LEBRON_JAMES_1_NBA" -> "LeBron James"

        Args:
            stat_entity_id: The statEntityID string

        Returns:
            Formatted player name
        """
        if not stat_entity_id:
            return ""

        # Remove _NBA suffix and number
        parts = stat_entity_id.replace("_NBA", "").split("_")

        # Filter out numeric parts
        name_parts = [p for p in parts if not p.isdigit()]

        # Title case each part
        formatted_name = " ".join(p.title() for p in name_parts)

        return formatted_name

    def _parse_american_odds(self, odds_str) -> int:
        """
        Parse American odds to integer

        Examples:
            "+100" -> 100
            "-130" -> -130
            "EVEN" -> 100
            100 -> 100 (already int)

        Args:
            odds_str: The odds from the API (can be string or int)

        Returns:
            Integer representation of odds, or -110 as default
        """
        if not odds_str:
            return -110

        # If already an int, return it
        if isinstance(odds_str, int):
            return odds_str

        if str(odds_str).upper() in ["EVEN", "EV"]:
            return 100

        # Remove '+' prefix and convert to int
        try:
            return int(str(odds_str).replace("+", ""))
        except (ValueError, AttributeError):
            return -110

    def fetch_season_to_date(
        self, start_date: str = "2024-10-22", end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch all NBA props from season start to present

        Args:
            start_date: Season start date (default: Oct 22, 2024)
            end_date: End date (default: Nov 23, 2024 - ~1 month of data)

        Returns:
            Combined DataFrame with all props
        """
        print("=" * 80)
        print("FETCHING NBA HISTORICAL PLAYER PROPS")
        print("Source: Sports Game Odds API")
        print(f"Season: 2024-25 ({start_date} to present)")
        print("=" * 80)
        print()

        # Generate list of dates from start to end
        # Default to Nov 23, 2024 (1 month of season data)
        if end_date is None:
            end_date = "2024-11-23"

        start = datetime.strptime(start_date, "%Y-%m-%d")
        today = datetime.strptime(end_date, "%Y-%m-%d")

        dates = []
        current = start
        while current <= today:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Total dates to fetch: {len(dates)}")
        print()

        all_props = []

        for i, date in enumerate(dates, 1):
            print(f"[{i}/{len(dates)}] Fetching {date}...")

            df = self.fetch_date(date)

            if not df.empty:
                all_props.append(df)

            # Rate limiting: Amateur tier = 10 requests/minute max
            # Use 7 second delay to stay well under limit (~8.5 requests/min)
            time.sleep(7.0)  # 7 seconds between dates

        print()
        print("=" * 80)
        print("FETCH COMPLETE")
        print("=" * 80)
        print()

        if all_props:
            combined = pd.concat(all_props, ignore_index=True)

            print(f"Total props fetched: {len(combined):,}")
            print()
            print("Breakdown by prop type:")
            print(combined["prop_type"].value_counts())
            print()
            print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
            print(f"Unique players: {combined['player'].nunique()}")
            print(f"Unique games: {combined['event_id'].nunique()}")
            print()

            return combined
        else:
            print("⚠️  No props fetched")
            return pd.DataFrame()


def main():
    """Main execution"""

    print("=" * 80)
    print("NBA HISTORICAL PLAYER PROPS FETCHER")
    print("Sports Game Odds API - 2024-25 Season")
    print("=" * 80)
    print()

    fetcher = NBAPropsFetcher(API_KEY)

    # Check for command line argument
    if len(sys.argv) > 1:
        date = sys.argv[1]
        print(f"Fetching single date: {date}")
        print()
        df = fetcher.fetch_date(date)

        if not df.empty:
            output_file = f"nba_props_{date}.csv"
            df.to_csv(output_file, index=False)
            print()
            print(f"Saved to: {output_file}")
    else:
        # Fetch all data from season start
        df = fetcher.fetch_season_to_date()

        if not df.empty:
            # Save to project data directory
            output_dir = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "raw"
                / "sportsgameodds"
                / "nba"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "all_nba_props_2024_2025.csv"
            df.to_csv(output_file, index=False)
            print(f"💾 Saved to: {output_file}")
            print()

    print()
    print("✅ Done")


if __name__ == "__main__":
    main()
