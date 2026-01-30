"""
Update Current NBA Rosters
Fetches current team rosters and updates player_profile table with correct team assignments.

This fixes Gap #1: Current Team Assignments
- Updates team_abbrev for all active players
- Marks players who are no longer active
- Fixes wrong team assignments (e.g., Simons on BOS -> POR)

Usage:
    python update_current_rosters.py
    python update_current_rosters.py --verify  # Dry run to see changes
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Set

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities"))
# For roster data, we need to use the nba_api directly
from nba_api.stats.endpoints.commonteamroster import CommonTeamRoster
from nba_api_wrapper import NBAApiWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database connection params
DB_CONFIG = {
    "host": "localhost",
    "port": 5536,
    "database": "nba_players",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}


class RosterUpdater:
    """Updates current NBA rosters in database"""

    def __init__(self, verify_only: bool = False):
        self.api = NBAApiWrapper(requests_per_minute=15)  # Conservative rate limit
        self.conn = None
        self.verify_only = verify_only
        self.changes = []

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logger.info("✅ Connected to nba_players database")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def fetch_team_roster(self, team_id: int, team_abbrev: str) -> pd.DataFrame:
        """
        Fetch current roster for a specific team

        Args:
            team_id: NBA team ID
            team_abbrev: Team abbreviation (e.g., "BOS")

        Returns:
            DataFrame with player roster data
        """
        logger.info(f"Fetching roster for {team_abbrev} (ID: {team_id})...")

        try:
            # Rate limit
            time.sleep(3)  # NBA API is sensitive, be conservative

            # Fetch roster using current season
            roster = CommonTeamRoster(team_id=team_id, season="2025-26")  # Current season

            roster_df = roster.get_data_frames()[0]

            if roster_df.empty:
                logger.warning(f"No roster data found for {team_abbrev}")
                return pd.DataFrame()

            # Add team abbreviation column
            roster_df["TEAM_ABBREV"] = team_abbrev

            logger.info(f"✅ Found {len(roster_df)} players on {team_abbrev}")
            return roster_df

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"❌ Failed to fetch roster for {team_abbrev}: {e}")
            return pd.DataFrame()

    def fetch_all_rosters(self) -> pd.DataFrame:
        """
        Fetch rosters for all 30 NBA teams

        Returns:
            DataFrame with all active players and their current teams
        """
        logger.info("Fetching all NBA team rosters...")

        # Get all teams
        teams_df = self.api.get_all_teams()
        logger.info(f"Found {len(teams_df)} teams")

        all_rosters = []

        for _, team in teams_df.iterrows():
            team_id = team["id"]
            team_abbrev = team["abbreviation"]

            roster_df = self.fetch_team_roster(team_id, team_abbrev)

            if not roster_df.empty:
                all_rosters.append(roster_df)

        if not all_rosters:
            logger.error("❌ No roster data fetched")
            return pd.DataFrame()

        # Combine all rosters
        combined_df = pd.concat(all_rosters, ignore_index=True)

        logger.info(f"✅ Total active players across all teams: {len(combined_df)}")
        return combined_df

    def get_current_db_rosters(self) -> Dict[int, str]:
        """
        Get current team assignments from database

        Returns:
            Dict mapping player_id -> team_abbrev
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT player_id, team_abbrev, full_name
            FROM player_profile
            WHERE team_abbrev IS NOT NULL
            ORDER BY player_id
        """
        )

        current_rosters = {}
        for player_id, team_abbrev, full_name in cursor.fetchall():
            current_rosters[player_id] = {"team": team_abbrev, "name": full_name}

        logger.info(f"Current database has {len(current_rosters)} players with team assignments")
        return current_rosters

    def identify_changes(self, api_rosters: pd.DataFrame, db_rosters: Dict[int, str]) -> List[Dict]:
        """
        Identify what needs to be updated

        Args:
            api_rosters: DataFrame from NBA API with current rosters
            db_rosters: Dict with current database rosters

        Returns:
            List of changes to make
        """
        changes = []

        # Track which players we've seen in API
        api_player_ids = set()

        # Check for updates and new players
        for _, row in api_rosters.iterrows():
            player_id = int(row["PLAYER_ID"])
            api_team = row["TEAM_ABBREV"]
            player_name = row.get("PLAYER", "Unknown")

            api_player_ids.add(player_id)

            if player_id in db_rosters:
                # Existing player - check if team changed
                db_team = db_rosters[player_id]["team"]
                db_name = db_rosters[player_id]["name"]

                if db_team != api_team:
                    changes.append(
                        {
                            "type": "UPDATE",
                            "player_id": player_id,
                            "name": db_name,
                            "old_team": db_team,
                            "new_team": api_team,
                        }
                    )
            else:
                # New player (drafted, signed, or previously not in DB)
                changes.append(
                    {
                        "type": "NEW",
                        "player_id": player_id,
                        "name": player_name,
                        "old_team": None,
                        "new_team": api_team,
                    }
                )

        # Check for players no longer on rosters (free agents, retired, etc.)
        for player_id, player_info in db_rosters.items():
            if player_id not in api_player_ids:
                changes.append(
                    {
                        "type": "REMOVED",
                        "player_id": player_id,
                        "name": player_info["name"],
                        "old_team": player_info["team"],
                        "new_team": None,
                    }
                )

        return changes

    def apply_changes(self, changes: List[Dict]):
        """
        Apply roster changes to database

        Args:
            changes: List of changes from identify_changes()
        """
        if not changes:
            logger.info("No changes to apply")
            return

        cursor = self.conn.cursor()
        updates = 0
        removals = 0
        new_players = 0

        for change in changes:
            change_type = change["type"]
            player_id = change["player_id"]

            if change_type == "UPDATE":
                # Update team assignment
                cursor.execute(
                    """
                    UPDATE player_profile
                    SET team_abbrev = %s
                    WHERE player_id = %s
                """,
                    (change["new_team"], player_id),
                )
                updates += 1

                logger.info(f"  ✏️  {change['name']}: {change['old_team']} → {change['new_team']}")

            elif change_type == "REMOVED":
                # Set team to NULL (player no longer on active roster)
                cursor.execute(
                    """
                    UPDATE player_profile
                    SET team_abbrev = NULL
                    WHERE player_id = %s
                """,
                    (player_id,),
                )
                removals += 1

                logger.info(
                    f"  ➖ {change['name']}: Removed from {change['old_team']} (inactive/FA)"
                )

            elif change_type == "NEW":
                # Insert new player (this shouldn't happen often if we loaded rosters properly)
                # For now, we'll just update if they exist, otherwise skip
                cursor.execute(
                    """
                    UPDATE player_profile
                    SET team_abbrev = %s
                    WHERE player_id = %s
                """,
                    (change["new_team"], player_id),
                )

                if cursor.rowcount > 0:
                    new_players += 1
                    logger.info(f"  ➕ {change['name']}: Added to {change['new_team']}")

        # Commit all changes
        if not self.verify_only:
            self.conn.commit()
            logger.info(f"\n✅ Changes committed to database")
        else:
            self.conn.rollback()
            logger.info(f"\n🔍 VERIFY MODE - No changes committed")

        # Summary
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Updates: {updates}")
        logger.info(f"   New: {new_players}")
        logger.info(f"   Removed: {removals}")
        logger.info(f"   Total: {len(changes)}")

    def run(self):
        """Main execution flow"""
        try:
            # Fetch current rosters from NBA API
            api_rosters = self.fetch_all_rosters()

            if api_rosters.empty:
                logger.error("❌ Failed to fetch rosters from NBA API")
                return

            # Get current database state
            db_rosters = self.get_current_db_rosters()

            # Identify changes
            logger.info("\n🔍 Analyzing changes...")
            changes = self.identify_changes(api_rosters, db_rosters)

            if not changes:
                logger.info("✅ All rosters are up to date!")
                return

            # Show changes
            logger.info(f"\n📝 Found {len(changes)} changes:")
            for change in changes:
                if change["type"] == "UPDATE":
                    logger.info(f"   {change['name']}: {change['old_team']} → {change['new_team']}")

            # Apply changes
            if self.verify_only:
                logger.info(f"\n🔍 VERIFY MODE - Run without --verify to apply changes")
            else:
                logger.info(f"\n💾 Applying changes...")

            self.apply_changes(changes)

            # Store changes for reporting
            self.changes = changes

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"❌ Roster update failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def verify_specific_players(self, player_names: List[str]):
        """
        Verify specific players' team assignments

        Args:
            player_names: List of player names to check
        """
        cursor = self.conn.cursor()

        logger.info(f"\n🔍 Verifying specific players:")

        for name in player_names:
            cursor.execute(
                """
                SELECT player_id, full_name, team_abbrev
                FROM player_profile
                WHERE full_name LIKE %s
            """,
                (f"%{name}%",),
            )

            results = cursor.fetchall()

            if results:
                for _player_id, full_name, team_abbrev in results:
                    logger.info(f"   {full_name}: {team_abbrev}")
            else:
                logger.warning(f"   {name}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Update current NBA rosters")
    parser.add_argument(
        "--verify", action="store_true", help="Dry run - show changes without applying"
    )
    parser.add_argument(
        "--check-players",
        type=str,
        nargs="+",
        help='Check specific players (e.g., "Anfernee Simons" "Chris Boucher")',
    )

    args = parser.parse_args()

    updater = RosterUpdater(verify_only=args.verify)

    try:
        updater.connect()

        # If checking specific players
        if args.check_players:
            updater.verify_specific_players(args.check_players)
        else:
            # Run full update
            updater.run()

        logger.info("\n🎉 Roster update complete!")

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"❌ Update failed: {e}")
        raise
    finally:
        updater.close()


if __name__ == "__main__":
    main()
