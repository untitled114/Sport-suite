"""
Calculate Minutes Projections for NBA Players
Projects minutes per game for current season based on historical data, age, and role changes.

This fixes Gap #4: Minutes Projections

Methodology:
1. Start with previous season's MPG as baseline
2. Apply age-based decline factor (for players 33+)
3. Apply experience-based adjustments (rookies, 2nd year players)
4. Flag high-uncertainty cases (role changes, new team, injury history)
5. Update with actual data after 5/10 games

Usage:
    # Calculate projections
    python calculate_minutes_projections.py

    # Update database
    python calculate_minutes_projections.py --update

    # Recalculate after N games
    python calculate_minutes_projections.py --update --min-games 5
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

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


class MinutesProjector:
    """Projects minutes per game for NBA players"""

    def __init__(self):
        self.conn = None
        self.player_data = None
        self.projections = []

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

    def load_player_data(self, min_games: int = 0) -> pd.DataFrame:
        """
        Load player historical data

        Args:
            min_games: Minimum games in current season to use actual data

        Returns:
            DataFrame with player stats
        """
        logger.info(f"\n📊 Loading player data...")

        query = """
            SELECT
                pp.player_id,
                pp.full_name,
                pp.team_abbrev,
                pp.position,
                pp.draft_year,

                -- Previous season stats (2024 = 2024-25 season)
                pss_prev.minutes_per_game as prev_mpg,
                pss_prev.games_played as prev_gp,
                pss_prev.season as prev_season,

                -- Current season stats (if any)
                pss_curr.minutes_per_game as curr_mpg,
                pss_curr.games_played as curr_gp,

                -- Calculate age (approx from draft year)
                CASE
                    WHEN pp.draft_year IS NOT NULL
                    THEN 2025 - pp.draft_year + 19  -- Assume drafted at ~19
                    ELSE NULL
                END as estimated_age,

                -- Calculate experience
                CASE
                    WHEN pp.draft_year IS NOT NULL
                    THEN 2025 - pp.draft_year
                    ELSE NULL
                END as years_experience

            FROM player_profile pp

            -- Previous season stats (2024-25)
            LEFT JOIN player_season_stats pss_prev
                ON pp.player_id = pss_prev.player_id
                AND pss_prev.season = 2024

            -- Current season stats (2025-26) - if games already played
            LEFT JOIN player_season_stats pss_curr
                ON pp.player_id = pss_curr.player_id
                AND pss_curr.season = 2025

            WHERE pp.team_abbrev IS NOT NULL  -- Only active players
        """

        df = pd.read_sql(query, self.conn)

        logger.info(f"✅ Loaded {len(df)} players")
        logger.info(f"   Players with prev season data: {df['prev_mpg'].notna().sum()}")
        logger.info(f"   Players with current season data: {df['curr_mpg'].notna().sum()}")

        self.player_data = df
        return df

    def apply_age_adjustment(self, mpg: float, age: int) -> Tuple[float, str]:
        """
        Apply age-based adjustment to minutes

        Args:
            mpg: Previous season MPG
            age: Player's current age

        Returns:
            (adjusted_mpg, reason)
        """
        if pd.isna(age) or age < 30:
            return mpg, "No adjustment"

        # Age-based decline factors (based on NBA historical data)
        if age >= 38:
            factor = 0.85  # ~15% reduction for very old players
            reason = "Age 38+ decline"
        elif age >= 35:
            factor = 0.92  # ~8% reduction
            reason = "Age 35+ decline"
        elif age >= 33:
            factor = 0.97  # ~3% reduction
            reason = "Age 33+ decline"
        else:
            return mpg, "No adjustment"

        adjusted = mpg * factor
        return adjusted, reason

    def apply_experience_adjustment(
        self, mpg: float, experience: int, prev_gp: int
    ) -> Tuple[float, str]:
        """
        Apply experience-based adjustment

        Args:
            mpg: Previous season MPG
            experience: Years in NBA
            prev_gp: Games played previous season

        Returns:
            (adjusted_mpg, reason)
        """
        if pd.isna(experience):
            return mpg, "Unknown experience"

        # Rookies/second year players often see minutes increase
        if experience == 1:  # Second year player
            if pd.notna(prev_gp) and prev_gp > 40 and mpg > 15:
                # Promising rookie likely gets more minutes
                adjusted = min(mpg * 1.10, 36)  # Cap at 36 MPG
                return adjusted, "2nd year bump"
            else:
                return mpg, "No adjustment"

        # Very experienced players (10+ years) often stable or declining
        if experience >= 10 and mpg > 30:
            # Veterans often get slight load management
            adjusted = mpg * 0.98
            return adjusted, "Veteran load management"

        return mpg, "No adjustment"

    def flag_uncertainty(self, row: pd.Series) -> str:
        """
        Flag cases with high projection uncertainty

        Returns:
            Uncertainty flag/reason
        """
        flags = []

        # Missing previous season data
        if pd.isna(row["prev_mpg"]):
            return "HIGH - No previous season data"

        # Very low games played last season (injury concerns)
        if pd.notna(row["prev_gp"]) and row["prev_gp"] < 20:
            flags.append("Injury history")

        # Extreme age
        if pd.notna(row["estimated_age"]) and row["estimated_age"] >= 38:
            flags.append("Advanced age")

        # Rookie (no baseline)
        if pd.notna(row["years_experience"]) and row["years_experience"] == 0:
            return "HIGH - Rookie (no NBA history)"

        # Low minutes last year (role player uncertainty)
        if row["prev_mpg"] < 10:
            flags.append("Low usage")

        if flags:
            return "MEDIUM - " + ", ".join(flags)

        return "LOW"

    def calculate_projections(self, min_games: int = 5) -> pd.DataFrame:
        """
        Calculate minutes projections for all players

        Args:
            min_games: Use actual current season data if >= this many games

        Returns:
            DataFrame with projections
        """
        logger.info(f"\n🔮 Calculating minutes projections...")

        df = self.player_data.copy()

        projections = []

        for _, row in df.iterrows():
            projection = {
                "player_id": row["player_id"],
                "player_name": row["full_name"],
                "team": row["team_abbrev"],
                "position": row["position"],
            }

            # If we have current season data (>= min_games), use that
            if pd.notna(row["curr_gp"]) and row["curr_gp"] >= min_games:
                projection["projected_mpg"] = row["curr_mpg"]
                projection["method"] = f'Actual (from {int(row["curr_gp"])} games)'
                projection["uncertainty"] = "LOW"

            # Otherwise, project from previous season
            elif pd.notna(row["prev_mpg"]):
                base_mpg = row["prev_mpg"]

                # Apply age adjustment
                mpg_after_age, age_reason = self.apply_age_adjustment(
                    base_mpg, row["estimated_age"]
                )

                # Apply experience adjustment
                mpg_final, exp_reason = self.apply_experience_adjustment(
                    mpg_after_age, row["years_experience"], row["prev_gp"]
                )

                # Combine reasons
                reasons = []
                if age_reason != "No adjustment":
                    reasons.append(age_reason)
                if exp_reason != "No adjustment":
                    reasons.append(exp_reason)

                method = "Baseline from 2024-25"
                if reasons:
                    method += f" + {', '.join(reasons)}"

                projection["projected_mpg"] = round(mpg_final, 1)
                projection["baseline_mpg"] = round(base_mpg, 1)
                projection["method"] = method
                projection["uncertainty"] = self.flag_uncertainty(row)

            else:
                # No data available
                projection["projected_mpg"] = None
                projection["baseline_mpg"] = None
                projection["method"] = "No data"
                projection["uncertainty"] = "HIGH - No historical data"

            projections.append(projection)

        proj_df = pd.DataFrame(projections)

        logger.info(f"✅ Calculated {len(proj_df)} projections")

        # Summary by uncertainty
        if "uncertainty" in proj_df.columns:
            logger.info(f"\n📊 UNCERTAINTY SUMMARY:")
            for unc_level in ["LOW", "MEDIUM", "HIGH"]:
                count = proj_df["uncertainty"].str.contains(unc_level, na=False).sum()
                logger.info(f"   {unc_level}: {count} players")

        self.projections = proj_df
        return proj_df

    def save_to_database(self):
        """Save projections to database"""
        logger.info(f"\n💾 Saving projections to database...")

        # Create table if not exists
        cursor = self.conn.cursor()

        create_table = """
            CREATE TABLE IF NOT EXISTS player_minutes_projections (
                player_id INTEGER PRIMARY KEY,
                projected_mpg NUMERIC(4,1),
                baseline_mpg NUMERIC(4,1),
                projection_method TEXT,
                uncertainty_level VARCHAR(50),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (player_id) REFERENCES player_profile(player_id) ON DELETE CASCADE
            );
        """

        cursor.execute(create_table)

        # Insert/update projections
        insert_data = []
        for _, row in self.projections.iterrows():
            if pd.notna(row["projected_mpg"]):
                insert_data.append(
                    (
                        int(row["player_id"]),
                        float(row["projected_mpg"]) if pd.notna(row["projected_mpg"]) else None,
                        float(row["baseline_mpg"]) if pd.notna(row["baseline_mpg"]) else None,
                        row["method"],
                        row["uncertainty"],
                    )
                )

        if insert_data:
            insert_query = """
                INSERT INTO player_minutes_projections
                (player_id, projected_mpg, baseline_mpg, projection_method, uncertainty_level)
                VALUES %s
                ON CONFLICT (player_id)
                DO UPDATE SET
                    projected_mpg = EXCLUDED.projected_mpg,
                    baseline_mpg = EXCLUDED.baseline_mpg,
                    projection_method = EXCLUDED.projection_method,
                    uncertainty_level = EXCLUDED.uncertainty_level,
                    last_updated = CURRENT_TIMESTAMP
            """

            execute_values(cursor, insert_query, insert_data)
            self.conn.commit()

            logger.info(f"✅ Saved {len(insert_data)} projections to database")
        else:
            logger.warning("No projections to save")

    def display_sample_projections(self, n: int = 20):
        """Display sample projections"""
        logger.info(f"\n📋 SAMPLE PROJECTIONS:")

        # Show a mix: stars, role players, rookies
        if len(self.projections) > 0:
            # Sort by projected MPG
            sorted_proj = self.projections.sort_values(
                "projected_mpg", ascending=False, na_position="last"
            )

            logger.info(f"\n   Top Minutes Players:")
            for _, row in sorted_proj.head(10).iterrows():
                logger.info(
                    f"      {row['player_name']} ({row['team']}): "
                    f"{row['projected_mpg']:.1f} MPG ({row['uncertainty']})"
                )

            logger.info(f"\n   High Uncertainty Players:")
            high_unc = sorted_proj[sorted_proj["uncertainty"].str.contains("HIGH", na=False)].head(
                5
            )
            for _, row in high_unc.iterrows():
                logger.info(
                    f"      {row['player_name']} ({row['team']}): "
                    f"{row['projected_mpg']} MPG - {row['uncertainty']}"
                )

    def run(self, update_db: bool = False, min_games: int = 5):
        """
        Main execution flow

        Args:
            update_db: Save to database
            min_games: Minimum games for using actual data
        """
        try:
            # Load data
            self.load_player_data(min_games=min_games)

            # Calculate projections
            self.calculate_projections(min_games=min_games)

            # Display samples
            self.display_sample_projections()

            # Save to database
            if update_db:
                self.save_to_database()
            else:
                logger.info(f"\n💡 Run with --update to save projections to database")

            logger.info(f"\n✅ Minutes projection complete!")

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"❌ Projection failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise


def main():
    parser = argparse.ArgumentParser(description="Calculate NBA player minutes projections")
    parser.add_argument("--update", action="store_true", help="Save projections to database")
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum current season games to use actual data (default: 5)",
    )

    args = parser.parse_args()

    projector = MinutesProjector()

    try:
        projector.connect()
        projector.run(update_db=args.update, min_games=args.min_games)

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"❌ Failed: {e}")
        raise
    finally:
        projector.close()


if __name__ == "__main__":
    main()
