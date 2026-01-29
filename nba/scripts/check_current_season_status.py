#!/usr/bin/env python3
"""
Check current season data status for 2025-2026 NBA season
"""

import psycopg2
from datetime import datetime
import os

print("=" * 80)
print("NBA 2025-2026 SEASON DATA STATUS CHECK")
print("=" * 80)
print()

print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
print()

# Database connections
DB_PLAYERS = {
    'host': 'localhost', 'port': 5536, 'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'), 'database': 'nba_players'
}

DB_GAMES = {
    'host': 'localhost', 'port': 5537, 'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'), 'database': 'nba_games'
}

DB_TEAM = {
    'host': 'localhost', 'port': 5538, 'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'), 'database': 'nba_team'
}

# Check player data
print("=" * 80)
print("PLAYER DATA STATUS")
print("=" * 80)
print()

conn = psycopg2.connect(**DB_PLAYERS)
cursor = conn.cursor()

# Latest games
cursor.execute("""
    SELECT MAX(game_date) as latest_game, COUNT(*) as total_games
    FROM player_game_logs
""")
result = cursor.fetchone()
print(f"Latest game in database: {result[0]}")
print(f"Total game logs: {result[1]:,}")
print()

# Games by recent months
cursor.execute("""
    SELECT
        DATE_TRUNC('month', game_date) as month,
        COUNT(*) as games
    FROM player_game_logs
    WHERE game_date >= '2025-01-01'
    GROUP BY DATE_TRUNC('month', game_date)
    ORDER BY month DESC
    LIMIT 12
""")
print("Games by month (2025):")
for row in cursor.fetchall():
    print(f"  {row[0].strftime('%Y-%m')}: {row[1]:,} game logs")
print()

# Current rosters
cursor.execute("""
    SELECT team_abbrev, COUNT(*) as players
    FROM player_profile
    WHERE team_abbrev IS NOT NULL
    GROUP BY team_abbrev
    ORDER BY team_abbrev
""")
rosters = cursor.fetchall()
print(f"Roster data: {len(rosters)} teams")
print("Players per team:")
for team, count in rosters[:5]:
    print(f"  {team}: {count} players")
print(f"  ... ({len(rosters)-5} more teams)")
print()

# Check for season stats
cursor.execute("""
    SELECT season, COUNT(DISTINCT player_id) as players
    FROM player_season_stats
    GROUP BY season
    ORDER BY season DESC
""")
print("Season stats by year:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,} players")
print()

conn.close()

# Check team data
print("=" * 80)
print("TEAM DATA STATUS")
print("=" * 80)
print()

conn = psycopg2.connect(**DB_TEAM)
cursor = conn.cursor()

cursor.execute("""
    SELECT season, COUNT(*) as teams
    FROM team_season_stats
    GROUP BY season
    ORDER BY season DESC
""")
print("Team season stats:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} teams")
print()

conn.close()

# Check games
print("=" * 80)
print("GAMES DATA STATUS")
print("=" * 80)
print()

conn = psycopg2.connect(**DB_GAMES)
cursor = conn.cursor()

cursor.execute("""
    SELECT MAX(game_date) as latest_game
    FROM games
""")
result = cursor.fetchone()
print(f"Latest game: {result[0]}")
print()

cursor.execute("""
    SELECT season, COUNT(*) as games
    FROM games
    GROUP BY season
    ORDER BY season DESC
""")
print("Games by season:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,} games")
print()

conn.close()

print("=" * 80)
print("DATA GAPS FOR 2025-2026 SEASON")
print("=" * 80)
print()

if result[0] < datetime.now().date():
    days_behind = (datetime.now().date() - result[0]).days
    print(f"🚨 DATA IS {days_behind} DAYS BEHIND!")
    print()
    print("Missing data:")
    print("  - Current season game logs")
    print("  - Updated rosters (trades, signings)")
    print("  - Current season stats")
    print("  - Team dynamics")
    print("  - Minutes/usage for new season")
    print()
else:
    print("✅ Data is current!")
    print()

print("=" * 80)
print("REQUIRED FOR PRODUCTION")
print("=" * 80)
print()

print("To deploy for 2025-2026 season, we need:")
print()
print("1. ROSTERS (CRITICAL)")
print("   - Updated player profiles (trades, signings, cuts)")
print("   - Current team assignments")
print("   - Position changes")
print()
print("2. MINUTES/USAGE (CRITICAL)")
print("   - Projected starters vs bench")
print("   - Minutes per game expectations")
print("   - Usage rates")
print("   - Role changes (promoted/demoted)")
print()
print("3. TEAM DYNAMICS (IMPORTANT)")
print("   - Pace projections")
print("   - Offensive/defensive ratings")
print("   - Rotation patterns")
print()
print("4. MATCHUP DATA (IMPORTANT)")
print("   - Player vs position historical")
print("   - Team defensive ratings vs position")
print()
print("5. CURRENT SEASON GAMES (AS THEY HAPPEN)")
print("   - Game logs")
print("   - Rolling stats")
print("   - Recent form")
print()

print("=" * 80)
print("END OF STATUS CHECK")
print("=" * 80)
