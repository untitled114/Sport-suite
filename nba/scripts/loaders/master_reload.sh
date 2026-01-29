#!/bin/bash
################################################################################
# NBA DATABASE MASTER RELOAD SCRIPT
################################################################################
# Completely reloads NBA player data with fixed loaders
# - Truncates player_game_logs and player_season_stats
# - Reloads all seasons (2021-2024) with complete columns
# - Recalculates rolling stats
# - Verifies no NULLs in critical columns
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "NBA DATABASE MASTER RELOAD"
echo "================================================================================"
echo ""

# Navigate to loaders directory
cd "$(dirname "$0")"

echo "📊 Step 1: Truncating existing data..."
echo "--------------------------------------------------------------------------------"
PGPASSWORD="${DB_PASSWORD}" psql -h localhost -p 5536 -U "${DB_USER:-nba_user}" -d nba_players << 'EOF'
TRUNCATE TABLE player_rolling_stats CASCADE;
TRUNCATE TABLE player_game_logs CASCADE;
TRUNCATE TABLE player_season_stats CASCADE;
SELECT 'player_game_logs' as table, COUNT(*) FROM player_game_logs
UNION ALL SELECT 'player_season_stats', COUNT(*) FROM player_season_stats
UNION ALL SELECT 'player_rolling_stats', COUNT(*) FROM player_rolling_stats;
EOF
echo "✅ Tables truncated"
echo ""

echo "📊 Step 2: Loading season stats for all years..."
echo "--------------------------------------------------------------------------------"
for season in "2020-21" "2021-22" "2022-23" "2023-24" "2024-25"; do
    echo "Loading season stats for $season..."
    python3 load_nba_players.py --season $season || echo "⚠️  Season $season failed or has no data"
done
echo "✅ Season stats loaded"
echo ""

echo "📊 Step 3: Loading game logs for all years (this will take 30-40 minutes)..."
echo "--------------------------------------------------------------------------------"
for season in "2020-21" "2021-22" "2022-23" "2023-24" "2024-25"; do
    echo "Loading game logs for $season..."
    python3 load_player_gamelogs_bulk.py --season $season || echo "⚠️  Season $season failed"
done
echo "✅ Game logs loaded"
echo ""

echo "📊 Step 4: Calculating rolling stats..."
echo "--------------------------------------------------------------------------------"
python3 calculate_rolling_stats.py
echo "✅ Rolling stats calculated"
echo ""

echo "📊 Step 5: Calculating team season stats (offensive/defensive ratings)..."
echo "--------------------------------------------------------------------------------"
python3 calculate_team_stats.py --season 2020 2021 2022 2023 2024
echo "✅ Team season stats calculated"
echo ""

echo "📊 Step 6: Final verification - checking for NULLs in critical columns..."
echo "--------------------------------------------------------------------------------"
PGPASSWORD="${DB_PASSWORD}" psql -h localhost -p 5536 -U "${DB_USER:-nba_user}" -d nba_players << 'EOF'
-- Check critical columns for NULLs
SELECT 'player_profile' as table_name,
       COUNT(*) as total,
       COUNT(CASE WHEN position IS NULL THEN 1 END) as null_position,
       COUNT(CASE WHEN team_abbrev IS NULL THEN 1 END) as null_team
FROM player_profile
UNION ALL
SELECT 'player_season_stats',
       COUNT(*),
       COUNT(CASE WHEN per IS NULL THEN 1 END),
       COUNT(CASE WHEN true_shooting_pct IS NULL THEN 1 END)
FROM player_season_stats
UNION ALL
SELECT 'player_game_logs',
       COUNT(*),
       COUNT(CASE WHEN ft_attempted IS NULL THEN 1 END),
       COUNT(CASE WHEN three_pt_attempted IS NULL THEN 1 END)
FROM player_game_logs
UNION ALL
SELECT 'player_rolling_stats',
       COUNT(*),
       COUNT(CASE WHEN ema_three_pt_pct IS NULL THEN 1 END),
       COUNT(CASE WHEN ema_fg_pct IS NULL THEN 1 END)
FROM player_rolling_stats;

-- Show record counts by season
SELECT 'Season Stats' as data_type, season, COUNT(*) as records
FROM player_season_stats GROUP BY season ORDER BY season
UNION ALL
SELECT 'Game Logs', season, COUNT(*) FROM player_game_logs GROUP BY season ORDER BY season;
EOF
echo ""

echo "================================================================================"
echo "✅ NBA DATABASE RELOAD COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Review NULL counts above"
echo "2. If acceptable, proceed with Phase 2 feature engineering"
echo "3. Run: cd ../../features && python3 build_nba_dataset.py"
echo ""
