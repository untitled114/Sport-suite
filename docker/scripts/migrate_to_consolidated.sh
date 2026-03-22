#!/usr/bin/env bash
# ==========================================================================
# Migrate from 6 separate PostgreSQL containers to 1 consolidated instance
#
# USAGE:
#   ./docker/scripts/migrate_to_consolidated.sh
#
# PREREQUISITES:
#   - Old containers running (ports 5536-5541)
#   - New container running (port 5432)
#   - DB_USER and DB_PASSWORD set in environment
#
# This script:
#   1. Dumps each old database's tables
#   2. Transforms dumps to use schema-qualified names
#   3. Loads into the consolidated database
#
# ROLLBACK:
#   Set DB_MODE=legacy in .env to revert to old containers.
#   Old containers and volumes are NOT removed by this script.
# ==========================================================================
set -euo pipefail

DB_USER="${DB_USER:-mlb_user}"
DB_PASSWORD="${DB_PASSWORD:?DB_PASSWORD is required}"
NEW_HOST="${DB_HOST:-localhost}"
NEW_PORT="${DB_PORT:-5432}"
NEW_DB="${DB_NAME:-sportsuite}"

DUMP_DIR="/tmp/sportsuite_migration_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DUMP_DIR"

export PGPASSWORD="$DB_PASSWORD"

echo "================================================"
echo "Sport-Suite Database Migration"
echo "================================================"
echo "Dump directory: $DUMP_DIR"
echo ""

# --------------------------------------------------------------------------
# Step 1: Dump old databases
# --------------------------------------------------------------------------
dump_old_db() {
    local port=$1 db=$2 schema=$3
    echo "  Dumping $db (port $port) → $schema..."

    # Build extra pg_dump flags per schema
    local extra_flags=""
    if [ "$schema" = "games" ]; then
        # blowout_flag is a GENERATED column — can't be in COPY
        extra_flags="--exclude-table-data=games"
    fi

    # Dump data only, no schema (new DB already has the schema)
    pg_dump -h localhost -p "$port" -U "$DB_USER" -d "$db" \
        --data-only --no-owner --no-privileges \
        --format=plain $extra_flags \
        > "$DUMP_DIR/${schema}_data.sql" 2>/dev/null || {
            echo "    WARNING: Could not dump $db (port $port) — container may not be running"
            return 0
        }

    # For games schema: dump games table with explicit columns (exclude generated blowout_flag)
    if [ "$schema" = "games" ]; then
        echo "    Dumping games table (excluding generated columns)..."
        psql -h localhost -p "$port" -U "$DB_USER" -d "$db" -c \
            "COPY (SELECT game_id, game_date, season, home_team, away_team, home_score, away_score,
                    total_possessions, pace, vegas_total, vegas_spread, game_status,
                    created_at, updated_at
             FROM games) TO STDOUT WITH (FORMAT csv, HEADER false);" \
            > "$DUMP_DIR/games_table.csv" 2>/dev/null

        # Append COPY command for games table to the dump
        {
            echo ""
            echo "COPY ${schema}.games (game_id, game_date, season, home_team, away_team, home_score, away_score, total_possessions, pace, vegas_total, vegas_spread, game_status, created_at, updated_at) FROM STDIN WITH (FORMAT csv);"
            cat "$DUMP_DIR/games_table.csv"
            echo "\\."
        } >> "$DUMP_DIR/${schema}_data.sql"
    fi

    # Prefix all table references with schema
    # This handles INSERT INTO table_name → INSERT INTO schema.table_name
    sed -i "s/^INSERT INTO \(public\.\)\?/INSERT INTO ${schema}./g" "$DUMP_DIR/${schema}_data.sql"

    # Also handle SET search_path and COPY commands (skip already-prefixed ones)
    sed -i "/^COPY ${schema}\./! s/^COPY \(public\.\)\?/COPY ${schema}./g" "$DUMP_DIR/${schema}_data.sql"

    # Handle SELECT setval for sequences
    sed -i "s/pg_catalog.setval('\(public\.\)\?/pg_catalog.setval('${schema}./g" "$DUMP_DIR/${schema}_data.sql"

    local size=$(wc -c < "$DUMP_DIR/${schema}_data.sql")
    echo "    Done: $(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes")"
}

echo "[1/4] Dumping old databases..."
dump_old_db 5536 nba_players players
dump_old_db 5537 nba_games games
dump_old_db 5538 nba_team teams
dump_old_db 5539 nba_intelligence intelligence
dump_old_db 5541 cephalon_axiom axiom
echo ""

# --------------------------------------------------------------------------
# Step 2: Verify new database is ready
# --------------------------------------------------------------------------
echo "[2/4] Verifying consolidated database..."
if ! pg_isready -h "$NEW_HOST" -p "$NEW_PORT" -U "$DB_USER" -d "$NEW_DB" > /dev/null 2>&1; then
    echo "  ERROR: Consolidated database not reachable at $NEW_HOST:$NEW_PORT/$NEW_DB"
    echo "  Start it with: cd docker && docker-compose -f docker-compose.new.yml up -d"
    exit 1
fi

# Verify schemas exist
for schema in players games teams intelligence axiom features; do
    count=$(psql -h "$NEW_HOST" -p "$NEW_PORT" -U "$DB_USER" -d "$NEW_DB" -t -c \
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = '$schema';" 2>/dev/null | tr -d ' ')
    if [ "$count" != "1" ]; then
        echo "  ERROR: Schema '$schema' not found in $NEW_DB"
        echo "  The init scripts may not have run. Check docker logs."
        exit 1
    fi
done
echo "  All 6 schemas verified."
echo ""

# --------------------------------------------------------------------------
# Step 3: Load data into consolidated database
# --------------------------------------------------------------------------
load_data() {
    local schema=$1
    local dump_file="$DUMP_DIR/${schema}_data.sql"

    if [ ! -s "$dump_file" ]; then
        echo "  Skipping $schema (empty or missing dump)"
        return 0
    fi

    echo "  Loading $schema..."

    # Disable FK triggers during load — must be in same session as the data load
    {
        echo "SET session_replication_role = 'replica';"
        cat "$dump_file"
        echo "SET session_replication_role = 'origin';"
    } | psql -h "$NEW_HOST" -p "$NEW_PORT" -U "$DB_USER" -d "$NEW_DB" \
        > "$DUMP_DIR/${schema}_load.log" 2>&1 || {
            echo "    WARNING: Some errors loading $schema (check $DUMP_DIR/${schema}_load.log)"
        }

    # Count loaded rows
    local tables=$(psql -h "$NEW_HOST" -p "$NEW_PORT" -U "$DB_USER" -d "$NEW_DB" -t -c \
        "SELECT string_agg(tablename, ',') FROM pg_tables WHERE schemaname = '$schema';" 2>/dev/null | tr -d ' ')

    echo "    Tables: $tables"
}

echo "[3/4] Loading data into consolidated database..."
load_data players
load_data games
load_data teams
load_data intelligence
load_data axiom
echo ""

# --------------------------------------------------------------------------
# Step 4: Verify migration
# --------------------------------------------------------------------------
echo "[4/4] Verifying migration..."

verify_schema() {
    local schema=$1
    local count=$(psql -h "$NEW_HOST" -p "$NEW_PORT" -U "$DB_USER" -d "$NEW_DB" -t -c \
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = '$schema';" 2>/dev/null | tr -d ' ')
    echo "  $schema: $count tables"
}

verify_schema players
verify_schema games
verify_schema teams
verify_schema intelligence
verify_schema axiom
verify_schema features

echo ""
echo "================================================"
echo "Migration complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Set DB_MODE=consolidated in .env (or remove DB_MODE — consolidated is default)"
echo "  2. Set DB_PORT=5432 in .env"
echo "  3. Test: python -c 'from nba.config.database import get_connection; c=get_connection(\"intelligence\"); print(\"OK\")'"
echo "  4. Run tests: make test"
echo "  5. Once verified, stop old containers: docker stop nba_players_db nba_games_db nba_team_db nba_intelligence_db cephalon_axiom_db"
echo ""
echo "To rollback: set DB_MODE=legacy in .env and restart old containers."
echo "Dumps saved in: $DUMP_DIR"
