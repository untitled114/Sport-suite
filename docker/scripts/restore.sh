#!/bin/bash
# =============================================================================
# Database Restore Script
# =============================================================================
# Restores databases from backup files
#
# Usage:
#   ./docker/scripts/restore.sh <backup_file>
#   ./docker/scripts/restore.sh nba_intelligence_20250115_120000.sql.gz
#   ./docker/scripts/restore.sh --list
#
# Environment:
#   BACKUP_DIR   - Backup directory (default: docker/backups)
#   DB_PASSWORD  - Database password (required)
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/docker/backups}"

# Database configuration
DB_USER="${DB_USER:-mlb_user}"
DB_HOST="${DB_HOST:-localhost}"

# Require password from environment
if [ -z "${DB_PASSWORD:-}" ]; then
    echo "Error: DB_PASSWORD environment variable is required"
    exit 1
fi

# PostgreSQL databases
declare -A PG_DATABASES=(
    ["nba_players"]="5536"
    ["nba_games"]="5537"
    ["nba_team"]="5538"
    ["nba_intelligence"]="5539"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# List available backups
list_backups() {
    log_info "Available backups in $BACKUP_DIR:"
    echo

    if ls "$BACKUP_DIR"/*.gz 1>/dev/null 2>&1; then
        ls -lh "$BACKUP_DIR"/*.gz | awk '{print "  " $9 " (" $5 ")"}'
    else
        log_warn "No backups found"
    fi
}

# Extract database name from backup filename
get_db_name() {
    local filename=$(basename "$1")
    echo "$filename" | sed 's/_[0-9]\{8\}_[0-9]\{6\}\.sql\.gz$//'
}

# Restore PostgreSQL database
restore_pg_database() {
    local backup_file=$1
    local db_name=$(get_db_name "$backup_file")
    local port

    # Get port for database
    if [[ -v "PG_DATABASES[$db_name]" ]]; then
        port="${PG_DATABASES[$db_name]}"
    else
        log_error "Unknown database: $db_name"
        log_info "Known databases: ${!PG_DATABASES[*]}"
        return 1
    fi

    log_info "Restoring $db_name from $backup_file..."

    # Confirm with user
    echo -e "${YELLOW}WARNING: This will OVERWRITE the existing $db_name database!${NC}"
    read -p "Continue? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    # Restore
    log_info "Restoring to $db_name (port $port)..."

    if gunzip -c "$backup_file" | PGPASSWORD="$DB_PASSWORD" psql \
        -h "$DB_HOST" \
        -p "$port" \
        -U "$DB_USER" \
        -d "$db_name" \
        --quiet 2>/dev/null; then

        log_info "Restore complete!"
        return 0
    else
        log_error "Restore failed"
        return 1
    fi
}

# Restore MongoDB
restore_mongodb() {
    local backup_file=$1

    log_info "Restoring MongoDB from $backup_file..."

    echo -e "${YELLOW}WARNING: This will OVERWRITE existing MongoDB data!${NC}"
    read -p "Continue? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    if [ -z "${MONGO_PASSWORD:-}" ]; then
        log_error "MONGO_PASSWORD environment variable is required"
        return 1
    fi

    if mongorestore \
        --uri="mongodb://${MONGO_USER:-nba_user}:${MONGO_PASSWORD}@localhost:27017" \
        --archive="$backup_file" \
        --gzip \
        --drop 2>/dev/null; then

        log_info "MongoDB restore complete!"
        return 0
    else
        log_error "MongoDB restore failed"
        return 1
    fi
}

# Main
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <backup_file> | --list"
        exit 1
    fi

    if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
        list_backups
        exit 0
    fi

    local backup_file="$1"

    # If relative path, prepend BACKUP_DIR
    if [[ ! "$backup_file" = /* ]]; then
        backup_file="$BACKUP_DIR/$backup_file"
    fi

    # Check file exists
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        list_backups
        exit 1
    fi

    # Determine type and restore
    if [[ "$backup_file" == *"mongodb"* ]]; then
        restore_mongodb "$backup_file"
    else
        restore_pg_database "$backup_file"
    fi
}

main "$@"
