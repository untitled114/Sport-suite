#!/bin/bash
# =============================================================================
# Database Backup Script
# =============================================================================
# Creates compressed backups of all NBA databases (5 PostgreSQL + 1 MongoDB)
#
# Usage:
#   ./docker/scripts/backup.sh              # Full backup
#   ./docker/scripts/backup.sh intelligence # Single database backup
#
# Environment:
#   BACKUP_DIR     - Backup directory (default: docker/backups)
#   RETENTION_DAYS - Days to keep backups (default: 7)
#   DB_PASSWORD    - Database password (required)
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/docker/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database configuration
DB_USER="${DB_USER:-mlb_user}"
DB_PASSWORD="${DB_PASSWORD:-${DB_PASSWORD}}"
DB_HOST="${DB_HOST:-localhost}"

# PostgreSQL databases
declare -A PG_DATABASES=(
    ["nba_players"]="5536"
    ["nba_games"]="5537"
    ["nba_team"]="5538"
    ["nba_intelligence"]="5539"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup single PostgreSQL database
backup_pg_database() {
    local db_name=$1
    local port=$2
    local backup_file="$BACKUP_DIR/${db_name}_${TIMESTAMP}.sql.gz"

    log_info "Backing up PostgreSQL: $db_name (port $port)"

    if PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$port" \
        -U "$DB_USER" \
        -d "$db_name" \
        --no-owner \
        --no-acl \
        2>/dev/null | gzip > "$backup_file"; then

        local size=$(du -h "$backup_file" | cut -f1)
        log_info "  Created: $backup_file ($size)"
        return 0
    else
        log_error "  Failed to backup $db_name"
        rm -f "$backup_file"
        return 1
    fi
}

# Backup MongoDB
backup_mongodb() {
    local backup_file="$BACKUP_DIR/mongodb_${TIMESTAMP}.archive.gz"

    log_info "Backing up MongoDB..."

    if mongodump \
        --uri="mongodb://${MONGO_USER:-nba_user}:${MONGO_PASSWORD:-${MONGO_PASSWORD}}@localhost:27017" \
        --archive \
        --gzip \
        --out="$backup_file" 2>/dev/null; then

        local size=$(du -h "$backup_file" | cut -f1)
        log_info "  Created: $backup_file ($size)"
        return 0
    else
        log_warn "  MongoDB backup failed (may not be running)"
        rm -f "$backup_file"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    log_info "Cleaning backups older than $RETENTION_DAYS days..."

    local count=$(find "$BACKUP_DIR" -name "*.gz" -mtime +"$RETENTION_DAYS" -type f | wc -l)

    if [ "$count" -gt 0 ]; then
        find "$BACKUP_DIR" -name "*.gz" -mtime +"$RETENTION_DAYS" -type f -delete
        log_info "  Removed $count old backup(s)"
    else
        log_info "  No old backups to remove"
    fi
}

# Main backup function
main() {
    local target="${1:-all}"
    local success=0
    local failed=0

    log_info "Starting backup at $(date)"
    log_info "Backup directory: $BACKUP_DIR"
    echo

    if [ "$target" == "all" ]; then
        # Backup all PostgreSQL databases
        for db_name in "${!PG_DATABASES[@]}"; do
            if backup_pg_database "$db_name" "${PG_DATABASES[$db_name]}"; then
                ((success++))
            else
                ((failed++))
            fi
        done

        # Backup MongoDB (optional - may not be running)
        backup_mongodb || true

        # Cleanup
        cleanup_old_backups
    else
        # Backup single database
        if [[ -v "PG_DATABASES[$target]" ]]; then
            backup_pg_database "$target" "${PG_DATABASES[$target]}"
        elif [ "$target" == "mongodb" ]; then
            backup_mongodb
        else
            log_error "Unknown database: $target"
            log_info "Available: ${!PG_DATABASES[*]} mongodb"
            exit 1
        fi
    fi

    echo
    log_info "Backup complete: $success succeeded, $failed failed"

    # List recent backups
    log_info "Recent backups:"
    ls -lh "$BACKUP_DIR"/*.gz 2>/dev/null | tail -10 || echo "  (none)"

    [ "$failed" -eq 0 ]
}

main "$@"
