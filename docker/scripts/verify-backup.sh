#!/bin/bash
# =============================================================================
# Backup Verification Script
# =============================================================================
# Verifies backup file integrity and contents
#
# Usage:
#   ./docker/scripts/verify-backup.sh <backup_file>
#   ./docker/scripts/verify-backup.sh --all
#
# Checks:
#   - File exists and is readable
#   - Gzip integrity
#   - SQL syntax (for PostgreSQL backups)
#   - Row count estimates
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/docker/backups}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# Verify single backup file
verify_backup() {
    local backup_file=$1
    local filename=$(basename "$backup_file")
    local checks_passed=0
    local checks_failed=0

    echo "Verifying: $filename"
    echo "----------------------------------------"

    # Check 1: File exists and is readable
    if [ -r "$backup_file" ]; then
        log_info "File readable"
        ((checks_passed++))
    else
        log_error "File not readable"
        ((checks_failed++))
        return 1
    fi

    # Check 2: File size
    local size=$(stat -c%s "$backup_file" 2>/dev/null || stat -f%z "$backup_file")
    if [ "$size" -gt 0 ]; then
        local human_size=$(du -h "$backup_file" | cut -f1)
        log_info "File size: $human_size"
        ((checks_passed++))
    else
        log_error "File is empty"
        ((checks_failed++))
    fi

    # Check 3: Gzip integrity
    if gzip -t "$backup_file" 2>/dev/null; then
        log_info "Gzip integrity OK"
        ((checks_passed++))
    else
        log_error "Gzip integrity FAILED"
        ((checks_failed++))
    fi

    # Check 4: SQL content check (for PostgreSQL backups)
    if [[ "$filename" == *.sql.gz ]]; then
        # Check for valid SQL statements
        local create_count=$(gunzip -c "$backup_file" 2>/dev/null | grep -c "CREATE TABLE" || echo 0)
        local insert_count=$(gunzip -c "$backup_file" 2>/dev/null | grep -c "^COPY\|^INSERT" || echo 0)

        if [ "$create_count" -gt 0 ]; then
            log_info "Contains $create_count CREATE TABLE statements"
            ((checks_passed++))
        else
            log_warn "No CREATE TABLE statements found"
        fi

        if [ "$insert_count" -gt 0 ]; then
            log_info "Contains $insert_count data operations"
            ((checks_passed++))
        else
            log_warn "No INSERT/COPY statements found"
        fi

        # Estimate row count
        local copy_rows=$(gunzip -c "$backup_file" 2>/dev/null | grep -E "^[0-9]" | wc -l || echo 0)
        log_info "Estimated rows: ~$copy_rows"
    fi

    # Check 5: MongoDB archive check
    if [[ "$filename" == *mongodb*.gz ]]; then
        # Check archive structure
        if gunzip -c "$backup_file" 2>/dev/null | head -c 100 | grep -q "BSON\|admin"; then
            log_info "Valid MongoDB archive structure"
            ((checks_passed++))
        else
            log_warn "Could not verify MongoDB archive structure"
        fi
    fi

    echo
    echo "Results: $checks_passed passed, $checks_failed failed"
    echo

    [ "$checks_failed" -eq 0 ]
}

# Verify all backups
verify_all() {
    local total=0
    local passed=0
    local failed=0

    echo "Verifying all backups in $BACKUP_DIR"
    echo "========================================"
    echo

    for backup_file in "$BACKUP_DIR"/*.gz; do
        if [ -f "$backup_file" ]; then
            ((total++))
            if verify_backup "$backup_file"; then
                ((passed++))
            else
                ((failed++))
            fi
        fi
    done

    echo "========================================"
    echo "Total: $total backups, $passed passed, $failed failed"

    [ "$failed" -eq 0 ]
}

# Main
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <backup_file> | --all"
        exit 1
    fi

    if [ "$1" == "--all" ] || [ "$1" == "-a" ]; then
        verify_all
        exit $?
    fi

    local backup_file="$1"

    # If relative path, prepend BACKUP_DIR
    if [[ ! "$backup_file" = /* ]]; then
        backup_file="$BACKUP_DIR/$backup_file"
    fi

    if [ ! -f "$backup_file" ]; then
        log_error "File not found: $backup_file"
        exit 1
    fi

    verify_backup "$backup_file"
}

main "$@"
