#!/bin/bash
################################################################################
#                       NBA XL PREDICTION SYSTEM                              #
#                      Hybrid Dual-Filter v2.0                                #
################################################################################

set -e
set -u
set -o pipefail

# Debug mode: Enable with DEBUG=1 ./nba-predictions.sh morning
# Or: ./nba-predictions.sh --debug morning
DEBUG=${DEBUG:-0}
export DEBUG  # Export so Python scripts can read it

# Modern terminal color palette - smooth, muted tones
TEAL='\033[38;5;80m'
MINT='\033[38;5;121m'
GREEN='\033[38;5;78m'
YELLOW='\033[38;5;221m'
RED='\033[38;5;203m'
ORANGE='\033[38;5;215m'
LAVENDER='\033[38;5;183m'
SEAFOAM='\033[38;5;116m'
GREY='\033[38;5;243m'
DARK_GREY='\033[38;5;237m'
SOFT_WHITE='\033[38;5;252m'
BRIGHT_WHITE='\033[38;5;255m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Semantic colors
PRIMARY="$TEAL"
SUCCESS="$GREEN"
WARNING="$YELLOW"
ERROR="$RED"
INFO="$MINT"
MUTED="$GREY"
SUBTLE="$DARK_GREY"

# Market colors
POINTS_COLOR="$LAVENDER"
REBOUNDS_COLOR="$SEAFOAM"
PLAYER_NAME_COLOR="$BRIGHT_WHITE"

# System Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/betting_xl/logs"
PREDICTIONS_DIR="$SCRIPT_DIR/betting_xl/predictions"
DATE_STR=$(date +%Y-%m-%d)
TIME_STR=$(date +%H:%M:%S)
LOG_FILE="$LOG_DIR/pipeline_${DATE_STR}.log"

# Season calculation: NBA season uses END year (2024-25 season = 2025, 2025-26 season = 2026)
# Season starts in October, so Oct-Dec uses next year's season number
current_month=$(date +%-m)
current_year=$(date +%Y)
if [ "$current_month" -ge 10 ]; then
    CURRENT_SEASON=$((current_year + 1))
else
    CURRENT_SEASON=$current_year
fi

# Database - use environment variables with sensible defaults
DB_HOST="${NBA_DB_HOST:-localhost}"
DB_PORT="${NBA_INT_DB_PORT:-5539}"
DB_NAME="${NBA_INT_DB_NAME:-nba_intelligence}"
DB_USER="${DB_USER:-nba_user}"
# Validate DB_PASSWORD is set (used per-command with PGPASSWORD=)
: "${DB_PASSWORD:?DB_PASSWORD environment variable is required}"

# Export DB credentials for Python scripts (they read from os.getenv)
export DB_USER DB_PASSWORD

# Python path for imports
export PYTHONPATH="$PROJECT_ROOT"

mkdir -p "$LOG_DIR" "$PREDICTIONS_DIR"

# Terminal width for centering
TERM_WIDTH=$(tput cols 2>/dev/null || echo 80)

################################################################################
# DISPLAY + AESTHETIC HELPERS
################################################################################

center() {
    local text="$1"
    local width=${#text}
    local padding=$(( (TERM_WIDTH - width) / 2 ))
    (( padding < 0 )) && padding=0
    printf "%${padding}s%s\n" "" "$text"
}

divider() {
    local char="${1:--}"
    printf "${SUBTLE}%${TERM_WIDTH}s${NC}\n" | tr ' ' '-'
}

light_divider() {
    printf "${SUBTLE}%${TERM_WIDTH}s${NC}\n" | tr ' ' '.'
}

section_divider() {
    echo ""
    printf "${MUTED}%${TERM_WIDTH}s${NC}\n" | tr ' ' '='
    echo ""
}

header() {
    local title="$1"
    local subtitle="${2:-}"
    echo ""
    echo -e "${BOLD}${PRIMARY}> ${title}${NC}"
    [ -n "$subtitle" ] && echo -e "  ${MUTED}${subtitle}${NC}"
    echo ""
}

section() {
    local title="$1"
    local detail="${2:-}"
    {
        echo ""
        echo -e "${BOLD}${INFO}* ${title}${NC}"
        [ -n "$detail" ] && echo -e "  ${MUTED}${detail}${NC}"
    } | tee -a "$LOG_FILE"
}

complete() {
    local title="$1"
    local detail="${2:-}"
    echo ""
    divider
    echo -e "${BOLD}${SUCCESS}✓ ${title}${NC}"
    [ -n "$detail" ] && echo -e "  ${MUTED}${detail}${NC}"
    divider
    echo ""
}

status() {
    local symbol="$1"
    local message="$2"
    local color="$3"
    echo -e "  ${color}${symbol}${NC} ${message}" | tee -a "$LOG_FILE"
}

success() { status "+" "$1" "$SUCCESS"; }
error()   { status "x" "$1" "$ERROR"; }
warning() { status "!" "$1" "$WARNING"; }
info()    { status "-" "$1" "$INFO"; }

progress() {
    local current=$1
    local total=$2
    local desc="$3"
    local percent=$((current * 100 / total))
    local segments=30
    local filled=$((percent * segments / 100))
    local bar=""
    for ((i=0; i<segments; i++)); do
        if [ $i -lt $filled ]; then
            bar+="#"
        else
            bar+="-"
        fi
    done
    echo -e "  ${INFO}${desc}${NC} ${PRIMARY}${bar}${NC} ${BOLD}${percent}%${NC}"
}

system_banner() {
    clear
    echo ""
    divider
    echo ""
    echo -e "${BOLD}${PRIMARY}  NBA XL Prediction System${NC}"
    echo -e "  ${MUTED}V3 OVER/UNDER (POINTS) | Tier A (REBOUNDS)${NC}"
    echo ""
    echo -e "  ${SOFT_WHITE}${DATE_STR}${NC}  ${MUTED}${TIME_STR}${NC}"
    echo ""
    divider
}

intro_sequence() {
    # Clean, minimal intro
    :
}

render_pick_section() {
    local label="$1"
    local accent="$2"
    local filter="$3"
    local file="$4"

    local picks
    picks=$(jq -c "$filter" "$file" 2>/dev/null)
    if [ -z "$picks" ]; then
        return
    fi

    local count
    count=$(echo "$picks" | grep -c '^') || count=0

    section_divider
    echo -e "  ${BOLD}${accent}${label}${NC}  ${MUTED}(${count} picks)${NC}"
    echo ""
    light_divider

    while IFS= read -r pick_json; do
        [ -z "$pick_json" ] && continue
        player=$(echo "$pick_json" | jq -r '.player_name')
        market=$(echo "$pick_json" | jq -r '.stat_type')
        side=$(echo "$pick_json" | jq -r '.side')
        best_line=$(echo "$pick_json" | jq -r '.best_line')
        best_book=$(echo "$pick_json" | jq -r '.best_book')
        edge=$(echo "$pick_json" | jq -r '.edge | tonumber | . * 100 | round / 100')
        prediction=$(echo "$pick_json" | jq -r '.prediction | tonumber | . * 10 | round / 10')
        prob=$(echo "$pick_json" | jq -r '.p_over | tonumber | . * 1000 | round / 1000')
        opp_rank=$(echo "$pick_json" | jq -r '.opp_rank // ""')
        expected_wr=$(echo "$pick_json" | jq -r '.expected_wr // ""')
        filter_tier=$(echo "$pick_json" | jq -r '.filter_tier // "unknown"')

        print_pick_card "$player" "$market" "$side" "$best_line" "$best_book" "$edge" "$prediction" "$prob" "$opp_rank" "$expected_wr" "$accent" "$filter_tier"
    done <<< "$picks"
}

print_pick_card() {
    local player="$1"
    local market="$2"
    local side="$3"
    local best_line="$4"
    local best_book="$5"
    local edge="$6"
    local prediction="$7"
    local prob="$8"
    local opp_rank="$9"
    local expected_wr="${10}"
    local accent="${11:-$PRIMARY}"
    local tier="${12:-unknown}"

    # Convert probability to percentage
    local prob_pct=$(echo "$prob * 100" | bc -l | xargs printf "%.0f")

    # Format expected WR from tier
    local wr_display="N/A"
    case "$tier" in
        JAN_PRIME_OVER) wr_display="91.7%" ;;
        JAN_CONFIDENT_OVER) wr_display="87.5%" ;;
        JAN_LINE_OVER) wr_display="82.4%" ;;
        V3_ELITE_OVER|V3_ELITE_UNDER) wr_display="85%" ;;
        V3_STANDARD_OVER|V3_STANDARD_UNDER) wr_display="75%" ;;
        STAR_V3|star_tier) wr_display="80%" ;;
        X|tier_x) wr_display="75%" ;;
        A|tier_a) wr_display="70%" ;;
        *)
            if [ -n "$expected_wr" ] && [ "$expected_wr" != "null" ]; then
                wr_display="${expected_wr}%"
            fi
            ;;
    esac

    echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}"
    echo -e "  ${MUTED}|${NC} ${market} ${side} ${BOLD}${best_line}${NC} @ ${best_book}"
    printf "  ${MUTED}|${NC}  %-12s %b\n" "Projection:" "${BOLD}${ORANGE}${prediction}${NC}"
    printf "  ${MUTED}|${NC}  %-12s %b\n" "Edge:" "${BOLD}${SUCCESS}+${edge}${NC}"

    # Opponent Defense Rank (if available)
    if [ -n "$opp_rank" ] && [ "$opp_rank" != "null" ]; then
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Opp Defense:" "#${opp_rank}"
    fi

    printf "  ${MUTED}|${NC}  %-12s %b\n" "Confidence:" "${BOLD}${prob_pct}%${NC}"
    printf "  ${MUTED}|${NC}  %-12s %b\n" "Expected WR:" "${BOLD}${SUCCESS}${wr_display}${NC}"
    echo ""
}

# Unified pick card for Pro Tier picks
# Format: Player | Market OVER line @ Book | Projection | Edge | L5 | Opp | Confidence | WR
print_pro_pick() {
    local player="$1"
    local market="$2"
    local line="$3"
    local book="${4:-Underdog}"
    local projection="$5"
    local edge="$6"           # projection_diff or "N/A"
    local l5_rate="$7"        # L5 hit rate % or "N/A"
    local opp_rank="$8"       # Opponent defense rank
    local confidence="$9"     # Season hit rate % or probability
    local expected_wr="${10}"

    echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}"
    echo -e "  ${MUTED}|${NC} ${market} OVER ${BOLD}${line}${NC} @ ${book}"
    printf "  ${MUTED}|${NC}  %-12s %b\n" "Projection:" "${BOLD}${ORANGE}${projection}${NC}"

    # Edge (projection diff)
    if [ "$edge" != "N/A" ] && [ -n "$edge" ] && [ "$edge" != "null" ]; then
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Edge:" "${BOLD}${SUCCESS}+${edge}${NC}"
    fi

    # L5 Hit Rate
    if [ "$l5_rate" != "N/A" ] && [ -n "$l5_rate" ] && [ "$l5_rate" != "null" ]; then
        printf "  ${MUTED}|${NC}  %-12s %b\n" "L5 Hit Rate:" "${BOLD}${l5_rate}%${NC}"
    fi

    # Opponent Defense
    if [ "$opp_rank" != "N/A" ] && [ -n "$opp_rank" ] && [ "$opp_rank" != "null" ]; then
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Opp Defense:" "#${opp_rank}"
    fi

    # Confidence (season hit rate or probability)
    if [ "$confidence" != "N/A" ] && [ -n "$confidence" ] && [ "$confidence" != "null" ]; then
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Confidence:" "${BOLD}${confidence}%${NC}"
    fi

    # Expected WR
    printf "  ${MUTED}|${NC}  %-12s %b\n" "Expected WR:" "${BOLD}${SUCCESS}${expected_wr}%${NC}"
    echo ""
}

################################################################################
# CORE FUNCTIONS
################################################################################

silent_check() {
    "$@" >> "$LOG_FILE" 2>&1
    return $?
}

verbose_run() {
    # Run command with output to both stdout and log file
    # Note: pipefail is set at script start, so we restore it after tee
    set +o pipefail  # Disable to prevent tee from masking exit codes
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail  # Restore (we know it was set at script start)
    return $exit_code
}

db_check() {
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1
}

get_coverage() {
    # DATE_STR comes from date command so it's safe
    # Note: Use <> instead of != to avoid shell escaping issues
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT ROUND(100.0 * COUNT(CASE WHEN opponent_team <> '' AND opponent_team IS NOT NULL AND is_home IS NOT NULL THEN 1 END) / NULLIF(COUNT(*), 0), 1)
         FROM nba_props_xl WHERE game_date = '$DATE_STR';" 2>/dev/null | tr -d ' '
}

get_props_count() {
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM nba_props_xl WHERE game_date = CURRENT_DATE;" 2>/dev/null | tr -d ' '
}

################################################################################
# MORNING PROTOCOL
################################################################################

morning_workflow() {
    header "Morning Workflow" "Data collection & enrichment"

    section "Fetching Props" "Multi-source collection from 7 sportsbooks"
    if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_all.py"; then
        echo ""
        success "Props fetched successfully"
    else
        error "Failed to fetch props"
        return 1
    fi

    section "Loading to Database"
    latest_props=$(ls -t "$SCRIPT_DIR/betting_xl/lines/all_sources_"*.json 2>/dev/null | head -1)
    if [ -f "$latest_props" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_props_to_db.py" --file "$latest_props" --skip-mongodb; then
            echo ""
            local count=$(get_props_count)
            success "Loaded ${count} props to database"
        else
            error "Database load failed"
            return 1
        fi
    else
        error "No props file found"
        return 1
    fi

    section "Fetching Cheatsheet Data" "BettingPros recommendations"
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" --platform underdog; then
            echo ""
            # Load to database
            latest_cheatsheet=$(ls -t "$SCRIPT_DIR/betting_xl/lines/cheatsheet_underdog_"*.json 2>/dev/null | head -1)
            if [ -f "$latest_cheatsheet" ]; then
                if verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_cheatsheet_to_db.py" --file "$latest_cheatsheet"; then
                    success "Cheatsheet data loaded"
                else
                    warning "Cheatsheet load failed"
                fi
            fi
        else
            warning "Cheatsheet fetch failed"
        fi
    else
        info "Cheatsheet fetcher not configured"
    fi

    section "Enriching Matchup Data" "Adding opponent & home/away context"

    if verbose_run python3 "$SCRIPT_DIR/betting_xl/enrich_props_with_matchups.py" --date "$DATE_STR"; then
        success "Enriched $DATE_STR"
    else
        warning "Enrichment incomplete for $DATE_STR"
    fi

    # Verify coverage for today
    echo ""
    local coverage
    coverage=$(get_coverage)
    coverage="${coverage:-0}"  # Default to 0 if empty
    if [ "${coverage%.*}" -ge 95 ]; then
        success "Enrichment complete (${coverage}% coverage for today)"
    else
        warning "Coverage below target: ${coverage}%"
        return 1
    fi

    section "Fetching Game Results" "Daily stats for completed games"
    if [ -f "$SCRIPT_DIR/scripts/fetch_daily_stats.py" ]; then
        # Fetch yesterday's games only (run daily = fresh data only)
        if verbose_run python3 "$SCRIPT_DIR/scripts/fetch_daily_stats.py" --days 1; then
            success "Game results updated"
        else
            info "No new games to process"
        fi
    else
        info "Game fetch script not found"
    fi

    section "Populating Actual Values" "Update props with game results"
    if [ -f "$SCRIPT_DIR/betting_xl/populate_actual_values.py" ]; then
        # Populate last 7 days to catch any gaps (season filter in script)
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/populate_actual_values.py" --days 7; then
            success "Actual values populated"
        else
            info "Actual values update skipped"
        fi
    else
        info "Populate script not found"
    fi

    section "JSONCalibrator Status" "Checking probability calibration from recent results"
    if [ -f "$SCRIPT_DIR/models/json_calibrator.py" ]; then
        if python3 "$SCRIPT_DIR/models/json_calibrator.py" status --lookback 21 >> "$LOG_FILE" 2>&1; then
            success "Calibrator status checked (21-day lookback)"
        else
            warning "Calibrator status check failed"
        fi
    else
        info "JSONCalibrator not configured"
    fi

    section "Updating Injury Reports"
    if [ -f "$SCRIPT_DIR/scripts/update_injuries_NOW.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/update_injuries_NOW.py"; then
            success "Injury data synchronized"
        else
            info "Injury update skipped"
        fi
    else
        info "Injury tracker not configured"
    fi

    section "Loading Team Games" "Incremental from NBA API (1 API call)"
    if [ -f "$SCRIPT_DIR/scripts/loaders/load_nba_games_incremental.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/load_nba_games_incremental.py"; then
            success "Team games loaded"
        else
            info "Team games load skipped"
        fi
    else
        info "Team games loader not found"
    fi

    section "Updating Team Season Stats" "Pace/ratings for current season"
    if [ -f "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" --season "$CURRENT_SEASON"; then
            success "Team season stats updated"
        else
            warning "Team season stats update skipped"
        fi
    else
        info "Team season stats script not found"
    fi

    section "Loading Team Advanced Stats" "Real PACE from NBA API (1 API call)"
    if [ -f "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py"; then
            success "Team pace/ratings loaded from NBA API"
        else
            info "Team advanced stats load skipped"
        fi
    else
        info "Team advanced stats loader not found"
    fi

    section "Fetching Vegas Lines" "Game spreads & totals for feature extraction"
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" --date "$DATE_STR" --save-to-db; then
            success "Vegas lines fetched and saved"
        else
            warning "Vegas lines fetch failed"
        fi
    else
        info "Vegas lines fetcher not configured"
    fi

    section "Updating Rolling Stats"
    local latest_game
    latest_game=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p 5536 -U "$DB_USER" -d nba_players -t -c \
        "SELECT MAX(game_date) FROM player_game_logs WHERE season = $CURRENT_SEASON;" 2>/dev/null | tr -d ' ')
    latest_game="${latest_game:-}"

    local yesterday
    yesterday=$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
    if [ -n "$latest_game" ] && { [ "$latest_game" = "$DATE_STR" ] || [ "$latest_game" = "$yesterday" ]; }; then
        if [ -f "$SCRIPT_DIR/scripts/update_rolling_stats_incremental.py" ]; then
            if verbose_run python3 "$SCRIPT_DIR/scripts/update_rolling_stats_incremental.py"; then
                success "Rolling statistics updated"
            else
                info "Stats update skipped"
            fi
        fi
    else
        info "No new games to process"
    fi

    section "Updating Minutes Projections"
    if [ -f "$SCRIPT_DIR/scripts/loaders/calculate_minutes_projections.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/calculate_minutes_projections.py" --update; then
            success "Minutes projections refreshed"
        else
            warning "Minutes projections update skipped"
        fi
    else
        info "Minutes projections script not found"
    fi

    section "Updating Prop Performance History" "Bayesian hit rate calculations"
    if [ -f "$SCRIPT_DIR/scripts/compute_prop_history.py" ]; then
        # Incremental update for the current season - only process last 7 days of props
        if verbose_run python3 "$SCRIPT_DIR/scripts/compute_prop_history.py" --season "$CURRENT_SEASON" --incremental --days 7; then
            success "Prop performance history updated"
        else
            warning "Prop history update skipped"
        fi
    else
        info "Prop history script not found"
    fi

    complete "Morning Workflow Complete" "Ready for evening predictions"
}

################################################################################
# EVENING PROTOCOL
################################################################################

evening_workflow() {
    header "Evening Workflow" "Generate predictions with hybrid dual-filter"

    section "Quick Health Check" "Verifying system readiness"
    if [ -f "$SCRIPT_DIR/betting_xl/health_check.py" ]; then
        if python3 "$SCRIPT_DIR/betting_xl/health_check.py" --quick >> "$LOG_FILE" 2>&1; then
            success "Health check passed"
        else
            warning "Health check failed - proceeding with caution"
        fi
    fi

    section "Pre-Flight Checks" "Validating data quality"

    if [ -f "$SCRIPT_DIR/betting_xl/config/data_freshness_validator.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/config/data_freshness_validator.py"; then
            success "Data freshness validated"
        else
            error "Stale data detected"
            return 1
        fi
    fi

    if [ -f "$SCRIPT_DIR/betting_xl/monitor.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/monitor.py"; then
            success "Performance thresholds OK"
        else
            error "Stop-loss triggered - system paused"
            return 1
        fi
    fi

    section "Refreshing Lines" "Capturing line movements"
    if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_all.py"; then
        echo ""
        success "Player prop lines refreshed"
    else
        error "Props refresh failed"
        return 1
    fi

    # Load refreshed props to database
    latest_props=$(ls -t "$SCRIPT_DIR/betting_xl/lines/all_sources_"*.json 2>/dev/null | head -1)
    if [ -f "$latest_props" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_props_to_db.py" --file "$latest_props" --skip-mongodb; then
            success "Refreshed props loaded to database"
        fi
    fi

    # Refresh vegas lines (capture spread/total movements)
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" --date "$DATE_STR" --save-to-db; then
            success "Vegas lines refreshed"
        else
            warning "Vegas refresh failed"
        fi
    fi

    # Refresh cheatsheet data (latest projections & hit rates)
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" --platform underdog; then
            latest_cheatsheet=$(ls -t "$SCRIPT_DIR/betting_xl/lines/cheatsheet_underdog_"*.json 2>/dev/null | head -1)
            if [ -f "$latest_cheatsheet" ] && [ -f "$SCRIPT_DIR/betting_xl/loaders/load_cheatsheet_to_db.py" ]; then
                if verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_cheatsheet_to_db.py" --file "$latest_cheatsheet"; then
                    success "Cheatsheet data refreshed"
                fi
            fi
        fi
    fi

    section "Enriching Matchup Data" "Adding opponent & home/away context"
    if verbose_run python3 "$SCRIPT_DIR/betting_xl/enrich_props_with_matchups.py" --date "$DATE_STR"; then
        echo ""
        success "Matchup data updated"
    fi

    # Verify coverage AFTER enrichment
    local coverage
    coverage=$(get_coverage)
    coverage="${coverage:-0}"  # Default to 0 if empty
    if [ "${coverage%.*}" -ge 95 ]; then
        success "Coverage ${coverage}% (target: 95%)"
    else
        error "Coverage ${coverage}% insufficient after enrichment"
        return 1
    fi

    section "Generating Predictions" "Odds API TIER 1/2 + Tier X + Tier A"

    if [ -f "$SCRIPT_DIR/betting_xl/generate_xl_predictions.py" ]; then
        # Odds API tiers (replaces V3 model - Jan 25, 2026):
        # TIER_1: mult<1.1 + R>=3 + opp>=11 + L5=100% → 100% WR
        # TIER_2: mult<1.2 + R>=4 + opp>=16 + L5>=80% + L15>=60% → 100% WR
        # XL model: Tier X (POINTS) + Tier A (REBOUNDS)
        # REBOUNDS: Tier A with spread>=2.0 → 69% WR
        if python3 "$SCRIPT_DIR/betting_xl/generate_xl_predictions.py" --output "$PREDICTIONS_DIR/xl_picks_${DATE_STR}.json" --underdog-only >> "$LOG_FILE" 2>&1; then
            local pick_file="$PREDICTIONS_DIR/xl_picks_${DATE_STR}.json"
            if [ -f "$pick_file" ] && command -v jq >/dev/null 2>&1; then
                local picks_count=$(jq '.total_picks // 0' "$pick_file")

                echo ""
                divider
                echo ""
                echo -e "  ${BOLD}${PRIMARY}Today's Picks${NC} | ${SOFT_WHITE}${DATE_STR}${NC}"
                echo -e "  ${MUTED}Odds API TIER 1/2 | POINTS: Tier X | REBOUNDS: Tier A${NC}"
                echo ""
                divider

                render_pick_section "POINTS" "$POINTS_COLOR" '.picks[] | select(.stat_type == "POINTS")' "$pick_file"
                render_pick_section "REBOUNDS" "$REBOUNDS_COLOR" '.picks[] | select(.stat_type == "REBOUNDS")' "$pick_file"
                render_pick_section "OTHER MARKETS" "$ORANGE" '.picks[] | select(.stat_type != "POINTS" and .stat_type != "REBOUNDS")' "$pick_file"

                divider
                echo ""
                echo -e "  ${BOLD}Summary${NC}"
                echo -e "  ${MUTED}|${NC} Total Picks: ${BOLD}${picks_count}${NC}"

                # Tier breakdown (Odds API + XL model tiers)
                local odds_tier1=$(jq '[.picks[] | select(.filter_tier == "TIER_1")] | length' "$pick_file")
                local odds_tier2=$(jq '[.picks[] | select(.filter_tier == "TIER_2")] | length' "$pick_file")
                local odds_total=$((odds_tier1 + odds_tier2))
                local tier_x=$(jq '.summary.by_tier.X // .summary.by_tier.tier_x // 0' "$pick_file")
                local tier_a=$(jq '.summary.by_tier.A // .summary.by_tier.tier_a // 0' "$pick_file")
                local star_tier=$(jq '.summary.by_tier.star_tier // 0' "$pick_file")

                echo -e "  ${MUTED}|${NC} Odds API: ${BOLD}${SUCCESS}${odds_total}${NC} (T1: ${odds_tier1}, T2: ${odds_tier2})  ${MUTED}|${NC}  Tier X: ${BOLD}${SUCCESS}${tier_x}${NC}  ${MUTED}|${NC}  Tier A: ${BOLD}${SUCCESS}${tier_a}${NC}  ${MUTED}|${NC}  STAR: ${BOLD}${YELLOW}${star_tier}${NC}"
                if [ "$odds_total" -gt 0 ]; then
                    echo -e "  ${MUTED}|${NC} Odds API: ${odds_tier1} TIER_1 (100% WR) + ${odds_tier2} TIER_2 (100% WR)"
                fi

                echo -e "  ${MUTED}|${NC} Odds API: Pick6 mult + BettingPros features"
                echo -e "  ${MUTED}|${NC} REBOUNDS: Tier A + Star tier"
                echo -e "  ${MUTED}+${NC} Target: Odds API 100% / Tier X 74% / Tier A 69%"
                echo ""
                divider
                echo ""

                if [ "$picks_count" -lt 2 ]; then
                    warning "Low volume - filtering may be too strict"
                elif [ "$picks_count" -gt 10 ]; then
                    warning "High volume - review filter settings"
                fi
            else
                success "Predictions generated"
            fi
        else
            error "Prediction generation failed"
            return 1
        fi
    else
        error "XL system not found"
        return 1
    fi

    section "Generating Pro Picks" "TIGHTENED Jan 16 + Injury Filter (75-88% WR, ~8 picks/day)"

    if [ -f "$SCRIPT_DIR/betting_xl/generate_cheatsheet_picks.py" ]; then
        local pro_file="$PREDICTIONS_DIR/pro_picks_${DATE_STR}.json"
        if python3 "$SCRIPT_DIR/betting_xl/generate_cheatsheet_picks.py" --output "$pro_file" >> "$LOG_FILE" 2>&1; then
            if [ -f "$pro_file" ] && command -v jq >/dev/null 2>&1; then
                local pro_picks_count=$(jq '.total_picks // 0' "$pro_file")

                if [ "$pro_picks_count" -gt 0 ]; then
                    # Get counts by stat type
                    local points_count=$(jq '[.picks[] | select(.stat_type == "POINTS")] | length' "$pro_file")
                    local assists_count=$(jq '[.picks[] | select(.stat_type == "ASSISTS")] | length' "$pro_file")

                    # Display POINTS Pro picks
                    if [ "$points_count" -gt 0 ]; then
                        echo ""
                        divider
                        echo ""
                        echo -e "  ${BOLD}${ORANGE}Pro Tier${NC} | ${MUTED}POINTS${NC}  ${MUTED}(${points_count} picks)${NC}"
                        echo -e "  ${MUTED}Filter: Season 70%+ | EV 20%+ | No injury returns | Expected 88% WR${NC}"
                        echo ""
                        divider

                        local points_picks
                        points_picks=$(jq -c '.picks[] | select(.stat_type == "POINTS")' "$pro_file" 2>/dev/null)
                        while IFS= read -r pick_json; do
                            [ -z "$pick_json" ] && continue
                            player=$(echo "$pick_json" | jq -r '.player_name')
                            line=$(echo "$pick_json" | jq -r '.line')
                            projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
                            edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
                            l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
                            opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
                            season_rate=$(echo "$pick_json" | jq -r '.hit_rate_season | tonumber | . * 100 | round')
                            expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

                            print_pro_pick "$player" "POINTS" "$line" "Underdog" "$projection" "$edge" "$l5_rate" "$opp_rank" "$season_rate" "$expected_wr"
                        done <<< "$points_picks"
                    fi

                    # Display ASSISTS Pro picks
                    if [ "$assists_count" -gt 0 ]; then
                        echo ""
                        divider
                        echo ""
                        echo -e "  ${BOLD}${ORANGE}Pro Tier${NC} | ${MUTED}ASSISTS${NC}  ${MUTED}(${assists_count} picks)${NC}"
                        echo -e "  ${MUTED}Filter: L5/L15 60%+ | Opp 21+ | Rating 3+ | Expected 73% WR${NC}"
                        echo ""
                        divider

                        local assists_picks
                        assists_picks=$(jq -c '.picks[] | select(.stat_type == "ASSISTS")' "$pro_file" 2>/dev/null)
                        while IFS= read -r pick_json; do
                            [ -z "$pick_json" ] && continue
                            player=$(echo "$pick_json" | jq -r '.player_name')
                            line=$(echo "$pick_json" | jq -r '.line')
                            projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
                            edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
                            l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
                            opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
                            season_rate=$(echo "$pick_json" | jq -r '.hit_rate_season | tonumber | . * 100 | round')
                            expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

                            print_pro_pick "$player" "ASSISTS" "$line" "Underdog" "$projection" "$edge" "$l5_rate" "$opp_rank" "$season_rate" "$expected_wr"
                        done <<< "$assists_picks"
                    fi

                    # Display REBOUNDS Pro picks
                    local rebounds_count=$(jq '[.picks[] | select(.stat_type == "REBOUNDS")] | length' "$pro_file")
                    if [ "$rebounds_count" -gt 0 ]; then
                        echo ""
                        divider
                        echo ""
                        echo -e "  ${BOLD}${ORANGE}Pro Tier${NC} | ${MUTED}REBOUNDS${NC}  ${MUTED}(${rebounds_count} picks)${NC}"
                        echo -e "  ${MUTED}Filter: L5/L15/Season 60%+ | Opp 11+ | Rating 3+ | Expected 80% WR${NC}"
                        echo ""
                        divider

                        local rebounds_picks
                        rebounds_picks=$(jq -c '.picks[] | select(.stat_type == "REBOUNDS")' "$pro_file" 2>/dev/null)
                        while IFS= read -r pick_json; do
                            [ -z "$pick_json" ] && continue
                            player=$(echo "$pick_json" | jq -r '.player_name')
                            line=$(echo "$pick_json" | jq -r '.line')
                            projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
                            edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
                            l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
                            opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
                            season_rate=$(echo "$pick_json" | jq -r '.hit_rate_season | tonumber | . * 100 | round')
                            expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

                            print_pro_pick "$player" "REBOUNDS" "$line" "Underdog" "$projection" "$edge" "$l5_rate" "$opp_rank" "$season_rate" "$expected_wr"
                        done <<< "$rebounds_picks"
                    fi

                    # Display COMBO STATS (PA, PR, RA) Pro picks
                    local combo_count=$(jq '[.picks[] | select(.stat_type == "PA" or .stat_type == "PR" or .stat_type == "RA")] | length' "$pro_file")
                    if [ "$combo_count" -gt 0 ]; then
                        echo ""
                        divider
                        echo ""
                        echo -e "  ${BOLD}${ORANGE}Pro Tier${NC} | ${MUTED}COMBO STATS${NC}  ${MUTED}(${combo_count} picks)${NC}"
                        echo -e "  ${MUTED}Filter: L15 70%+ Opp 11+ (PA/PR) | Opp 16+ Diff 2+ (RA) | Expected 71-86% WR${NC}"
                        echo ""
                        divider

                        local combo_picks
                        combo_picks=$(jq -c '.picks[] | select(.stat_type == "PA" or .stat_type == "PR" or .stat_type == "RA")' "$pro_file" 2>/dev/null)
                        while IFS= read -r pick_json; do
                            [ -z "$pick_json" ] && continue
                            player=$(echo "$pick_json" | jq -r '.player_name')
                            stat_type=$(echo "$pick_json" | jq -r '.stat_type')
                            line=$(echo "$pick_json" | jq -r '.line')
                            projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
                            edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
                            l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
                            opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
                            season_rate=$(echo "$pick_json" | jq -r '.hit_rate_season | tonumber | . * 100 | round')
                            expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

                            # Expand stat type names
                            case "$stat_type" in
                                PA) stat_label="PTS+AST" ;;
                                PR) stat_label="PTS+REB" ;;
                                RA) stat_label="REB+AST" ;;
                                *) stat_label="$stat_type" ;;
                            esac

                            print_pro_pick "$player" "$stat_label" "$line" "Underdog" "$projection" "$edge" "$l5_rate" "$opp_rank" "$season_rate" "$expected_wr"
                        done <<< "$combo_picks"
                    fi

                    divider
                    echo ""
                    echo -e "  ${BOLD}Pro Summary${NC}"
                    echo -e "  ${MUTED}|${NC} POINTS: ${BOLD}${points_count}${NC} picks (88% WR)"
                    echo -e "  ${MUTED}|${NC} ASSISTS: ${BOLD}${assists_count}${NC} picks (73% WR)"
                    echo -e "  ${MUTED}|${NC} REBOUNDS: ${BOLD}${rebounds_count}${NC} picks (80% WR)"
                    echo -e "  ${MUTED}|${NC} COMBO: ${BOLD}${combo_count}${NC} picks (71-86% WR)"
                    echo -e "  ${MUTED}|${NC} Total: ${BOLD}${pro_picks_count}${NC} picks"
                    echo -e "  ${MUTED}+${NC} Tier: ${BOLD}${ORANGE}PRO${NC} (TIGHTENED Jan 16 + Injury Filter)"
                    echo ""
                    divider
                    echo ""

                    success "Pro picks generated"
                else
                    info "No Pro picks today (filters not met)"
                fi
            else
                success "Pro picks generated"
            fi
        else
            warning "Pro picks generation skipped"
        fi
    else
        info "Pro generator not configured"
    fi

    section "Generating Odds API Picks" "Pick6 multipliers + BettingPros features"

    if [ -f "$SCRIPT_DIR/betting_xl/generate_odds_api_picks.py" ]; then
        local odds_file="$PREDICTIONS_DIR/odds_api_picks_${DATE_STR//-/}.json"
        if python3 "$SCRIPT_DIR/betting_xl/generate_odds_api_picks.py" \
            --date "$DATE_STR" \
            --output "$odds_file" >> "$LOG_FILE" 2>&1; then

            if [ -f "$odds_file" ] && command -v jq >/dev/null 2>&1; then
                local odds_picks_count=$(jq '.total_picks // 0' "$odds_file")

                if [ "$odds_picks_count" -gt 0 ]; then
                    echo ""
                    divider
                    echo ""
                    echo -e "  ${BOLD}${ORANGE}Odds API Picks${NC}  ${MUTED}(${odds_picks_count} picks)${NC}"
                    echo -e "  ${MUTED}Pick6 multipliers + BettingPros cheatsheet features${NC}"
                    echo ""
                    divider

                    local odds_picks
                    odds_picks=$(jq -c '.picks[]' "$odds_file" 2>/dev/null)
                    while IFS= read -r pick_json; do
                        [ -z "$pick_json" ] && continue
                        player=$(echo "$pick_json" | jq -r '.player_name')
                        market=$(echo "$pick_json" | jq -r '.stat_type')
                        line=$(echo "$pick_json" | jq -r '.line')
                        mult=$(echo "$pick_json" | jq -r '.pick6_multiplier | tonumber | . * 100 | round / 100')
                        projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
                        edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
                        l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
                        opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
                        filter_name=$(echo "$pick_json" | jq -r '.filter_name')
                        expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

                        echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}"
                        echo -e "  ${MUTED}|${NC} ${market} OVER ${BOLD}${line}${NC} @ Pick6"
                        printf "  ${MUTED}|${NC}  %-12s %b\n" "Multiplier:" "${BOLD}${ORANGE}${mult}x${NC}"
                        printf "  ${MUTED}|${NC}  %-12s %b\n" "Projection:" "${BOLD}${projection}${NC}"
                        [ "$edge" != "null" ] && [ -n "$edge" ] && printf "  ${MUTED}|${NC}  %-12s %b\n" "Edge:" "${BOLD}${SUCCESS}+${edge}${NC}"
                        [ "$l5_rate" != "null" ] && [ -n "$l5_rate" ] && printf "  ${MUTED}|${NC}  %-12s %b\n" "L5 Hit Rate:" "${BOLD}${l5_rate}%${NC}"
                        [ "$opp_rank" != "null" ] && [ -n "$opp_rank" ] && printf "  ${MUTED}|${NC}  %-12s %b\n" "Opp Defense:" "#${opp_rank}"
                        printf "  ${MUTED}|${NC}  %-12s %b\n" "Filter:" "${filter_name}"
                        printf "  ${MUTED}|${NC}  %-12s %b\n" "Expected WR:" "${BOLD}${SUCCESS}${expected_wr}%${NC}"
                        echo ""
                    done <<< "$odds_picks"

                    divider
                    echo ""
                    echo -e "  ${BOLD}Odds API Summary${NC}"
                    echo -e "  ${MUTED}|${NC} Total: ${BOLD}${odds_picks_count}${NC} picks"
                    echo -e "  ${MUTED}+${NC} Source: Pick6 multipliers + BettingPros cheatsheet"
                    echo ""
                    divider
                    echo ""

                    success "Odds API picks generated"
                else
                    info "No Odds API picks today (filters not met)"
                fi
            else
                success "Odds API picks generated"
            fi
        else
            warning "Odds API picks generation skipped"
        fi
    else
        info "Odds API generator not configured"
    fi

    complete "Evening Workflow Complete" "Picks ready for review"
}

################################################################################
# HEALTH CHECK
################################################################################

health_check() {
    header "System Health Check" "Diagnostics & status"

    echo ""
    echo -e "${BOLD}${INFO}* System Components${NC}" | tee -a "$LOG_FILE"

    local status=0

    if db_check; then
        success "Database connection"
    else
        error "Database offline"
        status=1
    fi

    local model_count
    model_count=$(ls "$SCRIPT_DIR/models/saved_xl"/*.pkl 2>/dev/null | wc -l)
    model_count="${model_count:-0}"
    if [ "$model_count" -ge 24 ]; then
        success "XL models (${model_count}/24 files)"
    else
        error "XL models incomplete (${model_count}/24)"
        status=1
    fi

    local props_count
    props_count=$(get_props_count)
    props_count="${props_count:-0}"
    if [ "$props_count" -gt 50 ]; then
        success "Props available (${props_count})"
    else
        warning "Low prop volume (${props_count})"
        status=1
    fi

    local coverage
    coverage=$(get_coverage)
    coverage="${coverage:-0}"  # Default to 0 if empty
    if [ "${coverage%.*}" -ge 95 ]; then
        success "Matchup coverage (${coverage}%)"
    else
        warning "Coverage below target (${coverage}%)"
        status=1
    fi

    # Check vegas data coverage for today
    local vegas_count total_games
    vegas_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p 5537 -U "$DB_USER" -d nba_games -t -c \
        "SELECT COUNT(*) FROM games WHERE game_date = '$DATE_STR' AND vegas_spread IS NOT NULL;" 2>/dev/null | tr -d ' ')
    vegas_count="${vegas_count:-0}"
    total_games=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p 5537 -U "$DB_USER" -d nba_games -t -c \
        "SELECT COUNT(*) FROM games WHERE game_date = '$DATE_STR';" 2>/dev/null | tr -d ' ')
    total_games="${total_games:-0}"
    if [ "$total_games" -gt 0 ] && [ "$vegas_count" -eq "$total_games" ]; then
        success "Vegas lines (${vegas_count}/${total_games} games)"
    elif [ "$total_games" -gt 0 ]; then
        warning "Vegas coverage: ${vegas_count}/${total_games} games"
    else
        info "No games scheduled today"
    fi

    # Check cheatsheet data coverage
    local cheatsheet_count
    cheatsheet_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM cheatsheet_data WHERE game_date = '$DATE_STR';" 2>/dev/null | tr -d ' ')
    cheatsheet_count="${cheatsheet_count:-0}"
    if [ "$cheatsheet_count" -gt 50 ]; then
        success "Cheatsheet data (${cheatsheet_count} props)"
    elif [ "$cheatsheet_count" -gt 0 ]; then
        info "Cheatsheet data: ${cheatsheet_count} props"
    else
        warning "No cheatsheet data for today"
    fi

    local disk_usage
    disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | tr -d '%')
    disk_usage="${disk_usage:-0}"
    if [ "$disk_usage" -lt 90 ]; then
        success "Disk space (${disk_usage}% used)"
    else
        error "Disk space critical (${disk_usage}% used)"
        status=1
    fi

    echo ""
    if [ $status -eq 0 ]; then
        divider
        echo -e "${BOLD}${SUCCESS}  [OK] All Systems Operational${NC}"
        divider
        echo ""
    else
        divider
        echo -e "${BOLD}${ERROR}  [!!] System Issues Detected${NC}"
        divider
        echo ""
    fi

    return $status
}

################################################################################
# MAIN
################################################################################

main() {
    local command="${1:-}"

    # Handle --debug flag
    if [ "$command" = "--debug" ]; then
        DEBUG=1
        export DEBUG
        command="${2:-}"
        echo -e "${YELLOW}[DEBUG] Debug mode enabled${NC}"
    fi

    intro_sequence
    system_banner

    # Show debug status in banner if enabled
    if [ "$DEBUG" = "1" ]; then
        echo -e "  ${YELLOW}** DEBUG MODE ACTIVE **${NC}"
        echo ""
    fi

    case "$command" in
        morning)
            morning_workflow
            ;;
        evening)
            evening_workflow
            ;;
        health)
            health_check
            ;;
        *)
            echo ""
            echo -e "${BOLD}${BRIGHT_WHITE}Usage:${NC} $0 ${PRIMARY}[--debug] {morning|evening|health}${NC}"
            echo ""
            echo -e "  ${PRIMARY}morning${NC}   Data collection workflow (run ~10am EST)"
            echo -e "            ${MUTED}|${NC} Fetch props from 7 sportsbooks"
            echo -e "            ${MUTED}|${NC} Load to database"
            echo -e "            ${MUTED}|${NC} Enrich matchup data"
            echo -e "            ${MUTED}+${NC} Update injuries & rolling stats"
            echo ""
            echo -e "  ${PRIMARY}evening${NC}   Generate predictions (run ~5pm EST)"
            echo -e "            ${MUTED}|${NC} Pre-flight validation"
            echo -e "            ${MUTED}|${NC} Refresh lines (capture movements)"
            echo -e "            ${MUTED}|${NC} Run V3 models (OVER/UNDER)"
            echo -e "            ${MUTED}|${NC} POINTS: V3 Elite (82% WR) + V3 Standard (67% WR)"
            echo -e "            ${MUTED}+${NC} REBOUNDS: Tier A (69% WR)"
            echo ""
            echo -e "  ${PRIMARY}health${NC}    System diagnostics"
            echo -e "            ${MUTED}+${NC} Check DB, models, props, coverage"
            echo ""
            echo -e "  ${PRIMARY}--debug${NC}   Enable verbose logging (DEBUG=1)"
            echo -e "            ${MUTED}+${NC} Also: DEBUG=1 $0 morning"
            echo ""
            exit 1
            ;;
    esac
}

main "$@"
