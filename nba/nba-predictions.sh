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

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

LOG_DIR="$SCRIPT_DIR/betting_xl/logs"
PREDICTIONS_DIR="$SCRIPT_DIR/betting_xl/predictions"

# Force Eastern Time for all date calculations (NBA operates on EST)
export TZ=America/New_York

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

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================
# Verify required tools are installed

check_dependencies() {
    local missing=()

    # Required tools
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v psql >/dev/null 2>&1 || missing+=("postgresql-client (psql)")
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    command -v bc >/dev/null 2>&1 || missing+=("bc")

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo -e "\033[38;5;203m[ERROR] Missing required tools:\033[0m"
        for tool in "${missing[@]}"; do
            echo -e "  - $tool"
        done
        echo ""
        echo -e "\033[38;5;221m[INSTALL] On Ubuntu/Debian:\033[0m"
        echo "  sudo apt install python3 postgresql-client jq bc"
        echo ""
        echo -e "\033[38;5;221m[INSTALL] On macOS:\033[0m"
        echo "  brew install python@3.10 postgresql jq"
        echo ""
        exit 1
    fi
}

# Run dependency check
check_dependencies

# =============================================================================
# AUTO-SOURCE .env FILE
# =============================================================================
# Automatically load environment variables from .env if it exists

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a  # Automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a
fi

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================
# Check for required environment variables and provide helpful setup instructions

check_env_setup() {
    local missing=()
    local warnings=()

    # Required: DB_PASSWORD
    if [ -z "${DB_PASSWORD:-}" ]; then
        missing+=("DB_PASSWORD")
    fi

    # Warn if API keys missing (not fatal, but limits functionality)
    if [ -z "${BETTINGPROS_API_KEY:-}" ]; then
        warnings+=("BETTINGPROS_API_KEY (props fetching will fail)")
    fi
    if [ -z "${ODDS_API_KEY:-}" ]; then
        warnings+=("ODDS_API_KEY (Pick6 picks will be unavailable)")
    fi

    # If missing required vars, show setup instructions
    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo -e "\033[38;5;203m[ERROR] Missing required environment variables:\033[0m"
        for var in "${missing[@]}"; do
            echo -e "  - $var"
        done
        echo ""
        echo -e "\033[38;5;221m[SETUP] To fix this:\033[0m"
        echo ""
        echo "  1. Create/edit your .env file in the project root:"
        echo "     cp .env.example .env"
        echo "     nano .env"
        echo ""
        echo "  2. Set the required variables:"
        echo "     DB_USER=mlb_user"
        echo "     DB_PASSWORD=your_secure_password"
        echo "     BETTINGPROS_API_KEY=your_api_key"
        echo "     ODDS_API_KEY=your_api_key"
        echo ""
        echo "  3. Source the file before running:"
        echo "     source .env"
        echo "     export DB_USER DB_PASSWORD BETTINGPROS_API_KEY ODDS_API_KEY"
        echo "     ./nba/nba-predictions.sh"
        echo ""
        echo "  See docs/SETUP.md for complete setup instructions."
        echo ""
        exit 1
    fi

    # Show warnings for optional but recommended vars
    if [ ${#warnings[@]} -gt 0 ] && [ "${1:-}" != "health" ]; then
        echo -e "\033[38;5;221m[WARN] Missing optional environment variables:\033[0m"
        for var in "${warnings[@]}"; do
            echo -e "  - $var"
        done
        echo ""
    fi
}

# Run environment check (pass command name for context)
check_env_setup "${1:-}"

# Database - use environment variables with sensible defaults
DB_HOST="${NBA_DB_HOST:-localhost}"
DB_PORT="${NBA_INT_DB_PORT:-5539}"
DB_NAME="${NBA_INT_DB_NAME:-nba_intelligence}"
DB_USER="${DB_USER:-mlb_user}"

# Export DB credentials and API keys for Python scripts (they read from os.getenv)
export DB_USER DB_PASSWORD ODDS_API_KEY BETTINGPROS_API_KEY

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
        if [ -n "$detail" ]; then
            echo -e "  ${MUTED}${detail}${NC}"
        fi
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
    clear 2>/dev/null || true
    echo ""
    divider
    echo ""
    echo -e "${BOLD}${PRIMARY}  NBA XL Prediction System${NC}"
    echo -e "  ${MUTED}XL + V3 Models | Risk-Adjusted Stake Sizing${NC}"
    echo ""
    echo -e "  ${SOFT_WHITE}${DATE_STR}${NC}  ${MUTED}${TIME_STR}${NC}"
    echo ""
    divider
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
        edge_pct=$(echo "$pick_json" | jq -r '.edge_pct | tonumber | . * 10 | round / 10')
        prediction=$(echo "$pick_json" | jq -r '.prediction | tonumber | . * 10 | round / 10')
        prob=$(echo "$pick_json" | jq -r '.p_over | tonumber | . * 1000 | round / 1000')
        opp_rank=$(echo "$pick_json" | jq -r '.opp_rank // ""')
        expected_wr=$(echo "$pick_json" | jq -r '.expected_wr // ""')
        filter_tier=$(echo "$pick_json" | jq -r '.filter_tier // "unknown"')

        # New fields for better betting decisions
        opponent=$(echo "$pick_json" | jq -r '.opponent_team // ""')
        is_home=$(echo "$pick_json" | jq -r '.is_home // true')
        consensus_line=$(echo "$pick_json" | jq -r '.consensus_line | tonumber | . * 10 | round / 10')
        line_spread=$(echo "$pick_json" | jq -r '.line_spread | tonumber | . * 10 | round / 10')
        num_books=$(echo "$pick_json" | jq -r '.num_books // 1')
        confidence=$(echo "$pick_json" | jq -r '.confidence // "STANDARD"')

        # Get line distribution summary (which books have which lines)
        line_dist=$(echo "$pick_json" | jq -r '[.line_distribution[] | "\(.line):\(.count)"] | join(" | ")')
        alt_books=$(echo "$pick_json" | jq -r '[.top_3_lines[1:3][] | .book] | join(", ")')

        # Stake sizing (POINTS only)
        stake=$(echo "$pick_json" | jq -r '.recommended_stake // 1.0')
        stake_reason=$(echo "$pick_json" | jq -r '.stake_reason // ""')
        risk_level=$(echo "$pick_json" | jq -r '.risk_level // ""')

        print_pick_card "$player" "$market" "$side" "$best_line" "$best_book" "$edge" "$edge_pct" "$prediction" "$prob" "$opp_rank" "$expected_wr" "$accent" "$filter_tier" "$opponent" "$is_home" "$consensus_line" "$line_spread" "$num_books" "$confidence" "$line_dist" "$alt_books" "$stake" "$stake_reason" "$risk_level"
    done <<< "$picks"
}

print_pick_card() {
    local player="$1"
    local market="$2"
    local side="$3"
    local best_line="$4"
    local best_book="$5"
    local edge="$6"
    local edge_pct="$7"
    local prediction="$8"
    local prob="$9"
    local opp_rank="${10}"
    local expected_wr="${11}"
    local accent="${12:-$PRIMARY}"
    local tier="${13:-unknown}"
    local opponent="${14:-}"
    local is_home="${15:-true}"
    local consensus_line="${16:-}"
    local line_spread="${17:-0}"
    local num_books="${18:-1}"
    local confidence="${19:-STANDARD}"
    local line_dist="${20:-}"
    local alt_books="${21:-}"
    local stake="${22:-1.0}"
    local stake_reason="${23:-}"
    local risk_level="${24:-}"

    # Convert probability to percentage
    local prob_pct=$(echo "$prob * 100" | bc -l | xargs printf "%.0f")

    # Format matchup (@ AWAY or vs HOME)
    local matchup=""
    if [ -n "$opponent" ] && [ "$opponent" != "null" ]; then
        if [ "$is_home" = "true" ]; then
            matchup="vs ${opponent}"
        else
            matchup="@ ${opponent}"
        fi
    fi

    # Confidence level color
    local conf_color="$MUTED"
    case "$confidence" in
        HIGH) conf_color="$SUCCESS" ;;
        MEDIUM) conf_color="$ORANGE" ;;
        STANDARD) conf_color="$PRIMARY" ;;
    esac

    # Edge color (green if positive, red if negative)
    local edge_color="$SUCCESS"
    if (( $(echo "$edge < 0" | bc -l) )); then
        edge_color="$ERROR"
    fi

    # Line spread indicator
    local spread_indicator=""
    if (( $(echo "$line_spread >= 2.0" | bc -l) )); then
        spread_indicator=" ${BOLD}${SUCCESS}[GOLDMINE]${NC}"
    elif (( $(echo "$line_spread >= 1.0" | bc -l) )); then
        spread_indicator=" ${MUTED}[spread: ${line_spread}]${NC}"
    fi

    # Tier color
    local tier_color="$MUTED"
    local tier_display="$tier"
    case "$tier" in
        X) tier_color="$SUCCESS"; tier_display="X" ;;
        Goldmine) tier_color="$YELLOW"; tier_display="GOLDMINE" ;;
        star_tier) tier_color="$LAVENDER"; tier_display="STAR" ;;
        A) tier_color="$PRIMARY"; tier_display="A" ;;
        Z) tier_color="$ORANGE"; tier_display="Z" ;;
        META) tier_color="$SEAFOAM"; tier_display="META" ;;
    esac

    # Header: Player + Matchup + Tier
    echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}  ${MUTED}${matchup}${NC}  ${BOLD}${tier_color}[${tier_display}]${NC}"

    # Main bet line
    echo -e "  ${MUTED}|${NC} ${market} ${side} ${BOLD}${best_line}${NC} @ ${BOLD}${best_book}${NC}${spread_indicator}"

    # Projection and edge
    printf "  ${MUTED}|${NC}  %-14s %b  ${MUTED}|${NC}  %-8s %b\n" \
        "Projection:" "${BOLD}${ORANGE}${prediction}${NC}" \
        "Edge:" "${BOLD}${edge_color}${edge_pct}%${NC} ${MUTED}(${edge})${NC}"

    # Consensus comparison and books count
    if [ -n "$consensus_line" ] && [ "$consensus_line" != "null" ]; then
        local consensus_diff=$(echo "$consensus_line - $best_line" | bc -l | xargs printf "%.1f")
        printf "  ${MUTED}|${NC}  %-14s %b  ${MUTED}|${NC}  %-8s %b\n" \
            "Consensus:" "${consensus_line}" \
            "Books:" "${num_books} offering"
    fi

    # Confidence and P(over)
    printf "  ${MUTED}|${NC}  %-14s %b  ${MUTED}|${NC}  %-8s %b\n" \
        "Confidence:" "${BOLD}${conf_color}${confidence}${NC}" \
        "P(over):" "${BOLD}${prob_pct}%${NC}"

    # Stake sizing (all markets)
    if [ -n "$stake" ] && [ "$stake" != "null" ]; then
        local stake_color="$SUCCESS"
        if (( $(echo "$stake < 1.0" | bc -l) )); then
            stake_color="$ORANGE"
        elif (( $(echo "$stake > 1.0" | bc -l) )); then
            stake_color="$SUCCESS"
        else
            stake_color="$NC"  # Standard 1u stake
        fi
        local risk_display=""
        if [ -n "$risk_level" ] && [ "$risk_level" != "null" ] && [ "$risk_level" != "LOW" ]; then
            risk_display=" ${MUTED}(${risk_level} risk)${NC}"
        fi
        echo -e "  ${MUTED}|${NC}  ${BOLD}Stake:${NC} ${stake_color}${stake}u${NC}${risk_display}"
    fi

    # Line distribution (where else to bet)
    if [ -n "$line_dist" ] && [ "$line_dist" != "null" ]; then
        echo -e "  ${MUTED}|${NC}  ${MUTED}Lines:${NC} ${line_dist}"
    fi

    # Alternative books
    if [ -n "$alt_books" ] && [ "$alt_books" != "null" ] && [ "$alt_books" != "" ]; then
        echo -e "  ${MUTED}|${NC}  ${MUTED}Also at:${NC} ${alt_books}"
    fi

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
    local filter_name="${11:-}"

    # Format filter name for display
    local filter_display=""
    if [ -n "$filter_name" ] && [ "$filter_name" != "null" ]; then
        filter_display="${filter_name}"
    fi

    echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}  ${BOLD}${ORANGE}[${filter_display}]${NC}"
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
# SHARED HELPERS
################################################################################

fetch_and_load_props() {
    section "Fetching Props" "Multi-source collection from 7 sportsbooks"
    if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_all.py"; then
        echo ""
        success "Props fetched"
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
            success "Loaded ${count} props"
        else
            error "Database load failed"
            return 1
        fi
    else
        error "No props file found"
        return 1
    fi
}

fetch_and_load_cheatsheet() {
    section "Fetching Cheatsheet" "BettingPros projections & hit rates"
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_cheatsheet.py" --platform underdog; then
            echo ""
            latest_cheatsheet=$(ls -t "$SCRIPT_DIR/betting_xl/lines/cheatsheet_underdog_"*.json 2>/dev/null | head -1)
            if [ -f "$latest_cheatsheet" ]; then
                if verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_cheatsheet_to_db.py" --file "$latest_cheatsheet"; then
                    success "Cheatsheet loaded"
                else
                    warning "Cheatsheet load failed"
                fi
            fi
        else
            warning "Cheatsheet fetch failed"
        fi
    fi
}

fetch_and_load_prizepicks() {
    section "Fetching PrizePicks" "Direct API (standard/goblin/demon)"

    # Count before fetch
    local before_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM nba_props_xl WHERE game_date = '$DATE_STR' AND book_name LIKE 'prizepicks%';" 2>/dev/null | tr -d ' ')

    # Try Direct API first
    if [ -f "$SCRIPT_DIR/betting_xl/loaders/load_prizepicks_to_db.py" ]; then
        verbose_run python3 "$SCRIPT_DIR/betting_xl/loaders/load_prizepicks_to_db.py" --fetch --quiet 2>/dev/null
    fi

    # Count after Direct API
    local after_direct=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM nba_props_xl WHERE game_date = '$DATE_STR' AND book_name LIKE 'prizepicks%';" 2>/dev/null | tr -d ' ')

    # Fallback to Odds API if Direct didn't add data
    if [ "$after_direct" = "$before_count" ]; then
        info "Direct API failed, trying Odds API fallback..."
        local odds_fetcher="$SCRIPT_DIR/betting_xl/fetchers/fetch_prizepicks.py"
        local pp_loader="$SCRIPT_DIR/betting_xl/loaders/load_prizepicks_to_db.py"

        if [ -f "$odds_fetcher" ]; then
            if verbose_run python3 "$odds_fetcher" --date "$DATE_STR"; then
                # Find the latest prizepicks JSON file for this date
                local pp_json=$(ls -t "$SCRIPT_DIR/betting_xl/lines/prizepicks_${DATE_STR}"*.json 2>/dev/null | head -1)
                if [ -n "$pp_json" ] && [ -f "$pp_json" ] && [ -f "$pp_loader" ]; then
                    verbose_run python3 "$pp_loader" --file "$pp_json"
                fi
            else
                warning "PrizePicks Odds API fallback also failed"
            fi
        fi
    fi

    # Get final counts by odds_type
    local pp_stats=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT odds_type, COUNT(*) FROM nba_props_xl
         WHERE game_date = '$DATE_STR' AND book_name LIKE 'prizepicks%'
         GROUP BY odds_type ORDER BY odds_type;" 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')

    if [ -n "$pp_stats" ]; then
        success "PrizePicks loaded: ${pp_stats}"
    else
        warning "No PrizePicks data loaded"
    fi
}

fetch_vegas_lines() {
    section "Fetching Vegas Lines" "Spreads & totals"
    if [ -f "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/fetchers/fetch_vegas_lines.py" --date "$DATE_STR" --save-to-db; then
            success "Vegas lines loaded"
        else
            warning "Vegas lines fetch failed"
        fi
    fi
}

update_injuries() {
    section "Updating Injuries" "Game-time decisions"
    if [ -f "$SCRIPT_DIR/scripts/update_injuries_NOW.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/update_injuries_NOW.py"; then
            success "Injuries updated"
        else
            info "Injury update skipped"
        fi
    fi
}

enrich_matchups() {
    section "Enriching Matchups" "Opponent & home/away context"
    if verbose_run python3 "$SCRIPT_DIR/betting_xl/enrich_props_with_matchups.py" --date "$DATE_STR"; then
        success "Matchups enriched"
    else
        warning "Enrichment incomplete"
    fi

    local coverage
    coverage=$(get_coverage)
    coverage="${coverage:-0}"
    if [ "${coverage%.*}" -ge 95 ]; then
        success "Coverage: ${coverage}%"
    else
        warning "Coverage below target: ${coverage}%"
        return 1
    fi
}

generate_all_predictions() {
    section "Generating XL Predictions" "POINTS + REBOUNDS"

    if [ -f "$SCRIPT_DIR/betting_xl/generate_xl_predictions.py" ]; then
        if python3 "$SCRIPT_DIR/betting_xl/generate_xl_predictions.py" --output "$PREDICTIONS_DIR/xl_picks_${DATE_STR}.json" --underdog-only >> "$LOG_FILE" 2>&1; then
            local pick_file="$PREDICTIONS_DIR/xl_picks_${DATE_STR}.json"
            if [ -f "$pick_file" ] && command -v jq >/dev/null 2>&1; then
                local picks_count=$(jq '.total_picks // 0' "$pick_file")
                display_xl_picks "$pick_file" "$picks_count"
            else
                success "XL predictions generated"
            fi
        else
            error "XL prediction failed"
            return 1
        fi
    else
        error "XL system not found"
        return 1
    fi

    section "Generating Pro Picks" "Cheatsheet filters"
    if [ -f "$SCRIPT_DIR/betting_xl/generate_cheatsheet_picks.py" ]; then
        local pro_file="$PREDICTIONS_DIR/pro_picks_${DATE_STR}.json"
        if python3 "$SCRIPT_DIR/betting_xl/generate_cheatsheet_picks.py" --output "$pro_file" >> "$LOG_FILE" 2>&1; then
            if [ -f "$pro_file" ] && command -v jq >/dev/null 2>&1; then
                local pro_count=$(jq '.total_picks // 0' "$pro_file")
                if [ "$pro_count" -gt 0 ]; then
                    display_pro_picks "$pro_file"
                    success "Pro picks: ${pro_count}"
                else
                    info "No Pro picks (filters not met)"
                fi
            fi
        else
            warning "Pro picks skipped"
        fi
    fi

    section "Generating Odds API Picks" "Pick6 multipliers"
    if [ -f "$SCRIPT_DIR/betting_xl/generate_odds_api_picks.py" ]; then
        local odds_file="$PREDICTIONS_DIR/odds_api_picks_${DATE_STR//-/}.json"
        if python3 "$SCRIPT_DIR/betting_xl/generate_odds_api_picks.py" --date "$DATE_STR" --output "$odds_file" >> "$LOG_FILE" 2>&1; then
            if [ -f "$odds_file" ] && command -v jq >/dev/null 2>&1; then
                local odds_count=$(jq '.total_picks // 0' "$odds_file")
                if [ "$odds_count" -gt 0 ]; then
                    display_odds_api_picks "$odds_file"
                    success "Odds API picks: ${odds_count}"
                else
                    info "No Odds API picks (filters not met)"
                fi
            fi
        else
            warning "Odds API picks skipped"
        fi
    fi

    section "Generating Two Energy Picks" "Goblin OVER + Inflated UNDER"
    if [ -f "$SCRIPT_DIR/betting_xl/generate_two_energy_picks.py" ]; then
        local energy_file="$PREDICTIONS_DIR/two_energy_picks_${DATE_STR}.json"
        if python3 -m nba.betting_xl.generate_two_energy_picks --date "$DATE_STR" --output "$energy_file" >> "$LOG_FILE" 2>&1; then
            if [ -f "$energy_file" ] && command -v jq >/dev/null 2>&1; then
                local energy_count=$(jq '.total_picks // 0' "$energy_file")
                if [ "$energy_count" -gt 0 ]; then
                    display_two_energy_picks "$energy_file"
                    success "Two Energy picks: ${energy_count}"
                else
                    info "No Two Energy picks (no goblin/inflated lines)"
                fi
            fi
        else
            warning "Two Energy picks skipped"
        fi
    fi
}

################################################################################
# DISPLAY HELPERS FOR PREDICTIONS
################################################################################

display_xl_picks() {
    local pick_file="$1"
    local picks_count="$2"

    echo ""
    divider
    echo ""

    # Show model breakdown (XL vs V3)
    local xl_count=$(jq '.summary.by_model.xl // 0' "$pick_file")
    local v3_count=$(jq '.summary.by_model.v3 // 0' "$pick_file")
    echo -e "  ${BOLD}${PRIMARY}XL + V3 Picks${NC} | ${SOFT_WHITE}${DATE_STR}${NC} | ${MUTED}${picks_count} picks (XL: ${xl_count}, V3: ${v3_count})${NC}"
    echo ""
    divider

    # Show XL picks first, then V3 picks
    render_pick_section "POINTS (XL)" "$POINTS_COLOR" '.picks[] | select(.stat_type == "POINTS" and .model_version == "xl")' "$pick_file"
    render_pick_section "POINTS (V3)" "$LAVENDER" '.picks[] | select(.stat_type == "POINTS" and .model_version == "v3")' "$pick_file"
    render_pick_section "REBOUNDS (XL)" "$REBOUNDS_COLOR" '.picks[] | select(.stat_type == "REBOUNDS" and .model_version == "xl")' "$pick_file"
    render_pick_section "REBOUNDS (V3)" "$LAVENDER" '.picks[] | select(.stat_type == "REBOUNDS" and .model_version == "v3")' "$pick_file"

    local star_tier=$(jq '.summary.by_tier.star_tier // 0' "$pick_file")
    local goldmine=$(jq '.summary.by_tier.Goldmine // 0' "$pick_file")

    divider
    echo ""
    echo -e "  ${MUTED}XL: ${xl_count} | V3: ${v3_count} | Star: ${star_tier} | Goldmine: ${goldmine} | Total: ${picks_count}${NC}"
    echo ""
}

display_pro_picks() {
    local pro_file="$1"

    echo ""
    divider
    echo ""
    echo -e "  ${BOLD}${ORANGE}Pro Picks${NC} | ${MUTED}Cheatsheet Filters${NC}"
    echo ""
    divider

    # Display all Pro picks using unified format
    local all_picks
    all_picks=$(jq -c '.picks[]' "$pro_file" 2>/dev/null)
    while IFS= read -r pick_json; do
        [ -z "$pick_json" ] && continue
        local player=$(echo "$pick_json" | jq -r '.player_name')
        local market=$(echo "$pick_json" | jq -r '.stat_type')
        local line=$(echo "$pick_json" | jq -r '.line')
        local projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
        local edge=$(echo "$pick_json" | jq -r '.projection_diff | tonumber | . * 10 | round / 10')
        local l5_rate=$(echo "$pick_json" | jq -r '.hit_rate_l5 | tonumber | . * 100 | round')
        local opp_rank=$(echo "$pick_json" | jq -r '.opp_rank')
        local season_rate=$(echo "$pick_json" | jq -r '.hit_rate_season | tonumber | . * 100 | round')
        local expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')
        local filter_name=$(echo "$pick_json" | jq -r '.filter_name // "unknown"')

        # Expand combo stat names
        case "$market" in
            PA) market="PTS+AST" ;;
            PR) market="PTS+REB" ;;
            RA) market="REB+AST" ;;
        esac

        print_pro_pick "$player" "$market" "$line" "Underdog" "$projection" "$edge" "$l5_rate" "$opp_rank" "$season_rate" "$expected_wr" "$filter_name"
    done <<< "$all_picks"

    divider
    echo ""
}

display_odds_api_picks() {
    local odds_file="$1"

    echo ""
    divider
    echo ""
    echo -e "  ${BOLD}${ORANGE}Odds API Picks${NC} | ${MUTED}Pick6 Multipliers${NC}"
    echo ""
    divider

    local odds_picks
    odds_picks=$(jq -c '.picks[]' "$odds_file" 2>/dev/null)
    while IFS= read -r pick_json; do
        [ -z "$pick_json" ] && continue
        local player=$(echo "$pick_json" | jq -r '.player_name')
        local market=$(echo "$pick_json" | jq -r '.stat_type')
        local line=$(echo "$pick_json" | jq -r '.line')
        local mult=$(echo "$pick_json" | jq -r '.pick6_multiplier | tonumber | . * 100 | round / 100')
        local projection=$(echo "$pick_json" | jq -r '.projection | tonumber | . * 10 | round / 10')
        local expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

        echo -e "  ${BOLD}${PLAYER_NAME_COLOR}${player}${NC}"
        echo -e "  ${MUTED}|${NC} ${market} OVER ${BOLD}${line}${NC} @ Pick6"
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Multiplier:" "${BOLD}${ORANGE}${mult}x${NC}"
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Projection:" "${BOLD}${projection}${NC}"
        printf "  ${MUTED}|${NC}  %-12s %b\n" "Expected WR:" "${BOLD}${SUCCESS}${expected_wr}%${NC}"
        echo ""
    done <<< "$odds_picks"

    divider
    echo ""
}

display_two_energy_picks() {
    local energy_file="$1"

    echo ""
    divider
    echo ""
    echo -e "  ${BOLD}${MINT}Two Energy Picks${NC} | ${MUTED}Goblin OVER + Inflated UNDER${NC}"
    echo ""
    divider

    # Display POSITIVE energy (OVER) picks
    echo -e "  ${BOLD}${GREEN}POSITIVE ENERGY (OVER)${NC}"
    local over_picks
    over_picks=$(jq -c '.picks[] | select(.side == "OVER")' "$energy_file" 2>/dev/null)
    while IFS= read -r pick_json; do
        [ -z "$pick_json" ] && continue
        local player=$(echo "$pick_json" | jq -r '.player_name')
        local market=$(echo "$pick_json" | jq -r '.stat_type')
        local line=$(echo "$pick_json" | jq -r '.line')
        local deflation=$(echo "$pick_json" | jq -r '.deflation // 0')
        local expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

        printf "  ${MUTED}|${NC} ${BOLD}${PLAYER_NAME_COLOR}%-20s${NC} ${market} O${BOLD}%-5s${NC} ${MUTED}(deflated -%.1f)${NC} ${GREEN}${expected_wr}%%${NC}\n" "$player" "$line" "$deflation"
    done <<< "$over_picks"

    echo ""

    # Display NEGATIVE energy (UNDER) picks
    echo -e "  ${BOLD}${RED}NEGATIVE ENERGY (UNDER)${NC}"
    local under_picks
    under_picks=$(jq -c '.picks[] | select(.side == "UNDER")' "$energy_file" 2>/dev/null)
    while IFS= read -r pick_json; do
        [ -z "$pick_json" ] && continue
        local player=$(echo "$pick_json" | jq -r '.player_name')
        local market=$(echo "$pick_json" | jq -r '.stat_type')
        local line=$(echo "$pick_json" | jq -r '.line')
        local book=$(echo "$pick_json" | jq -r '.book')
        local inflate=$(echo "$pick_json" | jq -r '.line_inflate // 0')
        local expected_wr=$(echo "$pick_json" | jq -r '.expected_wr')

        printf "  ${MUTED}|${NC} ${BOLD}${PLAYER_NAME_COLOR}%-20s${NC} ${market} U${BOLD}%-5s${NC} ${MUTED}(${book} +%.1f)${NC} ${GREEN}${expected_wr}%%${NC}\n" "$player" "$line" "$inflate"
    done <<< "$under_picks"

    echo ""
    divider
    echo ""
}

################################################################################
# FULL WORKFLOW (run once daily)
################################################################################

full_workflow() {
    header "Full Workflow" "Complete data collection + predictions"

    # Data collection
    fetch_and_load_props || return 1
    fetch_and_load_prizepicks
    fetch_and_load_cheatsheet
    enrich_matchups || return 1

    section "Fetching Game Results" "Yesterday's stats"
    if [ -f "$SCRIPT_DIR/scripts/fetch_daily_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/fetch_daily_stats.py" --days 1; then
            success "Game results updated"
        else
            info "No new games"
        fi
    fi

    section "Populating Actuals" "Update props with results"
    if [ -f "$SCRIPT_DIR/betting_xl/populate_actual_values.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/populate_actual_values.py" --days 7; then
            success "Actuals populated"
        fi
    fi

    section "Calibrator Status" "21-day lookback"
    if [ -f "$SCRIPT_DIR/models/json_calibrator.py" ]; then
        if python3 "$SCRIPT_DIR/models/json_calibrator.py" status --lookback 21 >> "$LOG_FILE" 2>&1; then
            success "Calibrator checked"
        fi
    fi

    update_injuries

    section "Loading Team Games" "NBA API"
    if [ -f "$SCRIPT_DIR/scripts/loaders/load_nba_games_incremental.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/load_nba_games_incremental.py"; then
            success "Team games loaded"
        fi
    fi

    section "Team Season Stats" "Pace/ratings"
    if [ -f "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" --season "$CURRENT_SEASON"; then
            success "Team stats updated"
        fi
    fi

    section "Team Advanced Stats" "NBA API PACE"
    if [ -f "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py"; then
            success "Advanced stats loaded"
        fi
    fi

    fetch_vegas_lines

    section "Minutes Projections"
    if [ -f "$SCRIPT_DIR/scripts/loaders/calculate_minutes_projections.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/loaders/calculate_minutes_projections.py" --update; then
            success "Minutes projections updated"
        fi
    fi

    section "Prop History" "Bayesian hit rates"
    if [ -f "$SCRIPT_DIR/scripts/compute_prop_history.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/compute_prop_history.py" --season "$CURRENT_SEASON" --incremental --days 7; then
            success "Prop history updated"
        fi
    fi

    # Generate predictions
    generate_all_predictions

    complete "Full Workflow Complete" "Predictions ready"
}

################################################################################
# UPDATE PROPS WORKFLOW (lightweight - just fetch lines, no predictions)
################################################################################

update_props_workflow() {
    header "Update Props" "Fetching latest lines for movement tracking"

    # Fetch from all sources - no predictions
    fetch_and_load_props || return 1
    fetch_and_load_prizepicks

    # Quick summary of what we captured
    local count=$(get_props_count)
    local fetch_num=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(DISTINCT fetch_timestamp::date || fetch_timestamp::time(0)) FROM nba_props_xl WHERE game_date = '$DATE_STR';" 2>/dev/null | tr -d ' ')

    complete "Props Updated" "${count} props | Fetch #${fetch_num} today"
}

################################################################################
# REFRESH WORKFLOW (run anytime for line movements)
################################################################################

refresh_workflow() {
    header "Refresh Workflow" "Line movements + injury updates + predictions"

    # Quick health check
    if [ -f "$SCRIPT_DIR/betting_xl/health_check.py" ]; then
        if python3 "$SCRIPT_DIR/betting_xl/health_check.py" --quick >> "$LOG_FILE" 2>&1; then
            success "Health check passed"
        else
            warning "Health check issues - proceeding"
        fi
    fi

    # Monitor (stop-loss check)
    if [ -f "$SCRIPT_DIR/betting_xl/monitor.py" ]; then
        if python3 "$SCRIPT_DIR/betting_xl/monitor.py" >> "$LOG_FILE" 2>&1; then
            success "Stop-loss check passed"
        else
            error "Stop-loss triggered"
            return 1
        fi
    fi

    # Refresh data that changes intraday
    fetch_and_load_props || return 1
    fetch_and_load_prizepicks
    fetch_and_load_cheatsheet
    update_injuries
    fetch_vegas_lines
    enrich_matchups || return 1

    # Generate predictions
    generate_all_predictions

    complete "Refresh Complete" "Predictions updated with latest lines"
}

################################################################################
# INDIVIDUAL STEPS (for Airflow DAG tasks)
################################################################################

# These can be called individually: ./nba-predictions.sh step:fetch_props
step_fetch_props() {
    header "Step: Fetch Props"
    fetch_and_load_props
}

step_fetch_cheatsheet() {
    header "Step: Fetch Cheatsheet"
    fetch_and_load_cheatsheet
}

step_fetch_prizepicks() {
    header "Step: Fetch PrizePicks"
    fetch_and_load_prizepicks
}

step_enrich() {
    header "Step: Enrich Matchups"
    enrich_matchups
}

step_injuries() {
    header "Step: Update Injuries"
    update_injuries
}

step_vegas() {
    header "Step: Fetch Vegas Lines"
    fetch_vegas_lines
}

step_predict() {
    header "Step: Generate Predictions"
    generate_all_predictions
}

step_game_results() {
    header "Step: Fetch Game Results"
    if [ -f "$SCRIPT_DIR/scripts/fetch_daily_stats.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/scripts/fetch_daily_stats.py" --days 1; then
            success "Game results updated"
        fi
    fi
}

step_populate_actuals() {
    header "Step: Populate Actuals"
    if [ -f "$SCRIPT_DIR/betting_xl/populate_actual_values.py" ]; then
        if verbose_run python3 "$SCRIPT_DIR/betting_xl/populate_actual_values.py" --days 7; then
            success "Actuals populated"
        fi
    fi
}

step_team_stats() {
    header "Step: Team Stats"
    if [ -f "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" ]; then
        verbose_run python3 "$SCRIPT_DIR/scripts/loaders/calculate_team_stats.py" --season "$CURRENT_SEASON"
    fi
    if [ -f "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py" ]; then
        verbose_run python3 "$SCRIPT_DIR/scripts/loaders/load_team_advanced_stats.py"
    fi
}

################################################################################
# VALIDATE PICKS
################################################################################

validate_workflow() {
    local target_date="${1:-}"
    local end_date="${2:-}"
    local verbose="${3:-}"

    # Default to yesterday if no date provided
    if [ -z "$target_date" ]; then
        target_date=$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
    fi

    header "Pick Validation" "Comparing predictions vs actual results"

    echo ""
    echo -e "${BOLD}${INFO}* Validation Parameters${NC}"

    if [ -n "$end_date" ]; then
        info "Date Range: $target_date to $end_date"
    else
        info "Date: $target_date"
    fi

    echo ""

    # Build command arguments
    local cmd_args=""

    if [ -n "$end_date" ]; then
        cmd_args="--start-date $target_date --end-date $end_date"
    else
        cmd_args="--date $target_date"
    fi

    if [ "$verbose" = "1" ]; then
        cmd_args="$cmd_args --verbose"
    fi

    # Run validation (from betting_xl directory so relative paths work)
    section "Running Validation" "Checking XL, PRO, and ODDS_API picks"

    pushd "$SCRIPT_DIR/betting_xl" > /dev/null
    if python3 validate_predictions.py $cmd_args 2>&1 | tee -a "$LOG_FILE"; then
        popd > /dev/null
        echo ""
        complete "Validation Complete" "Results displayed above"
    else
        popd > /dev/null
        echo ""
        error "Validation failed"
        return 1
    fi
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
    # XL (24) + V3 (14) + matchup (34) = 72 model files
    if [ "$model_count" -ge 38 ]; then
        success "XL + V3 models (${model_count} files)"
    else
        error "Models incomplete (${model_count}/38+ expected)"
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
    local all_args=("$@")

    # Handle --debug flag
    if [ "$command" = "--debug" ]; then
        DEBUG=1
        export DEBUG
        command="${2:-}"
        all_args=("${@:2}")  # Remove --debug from args
        echo -e "${YELLOW}[DEBUG] Debug mode enabled${NC}"
    fi

    # Quick commands that skip the intro
    if [ "$command" = "picks" ]; then
        python3 "$SCRIPT_DIR/betting_xl/show_picks.py" "${all_args[@]:1}"
        exit 0
    fi

    # Step commands for Airflow (skip banner for cleaner logs)
    if [[ "$command" == step:* ]]; then
        local step_name="${command#step:}"
        case "$step_name" in
            fetch_props)     step_fetch_props ;;
            fetch_cheatsheet) step_fetch_cheatsheet ;;
            fetch_prizepicks) step_fetch_prizepicks ;;
            enrich)          step_enrich ;;
            injuries)        step_injuries ;;
            vegas)           step_vegas ;;
            predict)         step_predict ;;
            game_results)    step_game_results ;;
            populate_actuals) step_populate_actuals ;;
            team_stats)      step_team_stats ;;
            *)
                error "Unknown step: $step_name"
                echo "Available steps: fetch_props, fetch_cheatsheet, fetch_prizepicks, enrich, injuries, vegas, predict, game_results, populate_actuals, team_stats"
                exit 1
                ;;
        esac
        exit $?
    fi

    system_banner

    # Show debug status in banner if enabled
    if [ "$DEBUG" = "1" ]; then
        echo -e "  ${YELLOW}** DEBUG MODE ACTIVE **${NC}"
        echo ""
    fi

    case "$command" in
        ""|full|run)
            full_workflow
            ;;
        refresh)
            refresh_workflow
            ;;
        update_props|update-props|fetch)
            update_props_workflow
            ;;
        health)
            health_check
            ;;
        validate)
            # Parse validate options
            local val_date=""
            local val_end=""
            local val_verbose=""
            local args=("${all_args[@]:1}")  # Remove 'validate' from args

            local i=0
            while [ $i -lt ${#args[@]} ]; do
                case "${args[$i]}" in
                    --date)
                        val_date="${args[$((i+1))]}"
                        i=$((i+2))
                        ;;
                    --start)
                        val_date="${args[$((i+1))]}"
                        i=$((i+2))
                        ;;
                    --end)
                        val_end="${args[$((i+1))]}"
                        i=$((i+2))
                        ;;
                    --verbose|-v)
                        val_verbose="1"
                        i=$((i+1))
                        ;;
                    --7d)
                        val_date=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d)
                        val_end=$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
                        i=$((i+1))
                        ;;
                    --30d)
                        val_date=$(date -d '30 days ago' +%Y-%m-%d 2>/dev/null || date -v-30d +%Y-%m-%d)
                        val_end=$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
                        i=$((i+1))
                        ;;
                    *)
                        # Assume it's a date if it looks like YYYY-MM-DD
                        if [[ "${args[$i]}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
                            val_date="${args[$i]}"
                        fi
                        i=$((i+1))
                        ;;
                esac
            done

            validate_workflow "$val_date" "$val_end" "$val_verbose"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${ERROR}Unknown command: ${command}${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

show_help() {
    echo ""
    echo -e "${BOLD}${BRIGHT_WHITE}NBA XL Prediction System${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 ${PRIMARY}[--debug] <command>${NC}"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo ""
    echo -e "  ${PRIMARY}(default)${NC}  Full workflow - all data + predictions"
    echo -e "  ${PRIMARY}full${NC}       Same as default"
    echo -e "             ${MUTED}Run once daily (morning/early afternoon)${NC}"
    echo ""
    echo -e "  ${PRIMARY}refresh${NC}    Quick refresh - lines + injuries + predictions"
    echo -e "             ${MUTED}Run anytime to capture line movements${NC}"
    echo ""
    echo -e "  ${PRIMARY}update_props${NC} Fetch lines only (no predictions)"
    echo -e "             ${MUTED}Lightweight - for line movement tracking cron${NC}"
    echo ""
    echo -e "  ${PRIMARY}validate${NC}   Check pick performance"
    echo -e "             ${MUTED}--date YYYY-MM-DD   Single date${NC}"
    echo -e "             ${MUTED}--start/--end       Date range${NC}"
    echo -e "             ${MUTED}--7d / --30d        Last N days${NC}"
    echo -e "             ${MUTED}-v, --verbose       Show all picks${NC}"
    echo ""
    echo -e "  ${PRIMARY}picks${NC}      View today's picks"
    echo ""
    echo -e "  ${PRIMARY}health${NC}     System diagnostics"
    echo ""
    echo -e "${BOLD}Airflow Steps:${NC} ${MUTED}(for DAG tasks)${NC}"
    echo ""
    echo -e "  ${PRIMARY}step:fetch_props${NC}      Fetch props from sportsbooks"
    echo -e "  ${PRIMARY}step:fetch_cheatsheet${NC} Fetch BettingPros data"
    echo -e "  ${PRIMARY}step:fetch_prizepicks${NC} Fetch PrizePicks direct API"
    echo -e "  ${PRIMARY}step:enrich${NC}           Add matchup context"
    echo -e "  ${PRIMARY}step:injuries${NC}         Update injury reports"
    echo -e "  ${PRIMARY}step:vegas${NC}            Fetch vegas lines"
    echo -e "  ${PRIMARY}step:predict${NC}          Generate all predictions"
    echo -e "  ${PRIMARY}step:game_results${NC}     Fetch yesterday's results"
    echo -e "  ${PRIMARY}step:populate_actuals${NC} Update props with results"
    echo -e "  ${PRIMARY}step:team_stats${NC}       Update team stats"
    echo ""
}

main "$@"
