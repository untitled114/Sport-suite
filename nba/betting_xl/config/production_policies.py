#!/usr/bin/env python3
"""
Production Policies Configuration
==================================
Defines stop-loss triggers, data freshness requirements, and escalation procedures
for the NBA XL betting system.

Part of Phase 9: Production Deployment - Operational Safeguards
Created: November 7, 2025
"""

from typing import Dict, Any


# =============================================================================
# STOP-LOSS TRIGGERS
# =============================================================================

# Win rate thresholds per market (7-day rolling WR)
# If rolling WR drops below threshold, STOP betting that market
STOP_LOSS_WR_THRESHOLDS = {
    'POINTS': 52.0,      # Stop if 7-day WR < 52% (below breakeven at -110)
    'REBOUNDS': 55.0,    # Stop if 7-day WR < 55% (conservative threshold)
    'ASSISTS': 999.0,    # Market disabled - no threshold needed
    'THREES': 999.0      # Market disabled - no threshold needed
}

# Warning thresholds (alert but continue betting)
WARNING_WR_THRESHOLDS = {
    'POINTS': 54.0,      # Warning if < 54% (approaching danger zone)
    'REBOUNDS': 58.0,    # Warning if < 58% (below expected 61.2%)
    'ASSISTS': 999.0,    # Disabled
    'THREES': 999.0      # Disabled
}

# Consecutive losing days before review
MAX_CONSECUTIVE_LOSING_DAYS = 3  # Review strategy if 3 days in a row negative ROI

# Edge anomaly detection (unusually high edges may indicate data quality issues)
MAX_REASONABLE_EDGE = 15.0  # Alert if any pick has edge > 15.0 points
MAX_AVG_EDGE_PER_DAY = 8.0  # Alert if average edge across all picks > 8.0


# =============================================================================
# DATA FRESHNESS REQUIREMENTS
# =============================================================================

DATA_FRESHNESS_POLICY = {
    # Props must be from today
    'props_date_required': 'today',  # Must match CURRENT_DATE
    'min_props_required': 50,        # At least 50 props needed to generate picks

    # Game results freshness (for rolling stats updates)
    'max_game_result_age_hours': 48,  # Game results within 48 hours (tightened threshold)

    # Player stats freshness
    'rolling_stats_required': True,   # Must have rolling stats (L3/L5/L10/L20)
    'max_rolling_stats_age_days': 7,  # Rolling stats updated within 7 days

    # Injury reports freshness
    'injuries_required': True,        # Must have injury data
    'max_injury_report_age_hours': 48,  # Injury reports updated within 48 hours (MAX per user request)

    # Model freshness
    # NOTE: Nov 7, 2025 models still achieving 68.2% WR as of Jan 3, 2026
    # Retrain only when performance degrades, not based on calendar time
    'max_model_age_days': 180,        # 6 months max before forced retrain
    'recommended_retrain_frequency_days': 90  # Warn at 90 days (3 months)
}


# =============================================================================
# ESCALATION PROCEDURES
# =============================================================================

ESCALATION_LEVELS = {
    # Level 1: WARNING - Increase monitoring, continue with caution
    'warning': {
        'triggers': [
            'WR drop of 5-10 percentage points vs validation',
            'Average edge < 1.0 (possible line efficiency)',
            'Picks volume < 5 per day (limited opportunities)'
        ],
        'actions': [
            'Log detailed warning with metrics',
            'Increase monitoring frequency',
            'Continue betting with caution',
            'Review edge distribution'
        ]
    },

    # Level 2: CAUTION - Pause market, analyze root cause
    'caution': {
        'triggers': [
            'WR drop of 10-15 percentage points vs validation',
            '2 consecutive losing days',
            'Edge anomaly detected (>15.0 on any pick)',
            'Data freshness failure (props > 24 hours old)'
        ],
        'actions': [
            'PAUSE affected market immediately',
            'Analyze recent picks for patterns',
            'Check data quality (injuries, rolling stats)',
            'Review model performance on recent data',
            'Manual review before resuming'
        ]
    },

    # Level 3: STOP - Emergency halt, full investigation required
    'stop': {
        'triggers': [
            'WR drop >15 percentage points vs validation',
            '3 consecutive losing days',
            'Negative ROI over 7-day rolling window',
            'Critical data freshness failure (missing props/stats)',
            'Model age > 30 days'
        ],
        'actions': [
            'STOP ALL BETTING immediately',
            'Full system investigation required',
            'Retrain models if needed',
            'Validate data pipeline integrity',
            'Manual approval required to resume'
        ]
    }
}


# =============================================================================
# PERFORMANCE TRACKING REQUIREMENTS
# =============================================================================

TRACKING_CONFIG = {
    'rolling_window_days': 7,        # Track performance over 7-day rolling window
    'min_bets_for_validation': 20,   # Minimum 20 bets to calculate reliable WR
    'track_metrics': [
        'win_rate',                  # Percentage of winning bets
        'roi',                       # Return on investment
        'avg_edge',                  # Average edge per pick
        'total_picks',               # Total number of picks
        'picks_by_market',           # Breakdown by market
        'consecutive_days_positive',  # Consecutive days with positive ROI
        'consecutive_days_negative',  # Consecutive days with negative ROI
        'max_drawdown'               # Maximum consecutive loss streak
    ],
    'alert_recipients': [
        'log_file',                  # Always log alerts
        'console'                    # Print to console for immediate visibility
    ]
}


# =============================================================================
# VALIDATION BENCHMARKS (from Oct 23 - Nov 4, 2024 validation)
# =============================================================================

VALIDATION_BENCHMARKS = {
    'POINTS': {
        'win_rate': 59.2,     # Tiered filter validation
        'roi': 13.1,
        'total_bets': 342,
        'avg_edge': 3.5       # Approximate from validation
    },
    'REBOUNDS': {
        'win_rate': 63.5,     # Tiered filter validation
        'roi': 21.4,
        'total_bets': 236,
        'avg_edge': 4.0       # Approximate from validation
    },
    'ASSISTS': {
        'win_rate': 14.6,     # Line shopping validation (DISABLED for XL model)
        'roi': -72.05,
        'total_bets': 41,
        'status': 'DISABLED'
    },
    'THREES': {
        'win_rate': 46.5,     # Line shopping validation (DISABLED)
        'roi': -11.23,
        'total_bets': 71,
        'status': 'DISABLED'
    },
    'overall': {
        'win_rate': 57.6,     # Tiered filter (POINTS + REBOUNDS)
        'roi': 10.04,
        'total_bets': 578
    }
}


# =============================================================================
# CHEATSHEET VALIDATION BENCHMARKS (from Nov 15 - Dec 31, 2025 validation)
# =============================================================================
# These are SEPARATE from XL model predictions - uses BettingPros recommendations
# with validated high-WR filters.

CHEATSHEET_BENCHMARKS = {
    'assists_primary': {
        'description': 'ASSISTS + L5_60+ + Opp_11+',
        'win_rate': 93.8,
        'total_bets': 129,      # 121W + 8L
        'avg_picks_per_day': 3.0,
        'validation_period': 'Nov 15 - Dec 31, 2025',
        'status': 'ACTIVE'
    },
    'l15_elite': {
        'description': 'L15 80%+ any stat',
        'win_rate': 77.4,
        'avg_picks_per_day': 1.0,
        'validation_period': 'Nov 15 - Dec 31, 2025',
        'status': 'BACKUP'  # Enable for more volume
    }
}

# Cheatsheet-specific stop-loss thresholds
# Uses higher thresholds since expected WR is much higher
CHEATSHEET_STOP_LOSS = {
    'assists_primary': 85.0,    # Stop if 7-day WR < 85% (expecting 93.8%)
    'l15_elite': 70.0           # Stop if 7-day WR < 70% (expecting 77.4%)
}

CHEATSHEET_WARNING_THRESHOLDS = {
    'assists_primary': 88.0,    # Warning if < 88%
    'l15_elite': 73.0           # Warning if < 73%
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_stop_loss_threshold(market: str) -> float:
    """Get stop-loss WR threshold for a market."""
    return STOP_LOSS_WR_THRESHOLDS.get(market, 999.0)


def get_warning_threshold(market: str) -> float:
    """Get warning WR threshold for a market."""
    return WARNING_WR_THRESHOLDS.get(market, 999.0)


def get_validation_benchmark(market: str, metric: str = 'win_rate') -> float:
    """Get validation benchmark for comparison."""
    return VALIDATION_BENCHMARKS.get(market, {}).get(metric, 0.0)


def should_stop_market(market: str, rolling_wr: float) -> bool:
    """
    Determine if a market should be stopped based on rolling WR.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)
        rolling_wr: 7-day rolling win rate (percentage)

    Returns:
        True if market should be stopped, False otherwise
    """
    threshold = get_stop_loss_threshold(market)
    return rolling_wr < threshold


def should_warn_market(market: str, rolling_wr: float) -> bool:
    """
    Determine if a market should trigger warning based on rolling WR.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)
        rolling_wr: 7-day rolling win rate (percentage)

    Returns:
        True if market should trigger warning, False otherwise
    """
    threshold = get_warning_threshold(market)
    return rolling_wr < threshold


def get_escalation_level(metrics: Dict[str, Any]) -> str:
    """
    Determine escalation level based on current metrics.

    Args:
        metrics: Dictionary with performance metrics

    Returns:
        'normal', 'warning', 'caution', or 'stop'
    """
    # Level 3: STOP
    if metrics.get('consecutive_losing_days', 0) >= 3:
        return 'stop'
    if metrics.get('rolling_roi_7d', 0) < 0:
        return 'stop'
    if metrics.get('wr_drop_vs_validation', 0) > 15:
        return 'stop'

    # Level 2: CAUTION
    if metrics.get('consecutive_losing_days', 0) >= 2:
        return 'caution'
    if metrics.get('wr_drop_vs_validation', 0) > 10:
        return 'caution'
    if metrics.get('max_edge', 0) > MAX_REASONABLE_EDGE:
        return 'caution'

    # Level 1: WARNING
    if metrics.get('wr_drop_vs_validation', 0) > 5:
        return 'warning'
    if metrics.get('avg_edge', 0) < 1.0:
        return 'warning'
    if metrics.get('picks_today', 0) < 5:
        return 'warning'

    return 'normal'


if __name__ == '__main__':
    # Print policy summary
    print("NBA XL Production Policies")
    print("=" * 80)
    print("\nSTOP-LOSS THRESHOLDS:")
    for market, threshold in STOP_LOSS_WR_THRESHOLDS.items():
        if threshold < 999.0:
            print(f"  {market:10s}: Stop if WR < {threshold}%")
        else:
            print(f"  {market:10s}: DISABLED")

    print("\nDATA FRESHNESS REQUIREMENTS:")
    for key, value in DATA_FRESHNESS_POLICY.items():
        print(f"  {key}: {value}")

    print("\nESCALATION LEVELS:")
    for level, config in ESCALATION_LEVELS.items():
        print(f"\n  {level.upper()}:")
        print(f"    Triggers: {', '.join(config['triggers'][:2])}")
