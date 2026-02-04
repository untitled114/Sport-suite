#!/usr/bin/env python3
"""
Risk Filters for NBA Predictions
=================================
Market-specific risk assessment:

REBOUNDS: Traditional risk filtering (skip high risk)
- Volatility = risk → filter it out
- Minutes volatility matters most
- Strict filtering works well

POINTS: Volatility-aware stake sizing (don't skip, adjust stakes)
- Volatility ≠ risk when lines are soft
- High vol + soft line = GOOD opportunity (bet smaller)
- High vol + sharp line = SKIP
- Stake sizing beats filtering for POINTS
"""

import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import psycopg2

logger = logging.getLogger(__name__)

# Database config for querying game logs
DB_PLAYERS = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv(
        "NBA_PLAYERS_DB_USER", os.getenv("NBA_DB_USER", os.getenv("DB_USER", "mlb_user"))
    ),
    "password": os.getenv(
        "NBA_PLAYERS_DB_PASSWORD", os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    ),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}


@dataclass
class RiskAssessment:
    """Container for risk assessment results"""

    # Market type
    stat_type: str = "POINTS"

    # Risk flags
    high_volatility: bool = False
    elite_defense: bool = False
    negative_trend: bool = False

    # Risk scores (0-1, higher = more risky)
    volatility_score: float = 0.0
    defense_score: float = 0.0
    trend_score: float = 0.0

    # Combined risk
    total_risk_score: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, EXTREME

    # Line softness (POINTS only)
    line_is_soft: bool = False
    line_softness_score: float = 0.0  # 0-1, higher = softer

    # Stake sizing (1.0 = full unit)
    recommended_stake: float = 1.0
    stake_reason: str = ""

    # Details for logging
    details: Dict[str, str] = field(default_factory=dict)

    @property
    def should_skip(self) -> bool:
        """Returns True if pick should be skipped"""
        if self.stat_type == "REBOUNDS":
            # REBOUNDS: Strict filtering - skip high risk
            if self.risk_level == "EXTREME":
                return True
            risk_count = sum([self.high_volatility, self.elite_defense, self.negative_trend])
            if risk_count >= 2:
                return True
            return False
        else:
            # POINTS: Only skip if high vol + sharp line (no edge)
            # Or if recommended_stake is 0
            return self.recommended_stake == 0.0

    @property
    def should_flag(self) -> bool:
        """Returns True if pick should be flagged as risky"""
        return self.risk_level in ("HIGH", "EXTREME")


class RiskFilter:
    """
    Market-specific risk assessment for NBA player prop predictions.

    REBOUNDS: Traditional filtering (skip high risk)
    POINTS: Stake sizing based on volatility + line softness
    """

    # Volatility thresholds
    POINTS_VOLATILITY_HIGH = 0.30
    POINTS_VOLATILITY_EXTREME = 0.45
    REBOUNDS_VOLATILITY_HIGH = 0.35
    REBOUNDS_VOLATILITY_EXTREME = 0.50

    # Defense thresholds
    ELITE_DEFENSE_RANK = 5
    GOOD_DEFENSE_RANK = 10

    # Trend thresholds
    SLUMP_THRESHOLD = 0.88
    HOT_THRESHOLD = 1.12

    # Line softness thresholds (POINTS)
    SOFT_LINE_SPREAD = 2.0  # Line spread >= 2 points = soft
    VERY_SOFT_LINE_SPREAD = 3.0  # Line spread >= 3 points = very soft
    SOFT_EDGE_PCT = 15.0  # Edge >= 15% = soft line indicator

    def __init__(self):
        self.stats_assessed = 0
        self.high_risk_skipped = 0
        self._conn = None
        self._volatility_cache: Dict[str, Dict] = {}

    def _get_connection(self):
        """Get database connection for game log queries"""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(**DB_PLAYERS)
            except psycopg2.Error as e:
                logger.warning(f"Could not connect to players database: {e}")
                return None
        return self._conn

    def _get_real_volatility(
        self, player_name: str, stat_type: str, game_date: str = None
    ) -> Optional[Dict]:
        """Calculate real volatility from game logs."""
        cache_key = f"{player_name}_{stat_type}_{game_date}"
        if cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]

        conn = self._get_connection()
        if conn is None:
            return None

        stat_col = stat_type.lower()

        try:
            query = f"""
            SELECT pgl.{stat_col}, pgl.minutes_played, pgl.game_date
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE pp.full_name = %s
              AND pgl.{stat_col} IS NOT NULL
              AND pgl.minutes_played >= 10
            ORDER BY pgl.game_date DESC
            LIMIT 15
            """
            with conn.cursor() as cur:
                cur.execute(query, (player_name,))
                rows = cur.fetchall()

            if len(rows) < 5:
                return None

            stats = [float(r[0]) for r in rows]
            minutes = [float(r[1]) for r in rows]

            L5_stats = stats[:5]
            L10_stats = stats[:10] if len(stats) >= 10 else stats
            L5_mins = minutes[:5]

            avg_L5 = statistics.mean(L5_stats)
            avg_L10 = statistics.mean(L10_stats) if len(L10_stats) >= 5 else avg_L5
            std_L5 = statistics.stdev(L5_stats) if len(L5_stats) >= 2 else 0
            std_L10 = statistics.stdev(L10_stats) if len(L10_stats) >= 2 else 0
            min_std_L5 = statistics.stdev(L5_mins) if len(L5_mins) >= 2 else 0
            min_avg_L5 = statistics.mean(L5_mins)

            # Calculate percentiles for upside detection
            percentile_75 = (
                sorted(L10_stats)[int(len(L10_stats) * 0.75)] if len(L10_stats) >= 4 else avg_L10
            )
            percentile_90 = (
                sorted(L10_stats)[int(len(L10_stats) * 0.9)]
                if len(L10_stats) >= 10
                else percentile_75
            )

            result = {
                "std_L5": std_L5,
                "std_L10": std_L10,
                "avg_L5": avg_L5,
                "avg_L10": avg_L10,
                "cv_L5": std_L5 / avg_L5 if avg_L5 > 0 else 0,
                "min_std_L5": min_std_L5,
                "min_avg_L5": min_avg_L5,
                "min_cv_L5": min_std_L5 / min_avg_L5 if min_avg_L5 > 0 else 0,
                "recent_games": L5_stats,
                "range_L5": max(L5_stats) - min(L5_stats),
                "percentile_75": percentile_75,
                "percentile_90": percentile_90,
                "median_L5": statistics.median(L5_stats),
            }

            self._volatility_cache[cache_key] = result
            return result

        except (psycopg2.Error, statistics.StatisticsError) as e:
            logger.debug(f"Error calculating volatility for {player_name}: {e}")
            return None

    def assess_risk(
        self,
        player_name: str,
        stat_type: str,
        features: Dict,
        opp_rank: Optional[int] = None,
        p_over: float = 0.5,
        line: float = 0.0,
        line_spread: float = 0.0,
        edge_pct: float = 0.0,
        prediction: float = 0.0,
        consensus_line: float = 0.0,
    ) -> RiskAssessment:
        """
        Assess risk with market-specific logic.

        For POINTS: Calculate stake sizing based on volatility + line softness
        For REBOUNDS: Traditional risk filtering
        """
        self.stats_assessed += 1
        assessment = RiskAssessment(stat_type=stat_type)
        assessment.details = {"player": player_name, "stat": stat_type}

        # 1. Volatility Assessment
        vol_score, vol_flag, vol_detail = self._assess_volatility(features, stat_type, player_name)
        assessment.volatility_score = vol_score
        assessment.high_volatility = vol_flag
        assessment.details["volatility"] = vol_detail

        # 2. Defense Assessment
        def_score, def_flag, def_detail = self._assess_defense(opp_rank, stat_type)
        assessment.defense_score = def_score
        assessment.elite_defense = def_flag
        assessment.details["defense"] = def_detail

        # 3. Trend Assessment
        trend_score, trend_flag, trend_detail = self._assess_trend(features, stat_type, player_name)
        assessment.trend_score = trend_score
        assessment.negative_trend = trend_flag
        assessment.details["trend"] = trend_detail

        # Calculate combined risk score
        weights = {"volatility": 0.4, "defense": 0.3, "trend": 0.3}
        assessment.total_risk_score = (
            weights["volatility"] * vol_score
            + weights["defense"] * def_score
            + weights["trend"] * trend_score
        )

        # Determine risk level
        assessment.risk_level = self._calculate_risk_level(assessment)

        # POINTS-specific: Line softness and stake sizing
        if stat_type == "POINTS":
            self._assess_line_softness(
                assessment, player_name, line, line_spread, edge_pct, prediction, consensus_line
            )
            self._calculate_points_stake(assessment)
        else:
            # REBOUNDS: Traditional approach
            assessment.recommended_stake = 1.0 if not assessment.should_skip else 0.0
            assessment.stake_reason = "Standard" if assessment.recommended_stake > 0 else "Filtered"

        if assessment.should_skip:
            self.high_risk_skipped += 1
            logger.debug(
                f"[RISK] SKIP {player_name} {stat_type}: {assessment.risk_level} "
                f"(stake={assessment.recommended_stake}, reason={assessment.stake_reason})"
            )

        return assessment

    def _assess_line_softness(
        self,
        assessment: RiskAssessment,
        player_name: str,
        line: float,
        line_spread: float,
        edge_pct: float,
        prediction: float,
        consensus_line: float,
    ) -> None:
        """
        Assess line softness for POINTS market.

        Soft line indicators:
        1. Large line spread (books disagree)
        2. High edge percentage
        3. Model prediction >> line
        4. Player's upside (75th percentile) clears line comfortably
        """
        softness_score = 0.0
        reasons = []

        # 1. Line spread check
        if line_spread >= self.VERY_SOFT_LINE_SPREAD:
            softness_score += 0.4
            reasons.append(f"very soft spread ({line_spread:.1f})")
        elif line_spread >= self.SOFT_LINE_SPREAD:
            softness_score += 0.25
            reasons.append(f"soft spread ({line_spread:.1f})")

        # 2. Edge percentage check
        if edge_pct >= self.SOFT_EDGE_PCT * 2:
            softness_score += 0.3
            reasons.append(f"huge edge ({edge_pct:.1f}%)")
        elif edge_pct >= self.SOFT_EDGE_PCT:
            softness_score += 0.15
            reasons.append(f"good edge ({edge_pct:.1f}%)")

        # 3. Model prediction vs line
        if prediction > 0 and line > 0:
            pred_edge = prediction - line
            if pred_edge >= 5.0:
                softness_score += 0.2
                reasons.append(f"model +{pred_edge:.1f}")
            elif pred_edge >= 3.0:
                softness_score += 0.1

        # 4. Upside check (does player's 75th percentile clear line?)
        real_vol = self._get_real_volatility(player_name, "POINTS")
        if real_vol and line > 0:
            p75 = real_vol.get("percentile_75", 0)
            p90 = real_vol.get("percentile_90", 0)
            if p75 > line * 1.1:  # 75th percentile clears line by 10%+
                softness_score += 0.15
                reasons.append(f"upside clears (p75={p75:.0f})")
            if p90 > line * 1.2:  # 90th percentile has big ceiling
                softness_score += 0.1
                reasons.append(f"ceiling (p90={p90:.0f})")

        assessment.line_softness_score = min(1.0, softness_score)
        assessment.line_is_soft = softness_score >= 0.4
        assessment.details["line_softness"] = ", ".join(reasons) if reasons else "sharp line"

    def _calculate_points_stake(self, assessment: RiskAssessment) -> None:
        """
        Calculate recommended stake for POINTS based on volatility + line softness.

        Calibrated based on backtest data:
        - 0.75u picks had 78% WR (best)
        - 0.5u picks had 71% WR (good)
        - 1.0u picks had 60% WR (needs soft line requirement)
        - 0.25u picks had 40% WR (DROP entirely)

        New logic:
        - EXTREME risk → SKIP (was 0.25u, too risky)
        - HIGH vol + sharp line → SKIP
        - HIGH vol + soft line → 0.75u (was 0.5u, proven profitable)
        - MEDIUM risk + soft line → 1.0u (best edge)
        - LOW risk + soft line → 1.0u
        - Any risk + NOT soft line → 0.5u or SKIP
        - SLUMP or ELITE DEF → reduce by 0.25u (not multiplicative)
        """
        base_stake = 1.0
        reasons = []

        # EXTREME risk → always skip (40% WR in backtest)
        if assessment.risk_level == "EXTREME":
            base_stake = 0.0
            reasons.append("EXTREME risk = SKIP")
            assessment.recommended_stake = 0.0
            assessment.stake_reason = reasons[0]
            return

        # Line softness is key - sharp lines without edge should be reduced/skipped
        if not assessment.line_is_soft:
            if assessment.high_volatility:
                base_stake = 0.0
                reasons.append("high vol + sharp line = SKIP")
            else:
                base_stake = 0.5
                reasons.append("sharp line = reduced")
        else:
            # Soft line - good opportunity
            if assessment.high_volatility:
                # HIGH vol + soft line was 71% WR at 0.5u, bump to 0.75u
                base_stake = 0.75
                reasons.append("high vol + soft line")
            elif assessment.risk_level == "MEDIUM":
                # MEDIUM risk + soft was 78% WR - this is the sweet spot
                base_stake = 1.0
                reasons.append("medium risk + soft line")
            else:
                # LOW risk + soft line
                base_stake = 1.0
                reasons.append("low risk + soft line")

        # Slump penalty (additive reduction, not multiplicative)
        if assessment.negative_trend and base_stake > 0:
            base_stake = max(0.5, base_stake - 0.25)
            reasons.append("slump -0.25u")

        # Elite defense penalty (additive reduction)
        if assessment.elite_defense and base_stake > 0:
            base_stake = max(0.5, base_stake - 0.25)
            reasons.append("elite def -0.25u")

        # Round to nearest 0.25
        assessment.recommended_stake = round(base_stake * 4) / 4
        assessment.stake_reason = " | ".join(reasons) if reasons else "standard"

    def _assess_volatility(
        self, features: Dict, stat_type: str, player_name: str = None
    ) -> Tuple[float, bool, str]:
        """Assess player volatility from game logs."""
        stat_key = stat_type.lower()

        real_vol = None
        if player_name:
            real_vol = self._get_real_volatility(player_name, stat_type)

        if real_vol:
            stat_cv = real_vol["cv_L5"]
            minutes_cv = real_vol["min_cv_L5"]
            std_L5 = real_vol["std_L5"]
            range_L5 = real_vol["range_L5"]
            recent = real_vol["recent_games"]
            avg_L5 = real_vol["avg_L5"]

            combined_cv = 0.7 * stat_cv + 0.3 * minutes_cv

            # Boost CV if range is extreme
            range_ratio = range_L5 / avg_L5 if avg_L5 > 0 else 0
            if range_ratio > 1.0:
                combined_cv = max(combined_cv, 0.5)
        else:
            std_L5 = features.get(f"{stat_key}_std_L5", 4.0)
            avg_L5 = features.get(f"ema_{stat_key}_L5", 15.0)
            minutes_std = features.get("minutes_std_L5", 3.0)
            minutes_avg = features.get("ema_minutes_L5", 28.0)

            stat_cv = std_L5 / avg_L5 if avg_L5 > 0 else 0.0
            minutes_cv = minutes_std / minutes_avg if minutes_avg > 0 else 0.0
            combined_cv = 0.7 * stat_cv + 0.3 * minutes_cv
            range_L5 = 0
            recent = []

        if stat_type == "POINTS":
            high_thresh = self.POINTS_VOLATILITY_HIGH
            extreme_thresh = self.POINTS_VOLATILITY_EXTREME
        else:
            high_thresh = self.REBOUNDS_VOLATILITY_HIGH
            extreme_thresh = self.REBOUNDS_VOLATILITY_EXTREME

        if combined_cv >= extreme_thresh:
            score = 1.0
            flag = True
            recent_str = f", recent={recent[:5]}" if recent else ""
            detail = f"EXTREME vol (CV={combined_cv:.2f}, std={std_L5:.1f}{recent_str})"
        elif combined_cv >= high_thresh:
            score = 0.7
            flag = True
            recent_str = f", recent={recent[:5]}" if recent else ""
            detail = f"HIGH vol (CV={combined_cv:.2f}, std={std_L5:.1f}{recent_str})"
        else:
            score = combined_cv / high_thresh
            flag = False
            detail = f"Normal vol (CV={combined_cv:.2f})"

        return score, flag, detail

    def _assess_defense(self, opp_rank: Optional[int], stat_type: str) -> Tuple[float, bool, str]:
        """Assess opponent defense quality."""
        if opp_rank is None:
            return 0.3, False, "Unknown defense"

        if opp_rank <= self.ELITE_DEFENSE_RANK:
            score = 1.0
            flag = True
            detail = f"ELITE defense (#{opp_rank})"
        elif opp_rank <= self.GOOD_DEFENSE_RANK:
            score = 0.6
            flag = False
            detail = f"Good defense (#{opp_rank})"
        elif opp_rank <= 20:
            score = 0.3
            flag = False
            detail = f"Average defense (#{opp_rank})"
        else:
            score = 0.1
            flag = False
            detail = f"Weak defense (#{opp_rank})"

        return score, flag, detail

    def _assess_trend(
        self, features: Dict, stat_type: str, player_name: str = None
    ) -> Tuple[float, bool, str]:
        """Assess recent trend using median L3 vs L10."""
        stat_key = stat_type.lower()

        real_vol = None
        if player_name:
            real_vol = self._get_real_volatility(player_name, stat_type)

        if real_vol and len(real_vol.get("recent_games", [])) >= 5:
            recent_games = real_vol["recent_games"]
            avg_L10 = real_vol["avg_L10"]
            L3_games = recent_games[:3]
            median_L3 = statistics.median(L3_games)
            trend_ratio = median_L3 / avg_L10 if avg_L10 > 0 else 1.0
        else:
            avg_L5 = features.get(f"ema_{stat_key}_L5", 15.0)
            avg_L10 = features.get(f"ema_{stat_key}_L10", 15.0)
            median_L3 = avg_L5
            trend_ratio = avg_L5 / avg_L10 if avg_L10 > 0 else 1.0
            recent_games = []

        if trend_ratio < self.SLUMP_THRESHOLD:
            decline_pct = (1 - trend_ratio) * 100
            score = min(1.0, decline_pct / 15)
            flag = True
            recent_str = f", recent={recent_games[:5]}" if recent_games else ""
            detail = (
                f"SLUMP (L3={median_L3:.1f} vs L10={avg_L10:.1f}, -{decline_pct:.0f}%{recent_str})"
            )
        elif trend_ratio > self.HOT_THRESHOLD:
            score = 0.0
            flag = False
            detail = f"HOT (L3={median_L3:.1f} vs L10={avg_L10:.1f}, +{(trend_ratio-1)*100:.0f}%)"
        else:
            score = 0.2
            flag = False
            detail = f"Stable (L3={median_L3:.1f} vs L10={avg_L10:.1f})"

        return score, flag, detail

    def _calculate_risk_level(self, assessment: RiskAssessment) -> str:
        """Calculate overall risk level."""
        high_risk_flags = sum(
            [
                assessment.high_volatility,
                assessment.elite_defense,
                assessment.negative_trend,
            ]
        )
        total_score = assessment.total_risk_score

        if high_risk_flags >= 2 or total_score >= 0.8:
            return "EXTREME"
        if high_risk_flags >= 1 and total_score >= 0.5:
            return "HIGH"
        if total_score >= 0.4:
            return "MEDIUM"
        return "LOW"

    def format_risk_summary(self, assessment: RiskAssessment) -> str:
        """Format risk assessment for display."""
        flags = []
        if assessment.high_volatility:
            flags.append("HIGH_VOL")
        if assessment.elite_defense:
            flags.append("ELITE_DEF")
        if assessment.negative_trend:
            flags.append("SLUMP")

        if assessment.stat_type == "POINTS":
            soft_str = "SOFT" if assessment.line_is_soft else "SHARP"
            return f"[{assessment.risk_level}] {', '.join(flags) or 'none'} | {soft_str} | {assessment.recommended_stake}u"
        else:
            if not flags:
                return None
            return f"[{assessment.risk_level}] {', '.join(flags)}"

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            "assessed": self.stats_assessed,
            "high_risk_skipped": self.high_risk_skipped,
            "skip_rate": (
                self.high_risk_skipped / self.stats_assessed * 100 if self.stats_assessed > 0 else 0
            ),
        }
