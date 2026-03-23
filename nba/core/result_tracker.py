"""
Result Tracker — automated performance tracking and anomaly detection.

Runs after validation to compute rolling metrics across multiple dimensions:
win rate, ROI, CLV, by market, by tier, by edge bucket.

Flags anomalies when metrics drop below thresholds.

Schema for performance_metrics table (to be created by Claude 1):
  CREATE TABLE performance_metrics (
      id SERIAL PRIMARY KEY,
      metric_date DATE NOT NULL,
      period TEXT NOT NULL,
      total_picks INTEGER,
      wins INTEGER,
      losses INTEGER,
      win_rate NUMERIC(5,2),
      roi NUMERIC(6,2),
      profit NUMERIC(8,2),
      avg_clv_cents NUMERIC(6,2),
      beat_close_rate NUMERIC(5,3),
      by_market JSONB,
      by_tier JSONB,
      by_edge_bucket JSONB,
      by_model JSONB,
      anomalies JSONB,
      computed_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE (metric_date, period)
  );
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nba.config.database import get_axiom_db_config, get_connection

log = logging.getLogger("nba.result_tracker")

_EST = ZoneInfo("America/New_York")

# Anomaly thresholds
_MIN_WIN_RATE_7D = 52.0
_MIN_WIN_RATE_30D = 50.0
_MIN_ROI_7D = -5.0
_MIN_PICKS_FOR_ALERT = 5


def _connect_axiom():
    return psycopg2.connect(**get_axiom_db_config())


def _grade(is_hit: bool) -> dict:
    """Standard -110 juice grading."""
    if is_hit:
        return {"outcome": "WIN", "profit": 0.909}
    return {"outcome": "LOSS", "profit": -1.0}


def _edge_bucket(edge: Optional[float]) -> str:
    """Classify edge into buckets."""
    if edge is None:
        return "unknown"
    e = float(edge)
    if e >= 10.0:
        return "10%+"
    elif e >= 7.0:
        return "7-10%"
    elif e >= 5.0:
        return "5-7%"
    elif e >= 3.0:
        return "3-5%"
    elif e >= 0:
        return "0-3%"
    return "negative"


class ResultTracker:
    """Tracks pick performance across multiple dimensions."""

    def compute_metrics(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Compute comprehensive metrics for a date range.

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            dict with total, wins, losses, win_rate, roi, profit,
            by_market, by_tier, by_edge_bucket, by_model
        """
        conn = _connect_axiom()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Deduplicated graded picks (latest run per pick)
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY run_date, player_name, stat_type
                                   ORDER BY run_number DESC
                               ) AS rn
                        FROM nba_prediction_history
                        WHERE run_date BETWEEN %s AND %s
                          AND is_hit IS NOT NULL
                    )
                    SELECT player_name, stat_type, model_version, tier,
                           line, p_over, edge, spread, book,
                           run_date, is_hit, context_snapshot
                    FROM ranked WHERE rn = 1
                    ORDER BY run_date
                    """,
                    (start_date, end_date),
                )
                picks = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        if not picks:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "roi": 0.0,
                "profit": 0.0,
                "by_market": {},
                "by_tier": {},
                "by_edge_bucket": {},
                "by_model": {},
            }

        by_market = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
        by_tier = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
        by_edge = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
        by_model = defaultdict(lambda: {"w": 0, "l": 0, "profit": 0.0})
        total_w, total_l, total_profit = 0, 0, 0.0

        for pick in picks:
            g = _grade(pick["is_hit"])
            is_win = g["outcome"] == "WIN"
            profit = g["profit"]

            market = pick.get("stat_type", "UNKNOWN")
            tier = pick.get("tier") or "unknown"
            model = pick.get("model_version") or "xl"
            edge = pick.get("edge")
            bucket = _edge_bucket(edge)

            def _update(d, is_win, profit):
                if is_win:
                    d["w"] += 1
                else:
                    d["l"] += 1
                d["profit"] += profit

            _update(by_market[market], is_win, profit)
            _update(by_tier[tier], is_win, profit)
            _update(by_edge[bucket], is_win, profit)
            _update(by_model[model], is_win, profit)

            if is_win:
                total_w += 1
            else:
                total_l += 1
            total_profit += profit

        total = total_w + total_l

        def _summarize(d):
            result = {}
            for key, stats in d.items():
                t = stats["w"] + stats["l"]
                result[key] = {
                    "w": stats["w"],
                    "l": stats["l"],
                    "total": t,
                    "win_rate": round(stats["w"] / t * 100, 1) if t else 0,
                    "roi": round(stats["profit"] / t * 100, 1) if t else 0,
                    "profit": round(stats["profit"], 2),
                }
            return result

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total": total,
            "wins": total_w,
            "losses": total_l,
            "win_rate": round(total_w / total * 100, 1) if total else 0,
            "roi": round(total_profit / total * 100, 1) if total else 0,
            "profit": round(total_profit, 2),
            "by_market": _summarize(by_market),
            "by_tier": _summarize(by_tier),
            "by_edge_bucket": _summarize(by_edge),
            "by_model": _summarize(by_model),
        }

    def compute_rolling(self, days: int = 7) -> dict[str, Any]:
        """Compute rolling metrics for the last N days.

        Args:
            days: Number of days to look back.

        Returns:
            Same format as compute_metrics with period added.
        """
        now = datetime.now(_EST)
        end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

        result = self.compute_metrics(start_date, end_date)
        result["period"] = f"{days}d"
        result["days"] = days
        return result

    def check_anomalies(self) -> list[str]:
        """Check for performance anomalies across multiple windows.

        Returns:
            List of alert messages (empty if all metrics are healthy).
        """
        alerts = []

        # 7-day check
        r7 = self.compute_rolling(7)
        if r7["total"] >= _MIN_PICKS_FOR_ALERT:
            if r7["win_rate"] < _MIN_WIN_RATE_7D:
                alerts.append(
                    f"7-day WR {r7['win_rate']:.1f}% below {_MIN_WIN_RATE_7D}% "
                    f"({r7['wins']}W-{r7['losses']}L)"
                )
            if r7["roi"] < _MIN_ROI_7D:
                alerts.append(f"7-day ROI {r7['roi']:+.1f}% below {_MIN_ROI_7D:+.1f}%")

        # 30-day check
        r30 = self.compute_rolling(30)
        if r30["total"] >= 10:
            if r30["win_rate"] < _MIN_WIN_RATE_30D:
                alerts.append(
                    f"30-day WR {r30['win_rate']:.1f}% below {_MIN_WIN_RATE_30D}% "
                    f"({r30['wins']}W-{r30['losses']}L) — consider model review"
                )

        # Per-market anomalies (7d)
        for market, stats in r7.get("by_market", {}).items():
            if stats["total"] >= 3 and stats["win_rate"] < 40.0:
                alerts.append(
                    f"{market} 7-day WR critically low: {stats['win_rate']:.1f}% "
                    f"({stats['w']}W-{stats['l']}L)"
                )

        # Per-model anomalies (7d)
        for model, stats in r7.get("by_model", {}).items():
            if stats["total"] >= 3 and stats["win_rate"] < 45.0:
                alerts.append(
                    f"Model {model} 7-day WR: {stats['win_rate']:.1f}% "
                    f"({stats['w']}W-{stats['l']}L)"
                )

        # Season-long check
        r_season = self.compute_rolling(90)
        if r_season["total"] >= 50 and r_season["win_rate"] < 52.0:
            alerts.append(
                f"Season WR declining: {r_season['win_rate']:.1f}% over "
                f"{r_season['total']} picks — trigger retraining review"
            )

        return alerts

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a comprehensive performance summary for the API.

        Returns:
            dict with rolling_7d, rolling_30d, season metrics and anomalies.
        """
        r7 = self.compute_rolling(7)
        r30 = self.compute_rolling(30)
        anomalies = self.check_anomalies()

        # Try CLV metrics
        clv_data = None
        try:
            from nba.core.clv_tracker import CLVTracker

            tracker = CLVTracker()
            clv_data = tracker.compute_rolling_clv(7)
        except Exception as e:
            log.debug(f"CLV tracking unavailable: {e}")

        return {
            "rolling_7d": r7,
            "rolling_30d": r30,
            "clv_7d": clv_data,
            "anomalies": anomalies,
            "computed_at": datetime.now(_EST).isoformat(),
        }

    def persist_metrics(self, metric_date: str | None = None) -> int:
        """Persist rolling performance metrics to features.performance_metrics.

        Computes 7d and 30d rolling metrics and writes one row per period.
        Creates the table if it doesn't exist.

        Args:
            metric_date: YYYY-MM-DD (defaults to yesterday EST).

        Returns:
            Number of rows upserted.
        """
        if metric_date is None:
            metric_date = (datetime.now(_EST) - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = get_connection("features", autocommit=False)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_date DATE NOT NULL,
                        period TEXT NOT NULL,
                        total_picks INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        win_rate NUMERIC(5,2),
                        roi NUMERIC(6,2),
                        profit NUMERIC(8,2),
                        avg_clv_cents NUMERIC(6,2),
                        beat_close_rate NUMERIC(5,3),
                        by_market JSONB,
                        by_tier JSONB,
                        by_edge_bucket JSONB,
                        by_model JSONB,
                        anomalies JSONB,
                        computed_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (metric_date, period)
                    )
                """
                )

            anomalies = self.check_anomalies()
            rows_written = 0

            for days, period_label in [(7, "7d"), (30, "30d")]:
                m = self.compute_rolling(days)
                if m.get("total", 0) == 0:
                    continue

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO performance_metrics
                            (metric_date, period, total_picks, wins, losses,
                             win_rate, roi, profit, by_market, by_tier,
                             by_edge_bucket, by_model, anomalies)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (metric_date, period) DO UPDATE SET
                            total_picks = EXCLUDED.total_picks,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            win_rate = EXCLUDED.win_rate,
                            roi = EXCLUDED.roi,
                            profit = EXCLUDED.profit,
                            by_market = EXCLUDED.by_market,
                            by_tier = EXCLUDED.by_tier,
                            by_edge_bucket = EXCLUDED.by_edge_bucket,
                            by_model = EXCLUDED.by_model,
                            anomalies = EXCLUDED.anomalies,
                            computed_at = NOW()
                        """,
                        (
                            metric_date,
                            period_label,
                            m.get("total", 0),
                            m.get("wins", 0),
                            m.get("losses", 0),
                            m.get("win_rate", 0),
                            m.get("roi", 0),
                            m.get("profit", 0),
                            json.dumps(m.get("by_market", {})),
                            json.dumps(m.get("by_tier", {})),
                            json.dumps(m.get("by_edge_bucket", {})),
                            json.dumps(m.get("by_model", {})),
                            json.dumps(anomalies),
                        ),
                    )
                    rows_written += cur.rowcount

            conn.commit()
            log.info(f"Persisted {rows_written} performance metric rows for {metric_date}")
            return rows_written
        except Exception:
            conn.rollback()
            log.exception("Failed to persist performance metrics")
            return 0
        finally:
            conn.close()
