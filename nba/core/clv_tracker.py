"""
CLV (Closing Line Value) Tracker
=================================
The most important metric for long-term profitability.

If you consistently beat the closing line, you're profitable long-term
regardless of short-term variance. CLV measures whether the market
moved in your direction after you made your pick.

Data sources:
  - nba_line_snapshots (intelligence schema): append-only line history
  - nba_prediction_history (axiom schema): pick records with lines

Persistence target: features.clv_tracking
  UNIQUE (player_name, game_date, stat_type, book_name)
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nba.config.database import get_axiom_db_config, get_connection, get_intelligence_db_config

log = logging.getLogger("nba.clv_tracker")

_EST = ZoneInfo("America/New_York")


def _connect_axiom():
    return psycopg2.connect(**get_axiom_db_config())


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability.

    -110 -> 0.5238 (risk 110 to win 100)
    +150 -> 0.4000 (risk 100 to win 150)
    """
    if odds is None or odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


class CLVTracker:
    """Tracks Closing Line Value for graded picks.

    CLV = closing_implied_prob - opening_implied_prob
    Positive CLV means the line moved in your direction (market agrees with you).
    """

    def get_closing_line(
        self,
        player_name: str,
        stat_type: str,
        game_date: str,
        book_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the closing line (last snapshot before game time).

        Returns dict with {over_line, over_odds, under_odds, snapshot_at} or None.
        """
        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if book_name:
                    cur.execute(
                        """
                        SELECT over_line, over_odds, under_odds, snapshot_at, book_name
                        FROM nba_line_snapshots
                        WHERE player_name = %s
                          AND stat_type = %s
                          AND game_date = %s
                          AND book_name = %s
                        ORDER BY snapshot_at DESC
                        LIMIT 1
                        """,
                        (player_name, stat_type, game_date, book_name),
                    )
                else:
                    # Use consensus (any book, latest snapshot)
                    cur.execute(
                        """
                        SELECT over_line, over_odds, under_odds, snapshot_at, book_name
                        FROM nba_line_snapshots
                        WHERE player_name = %s
                          AND stat_type = %s
                          AND game_date = %s
                        ORDER BY snapshot_at DESC
                        LIMIT 1
                        """,
                        (player_name, stat_type, game_date),
                    )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    def get_opening_line(
        self,
        player_name: str,
        stat_type: str,
        game_date: str,
        book_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the opening line (earliest snapshot for the day).

        Returns dict with {over_line, over_odds, under_odds, snapshot_at} or None.
        """
        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if book_name:
                    cur.execute(
                        """
                        SELECT over_line, over_odds, under_odds, snapshot_at, book_name
                        FROM nba_line_snapshots
                        WHERE player_name = %s
                          AND stat_type = %s
                          AND game_date = %s
                          AND book_name = %s
                        ORDER BY snapshot_at ASC
                        LIMIT 1
                        """,
                        (player_name, stat_type, game_date, book_name),
                    )
                else:
                    cur.execute(
                        """
                        SELECT over_line, over_odds, under_odds, snapshot_at, book_name
                        FROM nba_line_snapshots
                        WHERE player_name = %s
                          AND stat_type = %s
                          AND game_date = %s
                        ORDER BY snapshot_at ASC
                        LIMIT 1
                        """,
                        (player_name, stat_type, game_date),
                    )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    def compute_clv(self, pick: dict) -> Optional[dict]:
        """Compute CLV for a single graded pick.

        Args:
            pick: dict with player_name, stat_type, run_date, line, p_over, book

        Returns:
            dict with {opening_line, closing_line, clv_cents, beat_close,
                       opening_implied_prob, closing_implied_prob, model_prob}
            or None if line snapshots not available.
        """
        player_name = pick.get("player_name", "")
        stat_type = pick.get("stat_type", "")
        game_date = str(pick.get("run_date", ""))
        our_line = float(pick.get("line") or 0)
        model_prob = float(pick.get("p_over") or 0.5)
        book = pick.get("book")

        opening = self.get_opening_line(player_name, stat_type, game_date, book)
        closing = self.get_closing_line(player_name, stat_type, game_date, book)

        if not opening or not closing:
            return None

        opening_line = float(opening.get("over_line") or our_line)
        closing_line = float(closing.get("over_line") or our_line)

        opening_odds = opening.get("over_odds")
        closing_odds = closing.get("over_odds")

        opening_implied = _american_to_implied_prob(opening_odds)
        closing_implied = _american_to_implied_prob(closing_odds)

        # CLV in "cents": positive means line moved in our favor
        # For OVER bets: closing line dropping = good (easier to hit)
        # closing_implied rising = market now thinks OVER is more likely = validates us
        clv_cents = round((closing_implied - opening_implied) * 100, 2)

        # Did we beat the closing line? Our model prob > closing implied prob
        beat_close = model_prob > closing_implied

        return {
            "opening_line": opening_line,
            "closing_line": closing_line,
            "our_line": our_line,
            "opening_implied_prob": round(opening_implied, 4),
            "closing_implied_prob": round(closing_implied, 4),
            "model_prob": round(model_prob, 4),
            "clv_cents": clv_cents,
            "beat_close": beat_close,
            "line_movement": round(closing_line - opening_line, 1),
        }

    def compute_daily_clv(self, game_date: str) -> dict[str, Any]:
        """Compute aggregate CLV metrics for a given date.

        Returns:
            dict with {date, total_picks, picks_with_clv, avg_clv_cents,
                       beat_close_rate, clv_positive_rate, by_market}
        """
        conn = _connect_axiom()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get graded picks for the date (deduplicated, latest run)
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY player_name, stat_type
                                   ORDER BY run_number DESC
                               ) AS rn
                        FROM nba_prediction_history
                        WHERE run_date = %s AND is_hit IS NOT NULL
                    )
                    SELECT player_name, stat_type, line, p_over, book,
                           run_date, model_version, is_hit
                    FROM ranked WHERE rn = 1
                    """,
                    (game_date,),
                )
                picks = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        if not picks:
            return {
                "date": game_date,
                "total_picks": 0,
                "picks_with_clv": 0,
                "avg_clv_cents": 0.0,
                "beat_close_rate": 0.0,
                "clv_positive_rate": 0.0,
                "by_market": {},
            }

        clv_results = []
        by_market: dict[str, list] = {}

        for pick in picks:
            clv = self.compute_clv(pick)
            if clv is None:
                continue

            clv["is_hit"] = pick.get("is_hit")
            clv["stat_type"] = pick.get("stat_type")
            clv_results.append(clv)

            market = pick.get("stat_type", "UNKNOWN")
            by_market.setdefault(market, []).append(clv)

        if not clv_results:
            return {
                "date": game_date,
                "total_picks": len(picks),
                "picks_with_clv": 0,
                "avg_clv_cents": 0.0,
                "beat_close_rate": 0.0,
                "clv_positive_rate": 0.0,
                "by_market": {},
            }

        avg_clv = sum(c["clv_cents"] for c in clv_results) / len(clv_results)
        beat_count = sum(1 for c in clv_results if c["beat_close"])
        positive_count = sum(1 for c in clv_results if c["clv_cents"] > 0)

        market_summary = {}
        for market, clvs in by_market.items():
            market_avg = sum(c["clv_cents"] for c in clvs) / len(clvs)
            market_beat = sum(1 for c in clvs if c["beat_close"])
            market_summary[market] = {
                "count": len(clvs),
                "avg_clv_cents": round(market_avg, 2),
                "beat_close_rate": round(market_beat / len(clvs), 3),
            }

        return {
            "date": game_date,
            "total_picks": len(picks),
            "picks_with_clv": len(clv_results),
            "avg_clv_cents": round(avg_clv, 2),
            "beat_close_rate": round(beat_count / len(clv_results), 3),
            "clv_positive_rate": round(positive_count / len(clv_results), 3),
            "by_market": market_summary,
        }

    def persist_daily_clv(self, game_date: str) -> int:
        """Compute CLV for a date and persist per-pick results to features.clv_tracking.

        Fetches graded picks from axiom.nba_prediction_history, computes CLV
        for each pick, and UPSERTs into features.clv_tracking.

        Args:
            game_date: Date string in YYYY-MM-DD format.

        Returns:
            Number of rows persisted.
        """
        # Fetch graded picks (deduplicated, latest run per player+stat)
        axiom_conn = _connect_axiom()
        try:
            with axiom_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY player_name, stat_type
                                   ORDER BY run_number DESC
                               ) AS rn
                        FROM nba_prediction_history
                        WHERE run_date = %s AND is_hit IS NOT NULL
                    )
                    SELECT id, player_name, stat_type, line, p_over, book,
                           run_date, model_version, actual_result, is_hit
                    FROM ranked WHERE rn = 1
                    """,
                    (game_date,),
                )
                picks = [dict(r) for r in cur.fetchall()]
        finally:
            axiom_conn.close()

        if not picks:
            log.info("persist_daily_clv: no graded picks for %s", game_date)
            return 0

        # Compute CLV for each pick and collect rows to insert
        rows_to_upsert = []
        for pick in picks:
            clv = self.compute_clv(pick)
            if clv is None:
                continue

            rows_to_upsert.append(
                (
                    pick["id"],  # prediction_id
                    pick["player_name"],
                    pick["stat_type"],
                    game_date,
                    pick.get("book"),  # book_name
                    clv["opening_line"],
                    clv["closing_line"],
                    float(pick.get("line") or 0),  # model_line
                    clv["opening_implied_prob"],
                    clv["closing_implied_prob"],
                    clv["model_prob"],
                    clv["clv_cents"],
                    clv["beat_close"],  # beat_closing_line
                    float(pick["actual_result"]) if pick.get("actual_result") is not None else None,
                    pick.get("is_hit"),
                )
            )

        if not rows_to_upsert:
            log.info("persist_daily_clv: no CLV data for %s (no snapshots)", game_date)
            return 0

        # UPSERT into features.clv_tracking
        feat_conn = get_connection("features", autocommit=False)
        try:
            with feat_conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO features.clv_tracking (
                        prediction_id, player_name, stat_type, game_date,
                        book_name, opening_line, closing_line, model_line,
                        opening_implied_prob, closing_implied_prob, model_prob,
                        clv_cents, beat_closing_line, actual_value, is_hit,
                        computed_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        NOW()
                    )
                    ON CONFLICT (player_name, game_date, stat_type, book_name)
                    DO UPDATE SET
                        prediction_id = EXCLUDED.prediction_id,
                        opening_line = EXCLUDED.opening_line,
                        closing_line = EXCLUDED.closing_line,
                        model_line = EXCLUDED.model_line,
                        opening_implied_prob = EXCLUDED.opening_implied_prob,
                        closing_implied_prob = EXCLUDED.closing_implied_prob,
                        model_prob = EXCLUDED.model_prob,
                        clv_cents = EXCLUDED.clv_cents,
                        beat_closing_line = EXCLUDED.beat_closing_line,
                        actual_value = EXCLUDED.actual_value,
                        is_hit = EXCLUDED.is_hit,
                        computed_at = NOW()
                    """,
                    rows_to_upsert,
                )
            feat_conn.commit()
            log.info("persist_daily_clv: persisted %d rows for %s", len(rows_to_upsert), game_date)
            return len(rows_to_upsert)
        except Exception:
            feat_conn.rollback()
            raise
        finally:
            feat_conn.close()

    def backfill_clv(self, start_date: str, end_date: str) -> int:
        """Backfill CLV data for a date range (inclusive).

        Args:
            start_date: Start date string in YYYY-MM-DD format.
            end_date: End date string in YYYY-MM-DD format.

        Returns:
            Total number of rows persisted across all dates.
        """
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        total = 0
        current = start

        while current <= end:
            date_str = current.isoformat()
            count = self.persist_daily_clv(date_str)
            if count:
                print(f"  {date_str}: persisted {count} CLV rows")
            total += count
            current += timedelta(days=1)

        print(f"[CLV] Backfill complete: {total} rows across {(end - start).days + 1} days")
        return total

    def compute_rolling_clv(self, days: int = 7) -> dict[str, Any]:
        """Compute rolling CLV metrics over a window.

        Args:
            days: Number of days to look back.

        Returns:
            dict with {period, days, dates_checked, total_picks, picks_with_clv,
                       avg_clv_cents, beat_close_rate, clv_positive_rate, by_market, daily}
        """
        now = datetime.now(_EST)
        daily_results = []
        all_clv_cents = []
        all_beat = []
        all_positive = []
        by_market_agg: dict[str, list] = {}

        for days_ago in range(1, days + 1):
            date_str = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            daily = self.compute_daily_clv(date_str)

            if daily["picks_with_clv"] > 0:
                daily_results.append(daily)
                # We don't have individual CLV results here, but we can
                # aggregate from daily summaries weighted by count
                n = daily["picks_with_clv"]
                all_clv_cents.extend([daily["avg_clv_cents"]] * n)
                all_beat.extend([daily["beat_close_rate"]] * n)
                all_positive.extend([daily["clv_positive_rate"]] * n)

                for market, stats in daily.get("by_market", {}).items():
                    by_market_agg.setdefault(market, []).append(stats)

        total_picks_with_clv = sum(d["picks_with_clv"] for d in daily_results)

        if not all_clv_cents:
            return {
                "period": f"{days}d",
                "days": days,
                "dates_checked": days,
                "total_picks": sum(d["total_picks"] for d in daily_results),
                "picks_with_clv": 0,
                "avg_clv_cents": 0.0,
                "beat_close_rate": 0.0,
                "clv_positive_rate": 0.0,
                "by_market": {},
                "daily": daily_results,
            }

        avg_clv = sum(all_clv_cents) / len(all_clv_cents)

        # Weighted beat_close and positive rates
        total_beat = sum(d["beat_close_rate"] * d["picks_with_clv"] for d in daily_results)
        total_positive = sum(d["clv_positive_rate"] * d["picks_with_clv"] for d in daily_results)

        market_summary = {}
        for market, stats_list in by_market_agg.items():
            total_count = sum(s["count"] for s in stats_list)
            weighted_clv = sum(s["avg_clv_cents"] * s["count"] for s in stats_list)
            weighted_beat = sum(s["beat_close_rate"] * s["count"] for s in stats_list)
            market_summary[market] = {
                "count": total_count,
                "avg_clv_cents": round(weighted_clv / total_count, 2) if total_count else 0,
                "beat_close_rate": round(weighted_beat / total_count, 3) if total_count else 0,
            }

        return {
            "period": f"{days}d",
            "days": days,
            "dates_checked": days,
            "total_picks": sum(d["total_picks"] for d in daily_results),
            "picks_with_clv": total_picks_with_clv,
            "avg_clv_cents": round(avg_clv, 2),
            "beat_close_rate": round(total_beat / total_picks_with_clv, 3),
            "clv_positive_rate": round(total_positive / total_picks_with_clv, 3),
            "by_market": market_summary,
            "daily": daily_results,
        }
