"""
anomalies.py — Real-time anomaly detection for GET /stores/{id}/anomalies.

Detects three classes of anomaly:
  1. BILLING_QUEUE_SPIKE   — queue depth exceeds threshold in recent window
  2. CONVERSION_DROP       — today's conversion rate << 7-day rolling average
  3. DEAD_ZONE             — a named zone has had zero visits in the last 30 min

Each anomaly carries a severity (INFO / WARN / CRITICAL) and a suggested_action.
"""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func

from .database import get_conn, events_table, sessions_table, daily_stats_table, now_utc_str
from .models import (
    Anomaly, AnomalyType, AnomalySeverity, StoreAnomalies, EventType,
)

logger = logging.getLogger("anomalies")

# ── Thresholds ────────────────────────────────────────────────────────────────
QUEUE_SPIKE_THRESHOLD      = 5      # queue_depth >= this → WARN
QUEUE_SPIKE_CRITICAL       = 10     # queue_depth >= this → CRITICAL
CONVERSION_DROP_WARN_PCT   = 0.20   # 20% below 7-day avg → WARN
CONVERSION_DROP_CRIT_PCT   = 0.40   # 40% below → CRITICAL
DEAD_ZONE_WINDOW_MIN       = 30     # no visits in 30 min → INFO
DEAD_ZONE_WARN_MIN         = 60     # no visits in 60 min → WARN


def compute_anomalies(store_id: str) -> StoreAnomalies:
    now = now_utc_str()
    anomalies: list[Anomaly] = []

    with get_conn() as conn:
        anomalies += _check_queue_spike(conn, store_id, now)
        anomalies += _check_conversion_drop(conn, store_id, now)
        anomalies += _check_dead_zones(conn, store_id, now)

    return StoreAnomalies(store_id=store_id, anomalies=anomalies, as_of=now)


# ─────────────────────────────────────────────────────────────────────────────
# Queue spike
# ─────────────────────────────────────────────────────────────────────────────

def _check_queue_spike(conn, store_id: str, now: str) -> list[Anomaly]:
    five_min_ago = _ago_str(minutes=5)

    row = conn.execute(
        select(func.max(events_table.c.queue_depth))
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.event_type == EventType.BILLING_QUEUE_JOIN.value,
            events_table.c.timestamp  >= five_min_ago,
            events_table.c.queue_depth != None,
        )
    ).scalar()

    depth = int(row) if row else 0

    if depth >= QUEUE_SPIKE_CRITICAL:
        severity = AnomalySeverity.CRITICAL
    elif depth >= QUEUE_SPIKE_THRESHOLD:
        severity = AnomalySeverity.WARN
    else:
        return []

    return [Anomaly(
        anomaly_type=AnomalyType.BILLING_QUEUE_SPIKE,
        severity=severity,
        description=f"Billing queue depth reached {depth} in the last 5 minutes.",
        suggested_action=(
            "Open additional billing counter immediately."
            if severity == AnomalySeverity.CRITICAL
            else "Monitor queue — consider opening a second billing lane."
        ),
        detected_at=now,
        context={"max_queue_depth": depth, "window_minutes": 5},
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Conversion drop vs 7-day baseline
# ─────────────────────────────────────────────────────────────────────────────

def _check_conversion_drop(conn, store_id: str, now: str) -> list[Anomaly]:
    # Today's conversion rate
    today_start = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    today_visitors = conn.execute(
        select(func.count(func.distinct(events_table.c.visitor_id)))
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.event_type.in_([
                EventType.ENTRY.value, EventType.REENTRY.value,
            ]),
            events_table.c.timestamp  >= today_start,
        )
    ).scalar() or 0

    today_conversions = conn.execute(
        select(func.count())
        .where(
            sessions_table.c.store_id  == store_id,
            sessions_table.c.converted == True,
        )
    ).scalar() or 0

    if today_visitors == 0:
        return []

    today_rate = today_conversions / today_visitors

    # 7-day rolling average from daily_stats
    seven_days_ago = (
        datetime.now(tz=timezone.utc) - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    avg_row = conn.execute(
        select(func.avg(daily_stats_table.c.conversion_rate))
        .where(
            daily_stats_table.c.store_id == store_id,
            daily_stats_table.c.date     >= seven_days_ago,
        )
    ).scalar()

    if avg_row is None or float(avg_row) == 0.0:
        # No historical baseline yet — emit INFO
        return [Anomaly(
            anomaly_type=AnomalyType.CONVERSION_DROP,
            severity=AnomalySeverity.INFO,
            description="No 7-day baseline available yet for conversion comparison.",
            suggested_action="Continue collecting data to establish a baseline.",
            detected_at=now,
            context={"today_rate": round(today_rate, 4), "baseline": None},
        )]

    baseline = float(avg_row)
    drop_pct = (baseline - today_rate) / baseline if baseline > 0 else 0.0

    if drop_pct >= CONVERSION_DROP_CRIT_PCT:
        severity = AnomalySeverity.CRITICAL
    elif drop_pct >= CONVERSION_DROP_WARN_PCT:
        severity = AnomalySeverity.WARN
    else:
        return []

    return [Anomaly(
        anomaly_type=AnomalyType.CONVERSION_DROP,
        severity=severity,
        description=(
            f"Conversion rate is {drop_pct:.0%} below the 7-day average "
            f"({today_rate:.2%} vs baseline {baseline:.2%})."
        ),
        suggested_action=(
            "Escalate to store manager and review customer journey."
            if severity == AnomalySeverity.CRITICAL
            else "Investigate friction points — check funnel endpoint for drop-off detail."
        ),
        detected_at=now,
        context={
            "today_rate":   round(today_rate, 4),
            "baseline_rate": round(baseline, 4),
            "drop_pct":     round(drop_pct, 4),
        },
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Dead zone detection
# ─────────────────────────────────────────────────────────────────────────────

def _check_dead_zones(conn, store_id: str, now: str) -> list[Anomaly]:
    """
    A zone is 'dead' if it received at least one visit historically but has had
    no ZONE_ENTER events in the last 30 minutes.
    """
    thirty_min_ago = _ago_str(minutes=DEAD_ZONE_WINDOW_MIN)
    sixty_min_ago  = _ago_str(minutes=DEAD_ZONE_WARN_MIN)

    # Zones that have ever had visits for this store
    all_zones = {
        row[0]
        for row in conn.execute(
            select(func.distinct(events_table.c.zone_id))
            .where(
                events_table.c.store_id  == store_id,
                events_table.c.zone_id   != None,
                events_table.c.is_staff  == False,
            )
        ).fetchall()
        if row[0]
    }

    # Zones with a visit in the last 30 min
    recently_active = {
        row[0]
        for row in conn.execute(
            select(func.distinct(events_table.c.zone_id))
            .where(
                events_table.c.store_id   == store_id,
                events_table.c.zone_id    != None,
                events_table.c.is_staff   == False,
                events_table.c.event_type == EventType.ZONE_ENTER.value,
                events_table.c.timestamp  >= thirty_min_ago,
            )
        ).fetchall()
        if row[0]
    }

    # Zones with a visit in the last 60 min (for severity gradation)
    active_60 = {
        row[0]
        for row in conn.execute(
            select(func.distinct(events_table.c.zone_id))
            .where(
                events_table.c.store_id   == store_id,
                events_table.c.zone_id    != None,
                events_table.c.is_staff   == False,
                events_table.c.event_type == EventType.ZONE_ENTER.value,
                events_table.c.timestamp  >= sixty_min_ago,
            )
        ).fetchall()
        if row[0]
    }

    anomalies = []
    for zone in sorted(all_zones - recently_active):
        severity = (
            AnomalySeverity.WARN
            if zone not in active_60
            else AnomalySeverity.INFO
        )
        window_label = "60 minutes" if severity == AnomalySeverity.WARN else "30 minutes"
        anomalies.append(Anomaly(
            anomaly_type=AnomalyType.DEAD_ZONE,
            severity=severity,
            description=f"Zone '{zone}' has had no customer visits in the last {window_label}.",
            suggested_action=(
                f"Check if zone '{zone}' signage/stock is deterring entry, "
                "or verify camera coverage for this zone."
            ),
            detected_at=now,
            context={"zone_id": zone, "idle_window_minutes": DEAD_ZONE_WARN_MIN if severity == AnomalySeverity.WARN else DEAD_ZONE_WINDOW_MIN},
        ))

    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _ago_str(minutes: int = 0, hours: int = 0) -> str:
    dt = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes, hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
