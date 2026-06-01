"""
metrics.py — Real-time store metric computation for GET /stores/{id}/metrics.

All metrics are computed from the live events table — never served from yesterday's cache.
Staff events (is_staff=True) are excluded from all customer metrics.

Key metric: Conversion Rate = converted sessions / unique customer sessions
Re-entries are NOT double-counted (visitor_id is the dedup key).
"""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, and_, text

from .database import get_conn, events_table, sessions_table, now_utc_str, parse_ts
from .models import (
    StoreMetrics, ZoneDwellStats,
    StoreFunnel, FunnelStage,
    StoreHeatmap, ZoneHeatmapEntry,
    EventType,
)

logger = logging.getLogger("metrics")

# Window for "today" metrics: last 24 hours rolling
METRICS_WINDOW_HOURS = 24

# Minimum sessions for HIGH data confidence flag
MIN_SESSIONS_FOR_HIGH_CONFIDENCE = 20


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{id}/metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(store_id: str) -> StoreMetrics:
    window_start = _window_start()
    as_of = now_utc_str()

    with get_conn() as conn:
        # ── Unique customer visitors (exclude staff, no double-count on reentry) ──
        unique_visitors = _count_unique_visitors(conn, store_id, window_start)

        # ── Conversions ─────────────────────────────────────────────────────
        conversions = _count_conversions(conn, store_id, window_start)

        conversion_rate = (
            conversions / unique_visitors if unique_visitors > 0 else 0.0
        )

        # ── Zone dwell ──────────────────────────────────────────────────────
        zone_dwell = _zone_dwell_stats(conn, store_id, window_start)

        avg_dwell_ms = (
            sum(z.avg_dwell_ms for z in zone_dwell) / len(zone_dwell)
            if zone_dwell else 0.0
        )

        # ── Current queue depth ─────────────────────────────────────────────
        queue_depth = _current_queue_depth(conn, store_id)

        # ── Abandonment rate ────────────────────────────────────────────────
        abandonment_rate = _abandonment_rate(conn, store_id, window_start)

    data_confidence = (
        "HIGH" if unique_visitors >= MIN_SESSIONS_FOR_HIGH_CONFIDENCE else "LOW"
    )

    return StoreMetrics(
        store_id=store_id,
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_ms=round(avg_dwell_ms, 1),
        zone_dwell=zone_dwell,
        queue_depth=queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        as_of=as_of,
        data_confidence=data_confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{id}/funnel
# ─────────────────────────────────────────────────────────────────────────────

def compute_funnel(store_id: str) -> StoreFunnel:
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase
    Unit: unique visitor sessions. Re-entries don't double-count.
    """
    window_start = _window_start()
    as_of = now_utc_str()

    with get_conn() as conn:
        # Stage 1: Unique entrants
        entries = _count_unique_visitors(conn, store_id, window_start)

        # Stage 2: Visitors who entered at least one zone
        zone_visitors = _distinct_visitors_with_event(
            conn, store_id, window_start,
            event_types=[EventType.ZONE_ENTER, EventType.ZONE_DWELL],
        )

        # Stage 3: Visitors who joined billing queue
        billing_visitors = conn.execute(
            select(func.count(func.distinct(sessions_table.c.visitor_id)))
            .where(
                sessions_table.c.store_id == store_id,
                sessions_table.c.was_in_billing == True,
            )
        ).scalar() or 0

        # Stage 4: Conversions
        purchases = _count_conversions(conn, store_id, window_start)

    def drop_off(prev: int, curr: int) -> float:
        if prev == 0:
            return 0.0
        return round((prev - curr) / prev * 100, 1)

    stages = [
        FunnelStage(stage="Entry",        count=entries,         drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit",   count=zone_visitors,   drop_off_pct=drop_off(entries, zone_visitors)),
        FunnelStage(stage="Billing Queue",count=billing_visitors, drop_off_pct=drop_off(zone_visitors, billing_visitors)),
        FunnelStage(stage="Purchase",     count=purchases,        drop_off_pct=drop_off(billing_visitors, purchases)),
    ]

    return StoreFunnel(store_id=store_id, stages=stages, as_of=as_of)


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{id}/heatmap
# ─────────────────────────────────────────────────────────────────────────────

def compute_heatmap(store_id: str) -> StoreHeatmap:
    window_start = _window_start()
    as_of = now_utc_str()

    with get_conn() as conn:
        rows = conn.execute(
            select(
                events_table.c.zone_id,
                func.count(func.distinct(events_table.c.visitor_id)).label("visit_count"),
                func.avg(events_table.c.dwell_ms).label("avg_dwell_ms"),
            )
            .where(
                events_table.c.store_id   == store_id,
                events_table.c.is_staff   == False,
                events_table.c.zone_id    != None,
                events_table.c.event_type.in_([
                    EventType.ZONE_ENTER.value,
                    EventType.ZONE_DWELL.value,
                ]),
                events_table.c.timestamp >= window_start,
            )
            .group_by(events_table.c.zone_id)
        ).fetchall()

        unique_visitors = _count_unique_visitors(conn, store_id, window_start)

    zones: list[ZoneHeatmapEntry] = []
    if rows:
        max_visits = max(r.visit_count for r in rows) or 1
        for r in rows:
            zones.append(
                ZoneHeatmapEntry(
                    zone_id=r.zone_id,
                    visit_count=r.visit_count,
                    avg_dwell_ms=round(r.avg_dwell_ms or 0, 1),
                    score=round(r.visit_count / max_visits * 100, 1),
                )
            )
        zones.sort(key=lambda z: z.score, reverse=True)

    data_confidence = (
        "HIGH" if unique_visitors >= MIN_SESSIONS_FOR_HIGH_CONFIDENCE else "LOW"
    )

    return StoreHeatmap(
        store_id=store_id,
        zones=zones,
        data_confidence=data_confidence,
        as_of=as_of,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_start() -> str:
    dt = datetime.now(tz=timezone.utc) - timedelta(hours=METRICS_WINDOW_HOURS)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_unique_visitors(conn, store_id: str, window_start: str) -> int:
    """Count distinct non-staff visitor_ids with an ENTRY or REENTRY event."""
    result = conn.execute(
        select(func.count(func.distinct(events_table.c.visitor_id)))
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.event_type.in_([
                EventType.ENTRY.value,
                EventType.REENTRY.value,
            ]),
            events_table.c.timestamp  >= window_start,
        )
    ).scalar()
    return result or 0


def _count_conversions(conn, store_id: str, window_start: str) -> int:
    result = conn.execute(
        select(func.count(func.distinct(sessions_table.c.visitor_id)))
        .where(
            sessions_table.c.store_id  == store_id,
            sessions_table.c.converted == True,
        )
    ).scalar()
    return result or 0


def _zone_dwell_stats(conn, store_id: str, window_start: str) -> list[ZoneDwellStats]:
    rows = conn.execute(
        select(
            events_table.c.zone_id,
            func.count(func.distinct(events_table.c.visitor_id)).label("visit_count"),
            func.avg(events_table.c.dwell_ms).label("avg_dwell_ms"),
        )
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.zone_id    != None,
            events_table.c.event_type == EventType.ZONE_DWELL.value,
            events_table.c.timestamp  >= window_start,
        )
        .group_by(events_table.c.zone_id)
    ).fetchall()

    return [
        ZoneDwellStats(
            zone_id=r.zone_id,
            visit_count=r.visit_count,
            avg_dwell_ms=round(r.avg_dwell_ms or 0, 1),
        )
        for r in rows
        if r.zone_id
    ]


def _current_queue_depth(conn, store_id: str) -> int:
    """
    Most recent queue_depth from a BILLING_QUEUE_JOIN event in the last 5 minutes.
    Returns 0 if no recent billing events.
    """
    five_min_ago = (
        datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    row = conn.execute(
        select(events_table.c.queue_depth)
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.event_type == EventType.BILLING_QUEUE_JOIN.value,
            events_table.c.timestamp  >= five_min_ago,
            events_table.c.queue_depth != None,
        )
        .order_by(events_table.c.timestamp.desc())
        .limit(1)
    ).fetchone()

    return int(row[0]) if row and row[0] is not None else 0


def _abandonment_rate(conn, store_id: str, window_start: str) -> float:
    """
    abandonment_rate = abandoned_queue sessions / was_in_billing sessions
    """
    billing = conn.execute(
        select(func.count())
        .where(
            sessions_table.c.store_id      == store_id,
            sessions_table.c.was_in_billing == True,
        )
    ).scalar() or 0

    abandoned = conn.execute(
        select(func.count())
        .where(
            sessions_table.c.store_id       == store_id,
            sessions_table.c.abandoned_queue == True,
        )
    ).scalar() or 0

    return abandoned / billing if billing > 0 else 0.0


def _distinct_visitors_with_event(
    conn, store_id: str, window_start: str, event_types: list
) -> int:
    result = conn.execute(
        select(func.count(func.distinct(events_table.c.visitor_id)))
        .where(
            events_table.c.store_id   == store_id,
            events_table.c.is_staff   == False,
            events_table.c.event_type.in_([et.value for et in event_types]),
            events_table.c.timestamp  >= window_start,
        )
    ).scalar()
    return result or 0
