"""
health.py — GET /health endpoint implementation.

Returns service status, per-store feed freshness, and STALE_FEED warnings.
This is what an on-call engineer checks first — it must be accurate.

STALE_FEED: last event for a store was > 10 minutes ago.
NO_DATA: store_id has never received any events.
"""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func

from .database import get_conn, events_table, check_db, now_utc_str
from .models import HealthResponse, StoreFeedStatus

logger = logging.getLogger("health")

STALE_THRESHOLD_SECONDS = 600  # 10 minutes


def compute_health() -> HealthResponse:
    checked_at = now_utc_str()
    db_ok = check_db()

    stores: list[StoreFeedStatus] = []

    if db_ok:
        try:
            stores = _per_store_status()
        except Exception as exc:
            logger.error("Failed to compute per-store health: %s", exc)
            db_ok = False

    overall_status = "ok" if (db_ok and all(
        s.status in ("OK",) for s in stores
    )) else "degraded"

    return HealthResponse(
        status=overall_status,
        db_ok=db_ok,
        stores=stores,
        checked_at=checked_at,
    )


def _per_store_status() -> list[StoreFeedStatus]:
    now = datetime.now(tz=timezone.utc)
    statuses: list[StoreFeedStatus] = []

    with get_conn() as conn:
        rows = conn.execute(
            select(
                events_table.c.store_id,
                func.max(events_table.c.timestamp).label("last_ts"),
            )
            .group_by(events_table.c.store_id)
        ).fetchall()

    for row in rows:
        store_id = row.store_id
        last_ts_str = row.last_ts

        if last_ts_str is None:
            statuses.append(StoreFeedStatus(
                store_id=store_id,
                last_event_ts=None,
                lag_seconds=None,
                status="NO_DATA",
            ))
            continue

        try:
            last_ts = _parse_ts(last_ts_str)
            lag_s = (now - last_ts).total_seconds()
            status = "STALE_FEED" if lag_s > STALE_THRESHOLD_SECONDS else "OK"
        except Exception:
            lag_s = None
            status = "NO_DATA"

        statuses.append(StoreFeedStatus(
            store_id=store_id,
            last_event_ts=last_ts_str,
            lag_seconds=round(lag_s, 1) if lag_s is not None else None,
            status=status,
        ))

    return statuses


def _parse_ts(ts_str: str) -> datetime:
    if ts_str.endswith("Z"):
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
