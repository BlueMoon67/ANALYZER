"""
ingestion.py — Event ingest, deduplication, and session materialisation.

POST /events/ingest calls ingest_events().
Key behaviours:
  - Idempotent by event_id (duplicate → counted, not re-inserted)
  - Partial success: malformed events return errors; valid events are stored
  - Session materialisation: ENTRY/EXIT/REENTRY update visitor_sessions
  - Structured logging: every ingest logs trace_id, store_id, counts, latency_ms
"""

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select, insert, update

from .database import get_conn, events_table, sessions_table, now_utc_str, parse_ts
from .models import StoreEvent, IngestRequest, IngestResponse, EventError, EventType

logger = logging.getLogger("ingestion")


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def ingest_events(request: IngestRequest, trace_id: str) -> IngestResponse:
    """
    Validate, deduplicate, and persist a batch of events.
    Returns a structured response with accepted / rejected / duplicate counts.
    """
    t0 = time.perf_counter()

    accepted = 0
    rejected = 0
    duplicates = 0
    errors: list[EventError] = []

    # Collect all event_ids first for a bulk duplicate check
    incoming_ids = [e.event_id for e in request.events]
    existing_ids = _fetch_existing_ids(incoming_ids)

    rows_to_insert: list[dict] = []
    session_updates: list[StoreEvent] = []

    for idx, event in enumerate(request.events):
        if event.event_id in existing_ids:
            duplicates += 1
            continue

        # Mark as seen immediately to handle dupes within the same batch
        existing_ids.add(event.event_id)

        rows_to_insert.append(_event_to_row(event))
        session_updates.append(event)

    # Bulk insert
    if rows_to_insert:
        try:
            with get_conn() as conn:
                conn.execute(insert(events_table), rows_to_insert)
                conn.commit()
            accepted = len(rows_to_insert)
        except Exception as exc:
            logger.error("Bulk insert failed: %s", exc, extra={"trace_id": trace_id})
            # Fall back to row-by-row insert to isolate failures
            accepted, row_errors = _insert_row_by_row(rows_to_insert, trace_id)
            rejected += len(row_errors)
            errors.extend(row_errors)

    # Update session summaries
    if session_updates:
        _update_sessions(session_updates)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Ingest complete",
        extra={
            "trace_id":    trace_id,
            "store_ids":   list({e.store_id for e in request.events}),
            "accepted":    accepted,
            "rejected":    rejected,
            "duplicate":   duplicates,
            "latency_ms":  latency_ms,
        },
    )

    return IngestResponse(
        accepted=accepted,
        rejected=rejected,
        duplicate=duplicates,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_existing_ids(event_ids: list[str]) -> set[str]:
    if not event_ids:
        return set()
    with get_conn() as conn:
        stmt = select(events_table.c.event_id).where(
            events_table.c.event_id.in_(event_ids)
        )
        result = conn.execute(stmt)
        return {row[0] for row in result}


def _event_to_row(event: StoreEvent) -> dict:
    return {
        "event_id":    event.event_id,
        "store_id":    event.store_id,
        "camera_id":   event.camera_id,
        "visitor_id":  event.visitor_id,
        "event_type":  event.event_type.value,
        "timestamp":   event.timestamp,
        "zone_id":     event.zone_id,
        "dwell_ms":    event.dwell_ms,
        "is_staff":    event.is_staff,
        "confidence":  event.confidence,
        "queue_depth": event.metadata.queue_depth,
        "sku_zone":    event.metadata.sku_zone,
        "session_seq": event.metadata.session_seq,
        "ingested_at": now_utc_str(),
    }


def _insert_row_by_row(
    rows: list[dict], trace_id: str
) -> tuple[int, list[EventError]]:
    accepted = 0
    errors: list[EventError] = []
    with get_conn() as conn:
        for row in rows:
            try:
                conn.execute(insert(events_table).values(**row))
                accepted += 1
            except Exception as exc:
                errors.append(
                    EventError(
                        event_id=row.get("event_id"),
                        index=-1,
                        error=str(exc),
                    )
                )
        conn.commit()
    return accepted, errors


def _update_sessions(events: list[StoreEvent]):
    """
    Materialise visitor_sessions from ENTRY / EXIT / REENTRY / BILLING events.

    Session lifecycle:
    - ENTRY / REENTRY  → upsert row with entry_ts
    - EXIT             → set exit_ts on existing session
    - BILLING_QUEUE_JOIN    → set was_in_billing=True
    - BILLING_QUEUE_ABANDON → set abandoned_queue=True
    """
    with get_conn() as conn:
        for event in events:
            et = event.event_type

            if et in (EventType.ENTRY, EventType.REENTRY):
                # Check if session row exists
                existing = conn.execute(
                    select(sessions_table.c.id).where(
                        sessions_table.c.store_id   == event.store_id,
                        sessions_table.c.visitor_id == event.visitor_id,
                    )
                ).fetchone()

                if existing is None:
                    conn.execute(
                        insert(sessions_table).values(
                            store_id=event.store_id,
                            visitor_id=event.visitor_id,
                            entry_ts=event.timestamp,
                            exit_ts=None,
                            converted=False,
                            was_in_billing=False,
                            abandoned_queue=False,
                            is_reentry=(et == EventType.REENTRY),
                        )
                    )
                else:
                    # Re-entry: update entry_ts and flag
                    if et == EventType.REENTRY:
                        conn.execute(
                            update(sessions_table)
                            .where(sessions_table.c.id == existing[0])
                            .values(entry_ts=event.timestamp, is_reentry=True)
                        )

            elif et == EventType.EXIT:
                conn.execute(
                    update(sessions_table)
                    .where(
                        sessions_table.c.store_id   == event.store_id,
                        sessions_table.c.visitor_id == event.visitor_id,
                    )
                    .values(exit_ts=event.timestamp)
                )

            elif et == EventType.BILLING_QUEUE_JOIN:
                conn.execute(
                    update(sessions_table)
                    .where(
                        sessions_table.c.store_id   == event.store_id,
                        sessions_table.c.visitor_id == event.visitor_id,
                    )
                    .values(was_in_billing=True)
                )

            elif et == EventType.BILLING_QUEUE_ABANDON:
                conn.execute(
                    update(sessions_table)
                    .where(
                        sessions_table.c.store_id   == event.store_id,
                        sessions_table.c.visitor_id == event.visitor_id,
                    )
                    .values(abandoned_queue=True)
                )

        conn.commit()


def mark_visitor_converted(store_id: str, visitor_id: str):
    """Called by POS correlation to mark a session as converted."""
    with get_conn() as conn:
        conn.execute(
            update(sessions_table)
            .where(
                sessions_table.c.store_id   == store_id,
                sessions_table.c.visitor_id == visitor_id,
            )
            .values(converted=True)
        )
        conn.commit()
