"""
emit.py — Event schema definition and emission for Store Intelligence Pipeline.

Every event emitted by the detection pipeline passes through this module.
This is the single source of truth for the event schema.

The emitter writes JSONL (one JSON object per line) to the configured output path.
It also maintains an in-memory buffer for the last N events so that the
Live Dashboard (Part E) can consume them without re-reading the file.

Schema reference: DESIGN.md §3 — Event Schema
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("emit")


# ─────────────────────────────────────────────────────────────────────────────
# Valid event types
# ─────────────────────────────────────────────────────────────────────────────

class EventType:
    ENTRY                  = "ENTRY"
    EXIT                   = "EXIT"
    ZONE_ENTER             = "ZONE_ENTER"
    ZONE_EXIT              = "ZONE_EXIT"
    ZONE_DWELL             = "ZONE_DWELL"
    BILLING_QUEUE_JOIN     = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON  = "BILLING_QUEUE_ABANDON"
    REENTRY                = "REENTRY"


ALL_EVENT_TYPES = {v for k, v in vars(EventType).items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Schema builder
# ─────────────────────────────────────────────────────────────────────────────

def build_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 1.0,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: int = 0,
) -> dict:
    """
    Construct a fully-compliant event dict.

    Validation rules (enforced here, not deferred to ingestion):
    - event_id must be unique (uuid4)
    - timestamp must be UTC ISO-8601
    - confidence is not clamped — low-confidence events are emitted, not dropped
    - zone_id must be None for ENTRY/EXIT events
    """
    if event_type not in ALL_EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type!r}")

    if event_type in (EventType.ENTRY, EventType.EXIT, EventType.REENTRY):
        zone_id = None   # Zone is not applicable for entry/exit

    if not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be datetime, got {type(timestamp)}")

    ts_str = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "event_id":   str(uuid.uuid4()),
        "store_id":   store_id,
        "camera_id":  camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp":  ts_str,
        "zone_id":    zone_id,
        "dwell_ms":   dwell_ms,
        "is_staff":   is_staff,
        "confidence": round(confidence, 4),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone":    sku_zone,
            "session_seq": session_seq,
        },
    }


def validate_event(event: dict) -> list[str]:
    """
    Returns a list of validation errors. Empty list → valid.
    Used by POST /events/ingest to do partial-success validation.
    """
    errors = []
    required = ["event_id", "store_id", "camera_id", "visitor_id",
                "event_type", "timestamp", "dwell_ms", "is_staff",
                "confidence", "metadata"]
    for field in required:
        if field not in event:
            errors.append(f"Missing required field: {field}")

    if "event_type" in event and event["event_type"] not in ALL_EVENT_TYPES:
        errors.append(f"Invalid event_type: {event['event_type']!r}")

    if "timestamp" in event:
        try:
            datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            errors.append(f"timestamp must be ISO-8601 UTC: {event['timestamp']!r}")

    if "confidence" in event:
        c = event["confidence"]
        if not isinstance(c, (int, float)) or c < 0 or c > 1:
            errors.append(f"confidence must be float in [0, 1], got {c!r}")

    if "metadata" in event and isinstance(event["metadata"], dict):
        if "session_seq" not in event["metadata"]:
            errors.append("metadata.session_seq is required")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Emitter
# ─────────────────────────────────────────────────────────────────────────────

class EventEmitter:
    """
    Writes events to a JSONL file and keeps an in-memory buffer.

    Thread safety: not required — the pipeline is single-threaded per clip.
    If parallel clip processing is added, wrap _write with a threading.Lock.
    """

    BUFFER_MAX = 500  # In-memory ring buffer size

    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(output_path, "a", encoding="utf-8")
        self._buffer: list[dict] = []
        self._total = 0

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __del__(self):
        self.close()

    # ── Convenience emit methods (one per event type) ────────────────────────

    def emit_entry(self, **kw) -> dict:
        return self._emit(EventType.ENTRY, **kw)

    def emit_exit(self, **kw) -> dict:
        return self._emit(EventType.EXIT, **kw)

    def emit_reentry(self, **kw) -> dict:
        return self._emit(EventType.REENTRY, **kw)

    def emit_zone_enter(self, **kw) -> dict:
        return self._emit(EventType.ZONE_ENTER, **kw)

    def emit_zone_exit(self, **kw) -> dict:
        return self._emit(EventType.ZONE_EXIT, **kw)

    def emit_zone_dwell(self, **kw) -> dict:
        return self._emit(EventType.ZONE_DWELL, **kw)

    def emit_billing_queue_join(self, **kw) -> dict:
        return self._emit(EventType.BILLING_QUEUE_JOIN, **kw)

    def emit_billing_queue_abandon(self, **kw) -> dict:
        return self._emit(EventType.BILLING_QUEUE_ABANDON, **kw)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, **kw) -> dict:
        event = build_event(event_type=event_type, **kw)
        self._write(event)
        return event

    def _write(self, event: dict):
        line = json.dumps(event, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

        # In-memory ring buffer
        self._buffer.append(event)
        if len(self._buffer) > self.BUFFER_MAX:
            self._buffer = self._buffer[-self.BUFFER_MAX:]

        self._total += 1
        logger.debug(
            "Emitted %s visitor=%s store=%s",
            event["event_type"], event["visitor_id"], event["store_id"],
        )

    def recent_events(self, n: int = 50) -> list[dict]:
        """Return last n emitted events (for dashboard streaming)."""
        return self._buffer[-n:]

    def total_emitted(self) -> int:
        return self._total
