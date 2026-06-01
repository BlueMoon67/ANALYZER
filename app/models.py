"""
models.py — Pydantic schemas and SQLAlchemy ORM models for Store Intelligence API.

Every event ingested through POST /events/ingest is validated against StoreEvent.
The ORM layer uses SQLAlchemy Core (not ORM session) for lightweight async-ready access.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Event type enum
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, enum.Enum):
    ENTRY                 = "ENTRY"
    EXIT                  = "EXIT"
    ZONE_ENTER            = "ZONE_ENTER"
    ZONE_EXIT             = "ZONE_EXIT"
    ZONE_DWELL            = "ZONE_DWELL"
    BILLING_QUEUE_JOIN    = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY               = "REENTRY"


# ─────────────────────────────────────────────────────────────────────────────
# Event metadata sub-schema
# ─────────────────────────────────────────────────────────────────────────────

class EventMetadata(BaseModel):
    queue_depth:  Optional[int]   = None
    sku_zone:     Optional[str]   = None
    session_seq:  int             = 0


# ─────────────────────────────────────────────────────────────────────────────
# Core event schema (inbound)
# ─────────────────────────────────────────────────────────────────────────────

class StoreEvent(BaseModel):
    event_id:   str        = Field(..., description="UUID v4 — globally unique")
    store_id:   str        = Field(..., min_length=1)
    camera_id:  str        = Field(..., min_length=1)
    visitor_id: str        = Field(..., min_length=1)
    event_type: EventType
    timestamp:  str        = Field(..., description="ISO-8601 UTC e.g. 2026-03-03T14:22:10Z")
    zone_id:    Optional[str]  = None
    dwell_ms:   int        = Field(default=0, ge=0)
    is_staff:   bool       = False
    confidence: float      = Field(..., ge=0.0, le=1.0)
    metadata:   EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        # Accept both Z and +00:00 suffix
        try:
            if v.endswith("Z"):
                datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            else:
                datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO-8601 UTC, got {v!r}") from exc
        return v

    @model_validator(mode="after")
    def zone_rules(self) -> "StoreEvent":
        if self.event_type in (EventType.ENTRY, EventType.EXIT, EventType.REENTRY):
            self.zone_id = None
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Ingest request / response
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    events: list[StoreEvent] = Field(..., max_length=500)


class EventError(BaseModel):
    event_id: Optional[str] = None
    index:    int
    error:    str


class IngestResponse(BaseModel):
    accepted:  int
    rejected:  int
    duplicate: int
    errors:    list[EventError] = []


# ─────────────────────────────────────────────────────────────────────────────
# Metrics response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ZoneDwellStats(BaseModel):
    zone_id:      str
    visit_count:  int
    avg_dwell_ms: float


class StoreMetrics(BaseModel):
    store_id:          str
    unique_visitors:   int
    conversion_rate:   float           # 0.0–1.0
    avg_dwell_ms:      float
    zone_dwell:        list[ZoneDwellStats]
    queue_depth:       int
    abandonment_rate:  float           # 0.0–1.0
    as_of:             str             # ISO timestamp of computation
    data_confidence:   str = "HIGH"    # HIGH | LOW (< 20 sessions)


class FunnelStage(BaseModel):
    stage:       str
    count:       int
    drop_off_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    stages:   list[FunnelStage]
    as_of:    str


class ZoneHeatmapEntry(BaseModel):
    zone_id:       str
    visit_count:   int
    avg_dwell_ms:  float
    score:         float   # 0–100 normalised


class StoreHeatmap(BaseModel):
    store_id:         str
    zones:            list[ZoneHeatmapEntry]
    data_confidence:  str
    as_of:            str


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnomalySeverity(str, enum.Enum):
    INFO     = "INFO"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"


class AnomalyType(str, enum.Enum):
    BILLING_QUEUE_SPIKE = "BILLING_QUEUE_SPIKE"
    CONVERSION_DROP     = "CONVERSION_DROP"
    DEAD_ZONE           = "DEAD_ZONE"
    STALE_FEED          = "STALE_FEED"


class Anomaly(BaseModel):
    anomaly_type:     AnomalyType
    severity:         AnomalySeverity
    description:      str
    suggested_action: str
    detected_at:      str
    context:          dict[str, Any] = {}


class StoreAnomalies(BaseModel):
    store_id:  str
    anomalies: list[Anomaly]
    as_of:     str


# ─────────────────────────────────────────────────────────────────────────────
# Health schema
# ─────────────────────────────────────────────────────────────────────────────

class StoreFeedStatus(BaseModel):
    store_id:         str
    last_event_ts:    Optional[str]
    lag_seconds:      Optional[float]
    status:           str   # OK | STALE_FEED | NO_DATA


class HealthResponse(BaseModel):
    status:       str   # ok | degraded
    db_ok:        bool
    stores:       list[StoreFeedStatus]
    checked_at:   str
