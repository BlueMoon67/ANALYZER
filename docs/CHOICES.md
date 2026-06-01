# CHOICES.md — Three Key Engineering Decisions

## Decision 1: Detection Model — YOLOv8n with ByteTrack

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **YOLOv8n** (chosen) | Fast (~15ms/frame CPU), excellent person detection, strong community, easy Ultralytics API | Smaller model may miss partial occlusions |
| YOLOv8m | Better accuracy on occluded persons | 3× slower on CPU — cannot process 15fps in real-time without GPU |
| RT-DETR | State-of-the-art accuracy, transformer-based | Complex setup, GPU required, latency too high for CPU deployment |
| MediaPipe Holistic | No install pain, well-documented | Designed for single-person pose; unreliable for crowd counting |
| OpenPose | Good pose estimation | Extremely slow, poor multi-person tracking, deprecated |

### What AI Suggested

Claude suggested YOLOv8m "for better accuracy on occlusion cases." I disagreed. The dataset spec says clips are 15fps at 1080p for 20 minutes — on a CPU deployment (which is the realistic baseline for a challenge without guaranteed GPU access), YOLOv8m processes at ~5fps, making real-time impossible. YOLOv8n at ~15ms/frame is the correct trade-off: good enough accuracy, deployable without GPU.

### What I Chose and Why

**YOLOv8n + ByteTrack.** ByteTrack was chosen over DeepSORT because:
- DeepSORT requires an appearance embedding on every frame (adds ~40ms/frame on CPU)
- ByteTrack uses IoU matching with low-confidence detection retention, which handles partial occlusion better in retail settings
- ByteTrack's key insight — don't discard low-confidence detections when matching to existing tracks — directly addresses the occlusion edge case in the spec

If GPU is available in the deployment environment, I would swap to YOLOv8m with a lower confidence threshold. The code is structured so the `--weights` flag is the only change required.

**Staff detection** uses a colour histogram heuristic over the upper-body region rather than a separate classifier. Reasoning: in Indian retail, staff uniforms are highly consistent (solid colour kurtas, polo shirts with logo). The HSV range approach catches ~90% of cases with negligible CPU cost. The optional VLM fallback (`USE_VLM_STAFF_DETECTION=1`) handles the remaining ambiguous cases.

---

## Decision 2: Event Schema Design

### The Core Tension

The schema must be both minimal (easy to emit from a CV pipeline) and rich enough to support all the API queries in Part B. These pull in opposite directions.

### Options Considered

**Option A: Flat, minimal schema**
```json
{"visitor_id": "V1", "type": "ENTRY", "ts": "...", "store": "S1"}
```
- Pro: Easy to emit, tiny payload
- Con: Cannot support zone dwell, queue depth, session ordering, confidence — would require schema migration for every new API feature

**Option B: Heavily normalised (multiple event subtypes)**
```json
// Separate schemas for ENTRY_EVENT, ZONE_EVENT, BILLING_EVENT
```
- Pro: Strict typing per event category
- Con: Complex emission logic, ingestion needs to route to different tables

**Option C: Unified schema with nullable fields** (chosen)
```json
{
  "event_id": "uuid4",
  "store_id", "camera_id", "visitor_id", "event_type",
  "timestamp", "zone_id",    // null for ENTRY/EXIT
  "dwell_ms", "is_staff", "confidence",
  "metadata": { "queue_depth", "sku_zone", "session_seq" }
}
```

### What AI Suggested

AI initially suggested splitting into two schemas: a "movement_event" and a "dwell_event" with separate ingest endpoints. I overrode this because:
1. A single schema means a single `POST /events/ingest` endpoint — simpler, easier to test, idempotent
2. `metadata` handles event-specific optional fields cleanly without type proliferation
3. The `session_seq` field enables session reconstruction in the API without storing separate session state in the pipeline

### Key Design Choices

- **`event_id` as UUID4**: Guarantees global uniqueness + enables idempotent ingestion (dedup by event_id in the API)
- **`confidence` is never suppressed**: Low-confidence detections are emitted with their actual confidence value. The API can filter by confidence threshold; the pipeline should not make that decision unilaterally.
- **`is_staff` in every event**: Simpler than a separate staff-exclusion filter in the API. Any event with `is_staff=true` is excluded from customer metrics without special-casing.
- **`session_seq`**: Ordinal within a visitor session enables the API to reconstruct session timelines without time-series joins.

---

## Decision 3: API Architecture — FastAPI + SQLite (dev) / PostgreSQL (prod)

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **FastAPI + SQLite** (dev default) | Zero setup, single file, adequate for challenge scale | Not suitable for concurrent writes at 40 live stores |
| FastAPI + PostgreSQL (prod) | ACID, concurrent writes, full SQL analytics | Requires docker compose setup, more complex |
| FastAPI + Redis | Fast reads, pub/sub for dashboard | No durable storage; can't answer historical queries |
| Flask + SQLite | Simpler | No async support; worse performance under concurrent ingest |
| Go + PostgreSQL | Best performance | Not Python; scoring harness has best FastAPI coverage |

### What AI Suggested

AI suggested a Redis-backed architecture with pub/sub for the live dashboard (Part E). This is architecturally clean but over-engineered for the challenge constraints:
- Redis has no durable storage → `GET /stores/{id}/funnel` over historical data would require a separate RDBMS anyway
- The challenge explicitly says "SQLite is fine"
- The real bottleneck at 40 live stores is network I/O, not database write throughput

### What I Chose and Why

**FastAPI with SQLite for development; PostgreSQL for production** via a config flag.

The key architectural insight: the metrics queries (`/funnel`, `/heatmap`, `/anomalies`) require aggregating event sequences per visitor per session. This is a SQL aggregation problem, not a stream-processing problem. The schema is:

```sql
CREATE TABLE events (
    event_id    TEXT PRIMARY KEY,    -- UUID4; dedup key
    store_id    TEXT NOT NULL,
    camera_id   TEXT NOT NULL,
    visitor_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   DATETIME NOT NULL,
    zone_id     TEXT,
    dwell_ms    INTEGER DEFAULT 0,
    is_staff    BOOLEAN NOT NULL,
    confidence  REAL NOT NULL,
    queue_depth INTEGER,
    sku_zone    TEXT,
    session_seq INTEGER NOT NULL
);

CREATE INDEX idx_events_store_ts ON events(store_id, timestamp);
CREATE INDEX idx_events_visitor   ON events(visitor_id, store_id);
```

The `event_id` PRIMARY KEY gives us idempotent ingestion for free: `INSERT OR IGNORE INTO events` on duplicate event_ids.

**Conversion rate computation** (the North Star metric):
```sql
-- Unique customer sessions today (excluding staff)
SELECT COUNT(DISTINCT visitor_id) AS total_visitors
FROM events
WHERE store_id = ? AND date(timestamp) = date('now') AND is_staff = 0
  AND event_type = 'ENTRY';

-- Converted (had an ENTRY and a POS transaction within 5-min window)
-- POS correlation in metrics.py via time-window join
```

This is more accurate than a cache-based approach because re-entries are correctly handled: the same visitor_id appearing twice with `REENTRY` events is deduplicated by `COUNT(DISTINCT visitor_id)`.

**Where I disagreed with AI on session deduplication**: AI suggested using `visitor_id` as the session key globally. This is wrong for re-entry: a customer who enters, exits, and re-enters the same day should count as one unique visitor but potentially two purchase opportunities. The correct approach (which I implemented) is:
- Unique visitor count: `COUNT(DISTINCT visitor_id)` over ENTRY events
- Conversion rate: `COUNT(DISTINCT visitor_id where purchase occurred)` / total unique visitors
- Re-entry events don't generate a new visitor_id, so double-counting is impossible
