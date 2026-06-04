# DESIGN.md — Store Intelligence Platform

## Overview

Store Intelligence is an edge-ready retail analytics platform that fuses real-time computer vision (YOLOv8n + ByteTrack) with transactional POS data to produce actionable store metrics. This document describes the system architecture, data flows, and the AI-assisted engineering decisions made during development.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Edge Node (Host)                      │
│                                                              │
│   CCTV / RTSP Feeds                                          │
│         │                                                    │
│         ▼                                                    │
│   pipeline/detect.py                                         │
│   ┌─────────────────────────────────────────────┐           │
│   │  YOLOv8n  →  ByteTrack  →  Re-ID            │           │
│   │  zone_classifier.py (point-in-polygon)       │           │
│   │  staff_classifier.py  (colour heuristic)     │           │
│   │  emit.py  →  JSONL event files               │           │
│   └─────────────────────────────────────────────┘           │
│         │                                                    │
│         ▼                                                    │
│   pipeline/feed.py  (batch or stream mode)                   │
│         │                                                    │
└─────────┼───────────────────────────────────────────────────┘
          │  POST /events/ingest
          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Docker Container                            │
│                                                              │
│   FastAPI  (app/main.py)                                     │
│   ├── /events/ingest      → ingestion.py                    │
│   ├── /stores/{id}/metrics  → metrics.py                    │
│   ├── /stores/{id}/funnel   → metrics.py                    │
│   ├── /stores/{id}/heatmap  → metrics.py                    │
│   ├── /stores/{id}/anomalies → anomalies.py                 │
│   ├── /cameras/*           → cameras.py                     │
│   ├── /health              → health.py                      │
│   └── scheduler.py  (background: camera poll + dispatch)    │
│                                                              │
│   SQLAlchemy ORM  →  SQLite (dev) / PostgreSQL (prod)        │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│   store_dashboard.html  (browser, no build step)            │
│   Chart.js + HLS.js + WebRTC WHEP                           │
│   Polls metrics / funnel / anomaly endpoints every 30s      │
│   Live RTSP via MediaMTX  (WebRTC → HLS → iframe cascade)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### `pipeline/detect.py`
Entry point for the vision pipeline. Accepts a single `--clip` or a `--store` + `--clips-dir` batch. Opens each source with OpenCV, runs YOLOv8n inference per frame, feeds detections into the ByteTrack multi-object tracker, resolves zone membership via `zone_classifier.py`, filters staff via `staff_classifier.py`, and emits structured events through `emit.py`.

### `pipeline/tracker.py`
Implements multi-camera Re-ID using an appearance-feature cosine similarity threshold (0.6). Maintains a cross-camera identity map so that a visitor who leaves CAM_ENTRY_01 and appears on CAM_FLOOR_01 is counted only once. Re-entry detection emits a `REENTRY` event when a previously exited visitor ID reappears within a configurable grace window (default 30 minutes).

### `pipeline/emit.py`
Defines the canonical event schema and writes JSONL. Every event carries: `event_id` (UUID), `store_id`, `camera_id`, `visitor_id`, `event_type`, `timestamp` (ISO-8601 UTC), `zone_id`, `dwell_ms`, `confidence`, `queue_depth`, `is_staff`. The `event_id` is used for idempotent deduplication in the API.

### `pipeline/staff_classifier.py`
Primary signal: dominant colour histogram of the upper-body bounding box compared against a per-store configurable HSV range for staff uniforms. Confidence score is passed through; if below threshold and `USE_VLM_STAFF_DETECTION=1`, falls back to a VLM prompt describing the torso region.

### `pipeline/zone_classifier.py`
Point-in-polygon lookup using `store_layout.json`. Each zone is a named polygon in pixel space relative to the camera frame. Supports overlapping zones with priority ordering so a visitor at the BILLING counter is preferentially assigned to `BILLING` over `FLOOR`.

### `pipeline/feed.py`
Batch mode: reads a completed `.jsonl` file and POSTs batches of up to 500 events to `/events/ingest`. Stream mode (`--stream`): tails the file as it is written and replays events with frame-accurate timing delays, enabling the live dashboard to reflect detection in near real-time.

### `pipeline/pos_correlate.py`
Reads a CSV of POS transactions (`timestamp`, `store_id`, `amount`). For each transaction, finds all visitor sessions that were in the `BILLING` zone within the preceding 5-minute window and marks them as converted. This bridges CCTV-derived sessions to actual purchase data.

### `app/ingestion.py`
Receives event batches, deduplicates by `event_id`, and materialises visitor sessions. A session is a (store_id, visitor_id) pair; `ENTRY` opens it, `EXIT` closes it. Zone dwell times are accumulated per zone per session. Converted sessions are flagged by `pos_correlate.py` or inferred when a visitor reaches the `BILLING` zone.

### `app/metrics.py`
Computes on-demand aggregates from the session and event tables:
- **unique_visitors**: count of distinct visitor sessions in the last 24 h, excluding staff.
- **conversion_rate**: converted sessions / total sessions. `data_confidence` is `HIGH` when ≥ 30 sessions exist, `LOW` otherwise.
- **avg_dwell_ms**: mean of total session dwell time across all visitors.
- **queue_depth**: count of active `BILLING_QUEUE_JOIN` minus `BILLING_QUEUE_ABANDON` events in the last 15 minutes.
- **funnel stages**: Entry → Zone Visit → Billing Queue → Purchase, with drop-off percentages between each stage.
- **heatmap**: per-zone visit frequency and average dwell, normalised 0–100 relative to the busiest zone.

### `app/anomalies.py`
Runs four detectors on each API call:
1. **Queue spike** — queue depth exceeds 2× the 7-day rolling average for that hour.
2. **Conversion drop** — current hour conversion rate is more than 20 pp below the rolling 7-day baseline.
3. **Dead zone** — a zone that normally receives visits has had zero traffic for > 2 hours during open hours.
4. **Stale feed** — no events ingested for a camera in > 5 minutes.

Each anomaly carries a severity (`CRITICAL` / `WARN` / `INFO`) and a `suggested_action` string.

### `app/scheduler.py`
Two daemon threads: `fetch_cameras` polls `GET /cameras` every 10 seconds and refreshes the active-camera cache. `scheduler_loop` iterates the cache and dispatches each camera to `POST /cameras/detect/camera` every 0.2 seconds. A per-camera guard set (`_active_cameras`) prevents duplicate stream sessions.

### `store_dashboard.html`
Zero-dependency single-file dashboard. Uses Syne (display) and IBM Plex Mono (data) from Google Fonts. Chart.js for bar charts. HLS.js + native WebRTC WHEP for live stream playback. RTSP feeds are re-exposed by MediaMTX; the dashboard cascades: WebRTC WHEP → HLS → iframe player page. Design tokens are dark-first (`--bg: #07080c`) with semantic accent colours: blue for primary data, green for positive metrics, amber for warnings, red for critical anomalies.

---

## Data Flow: End-to-End

```
Video clip / RTSP stream
    → YOLOv8n bounding boxes (per frame)
    → ByteTrack track IDs (persistent across frames)
    → Re-ID (cross-camera identity resolution)
    → Zone assignment (point-in-polygon)
    → Staff filter (colour heuristic)
    → JSONL event emission (one event per state change)
    → POST /events/ingest (batched, idempotent)
    → Session materialisation (SQLAlchemy, SQLite/PostgreSQL)
    → Metric aggregation (on-demand SQL queries)
    → Anomaly detection (threshold + rolling baseline)
    → Dashboard polling (every 30 s or triggered)
    → POS correlation (offline, per transaction CSV)
```

---

## Database Schema

```sql
-- Immutable event log (append-only)
CREATE TABLE events (
    event_id    TEXT PRIMARY KEY,
    store_id    TEXT NOT NULL,
    camera_id   TEXT NOT NULL,
    visitor_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,        -- ENTRY, EXIT, ZONE_ENTER, ...
    timestamp   DATETIME NOT NULL,
    zone_id     TEXT,
    dwell_ms    INTEGER DEFAULT 0,
    confidence  REAL DEFAULT 1.0,
    queue_depth INTEGER DEFAULT 0,
    is_staff    INTEGER DEFAULT 0
);

-- Materialised visitor sessions
CREATE TABLE sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id        TEXT NOT NULL,
    visitor_id      TEXT NOT NULL,
    entry_time      DATETIME,
    exit_time       DATETIME,
    total_dwell_ms  INTEGER DEFAULT 0,
    converted       INTEGER DEFAULT 0,
    is_staff        INTEGER DEFAULT 0
);

-- Zone dwell summary per session
CREATE TABLE zone_dwells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id    TEXT NOT NULL,
    visitor_id  TEXT NOT NULL,
    zone_id     TEXT NOT NULL,
    visit_count INTEGER DEFAULT 0,
    total_ms    INTEGER DEFAULT 0
);

-- Camera registry
CREATE TABLE cameras (
    camera_id    TEXT NOT NULL,
    store_id     TEXT NOT NULL,
    rtsp_url     TEXT,
    camera_type  TEXT DEFAULT 'floor',
    label        TEXT,
    is_active    INTEGER DEFAULT 0,
    registered_at DATETIME,
    PRIMARY KEY (store_id, camera_id)
);
```

---

## AI-Assisted Decisions

### 1. Choosing YOLOv8n over heavier variants

**Context**: The pipeline must run on edge hardware (a single-board GPU or a modest workstation) without a dedicated inference server, and still process 15–30 fps CCTV footage in near real-time.

**AI consultation**: Claude was asked to compare detection accuracy vs. inference speed trade-offs for the YOLOv8 model family (n / s / m / l / x) on CPU and entry-level GPU targets (RTX 3060, Jetson Orin Nano).

**Decision**: YOLOv8n at 640 px input. On an RTX 3060 it sustains ~120 fps, leaving headroom for ByteTrack and zone classification. The accuracy gap vs. YOLOv8s for pedestrian detection in a retail setting (well-lit, mostly upright figures) is negligible at distances < 5 m. If accuracy is later insufficient, the weights path is a single config flag away from swapping to `yolov8s.pt`.

**AI contribution**: Claude generated a structured comparison table of mAP@0.5 vs. FPS across the family, and suggested the "detect then confirm with a larger model on uncertain crops" pattern as a future upgrade path. It also flagged the ByteTrack dependency on a minimum detection confidence of 0.25 to avoid track proliferation on false positives.

---

### 2. ByteTrack for multi-object tracking instead of SORT or DeepSORT

**Context**: Re-ID accuracy depends on the tracker's ability to maintain consistent IDs through occlusion (e.g. a visitor blocked by a shelf) and brief exits from the camera frame.

**AI consultation**: Claude was asked to compare SORT, DeepSORT, and ByteTrack for retail CCTV specifically, where density is low (< 20 people simultaneously) but occlusion from shelving is frequent.

**Decision**: ByteTrack. Unlike SORT, it uses low-confidence detections in a second association pass specifically to recover occluded tracks, which directly addresses the shelving-occlusion problem. Unlike DeepSORT, it does not require a separately trained appearance model, keeping the dependency footprint small. The Re-ID appearance features for cross-camera matching are computed independently in `tracker.py` using ResNet50 embeddings cropped to the upper-body box.

**AI contribution**: Claude outlined the two-pass association algorithm and helped write the `tracker.py` cosine-similarity Re-ID logic, including the choice of a 0.6 similarity threshold as the practical sweet spot between false merges and false splits, based on published benchmarks.

---

### 3. Staff exclusion via colour histogram instead of a dedicated classifier

**Context**: Staff wear identifiable uniforms. A fully supervised classifier would require labelled training data that changes per store. The pipeline must work out-of-the-box for any store with a brief HSV range configuration.

**AI consultation**: Claude was asked to compare: (a) a dedicated uniform classifier (ResNet fine-tune), (b) a zero-shot VLM prompt, and (c) HSV colour histogram on the upper-body bounding box.

**Decision**: HSV histogram as the primary signal, with an optional VLM fallback (`USE_VLM_STAFF_DETECTION=1`). The histogram approach is zero-training, deterministic, fast (< 1 ms per crop), and surprisingly robust when the uniform colour is distinctive (most retail chains have a single dominant colour). The VLM fallback handles edge cases (part-time staff in civilian clothes, high-vis vests).

**AI contribution**: Claude suggested the upper-body crop ratio (top 40% of bounding box height) to focus on torso/shirt colour and exclude floor reflections, and proposed computing the histogram only on the saturation-weighted pixels above an HSV saturation threshold of 40 to avoid counting grey/black neutrals as a colour signal.

---

### 4. Idempotent ingestion via `event_id` UUID

**Context**: The pipeline feed script may crash and restart mid-file, or the same JSONL file may be ingested twice if an operator re-runs the feed command. Duplicate events would inflate all metrics.

**AI consultation**: Claude was asked for the simplest production-safe deduplication strategy for an append-oriented event log.

**Decision**: Each event carries a deterministic UUID v5 derived from `(store_id, camera_id, visitor_id, event_type, timestamp_ms)`. The API uses `INSERT OR IGNORE` (SQLite) / `INSERT … ON CONFLICT DO NOTHING` (PostgreSQL). The batch endpoint accepts up to 500 events and returns per-event `inserted` / `skipped` counts.

**AI contribution**: Claude recommended UUID v5 (namespace + deterministic hash) over UUID v4 (random) specifically because it allows the pipeline to re-emit the same logical event with the same ID, making feed retries safe without any server-side seen-ID cache.

---

### 5. On-demand SQL aggregation vs. pre-materialised metrics

**Context**: Metrics (unique visitors, conversion rate, funnel, heatmap) could be computed either continuously via a background worker or on-demand per API call.

**AI consultation**: Claude was asked to evaluate the trade-offs for a store that ingests ~5 000 events/day with a 30-second dashboard refresh interval.

**Decision**: On-demand SQL aggregation with a 10-second in-process cache. At 5 000 events/day, the event table has ~150 000 rows after 30 days — trivially fast for indexed SQLite queries (< 5 ms). Pre-materialisation adds operational complexity (background task failure modes, staleness windows) with no meaningful latency benefit at this scale. The cache prevents query storms during a rapid dashboard refresh.

**AI contribution**: Claude provided query plans for the key aggregations and recommended adding composite indexes on `(store_id, timestamp)` and `(store_id, visitor_id)` as the two hot paths. It also flagged the `STALE_FEED` health check as a proxy for detecting pipeline crashes, replacing the need for a separate heartbeat mechanism.

---

### 6. Single-file HTML dashboard (no build step)

**Context**: The dashboard must be openable by a store manager with zero toolchain setup — no Node, no npm, no build.

**AI consultation**: Claude was asked to evaluate React + Vite, vanilla HTML with CDN imports, and Streamlit for the live dashboard component.

**Decision**: Vanilla HTML with CDN-hosted Chart.js and HLS.js. Streamlit was rejected because it requires a running Python process and introduces a second server. React was rejected because even a CDN-only React app requires understanding JSX or a compiled build. The HTML file is self-contained, loads in any browser, and the entire UI state is managed with plain DOM manipulation and `fetch()`.

**AI contribution**: Claude designed the CSS custom-property token system (`--bg`, `--blue`, `--green`, `--amber`, `--red`) that unifies the colour language across KPI cards, anomaly badges, funnel bars, and the heatmap, and generated the WebRTC WHEP → HLS → iframe cascade logic for RTSP stream playback without a browser plugin.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | SQLAlchemy DSN; swap to `postgresql://...` for production |
| `LOG_LEVEL` | `INFO` | Python logging verbosity |
| `USE_VLM_STAFF_DETECTION` | `0` | Set to `1` to enable VLM fallback in staff classifier |

---

## Production Upgrade Path

- **Database**: swap `DATABASE_URL` to PostgreSQL; no code changes required.
- **Detection accuracy**: swap `yolov8n.pt` to `yolov8s.pt` or `yolov8m.pt` via `--weights` flag.
- **Staff detection**: set `USE_VLM_STAFF_DETECTION=1` and configure a VLM endpoint.
- **Horizontal scaling**: the FastAPI app is stateless (all state in DB); multiple replicas behind a load balancer are safe with PostgreSQL.
- **Metrics pre-materialisation**: add a Celery task or APScheduler job calling `metrics.py` aggregations on a 1-minute cron and caching results in Redis, if query latency becomes a concern at > 10 M events.
- <img width="50" height="150" alt="store_intelligence_full_workflow" src="https://github.com/user-attachments/assets/d64e7b61-ed10-472d-a4e0-d25af2a6b627" />
