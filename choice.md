# CHOICES.md — Engineering Decisions

This document records the key decisions made during the design and implementation of Store Intelligence, covering model selection, event schema design, and API architecture. Each entry explains what was chosen, what the realistic alternatives were, and why the chosen approach best fits the project constraints.

---

## 1. Model Selection

### 1.1 Object Detector — YOLOv8n

**Choice:** YOLOv8n (Ultralytics)

**Alternatives considered:**

| Model | Pros | Cons |
|---|---|---|
| YOLOv8n | Fast, edge-friendly, easy Python API, ONNX export | Lower accuracy than larger variants |
| YOLOv8s / YOLOv8m | Better mAP | Higher VRAM and latency — not edge-safe |
| RT-DETR | State-of-the-art accuracy | Much heavier; slow on CPU |
| MobileNet SSD | Very light | Inferior accuracy; abandonware ecosystem |

**Rationale:** Retail edge cameras are typically connected to NUCs or mid-range workstations, not data-centre GPUs. YOLOv8n runs at ≥ 30 fps on a single NVIDIA RTX 3060 and ≥ 10 fps on CPU, which is acceptable for 1-second event granularity. Ultralytics' Python API also simplifies the integration path considerably. Accuracy at the `n` scale is sufficient for counting and zone-classification tasks where precise keypoint localisation is not required.

**Trade-off accepted:** For crowded scenes or very wide-angle lenses, small detections may be missed. A `YOLOv8s` upgrade is a one-line config change if accuracy needs to improve.

---

### 1.2 Multi-Object Tracker — ByteTrack

**Choice:** ByteTrack (integrated via Ultralytics tracker API)

**Alternatives considered:**

| Tracker | Pros | Cons |
|---|---|---|
| ByteTrack | Strong low-confidence recovery; no Re-ID model required | Relies on IoU; struggles with identical-looking people in crowded scenes |
| DeepSORT | Appearance features improve ID continuity | Requires a separate Re-ID model; higher latency |
| StrongSORT | More accurate than DeepSORT | Even heavier; overkill for coarse dwell/zone tasks |
| SORT (original) | Ultra-lightweight | Poor track continuity when detections drop briefly |

**Rationale:** ByteTrack's two-stage matching (high-confidence first, then low-confidence rescue) dramatically reduces ID switches caused by brief occlusions — common in retail aisles. It needs no additional neural Re-ID model, keeping the pipeline dependency footprint small. For the zone-classification use case, short ID switches are acceptable as long as zone-entry and zone-exit events are correctly paired; ByteTrack's continuity is sufficient for this.

---

### 1.3 Cross-Camera Re-ID — Colour Histogram Similarity

**Choice:** HSV colour histogram cosine similarity (custom implementation in `pipeline/tracker.py`)

**Alternatives considered:**

| Approach | Pros | Cons |
|---|---|---|
| Colour histogram (chosen) | Zero training data; fast; interpretable | Fails when multiple people wear similar colours |
| OSNet / ResNet Re-ID | High accuracy | Requires labelled training data or fine-tuning; adds heavy dependency |
| VLM description matching | Flexible; zero-shot | High latency per person; API cost if cloud-based |

**Rationale:** No labelled Re-ID dataset for the target store was available at challenge time. A colour histogram baseline is interpretable, tunable, and fast. It is sufficient for the typical retail scenario where the same person re-appears in an adjacent camera zone within seconds. The `USE_VLM_STAFF_DETECTION` flag pattern demonstrates how a stronger model can be plugged in without changing the pipeline interface.

---

### 1.4 Staff Classifier — HSV Pixel Fraction Heuristic

**Choice:** HSV range + pixel-fraction threshold (zero training data required)

**Alternatives considered:**

| Approach | Pros | Cons |
|---|---|---|
| HSV heuristic (chosen) | No labelled data; fast; store-specific tunable | Brittle to lighting changes; fails with dark/generic uniforms |
| Fine-tuned binary classifier | Accurate and robust | Requires annotated images of staff |
| VLM (e.g., GPT-4o) | Zero-shot; flexible description | Cloud dependency; per-image cost; 100–500 ms latency |

**Rationale:** Staff uniforms in retail are typically a single distinctive colour (red apron, blue shirt). The HSV heuristic covers the majority of cases with no data collection cost. The optional VLM fallback (`USE_VLM_STAFF_DETECTION=1`) is available for edge cases and can be enabled per-store without touching core pipeline code.

---

## 2. Schema Design

### 2.1 Event Schema — Flat JSONL

**Choice:** One JSON object per line; self-contained with `event_id`, `store_id`, `camera_id`, `person_id`, `event_type`, `zone`, `timestamp`, `is_staff`, `metadata`.

**Alternatives considered:**

| Approach | Pros | Cons |
|---|---|---|
| Flat JSONL (chosen) | Streamable; append-only; trivial to validate | Repeated fields increase file size slightly |
| Nested JSON (per-person arrays) | Compact per person | Not streamable; requires full file load to process |
| CSV | Tiny files | No native support for nested `metadata`; fragile for optional fields |
| Protobuf / Avro | Compact binary | Complex tooling; not human-readable for debugging |

**Rationale:** JSONL is the natural format for an event stream. Each line is independently parseable, making it safe to tail or stream a file as it is being written by `detect.py`. The flat structure keeps the schema obvious and validation straightforward (one `jsonschema` check per line).

---

### 2.2 `event_id` as UUID4 Idempotency Key

**Choice:** Each event carries a client-generated UUID4 `event_id`. The ingest endpoint uses `INSERT OR IGNORE` (SQLite) / `ON CONFLICT DO NOTHING` (PostgreSQL) on this column.

**Rationale:** Re-running `feed.py` (e.g., after a network failure) must not inflate visitor counts. A client-side UUID4 generated at emit time means the key is stable across retries without any server-side state or sequence coordination. The UUID is generated once in `pipeline/emit.py` and written into the JSONL file, so replaying the same file always produces the same set of event IDs.

---

### 2.3 `is_staff` Flag on Every Event (Not a Separate Table)

**Choice:** Include `is_staff: bool` on the event itself rather than maintaining a staff-ID registry.

**Rationale:** Staff `person_id`s are not stable across days (Re-ID resets between sessions), so a long-lived staff registry would be unreliable. Embedding the flag at detection time is authoritative and avoids a join. All metrics queries filter with `WHERE is_staff = false`, which is index-friendly.

---

### 2.4 Session Materialisation at Ingest Time

**Choice:** `app/ingestion.py` creates or updates a session record for each `(store_id, person_id)` pair when events are ingested.

**Alternatives considered:**

| Approach | Pros | Cons |
|---|---|---|
| Eager session (chosen) | O(sessions) reads; low dashboard latency | Slightly higher write complexity |
| Lazy (compute on query) | Simpler writes | O(raw events) per query; slow dashboards at scale |
| Materialised views (PostgreSQL) | Best of both | Adds DB-specific logic; SQLite incompatible |

**Rationale:** The dashboard polls metrics every few seconds. Recomputing sessions over raw events on every poll would be unacceptably slow once event volume grows. Eager materialisation keeps all read endpoints fast. The added write complexity is isolated to `app/ingestion.py` and is well-covered by integration tests.

---

## 3. API Architecture Decisions

### 3.1 FastAPI + SQLAlchemy (SQLite / PostgreSQL)

**Choice:** FastAPI for the HTTP layer; SQLAlchemy Core for DB access; SQLite for dev and test, PostgreSQL for production.

**Alternatives considered:**

| Stack | Pros | Cons |
|---|---|---|
| FastAPI + SQLAlchemy (chosen) | Async-capable; Pydantic validation; dual-DB support | ORM overhead vs. raw SQL |
| Flask + SQLAlchemy | Simpler | Synchronous; lower throughput |
| FastAPI + raw asyncpg | Maximum PostgreSQL performance | PostgreSQL-only; no SQLite for dev/test |
| Django REST Framework | Batteries-included | Too heavyweight; ORM tightly coupled to Django |

**Rationale:** FastAPI's native Pydantic integration validates ingest payloads for free. SQLAlchemy's abstraction means the test suite runs against an in-memory SQLite DB without mocking — this is essential for reliable integration tests in CI environments without a database server.

---

### 3.2 Bulk Ingest Endpoint (up to 500 events per call)

**Choice:** `POST /events/ingest` accepts a JSON array of up to 500 events per request.

**Rationale:** `feed.py` reads JSONL files that may contain tens of thousands of events. Sending one HTTP request per event would saturate the API with connection overhead. A batch of 500 keeps individual requests under ~200 KB while reducing round-trips by three orders of magnitude. The 500-event cap prevents single requests from monopolising the DB write lock under SQLite.

---

### 3.3 Background Camera Scheduler in-process

**Choice:** Two daemon threads inside the FastAPI process (`app/scheduler.py`) rather than an external task queue.

**Alternatives considered:**

| Approach | Pros | Cons |
|---|---|---|
| In-process threads (chosen) | Zero extra infrastructure; simple deployment | Blocked by Python GIL for CPU-bound work (not a concern here — I/O-bound) |
| Celery + Redis | Robust distributed task queue | Requires Redis; overkill for single-node deployment |
| APScheduler | Clean cron-like API | Another dependency; similar capability |

**Rationale:** The scheduler tasks (`poll camera list`, `dispatch detect jobs`) are network I/O-bound, not CPU-bound. Python daemon threads are sufficient. Keeping everything in-process simplifies Docker deployment — one container, one process, no external broker.

---

### 3.4 POS Correlation as a Post-Processing Script

**Choice:** `pipeline/pos_correlate.py` is a standalone script run after ingest rather than inline logic.

**Rationale:** POS transaction data may arrive on a different schedule than CCTV events (e.g., end-of-day batch export vs. near-real-time video). Decoupling correlation into a script allows it to be re-run if POS data is corrected or arrives late, without re-ingesting any video events. The correlation logic (BILLING zone presence in the 5-minute window before a transaction) is pure business logic that belongs outside the real-time pipeline.

---

### 3.5 Analytics Computed in SQL, Not Python

**Choice:** Metrics (dwell, conversion rate, queue depth, funnel drop-off) are computed with SQL aggregations inside `app/metrics.py` rather than fetching rows to Python and aggregating in memory.

**Rationale:** Pushing aggregations to the DB engine scales with data volume and benefits from indices. Fetching all session rows to Python for a busy store would eventually exhaust API memory. SQL aggregations are also easier to test with known fixtures.

---

## 4. Production Readiness Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Containerisation | Docker + `docker-compose.yml` | Reproducible environment; single-command startup |
| Configuration | Environment variables only | 12-factor compliant; no secrets in code |
| Logging | Structured via Python `logging` + FastAPI request middleware | Machine-parseable; `LOG_LEVEL` adjustable without rebuild |
| Health endpoint | `GET /health` returns DB status + per-store feed freshness | Enables uptime monitoring and `STALE_FEED` alerting |
| Test coverage | `pytest --cov=app` with in-memory SQLite | CI-friendly; no external service required |
