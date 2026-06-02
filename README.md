# Store Intelligence API

Offline retail analytics for Apex Retail — from raw CCTV to live store metrics.

## Quick Start (5 commands)

```bash
# 1. Clone and enter the repo
git clone https://github.com/BlueMoon67/ANALYZER.git  store-intelligence && cd store-intelligence

# 2. Start the API
docker compose up -d

# 3. Run the detection pipeline on your clips
python pipeline/detect.py \
  --store STORE_BLR_002 \
  --clips-dir clips/ \
  --layout pipeline/store_layout.json \
  --output-dir events/

# 4. Feed events into the API
python pipeline/feed.py --dir events/ --api http://localhost:8000

# 5. Verify it's working
curl http://localhost:8000/health
```

> **Note:** The detection pipeline runs on the host machine (not inside Docker) and requires
> its own dependencies — see [Pipeline dependencies](#pipeline-dependencies) below.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/events/ingest` | Ingest up to 500 events per call (idempotent by `event_id`) |
| `GET`  | `/stores/{store_id}/metrics` | Unique visitors, conversion rate, avg dwell, queue depth |
| `GET`  | `/stores/{store_id}/funnel` | Entry → Zone Visit → Billing Queue → Purchase with drop-off % |
| `GET`  | `/stores/{store_id}/heatmap` | Zone visit frequency + dwell, normalised 0–100 |
| `GET`  | `/stores/{store_id}/anomalies` | Queue spike, conversion drop, dead zone, stale feed |
| `GET`  | `/health` | DB status, last event timestamp per store, STALE_FEED warnings |
| `GET`  | `/cameras` | List all registered cameras |
| `GET`  | `/cameras/{store_id}` | Cameras for a specific store |
| `POST` | `/cameras/register` | Register a new RTSP camera |
| `POST` | `/cameras/{store_id}/{cam_id}/activate` | Set a camera as the active stream |

---

## Detection Pipeline

The pipeline runs on the host — it is **not** included in the Docker image. It reads CCTV
clips (or RTSP streams), runs YOLOv8n + ByteTrack, and writes structured JSONL event files.

### Pipeline dependencies

```bash
pip install opencv-python-headless ultralytics numpy
# or, once requirements.txt includes the pipeline section:
pip install -r requirements.txt
```

### Process a single clip

```bash
python pipeline/detect.py \
  --clip clips/STORE_BLR_002/CAM_ENTRY_01.mp4 \
  --layout pipeline/store_layout.json \
  --output events/STORE_BLR_002_CAM_ENTRY_01.jsonl \
  --weights pipeline/yolov8n.pt
```

### Process all cameras for a store

```bash
python pipeline/detect.py \
  --store STORE_BLR_002 \
  --clips-dir clips/ \
  --layout pipeline/store_layout.json \
  --output-dir events/
```

### Feed events into the API

```bash
# Batch — ingest a single file
python pipeline/feed.py --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl

# Batch — ingest all .jsonl files in a directory
python pipeline/feed.py --dir events/ --api http://localhost:8000

# Stream mode — replay events with real-time delay (for the live dashboard)
python pipeline/feed.py \
  --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl \
  --stream \
  --api http://localhost:8000
```

### Correlate POS transactions

Run this after events are ingested. It marks sessions as converted when a visitor was
in the BILLING zone in the 5-minute window before a transaction timestamp.

```bash
python pipeline/pos_correlate.py \
  --pos pos_transactions.csv \
  --api http://localhost:8000
```

### One-command pipeline (all stores)

```bash
# Requires: docker compose up running, clips/ populated
./pipeline/run.sh \
  --clips-dir clips/ \
  --api-url http://localhost:8000
```

---

## Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install -r requirements.txt

# Run all tests with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# API integration tests only
pytest tests/test_metrics.py -v

# Pipeline unit tests only (ultralytics is mocked; numpy required)
pytest tests/test_pipeline.py -v
```

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py               # YOLOv8n detection + ByteTrack + Re-ID (host only)
│   ├── tracker.py              # Multi-camera tracker + Re-ID logic
│   ├── emit.py                 # Event schema + JSONL emission
│   ├── staff_classifier.py     # Uniform colour heuristic + optional VLM fallback
│   ├── zone_classifier.py      # Point-in-polygon zone lookup
│   ├── feed.py                 # JSONL → API bridge (batch + stream modes)
│   ├── pos_correlate.py        # POS ↔ visitor session correlation
│   ├── run.sh                  # One-command: detect + ingest for all stores
│   ├── store_layout.json       # Zone polygon definitions
│   └── yolov8n.pt              # YOLOv8 weights (downloaded on first run if absent)
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entrypoint + request logging middleware
│   ├── models.py               # Pydantic schemas + event enums
│   ├── database.py             # SQLAlchemy engine + table definitions (lazy init)
│   ├── ingestion.py            # Event ingest, deduplication, session materialisation
│   ├── metrics.py              # Real-time metrics, funnel, heatmap
│   ├── anomalies.py            # Anomaly detection (queue spike, dead zone, etc.)
│   ├── health.py               # Health check + feed freshness
│   ├── cameras.py              # Camera registry + RTSP management endpoints
│   └── scheduler.py            # Background threads: camera polling + detect dispatch
├── tests/
│   ├── conftest.py             # sys.path setup + DB engine reset for test isolation
│   ├── test_metrics.py         # API integration tests (in-memory SQLite)
│   └── test_pipeline.py        # Pipeline unit tests
├── events/                     # JSONL event output from detect.py (gitignored)
├── docs/
│   ├── DESIGN.md               # Architecture overview + AI-assisted decisions
│   └── CHOICES.md              # Model selection, schema design, storage decision
├── store_dashboard.html        # Part E live dashboard (open in browser)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | SQLAlchemy connection string — set to `postgresql://...` for production |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `USE_VLM_STAFF_DETECTION` | `0` | Set to `1` to enable VLM fallback for ambiguous staff detection |

---

## Part E — Live Dashboard

Open `store_dashboard.html` in a browser while the API is running. It polls the metrics,
funnel, and anomaly endpoints every few seconds and updates in real time.

To feed events live as they are processed (simulated real-time):

```bash
# Terminal 1 — start the API
docker compose up

# Terminal 2 — process a clip and write events to disk as they are detected
python pipeline/detect.py \
  --clip clips/STORE_BLR_002/CAM_ENTRY_01.mp4 \
  --layout pipeline/store_layout.json \
  --output events/live.jsonl &

# Stream events to the API as they arrive (15fps replay)
python pipeline/feed.py \
  --file events/live.jsonl \
  --stream \
  --api http://localhost:8000

# Terminal 3 — or watch metrics update via CLI
watch -n 2 'curl -s http://localhost:8000/stores/STORE_BLR_002/metrics | python3 -m json.tool'
```

---

## How the Camera Scheduler Works

At startup, `scheduler.py` launches two background daemon threads:

1. **`fetch_cameras`** — polls `GET /cameras` every 10 seconds and caches the list of
   active cameras that have an RTSP URL registered.
2. **`scheduler_loop`** — iterates the cache and POSTs each camera to
   `POST /cameras/detect/camera` every 0.2 seconds.

The detect endpoint (in `cameras.py`) receives each assignment and spawns a
`BackgroundTask` that calls `detect.process_camera()` — which opens the RTSP stream,
runs YOLOv8n, and ingests events. A per-camera guard (`_active_cameras` set) ensures
only one stream session runs per camera at any time.

To use live RTSP streams, register your cameras first:

```bash
curl -X POST http://localhost:8000/cameras/register \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "STORE_BLR_002",
    "camera_id": "CAM_ENTRY_01",
    "rtsp_url": "rtsp://192.168.1.10:554/stream1",
    "camera_type": "entry"
  }'
```
