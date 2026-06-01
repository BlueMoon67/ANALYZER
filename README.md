# Store Intelligence API

Offline retail analytics for Apex Retail — from raw CCTV to live store metrics.

## Quick Start (5 commands)

```bash
git clone <your-repo-url> store-intelligence && cd store-intelligence

# 1. Start the API
docker compose up -d

# 2. Run detection pipeline on all clips for a store
python pipeline/detect.py \
  --store STORE_BLR_002 \
  --clips-dir clips/ \
  --layout store_layout.json \
  --output-dir events/

# 3. Feed events into the API
python pipeline/feed.py --dir events/ --api http://localhost:8000

# 4. Correlate POS transactions → mark conversions
python pipeline/pos_correlate.py --pos pos_transactions.csv

# 5. Check it's working
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/events/ingest` | Ingest up to 500 events (idempotent by `event_id`) |
| `GET`  | `/stores/{id}/metrics` | Unique visitors, conversion rate, dwell, queue depth |
| `GET`  | `/stores/{id}/funnel` | Entry → Zone → Billing → Purchase with drop-off % |
| `GET`  | `/stores/{id}/heatmap` | Zone frequency + dwell, normalised 0–100 |
| `GET`  | `/stores/{id}/anomalies` | Queue spike, conversion drop, dead zone alerts |
| `GET`  | `/health` | DB status, per-store feed freshness, STALE_FEED warning |

## Detection Pipeline

The pipeline runs independently of the API. It reads video clips and writes JSONL event files.
Pipeline dependencies must be installed separately (they're not included in the Docker image).

### Install pipeline dependencies

```bash
pip install -r pipeline/requirements-pipeline.txt
```

### Single clip

```bash
python pipeline/detect.py \
  --clip clips/STORE_BLR_002/CAM_ENTRY_01.mp4 \
  --layout store_layout.json \
  --output events/STORE_BLR_002_CAM_ENTRY_01.jsonl \
  --weights yolov8n.pt
```

### Batch (all cameras for a store)

```bash
python pipeline/detect.py \
  --store STORE_BLR_002 \
  --clips-dir clips/ \
  --layout store_layout.json \
  --output-dir events/
```

### Feed into API (batch)

```bash
python pipeline/feed.py --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl
```

### Feed with real-time streaming (Part E dashboard)

```bash
python pipeline/feed.py \
  --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl \
  --stream \
  --api http://localhost:8000
```

### POS correlation (mark converted sessions)

```bash
python pipeline/pos_correlate.py --pos pos_transactions.csv
```

## Running Tests

```bash
# Install API + test deps (already included in requirements.txt)
pip install -r requirements.txt

# Run all tests with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Run only API integration tests
pytest tests/test_metrics.py -v

# Run only pipeline unit tests (requires numpy; ultralytics is mocked)
pytest tests/test_pipeline.py -v
```

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py              # YOLOv8 detection + ByteTrack + Re-ID
│   ├── tracker.py             # Multi-camera tracker + Re-ID
│   ├── emit.py                # Event schema + JSONL emission
│   ├── staff_classifier.py    # Uniform colour heuristic + optional VLM
│   ├── zone_classifier.py     # Point-in-polygon zone lookup
│   ├── feed.py                # JSONL → API bridge (batch + stream)
│   ├── pos_correlate.py       # POS ↔ visitor session correlation
│   ├── run.sh                 # One-command: detect + ingest for all stores
│   └── requirements-pipeline.txt
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI entrypoint + middleware
│   ├── models.py              # Pydantic schemas + enums
│   ├── database.py            # SQLAlchemy Core + table definitions (lazy engine)
│   ├── ingestion.py           # Ingest, dedup, session materialisation
│   ├── metrics.py             # Real-time metric computation
│   ├── anomalies.py           # Anomaly detection
│   └── health.py              # Health check
├── tests/
│   ├── conftest.py            # sys.path + engine reset for test isolation
│   ├── test_pipeline.py       # Pipeline unit tests
│   └── test_metrics.py        # API integration tests (in-memory SQLite)
├── docs/
│   ├── DESIGN.md
│   └── CHOICES.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt           # API + test deps
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | SQLAlchemy connection string |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `USE_VLM_STAFF_DETECTION` | `0` | Set to `1` to enable VLM fallback for ambiguous staff detection |

## Part E — Live Dashboard

Run detection and feed in stream mode simultaneously:

```bash
# Terminal 1: start API
docker compose up

# Terminal 2: process clip and stream events
python pipeline/detect.py --clip clips/STORE_BLR_002/CAM_ENTRY_01.mp4 \
  --layout store_layout.json --output events/live.jsonl &

python pipeline/feed.py --file events/live.jsonl --stream

# Terminal 3: watch metrics update live
watch -n 2 'curl -s http://localhost:8000/stores/STORE_BLR_002/metrics | python3 -m json.tool'
```

## One-Command Pipeline (all stores)

```bash
# Requires: docker compose up already running, clips/ populated
./pipeline/run.sh --clips-dir clips/ --api-url http://localhost:8000
```
