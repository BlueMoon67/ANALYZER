# DESIGN.md — Store Intelligence Pipeline Architecture

## 1. System Overview

The pipeline converts raw CCTV footage into structured behavioural events that feed a live analytics API.

```
CCTV Clips (3 cameras × N stores)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                   DETECTION LAYER (Module 1)                 │
│                                                              │
│  YOLOv8n Person Detection                                    │
│       │                                                      │
│  KalmanBoxTracker (per-camera IoU matching)                  │
│       │                                                      │
│  MultiCameraTracker (Re-ID + cross-camera dedup)             │
│       │                                                      │
│  StaffClassifier (HSV colour histogram + VLM fallback)       │
│       │                                                      │
│  ZoneClassifier (point-in-polygon vs store_layout.json)      │
│       │                                                      │
│  EventEmitter → JSONL files                                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼  (batch or streaming via run.sh)
┌─────────────────────────────────────────────────────────────┐
│              INTELLIGENCE API (Module 2 — app/)              │
│                                                              │
│  POST /events/ingest  →  SQLite/PostgreSQL                   │
│  GET  /stores/{id}/metrics                                   │
│  GET  /stores/{id}/funnel                                    │
│  GET  /stores/{id}/heatmap                                   │
│  GET  /stores/{id}/anomalies                                 │
│  GET  /health                                                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│               LIVE DASHBOARD (Module 3 — dashboard/)         │
│  Real-time metric updates (rich terminal / web UI)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Detection Layer Design

### 2.1 Person Detection

**Model:** YOLOv8n via Ultralytics Python API.

Inference is called on every frame at the native clip resolution (1080p, 15fps). YOLO outputs bounding boxes for class 0 (person). Detections below 0.35 confidence are excluded from tracking input but are *not* suppressed from event emission if a tracker already owns that region — this handles partial occlusion (see §2.3).

### 2.2 Tracking — KalmanBoxTracker + CameraTracker

Each camera maintains its own `CameraTracker` instance.

The tracker uses:
- A minimal 8-state constant-velocity Kalman filter: `[cx, cy, w, h, vx, vy, vw, vh]`
- Greedy IoU matching (Hungarian-style, ascending cost). Threshold: 0.35.
- ByteTrack-inspired logic: unmatched low-confidence detections are kept alive as "tentative" tracks for 3 seconds before deletion. This prevents tracking loss during momentary occlusion.

Track lifecycle:
```
Unconfirmed (hits < 2) → Confirmed (hits ≥ 2) → Lost (time_since_update > MAX_AGE) → Deleted
```

### 2.3 Re-ID and Cross-Camera Deduplication

The `MultiCameraTracker` maintains a global visitor registry per store.

**Within-session Re-ID** (same camera, brief disappearance):
When a `CameraTracker` reports a lost track, the corresponding `visitor_id` is held in registry for 30 minutes. If a new detection appears with a similar appearance embedding, it receives the same `visitor_id`.

**Cross-camera deduplication:**
A visitor visible on both `CAM_ENTRY_01` and `CAM_FLOOR_01` simultaneously would be double-counted without dedup. The appearance embedding similarity check (threshold: 0.78 cosine similarity) prevents this.

**Appearance embedding:**
Lightweight torso-region colour histogram (16 bins × 3 channels = 48-dimensional vector, L2-normalised). Computed over the middle 50% of bbox height to avoid head/feet variability.

**Re-entry detection:**
When a visitor with a prior `EXIT` event is matched by appearance, a `REENTRY` event is emitted instead of a second `ENTRY`. This is the primary defence against re-entry inflation.

### 2.4 Entry/Exit Direction

Applies only to `camera_type = "entry"`.

A horizontal tripwire at `y = 0.55 × frame_height` is derived from camera calibration. A track crossing from above to below = `ENTRY`; below to above = `EXIT`. The 0.55 fraction positions the wire at the doorframe midpoint in standard retail entrance camera configurations.

**Group entry handling:** Each Kalman track represents one person. Three people entering simultaneously produce three independent `ENTRY` events. The tracker handles this correctly because IoU matching operates on individual bounding boxes.

### 2.5 Staff Classification

`StaffClassifier` computes the fraction of upper-body pixels matching any configured HSV range. Default ranges cover navy blue, black, forest green, and burgundy — common Indian retail uniform colours.

If coverage ≥ 38% → `is_staff = True`. This threshold was chosen empirically to balance:
- Low false positives (customers in dark blue jeans): upper-body crop excludes jeans
- Low false negatives (staff in mixed lighting): 38% allows for partial shadowing

Optional VLM fallback (`USE_VLM_STAFF_DETECTION=1`): for confidence in range [0.40, 0.65], a Claude vision API call is made with a binary prompt. Results are cached per bbox hash.

### 2.6 Zone Classification

`ZoneClassifier` performs point-in-polygon tests using the ray-casting algorithm. Zone polygons are loaded from `store_layout.json` and cached per camera.

Fallback: if no polygon data is available for a camera, the frame is divided into a 3×2 grid of generic zones. This prevents crashes on uncalibrated stores.

---

## 3. Event Schema

See `emit.py` for the authoritative schema and `CHOICES.md §2` for design rationale.

Key design invariants:
- `event_id` is UUID4 — globally unique, enables idempotent ingestion
- `confidence` is never clamped or suppressed
- `zone_id` is always `null` for ENTRY/EXIT/REENTRY events
- `is_staff` is set on every event — the API filters on this field

---

## 4. Data Flow

```
Frame 0..N (15fps, 1080p)
  │
  ▼ detect.py: YOLO inference
[detection list: [{bbox, confidence}, ...]]
  │
  ▼ tracker.py: CameraTracker.update()
[raw tracks: [{track_id, bbox, confidence, hits}, ...]]
  │
  ▼ tracker.py: MultiCameraTracker enrichment
[enriched tracks: [{...raw, visitor_id}, ...]]
  │
  ▼ detect.py: ClipProcessor._process_track()
  ├─ StaffClassifier.classify(frame, bbox) → is_staff
  ├─ Entry/exit tripwire check (entry cameras)
  ├─ ZoneClassifier.classify() → zone_id (floor/billing cameras)
  └─ Dwell timer check
  │
  ▼ emit.py: EventEmitter
[JSONL line: {event_id, store_id, ..., metadata}]
  │
  ▼ POST /events/ingest (via run.sh)
[SQLite / PostgreSQL]
```

---

## 5. Handling Known Edge Cases

| Edge Case | Approach |
|-----------|----------|
| **Group entry** | Individual Kalman tracks per person; IoU matching with non-overlapping bboxes → N separate ENTRY events |
| **Staff movement** | HSV colour heuristic on upper body; `is_staff=true` propagated to all their events; API excludes `is_staff=true` from customer metrics |
| **Re-entry** | Appearance embedding match against exited visitor registry → `REENTRY` event instead of new `ENTRY` |
| **Partial occlusion** | ByteTrack-style low-confidence track retention; confidence degraded gracefully in event, not suppressed |
| **Empty store** | Zero detections → zero events; API handles `NULL`/empty aggregates without crash (tested in `test_pipeline.py`) |
| **Camera overlap** | Cross-camera appearance dedup in `MultiCameraTracker`; 0.78 cosine similarity threshold |
| **Billing queue** | Queue depth tracked as active non-staff tracks in billing zone; `BILLING_QUEUE_ABANDON` emitted when billing track disappears without POS correlation |

---

## 6. AI-Assisted Decisions

### 6.1 Tripwire Calibration
AI suggested using optical flow to detect entry direction, reasoning that tripwires require calibration per store. I **agreed partially**: optical flow is more robust but adds ~50ms/frame on CPU. The tripwire approach at 0.55 × frame_height is calibration-free for standard entrance cameras where the camera is mounted above and angled down. I added the fraction as a configurable parameter for cameras with unusual angles.

### 6.2 Re-ID Embedding Choice
I asked Claude to compare OSNet, colour histograms, and CLIP embeddings for Re-ID in retail CCTV. Claude's analysis:
- OSNet: best accuracy, needs GPU, adds 40ms/frame
- CLIP: surprisingly good but 200ms/frame — ruled out immediately
- Colour histogram: adequate for controlled lighting, <1ms/frame

I **agreed** with the colour histogram recommendation for CPU deployment and implemented it. Claude's suggestion to use only the torso (not full body) to reduce footwear/flooring noise was a useful refinement I adopted.

### 6.3 Event Schema `metadata` Structure
AI initially suggested a flat schema without the `metadata` envelope. I **overrode** this: the flat schema would make the `queue_depth` field appear on all event types (confusing for consumers), and `session_seq` would conflict with other ordinal fields if schema evolves. Nesting optional per-event-type fields in `metadata` keeps the base schema stable while allowing event-type-specific enrichment.

---

## 7. Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|------------|
| Detection latency | ~15ms/frame | YOLOv8n, CPU (Intel i7), 1080p |
| Tracking overhead | ~2ms/frame | KalmanBoxTracker + greedy IoU matching |
| Staff classification | ~0.3ms/bbox | HSV histogram (no VLM) |
| Zone classification | ~0.05ms/bbox | Point-in-polygon, cached zones |
| Total throughput | ~17ms/frame | ≈ real-time at 15fps on CPU |
| Event emission | ~0.1ms/event | JSONL append, buffered |

With GPU (NVIDIA RTX 3080): ~3ms/frame for YOLOv8n → room to upgrade to YOLOv8m without breaking real-time.

---

## 8. Assumptions

1. Clip filename convention encodes store_id and camera_id: `STORE_BLR_002_CAM_ENTRY_01_<timestamp>.mp4`
2. Camera angles are fixed (no pan/tilt/zoom)
3. Staff wear consistent uniforms (HSV heuristic assumption)
4. POS correlation window is 5 minutes (spec requirement)
5. Re-entry within 30 minutes of exit is the same visitor (configurable via `REENTRY_WINDOW_MIN`)
6. The entry camera is calibrated such that door threshold is at y ≈ 0.55 × frame_height
