"""
detect.py — Detection worker for Store Intelligence API (app module).

Flow:
    scheduler.py
        → POST /cameras/detect/camera   (every 0.2 s per active camera)
        → cameras.py detect_camera()
        → BackgroundTasks.add_task(process_camera, payload)
        → process_camera()              (runs in FastAPI thread-pool)
        → ClipProcessor.process()       (RTSP loop, emits + ingests periodically)

One background task is allowed per camera_id at a time.
If a task for the same camera is already running the new tick is silently dropped.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import cv2
import requests

# ── Resolve project root (the dir that contains both app/ and pipeline/) ──────
_APP_DIR      = Path(__file__).resolve().parent          # …/app
_PROJECT_ROOT = _APP_DIR.parent                          # …/store-intelligence

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.tracker          import MultiCameraTracker
from pipeline.emit             import EventEmitter
from pipeline.staff_classifier import StaffClassifier
from pipeline.zone_classifier  import ZoneClassifier

logger = logging.getLogger("detect")

# ── Singletons: model + classifiers, initialised once ─────────────────────────
_model:     object | None = None
_staff_clf: StaffClassifier | None = None
_zone_clf:  ZoneClassifier  | None = None
_init_lock  = Lock()

# ── Per-store trackers ────────────────────────────────────────────────────────
_store_trackers: dict[str, MultiCameraTracker] = {}

# ── Per-camera guard: prevents duplicate concurrent stream tasks ───────────────
_active_cameras: set[str] = set()
_active_lock     = Lock()

# ── Tuneables ─────────────────────────────────────────────────────────────────
YOLO_WEIGHTS      = "yolov8n.pt"
LAYOUT_PATH       = _PROJECT_ROOT / "store_layout.json"
EVENTS_DIR        = _PROJECT_ROOT / "events"
API_INGEST_URL    = "http://localhost:8000/events/ingest"
INGEST_INTERVAL_S = 30      # flush accumulated events to API every N seconds
INGEST_BATCH_SIZE = 200     # max events per POST


# ─────────────────────────────────────────────────────────────────────────────
# Lazy global initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_globals() -> None:
    global _model, _staff_clf, _zone_clf
    with _init_lock:
        if _model is not None:
            return
        try:
            from ultralytics import YOLO
            _model = YOLO(YOLO_WEIGHTS)
            logger.info("[DETECT] YOLO model loaded: %s", YOLO_WEIGHTS)
        except Exception as exc:
            logger.error("[DETECT] YOLO load failed: %s", exc)
            _model = None
        _staff_clf = StaffClassifier()
        _zone_clf  = ZoneClassifier()


def _get_tracker(store_id: str) -> MultiCameraTracker:
    if store_id not in _store_trackers:
        _store_trackers[store_id] = MultiCameraTracker(store_id=store_id)
        logger.info("[DETECT] Created tracker for store=%s", store_id)
    return _store_trackers[store_id]


def _load_layout() -> dict:
    try:
        with open(LAYOUT_PATH) as f:
            data = json.load(f)
            logger.info("[DETECT] Loaded store_layout.json from %s", LAYOUT_PATH)
            return data
    except FileNotFoundError:
        logger.warning("[DETECT] store_layout.json not found at %s — using {}", LAYOUT_PATH)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint — called by BackgroundTasks in cameras.py
# ─────────────────────────────────────────────────────────────────────────────

def process_camera(payload) -> None:
    """
    Runs in FastAPI's thread-pool via BackgroundTasks.add_task().

    Guard: if this camera_id is already running, silently skip.
    This is expected — the scheduler fires every 0.2 s but a stream session
    lasts much longer.
    """
    camera_key = f"{payload.store_id}:{payload.camera_id}"

    with _active_lock:
        if camera_key in _active_cameras:
            logger.debug("[DETECT] Already running for %s — skipping tick", camera_key)
            return
        _active_cameras.add(camera_key)

    logger.info(
        "[DETECT] Starting  store=%s  camera=%s  rtsp=%s  type=%s",
        payload.store_id, payload.camera_id,
        payload.rtsp_url, payload.camera_type,
    )

    try:
        _run_stream(payload)
    except Exception:
        logger.exception("[DETECT] Unhandled error in process_camera(%s)", camera_key)
    finally:
        with _active_lock:
            _active_cameras.discard(camera_key)
        logger.info("[DETECT] Stream ended — %s", camera_key)


# ─────────────────────────────────────────────────────────────────────────────
# Stream runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_stream(payload) -> None:
    _init_globals()

    if _model is None:
        logger.error("[DETECT] Model not available — aborting %s", payload.camera_id)
        return

    store_layout = _load_layout()
    tracker      = _get_tracker(payload.store_id)

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVENTS_DIR / f"{payload.store_id}_{payload.camera_id}.jsonl"

    emitter = EventEmitter(output_path=str(out_path))

    cam_type = payload.camera_type or _camera_type_from_id(payload.camera_id)

    processor = ClipProcessor(
        model             = _model,
        store_id          = payload.store_id,
        camera_id         = payload.camera_id,
        camera_type       = cam_type,
        store_layout      = store_layout,
        clip_start_utc    = datetime.now(tz=timezone.utc),
        fps               = 15.0,
        emitter           = emitter,
        tracker           = tracker,
        staff_classifier  = _staff_clf,
        zone_classifier   = _zone_clf,
        source            = payload.rtsp_url,
        on_ingest_tick    = lambda: _ingest_events_to_api(str(out_path)),
    )

    processor.process(payload.rtsp_url)

    # Final flush after stream ends (file sources only; live streams loop forever)
    _ingest_events_to_api(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Ingest helper — cursor-based so each line is posted exactly once
# ─────────────────────────────────────────────────────────────────────────────

_ingest_cursors: dict[str, int] = {}   # jsonl_path → byte offset already ingested
_ingest_lock = Lock()


def _ingest_events_to_api(jsonl_path: str) -> None:
    with _ingest_lock:
        cursor = _ingest_cursors.get(jsonl_path, 0)

    events: list[dict] = []
    new_cursor = cursor

    try:
        with open(jsonl_path, "rb") as f:
            f.seek(cursor)
            for raw in f:
                line = raw.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                new_cursor += len(raw)
    except FileNotFoundError:
        return

    if not events:
        with _ingest_lock:
            _ingest_cursors[jsonl_path] = new_cursor
        return

    total_accepted = 0
    success = True

    for i in range(0, len(events), INGEST_BATCH_SIZE):
        batch = events[i : i + INGEST_BATCH_SIZE]
        try:
            r    = requests.post(API_INGEST_URL, json={"events": batch}, timeout=30)
            body = r.json()
            total_accepted += body.get("accepted", 0)
            logger.info(
                "[DETECT] Ingest batch %d  status=%s  accepted=%d  duplicate=%d",
                i // INGEST_BATCH_SIZE + 1,
                r.status_code,
                body.get("accepted", 0),
                body.get("duplicate", 0),
            )
        except Exception as exc:
            logger.error("[DETECT] Ingest POST failed: %s", exc)
            success = False
            break

    # Only advance cursor when all batches succeeded (retry on next tick otherwise)
    if success:
        with _ingest_lock:
            _ingest_cursors[jsonl_path] = new_cursor
        logger.info(
            "[DETECT] Ingest flush done — accepted=%d / total=%d",
            total_accepted, len(events),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stream utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_live_source(source: str) -> bool:
    s = source.lower()
    return any(s.startswith(p) for p in (
        "rtsp://", "rtsps://", "rtmp://", "rtmps://",
        "http://", "https://", "hls://",
    )) or s.endswith(".m3u8")


def _open_capture(
    source: str,
    retries: int = 5,
    retry_delay: float = 3.0,
) -> cv2.VideoCapture:
    for attempt in range(1, retries + 1):
        cap = (
            cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if _is_live_source(source)
            else cv2.VideoCapture(source)
        )
        if _is_live_source(source):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "[DETECT] Opened %s  [%dx%d @ %.1ffps]  attempt=%d",
                source, w, h, fps, attempt,
            )
            return cap

        logger.warning(
            "[DETECT] Cannot open %s (attempt %d/%d) — retrying in %.1fs",
            source, attempt, retries, retry_delay,
        )
        cap.release()
        time.sleep(retry_delay)

    raise RuntimeError(
        f"Cannot open video source after {retries} attempts: {source}"
    )


def _camera_type_from_id(camera_id: str) -> str:
    cid = camera_id.upper()
    if "ENTRY" in cid or "EXIT" in cid:
        return "entry"
    if "BILLING" in cid or "POS" in cid:
        return "billing"
    return "floor"


# ─────────────────────────────────────────────────────────────────────────────
# ClipProcessor — RTSP / file stream loop
# ─────────────────────────────────────────────────────────────────────────────

class ClipProcessor:
    PERSON_CLASS            = 0
    CONF_THRESHOLD          = 0.35
    ENTRY_TRIPWIRE_FRACTION = 0.55
    DWELL_INTERVAL_MS       = 30_000
    MAX_RECONNECTS          = 5
    RECONNECT_DELAY         = 5.0
    SKIP_FRAMES_RTSP        = 2

    def __init__(
        self,
        model,
        store_id:          str,
        camera_id:         str,
        camera_type:       str,
        store_layout:      dict,
        clip_start_utc:    datetime,
        fps:               float,
        emitter:           EventEmitter,
        tracker:           MultiCameraTracker,
        staff_classifier:  StaffClassifier,
        zone_classifier:   ZoneClassifier,
        source:            str = "",
        stream_duration_s: float | None = None,
        on_ingest_tick:    callable = None,
    ):
        self.model             = model
        self.store_id          = store_id
        self.camera_id         = camera_id
        self.camera_type       = camera_type
        self.store_layout      = store_layout
        self.clip_start_utc    = clip_start_utc
        self.fps               = fps
        self.emitter           = emitter
        self.tracker           = tracker
        self.staff_classifier  = staff_classifier
        self.zone_classifier   = zone_classifier
        self.source            = source
        self.stream_duration_s = stream_duration_s
        self.on_ingest_tick    = on_ingest_tick
        self._is_live          = _is_live_source(source)
        self._track_state: dict[int, dict] = {}

    # ── main loop ─────────────────────────────────────────────────────────────

    def process(self, video_path: str) -> int:
        reconnects    = 0
        events_emitted = 0
        wall_start    = time.monotonic()
        last_ingest_t = wall_start
        frame_idx     = 0

        while True:
            try:
                cap = _open_capture(video_path)
            except RuntimeError as exc:
                logger.error("[DETECT] %s", exc)
                return events_emitted

            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps and detected_fps > 0:
                self.fps = detected_fps

            consecutive_failures = 0
            skip_counter         = 0

            while True:
                now = time.monotonic()

                # Duration cap
                if self.stream_duration_s and now - wall_start >= self.stream_duration_s:
                    logger.info("[DETECT] Duration cap reached.")
                    cap.release()
                    events_emitted += self._handle_lost_tracks(datetime.now(tz=timezone.utc))
                    return events_emitted

                # Periodic ingest flush
                if self.on_ingest_tick and now - last_ingest_t >= INGEST_INTERVAL_S:
                    self.on_ingest_tick()
                    last_ingest_t = now

                ret, frame = cap.read()

                if not ret:
                    if self._is_live and reconnects < self.MAX_RECONNECTS:
                        logger.warning(
                            "[DETECT] Read failed — reconnecting (%d/%d)…",
                            reconnects + 1, self.MAX_RECONNECTS,
                        )
                        cap.release()
                        time.sleep(self.RECONNECT_DELAY)
                        reconnects += 1
                        break   # re-open capture
                    logger.info("[DETECT] End of source or max reconnects.")
                    cap.release()
                    events_emitted += self._handle_lost_tracks(self._frame_to_utc(frame_idx))
                    return events_emitted

                # Skip frames on live streams
                if self._is_live:
                    skip_counter += 1
                    if skip_counter % self.SKIP_FRAMES_RTSP != 0:
                        frame_idx += 1
                        continue

                frame_ts = (
                    datetime.now(tz=timezone.utc)
                    if self._is_live
                    else self._frame_to_utc(frame_idx)
                )

                try:
                    detections = self._detect_persons(frame)
                    tracks     = self.tracker.update(
                        detections, frame, self.camera_id, frame_ts
                    )
                    for track in tracks:
                        events_emitted += len(
                            self._process_track(track, frame, frame_ts)
                        )
                    events_emitted += self._handle_lost_tracks(frame_ts)
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    logger.warning("[DETECT] Frame error (x%d): %s", consecutive_failures, exc)
                    if consecutive_failures > 30:
                        logger.error("[DETECT] Too many frame errors — aborting.")
                        cap.release()
                        return events_emitted

                frame_idx += 1

                if frame_idx % 150 == 0:
                    logger.info(
                        "[DETECT] frame=%d  events=%d  elapsed=%.1fs  store=%s  cam=%s",
                        frame_idx, events_emitted,
                        time.monotonic() - wall_start,
                        self.store_id, self.camera_id,
                    )

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect_persons(self, frame) -> list[dict]:
        results    = self.model(frame, classes=[self.PERSON_CLASS], verbose=False)
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return detections

    # ── track dispatch ────────────────────────────────────────────────────────

    def _process_track(self, track: dict, frame, frame_ts: datetime) -> list:
        tid        = track["track_id"]
        visitor_id = track["visitor_id"]
        bbox       = track["bbox"]
        confidence = track["confidence"]
        is_staff   = self.staff_classifier.classify(frame, bbox)

        state = self._track_state.setdefault(tid, {
            "visitor_id":      visitor_id,
            "is_staff":        is_staff,
            "entered":         False,
            "current_zone":    None,
            "zone_enter_ts":   None,
            "last_dwell_ts":   None,
            "session_seq":     0,
            "first_seen_ts":   frame_ts,
            "reentry_handled": False,
            "prev_centroid_y": None,
        })

        if is_staff and not state["is_staff"]:
            state["is_staff"] = True

        events = (
            self._handle_entry_exit(track, state, bbox, frame_ts, confidence)
            if self.camera_type == "entry"
            else self._handle_zone_tracking(track, state, frame, bbox, frame_ts, confidence)
        )
        state["session_seq"] += len(events)
        return events

    # ── entry / exit ──────────────────────────────────────────────────────────

    def _handle_entry_exit(self, track, state, bbox, frame_ts, confidence) -> list:
        events       = []
        _, y1, _, y2 = bbox
        centroid_y   = (y1 + y2) / 2
        frame_height = getattr(self.tracker, "frame_height", None) or 1080
        tripwire_y   = frame_height * self.ENTRY_TRIPWIRE_FRACTION
        prev_y       = state["prev_centroid_y"]
        state["prev_centroid_y"] = centroid_y

        if prev_y is None:
            return events

        crossed_down = prev_y < tripwire_y <= centroid_y
        crossed_up   = prev_y > tripwire_y >= centroid_y

        if crossed_down and not state["entered"]:
            visitor_id = track["visitor_id"]
            is_reentry = self.tracker.is_reentry(visitor_id, self.store_id)
            if is_reentry and not state["reentry_handled"]:
                state["reentry_handled"] = True
                events.append(self.emitter.emit_reentry(
                    store_id=self.store_id, camera_id=self.camera_id,
                    visitor_id=visitor_id, timestamp=frame_ts,
                    confidence=confidence, is_staff=state["is_staff"],
                    session_seq=state["session_seq"],
                ))
            else:
                state["entered"] = True
                events.append(self.emitter.emit_entry(
                    store_id=self.store_id, camera_id=self.camera_id,
                    visitor_id=visitor_id, timestamp=frame_ts,
                    confidence=confidence, is_staff=state["is_staff"],
                    session_seq=state["session_seq"],
                ))

        elif crossed_up and state["entered"]:
            state["entered"] = False
            events.append(self.emitter.emit_exit(
                store_id=self.store_id, camera_id=self.camera_id,
                visitor_id=track["visitor_id"], timestamp=frame_ts,
                confidence=confidence, is_staff=state["is_staff"],
                session_seq=state["session_seq"],
            ))

        return events

    # ── zone tracking ─────────────────────────────────────────────────────────

    def _handle_zone_tracking(self, track, state, frame, bbox, frame_ts, confidence) -> list:
        events     = []
        visitor_id = track["visitor_id"]
        zone       = self.zone_classifier.classify(bbox, self.camera_id, self.store_layout)
        prev_zone  = state["current_zone"]

        if zone != prev_zone:
            if prev_zone is not None:
                events.append(self.emitter.emit_zone_exit(
                    store_id=self.store_id, camera_id=self.camera_id,
                    visitor_id=visitor_id, timestamp=frame_ts, zone_id=prev_zone,
                    confidence=confidence, is_staff=state["is_staff"],
                    session_seq=state["session_seq"],
                ))
            if zone is not None:
                state["zone_enter_ts"] = frame_ts
                state["last_dwell_ts"] = frame_ts
                queue_depth = self.tracker.get_queue_depth(self.store_id, frame_ts)
                if self.camera_type == "billing" and zone == "BILLING" and queue_depth > 0:
                    events.append(self.emitter.emit_billing_queue_join(
                        store_id=self.store_id, camera_id=self.camera_id,
                        visitor_id=visitor_id, timestamp=frame_ts,
                        confidence=confidence, is_staff=state["is_staff"],
                        queue_depth=queue_depth, session_seq=state["session_seq"],
                    ))
                else:
                    events.append(self.emitter.emit_zone_enter(
                        store_id=self.store_id, camera_id=self.camera_id,
                        visitor_id=visitor_id, timestamp=frame_ts, zone_id=zone,
                        confidence=confidence, is_staff=state["is_staff"],
                        session_seq=state["session_seq"],
                    ))
            state["current_zone"] = zone

        if zone is not None and state["last_dwell_ts"] is not None:
            elapsed_ms = (frame_ts.timestamp() - state["last_dwell_ts"].timestamp()) * 1000
            if elapsed_ms >= self.DWELL_INTERVAL_MS:
                state["last_dwell_ts"] = frame_ts
                total_dwell_ms = (
                    frame_ts.timestamp() - state["zone_enter_ts"].timestamp()
                ) * 1000
                events.append(self.emitter.emit_zone_dwell(
                    store_id=self.store_id, camera_id=self.camera_id,
                    visitor_id=visitor_id, timestamp=frame_ts, zone_id=zone,
                    dwell_ms=int(total_dwell_ms), confidence=confidence,
                    is_staff=state["is_staff"], session_seq=state["session_seq"],
                ))

        return events

    # ── lost track cleanup ────────────────────────────────────────────────────

    def _handle_lost_tracks(self, frame_ts: datetime) -> int:
        lost  = self.tracker.get_lost_tracks(self.camera_id)
        count = 0
        for tid in lost:
            state = self._track_state.pop(tid, None)
            if state is None:
                continue
            if (
                self.camera_type == "billing"
                and state.get("current_zone") == "BILLING"
                and not state["is_staff"]
            ):
                self.emitter.emit_billing_queue_abandon(
                    store_id=self.store_id, camera_id=self.camera_id,
                    visitor_id=state["visitor_id"], timestamp=frame_ts,
                    confidence=0.0, is_staff=False,
                    session_seq=state["session_seq"],
                )
                count += 1
        return count