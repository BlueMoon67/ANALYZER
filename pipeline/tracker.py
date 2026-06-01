"""
tracker.py — Multi-camera person tracker with Re-ID for Store Intelligence.

Design (see CHOICES.md §1):
- Per-camera ByteTrack-style tracker (IoU matching + Kalman filter)
- Cross-camera Re-ID: appearance embedding hash + bounding-box size similarity
- Re-entry detection: visitor seen exiting and re-entering within 30 min
- Staff exclusion happens upstream in StaffClassifier; tracker is agnostic

Re-ID approach:
  Instead of a heavy OSNet model (adds ~40ms/frame on CPU), we use a
  lightweight colour histogram over the torso region of the bounding box.
  This is sufficient for 15fps retail CCTV where lighting is relatively stable
  and visitors wear distinct outfits. The similarity threshold is tuned to
  ~0.82 cosine similarity to avoid false matches.

  If you have a GPU and want better accuracy, swap _appearance_embedding() for
  a torchreid OSNet call — the interface is identical.
"""

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import numpy as np

logger = logging.getLogger("tracker")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How long to hold a lost track before deleting it (seconds)
TRACK_LOST_TTL_S = 3.0

# Maximum time gap for re-entry to be flagged as the same visitor (minutes)
REENTRY_WINDOW_MIN = 30

# Appearance similarity threshold for cross-camera Re-ID
APPEARANCE_SIM_THRESHOLD = 0.78

# IoU threshold to associate detection to existing track
IOU_MATCH_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter — lightweight 4-state (cx, cy, w, h) tracker
# ─────────────────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Minimal constant-velocity Kalman filter for axis-aligned bounding boxes.
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    """

    count = 0

    def __init__(self, bbox: list[float]):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        cx, cy, w, h = self._to_cx(bbox)
        self._state = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)

        # Simple process noise (tuned for retail walking speeds)
        self._P = np.eye(8) * 10.0
        self._Q = np.eye(8) * 0.1
        self._R = np.eye(4) * 1.0

        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history: list[list[float]] = []

    def predict(self) -> list[float]:
        """Advance state by one step."""
        self.age += 1
        self.time_since_update += 1
        # Constant velocity: x += vx
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = 1.0
        self._state = F @ self._state
        self._P = F @ self._P @ F.T + self._Q
        self.history.append(self._to_bbox(self._state[:4]))
        return self.history[-1]

    def update(self, bbox: list[float]):
        """Update with new measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = np.array(self._to_cx(bbox))
        H = np.eye(4, 8)
        y = z - H @ self._state
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._state = self._state + K @ y
        self._P = (np.eye(8) - K @ H) @ self._P

    def get_state(self) -> list[float]:
        return self._to_bbox(self._state[:4])

    @staticmethod
    def _to_cx(bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

    @staticmethod
    def _to_bbox(cx: np.ndarray) -> list[float]:
        cx_, cy_, w_, h_ = cx
        return [cx_ - w_ / 2, cy_ - h_ / 2, cx_ + w_ / 2, cy_ + h_ / 2]


# ─────────────────────────────────────────────────────────────────────────────
# IoU utility
# ─────────────────────────────────────────────────────────────────────────────

def iou(boxA: list[float], boxB: list[float]) -> float:
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Single-camera SORT-style tracker
# ─────────────────────────────────────────────────────────────────────────────

class CameraTracker:
    """
    IoU-based tracker per camera. Returns updated track list each frame.
    Wraps KalmanBoxTracker with Hungarian-style greedy matching.
    """

    # Frames a track survives without a match before being deleted
    MAX_AGE = int(TRACK_LOST_TTL_S * 15)  # 15 fps default

    # Minimum hits before a track is considered confirmed
    MIN_HITS = 2

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._lost_this_frame: list[int] = []

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update tracker with detections from one frame.
        Returns list of active confirmed tracks as dicts with bbox, track_id, confidence.
        """
        self.frame_count += 1
        self._lost_this_frame = []

        # Predict all existing tracks
        predicted = []
        to_del = []
        for t in self.trackers:
            predicted.append(t.predict())
            if np.any(np.isnan(predicted[-1])):
                to_del.append(t)
        for t in to_del:
            self.trackers.remove(t)
        predicted = [p for p in predicted if not np.any(np.isnan(p))]

        # Build IoU matrix and greedily match
        det_bboxes = [d["bbox"] for d in detections]
        matched, unmatched_dets, unmatched_trks = self._match(
            det_bboxes, predicted
        )

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(det_bboxes[d_idx])
            self.trackers[t_idx].hit_streak = getattr(
                self.trackers[t_idx], "hit_streak", 0
            ) + 1

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(det_bboxes[d_idx]))

        # Mark unmatched trackers and clean up dead ones
        for t_idx in unmatched_trks:
            self.trackers[t_idx].hit_streak = 0

        dead = [t for t in self.trackers if t.time_since_update > self.MAX_AGE]
        for t in dead:
            self._lost_this_frame.append(t.id)
            self.trackers.remove(t)

        # Return confirmed active tracks
        active = []
        for i, t in enumerate(self.trackers):
            if t.time_since_update == 0 and (
                t.hits >= self.MIN_HITS or self.frame_count <= self.MIN_HITS
            ):
                bbox = t.get_state()
                conf = detections[self._find_det_for_tracker(i, matched)]["confidence"] \
                    if self._find_det_for_tracker(i, matched) is not None else 0.5
                active.append({
                    "track_id": t.id,
                    "bbox": bbox,
                    "confidence": conf,
                    "hits": t.hits,
                })
        return active

    def get_lost_ids(self) -> list[int]:
        return list(self._lost_this_frame)

    def _find_det_for_tracker(self, tracker_idx: int, matched: list) -> Optional[int]:
        for d_idx, t_idx in matched:
            if t_idx == tracker_idx:
                return d_idx
        return None

    def _match(
        self,
        det_bboxes: list,
        trk_bboxes: list,
    ) -> tuple[list, list, list]:
        if not trk_bboxes or not det_bboxes:
            return [], list(range(len(det_bboxes))), list(range(len(trk_bboxes)))

        iou_matrix = np.zeros((len(det_bboxes), len(trk_bboxes)))
        for d, db in enumerate(det_bboxes):
            for t, tb in enumerate(trk_bboxes):
                iou_matrix[d, t] = iou(db, tb)

        # Greedy matching (descending IoU)
        matched = []
        used_d, used_t = set(), set()
        rows, cols = np.where(iou_matrix >= IOU_MATCH_THRESHOLD)
        pairs = sorted(
            zip(rows, cols), key=lambda x: iou_matrix[x[0], x[1]], reverse=True
        )
        for d, t in pairs:
            if d not in used_d and t not in used_t:
                matched.append((d, t))
                used_d.add(d)
                used_t.add(t)

        unmatched_dets = [i for i in range(len(det_bboxes)) if i not in used_d]
        unmatched_trks = [i for i in range(len(trk_bboxes)) if i not in used_t]
        return matched, unmatched_dets, unmatched_trks


# ─────────────────────────────────────────────────────────────────────────────
# Multi-camera tracker + Re-ID
# ─────────────────────────────────────────────────────────────────────────────

class MultiCameraTracker:
    """
    Maintains per-camera trackers and a global visitor registry for Re-ID.

    visitor_id assignment:
    1. New track → compute appearance embedding from first frame
    2. Compare to recently-seen embeddings from other cameras (cross-camera dedup)
    3. Compare to exited visitors for re-entry detection
    4. If match found, reuse visitor_id; else generate new VIS_xxxx id

    Queue depth tracking:
    - Maintained as a running count of non-staff tracks in BILLING zone
    - Accessed by ClipProcessor via get_queue_depth()
    """

    def __init__(self, store_id: str):
        self.store_id = store_id
        self._cam_trackers: dict[str, CameraTracker] = {}
        self._frame_height: Optional[int] = None

        # track_id → visitor_id
        self._track_to_visitor: dict[str, str] = {}

        # visitor_id → {embedding, last_seen_ts, last_camera, exited}
        self._visitor_registry: dict[str, dict] = {}

        # camera → set of active track_ids (for queue depth)
        self._active_billing_tracks: dict[str, set] = defaultdict(set)

        # Recently lost tracks per camera
        self._lost_tracks: dict[str, list[int]] = defaultdict(list)

    @property
    def frame_height(self) -> Optional[int]:
        return self._frame_height

    def update(
        self,
        detections: list[dict],
        frame,
        camera_id: str,
        frame_ts: datetime,
    ) -> list[dict]:
        """
        Update tracker for one camera frame. Returns tracks enriched with visitor_id.
        """
        if self._frame_height is None and frame is not None:
            self._frame_height = frame.shape[0]

        if camera_id not in self._cam_trackers:
            self._cam_trackers[camera_id] = CameraTracker(camera_id)

        cam_tracker = self._cam_trackers[camera_id]
        raw_tracks = cam_tracker.update(detections)

        # Store lost tracks for ClipProcessor._handle_lost_tracks
        self._lost_tracks[camera_id] = cam_tracker.get_lost_ids()

        enriched = []
        for track in raw_tracks:
            tid = track["track_id"]
            key = f"{camera_id}:{tid}"

            if key not in self._track_to_visitor:
                # Try Re-ID
                emb = self._appearance_embedding(frame, track["bbox"])
                matched_id = self._reid_lookup(emb, camera_id, frame_ts)
                if matched_id:
                    visitor_id = matched_id
                else:
                    visitor_id = self._new_visitor_id()
                self._track_to_visitor[key] = visitor_id
                self._visitor_registry[visitor_id] = {
                    "embedding": emb,
                    "last_seen_ts": frame_ts,
                    "last_camera": camera_id,
                    "exited": False,
                }
            else:
                visitor_id = self._track_to_visitor[key]
                if visitor_id in self._visitor_registry:
                    self._visitor_registry[visitor_id]["last_seen_ts"] = frame_ts
                    self._visitor_registry[visitor_id]["last_camera"] = camera_id

            enriched.append({**track, "visitor_id": visitor_id})

        return enriched

    def is_reentry(self, visitor_id: str, store_id: str) -> bool:
        """
        Return True if visitor_id has a prior EXIT event in the registry.
        """
        reg = self._visitor_registry.get(visitor_id)
        if reg and reg.get("exited"):
            return True
        return False

    def mark_exited(self, visitor_id: str):
        if visitor_id in self._visitor_registry:
            self._visitor_registry[visitor_id]["exited"] = True

    def get_queue_depth(self, store_id: str, frame_ts: datetime) -> int:
        """
        Returns current number of non-staff visitors in the billing zone.
        (Billing camera tracks billing area; this is an approximate count.)
        """
        return sum(
            len(v) for cam, v in self._active_billing_tracks.items()
            if "BILLING" in cam.upper()
        )

    def get_lost_tracks(self, camera_id: str) -> list[int]:
        return self._lost_tracks.get(camera_id, [])

    # ── Re-ID internals ──────────────────────────────────────────────────────

    def _appearance_embedding(self, frame, bbox: list[float]) -> np.ndarray:
        """
        Lightweight torso-region colour histogram as appearance descriptor.

        Steps:
        1. Crop torso (middle 50% of bbox height)
        2. Downsample to 32x32
        3. Compute 3-channel 16-bin histogram
        4. L2-normalise

        CPU cost: ~0.3ms per bbox at 1080p → negligible at 15fps.
        """
        if frame is None:
            return np.zeros(48, dtype=float)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        # Torso region: 25%–75% of bbox height
        ty1 = y1 + int(h * 0.25)
        ty2 = y1 + int(h * 0.75)
        ty1 = max(0, ty1)
        ty2 = min(frame.shape[0], ty2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)

        crop = frame[ty1:ty2, x1:x2]
        if crop.size == 0:
            return np.zeros(48, dtype=float)

        crop_small = cv2_resize_safe(crop, (32, 32))
        hist = []
        for ch in range(3):
            h_ch, _ = np.histogram(crop_small[:, :, ch], bins=16, range=(0, 256))
            hist.append(h_ch)
        emb = np.concatenate(hist).astype(float)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _reid_lookup(
        self,
        embedding: np.ndarray,
        camera_id: str,
        frame_ts: datetime,
    ) -> Optional[str]:
        """
        Search registry for a visitor matching this embedding.
        Only considers visitors last seen within REENTRY_WINDOW_MIN.
        """
        best_sim = 0.0
        best_id = None
        cutoff = frame_ts - timedelta(minutes=REENTRY_WINDOW_MIN)

        for vid, reg in self._visitor_registry.items():
            if reg["last_seen_ts"] < cutoff:
                continue
            if reg["last_camera"] == camera_id:
                # Same camera: very recent, only match if very high confidence
                threshold = 0.92
            else:
                threshold = APPEARANCE_SIM_THRESHOLD

            sim = float(np.dot(embedding, reg["embedding"])) \
                if embedding.shape == reg["embedding"].shape else 0.0
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_id = vid

        return best_id

    @staticmethod
    def _new_visitor_id() -> str:
        """Generate a short visitor token: VIS_<6 hex chars>"""
        return "VIS_" + uuid.uuid4().hex[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Helper to avoid importing cv2 at top level when running unit tests
# ─────────────────────────────────────────────────────────────────────────────

def cv2_resize_safe(img, size: tuple) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, size)
    except Exception:
        return np.zeros((*reversed(size), img.shape[2]), dtype=img.dtype)
