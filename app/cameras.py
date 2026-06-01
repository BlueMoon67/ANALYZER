"""
cameras.py — Camera Distribution Manager for Store Intelligence API.

Endpoints:
  GET    /cameras                              — list all registered cameras
  GET    /cameras/{store_id}                   — cameras for a store
  POST   /cameras/register                     — register a new RTSP camera
  DELETE /cameras/{store_id}/{cam_id}          — remove a camera
  POST   /cameras/{store_id}/{cam_id}/activate — set as active stream
  POST   /cameras/{store_id}/{cam_id}/deactivate — deactivate
  GET    /cameras/{store_id}/active            — get current active camera for store
  GET    /cameras/{store_id}/snapshots         — latest frame metadata per camera
  POST   /cameras/{store_id}/{cam_id}/snapshot — push frame metadata (from pipeline)
  POST   /cameras/detect/camera               — scheduler posts here; spawns detector
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("cameras")

router = APIRouter(prefix="/cameras", tags=["cameras"])

# ─── In-memory store ──────────────────────────────────────────────────────────
_lock      = Lock()
_cameras:   dict[str, dict] = {}   # key: "{store_id}:{cam_id}"
_active:    dict[str, str]  = {}   # store_id → cam_id
_snapshots: dict[str, dict] = {}   # "{store_id}:{cam_id}" → snapshot meta


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key(store_id: str, cam_id: str) -> str:
    return f"{store_id}:{cam_id}"


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class CameraRegisterRequest(BaseModel):
    store_id:      str            = Field(..., min_length=1)
    camera_id:     str            = Field(..., min_length=1)
    rtsp_url:      str            = Field(..., min_length=1)
    label:         Optional[str]  = None
    camera_type:   str            = Field(default="floor")
    zone_ids:      list[str]      = Field(default_factory=list)
    auto_activate: bool           = False


class ActiveCameraResponse(BaseModel):
    store_id:    str
    camera_id:   Optional[str]
    rtsp_url:    Optional[str]
    label:       Optional[str]
    camera_type: Optional[str]
    is_active:   bool


class DetectCameraRequest(BaseModel):
    id:          str
    store_id:    str
    camera_id:   str
    rtsp_url:    str
    camera_type: str
    label:       Optional[str] = None
    zone_ids:    list[str]     = []


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("", response_model=dict)
def list_all_cameras():
    with _lock:
        records = _build_records()
    return {"cameras": records, "total": len(records), "fetched_at": _now()}


@router.get("/{store_id}", response_model=dict)
def list_store_cameras(store_id: str):
    with _lock:
        records = [r for r in _build_records() if r["store_id"] == store_id]
    return {
        "store_id":         store_id,
        "cameras":          records,
        "active_camera_id": _active.get(store_id),
        "total":            len(records),
        "fetched_at":       _now(),
    }


@router.post("/register", response_model=dict, status_code=201)
def register_camera(req: CameraRegisterRequest):
    k = _key(req.store_id, req.camera_id)
    with _lock:
        if k in _cameras:
            _cameras[k].update({
                "rtsp_url":    req.rtsp_url,
                "label":       req.label or req.camera_id,
                "camera_type": req.camera_type,
                "zone_ids":    req.zone_ids,
                "updated_at":  _now(),
            })
            action = "updated"
        else:
            _cameras[k] = {
                "id":            str(uuid.uuid4()),
                "store_id":      req.store_id,
                "camera_id":     req.camera_id,
                "rtsp_url":      req.rtsp_url,
                "label":         req.label or req.camera_id,
                "camera_type":   req.camera_type,
                "zone_ids":      req.zone_ids,
                "registered_at": _now(),
                "updated_at":    _now(),
                "last_event_ts": None,
            }
            action = "created"

        if req.auto_activate:
            _active[req.store_id] = req.camera_id

    logger.info("Camera %s %s: %s", k, action, req.rtsp_url[:60])
    return {
        "action":    action,
        "store_id":  req.store_id,
        "camera_id": req.camera_id,
        "active":    _active.get(req.store_id) == req.camera_id,
    }


@router.delete("/{store_id}/{cam_id}", response_model=dict)
def delete_camera(store_id: str, cam_id: str):
    k = _key(store_id, cam_id)
    with _lock:
        if k not in _cameras:
            raise HTTPException(404, f"Camera {k} not found")
        del _cameras[k]
        _snapshots.pop(k, None)
        if _active.get(store_id) == cam_id:
            remaining = [c for ck, c in _cameras.items() if c["store_id"] == store_id]
            _active[store_id] = remaining[0]["camera_id"] if remaining else None
    return {"deleted": cam_id, "store_id": store_id}


@router.post("/{store_id}/{cam_id}/activate", response_model=dict)
def activate_camera(store_id: str, cam_id: str):
    k = _key(store_id, cam_id)
    with _lock:
        if k not in _cameras:
            raise HTTPException(404, f"Camera {k} not found")
        prev = _active.get(store_id)
        _active[store_id] = cam_id
    logger.info("Store %s active camera: %s → %s", store_id, prev, cam_id)
    return {"store_id": store_id, "active_camera_id": cam_id, "previous": prev}


@router.post("/{store_id}/{cam_id}/deactivate", response_model=dict)
def deactivate_camera(store_id: str, cam_id: str):
    with _lock:
        _active.pop(store_id, None)
    return {"store_id": store_id, "active_camera_id": None}


@router.get("/{store_id}/active", response_model=ActiveCameraResponse)
def get_active_camera(store_id: str):
    with _lock:
        cam_id = _active.get(store_id)
        if not cam_id:
            return ActiveCameraResponse(
                store_id=store_id, camera_id=None, rtsp_url=None,
                label=None, camera_type=None, is_active=False,
            )
        k   = _key(store_id, cam_id)
        rec = _cameras.get(k)
        if not rec:
            return ActiveCameraResponse(
                store_id=store_id, camera_id=cam_id, rtsp_url=None,
                label=None, camera_type=None, is_active=False,
            )
    return ActiveCameraResponse(
        store_id=store_id,
        camera_id=cam_id,
        rtsp_url=rec["rtsp_url"],
        label=rec.get("label", cam_id),
        camera_type=rec.get("camera_type", "floor"),
        is_active=True,
    )


@router.post("/{store_id}/{cam_id}/snapshot", response_model=dict)
def update_snapshot(store_id: str, cam_id: str, body: dict):
    """Called by the pipeline to push latest frame metadata."""
    k = _key(store_id, cam_id)
    with _lock:
        if k in _cameras:
            _cameras[k]["last_event_ts"] = _now()
        _snapshots[k] = {
            "store_id":     store_id,
            "camera_id":    cam_id,
            "frame_ts":     _now(),
            "fps":          body.get("fps"),
            "resolution":   body.get("resolution"),
            "person_count": body.get("person_count"),
            "status":       body.get("status", "streaming"),
        }
    return {"accepted": True}


@router.get("/{store_id}/snapshots", response_model=dict)
def get_snapshots(store_id: str):
    with _lock:
        snaps = [v for k, v in _snapshots.items() if v["store_id"] == store_id]
    return {"store_id": store_id, "snapshots": snaps, "fetched_at": _now()}


# ─────────────────────────────────────────────────────────────────────────────
# POST /cameras/detect/camera
# Scheduler POSTs camera assignments here every 0.2 s.
# BackgroundTasks runs process_camera() in FastAPI's thread-pool.
# The per-camera guard inside process_camera() drops duplicate ticks.
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/detect/camera", response_model=dict)
def detect_camera(req: DetectCameraRequest, bg: BackgroundTasks):
    # Import here to avoid circular imports at module load time
    from .detect import process_camera

    # Touch last_event_ts so stream_health stays OK while stream is running
    k = _key(req.store_id, req.camera_id)
    with _lock:
        if k in _cameras:
            _cameras[k]["last_event_ts"] = _now()

    bg.add_task(process_camera, req)

    logger.debug(
        "[DETECT] Queued background task  store=%s  camera=%s",
        req.store_id, req.camera_id,
    )
    return {"accepted": True, "camera_id": req.camera_id, "store_id": req.store_id}


# ─── Seed from store_layout.json ─────────────────────────────────────────────

def seed_from_layout(layout: dict):
    """Pre-populate camera registrations from store_layout.json at startup."""
    for store_id, store in layout.get("stores", {}).items():
        for cam_id, cam_data in store.get("cameras", {}).items():
            k = _key(store_id, cam_id)
            if k not in _cameras:
                _cameras[k] = {
                    "id":            str(uuid.uuid4()),
                    "store_id":      store_id,
                    "camera_id":     cam_id,
                    "rtsp_url":      "",
                    "label":         f"{cam_id} ({cam_data.get('type', '?')})",
                    "camera_type":   cam_data.get("type", "floor"),
                    "zone_ids":      [z["zone_id"] for z in cam_data.get("zones", [])],
                    "registered_at": _now(),
                    "updated_at":    _now(),
                    "last_event_ts": None,
                }
    logger.info("Seeded %d cameras from store_layout.json", len(_cameras))


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_records() -> list[dict]:
    now_ts = time.time()
    out    = []
    for k, c in _cameras.items():
        snap    = _snapshots.get(k, {})
        last_ts = snap.get("frame_ts") or c.get("last_event_ts")
        if last_ts:
            try:
                age    = now_ts - datetime.strptime(
                    last_ts, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc).timestamp()
                health = "OK" if age < 60 else ("STALE" if age < 300 else "UNKNOWN")
            except Exception:
                health = "UNKNOWN"
        else:
            health = "UNKNOWN"

        out.append({
            **c,
            "active":        _active.get(c["store_id"]) == c["camera_id"],
            "stream_health": health,
            "snapshot":      snap or None,
        })
    return out