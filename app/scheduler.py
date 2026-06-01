"""
scheduler.py — Camera scheduler for Store Intelligence API.

Runs two daemon threads at startup (called from main.py lifespan):

  fetch_cameras()   — polls GET /cameras every 10 s and caches active cameras
  scheduler_loop()  — iterates the cache and POSTs each camera to
                       POST /cameras/detect/camera every 0.2 s

The detect/camera endpoint (cameras.py) receives each assignment and spins up
a BackgroundTask that calls detect.process_camera() — which opens the RTSP
stream, runs YOLOv8, and ingests events.

The 0.2 s slice per camera is intentional: it means the scheduler visits each
camera frequently enough to restart a crashed task, without hammering the API.
The duplicate-guard in detect.py (_active_cameras set) ensures only one stream
session runs per camera at any time.
"""

import logging
import threading
import time

import requests

logger = logging.getLogger("scheduler")

API_BASE               = "http://127.0.0.1:8000"
CAMERA_REFRESH_SECONDS = 10     # how often to re-fetch the camera list
CAMERA_SLICE_SECONDS   = 0.2    # pause between posting each camera to detect endpoint

_cameras_cache: list[dict] = []
_cache_lock = threading.Lock()


def fetch_cameras() -> None:
    """Background thread: keep _cameras_cache in sync with the API."""
    global _cameras_cache

    while True:
        try:
            resp = requests.get(f"{API_BASE}/cameras", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            active = [
                cam for cam in data.get("cameras", [])
                if cam.get("active", False) and cam.get("rtsp_url", "")
            ]

            with _cache_lock:
                _cameras_cache = active

            logger.info("[Scheduler] Loaded %d active camera(s)", len(active))

        except Exception as exc:
            logger.warning("[Scheduler] Camera fetch failed: %s", exc)

        time.sleep(CAMERA_REFRESH_SECONDS)


def scheduler_loop() -> None:
    """Background thread: POST each active camera to the detect endpoint."""

    while True:
        with _cache_lock:
            cameras = list(_cameras_cache)

        if not cameras:
            time.sleep(1)
            continue

        for cam in cameras:
            # Skip cameras with no RTSP URL registered yet
            if not cam.get("rtsp_url"):
                continue

            payload = {
                "id":          cam["id"],
                "store_id":    cam["store_id"],
                "camera_id":   cam["camera_id"],
                "rtsp_url":    cam["rtsp_url"],
                "camera_type": cam.get("camera_type", "floor"),
                "label":       cam.get("label"),
                "zone_ids":    cam.get("zone_ids", []),
            }

            try:
                resp = requests.post(
                    f"{API_BASE}/cameras/detect/camera",
                    json=payload,
                    timeout=5,
                )
                logger.debug(
                    "[Scheduler] → %s %s  HTTP %s",
                    payload["camera_id"],
                    payload["rtsp_url"],
                    resp.status_code,
                )
            except Exception as exc:
                logger.warning(
                    "[Scheduler] POST failed for %s: %s",
                    payload["camera_id"], exc,
                )

            time.sleep(CAMERA_SLICE_SECONDS)


def start_scheduler() -> None:
    """
    Launch both threads as daemons so they die when the main process exits.
    Called once from main.py lifespan at startup.
    """
    threading.Thread(target=fetch_cameras,   daemon=True, name="cam-fetcher").start()
    threading.Thread(target=scheduler_loop,  daemon=True, name="cam-scheduler").start()
    logger.info("[Scheduler] Started (refresh=%ds slice=%.1fs)",
                CAMERA_REFRESH_SECONDS, CAMERA_SLICE_SECONDS)