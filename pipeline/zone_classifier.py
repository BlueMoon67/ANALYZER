"""
zone_classifier.py — Maps detected bounding box positions to named store zones.

Zones are defined in store_layout.json as polygons or bounding rectangles
in the camera's coordinate space. This module performs point-in-polygon
lookup for each detected centroid.

Design (CHOICES.md §3):
  Rule-based zone classification was preferred over a VLM because:
  (a) Store layout is known at deploy time from store_layout.json
  (b) Point-in-polygon is O(n) and deterministic — no latency spikes
  (c) VLM zone classification would require a per-frame image → too slow
  The trade-off is that zones must be hand-calibrated per store, which
  is acceptable given the fixed camera positions.

store_layout.json format expected (per camera):
  {
    "stores": {
      "STORE_BLR_002": {
        "cameras": {
          "CAM_FLOOR_01": {
            "zones": [
              {
                "zone_id": "SKINCARE",
                "sku_zone": "MOISTURISER",
                "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
              }
            ]
          }
        }
      }
    }
  }

If the layout lacks polygon data (common in early testing), we fall back
to a uniform grid partition of the frame — floor/ceiling divided into
equal-width columns assigned to sequentially-named zones.
"""

import logging
from typing import Optional

logger = logging.getLogger("zone_classifier")


class ZoneClassifier:
    """
    Looks up which named zone a bounding box centroid falls within.

    Usage:
        clf = ZoneClassifier()
        zone_id = clf.classify(bbox, camera_id, store_layout)
    """

    def __init__(self):
        # Cache parsed zone polygons per camera to avoid re-parsing each frame
        self._zone_cache: dict[str, list[dict]] = {}

    def classify(
        self,
        bbox: list[float],
        camera_id: str,
        store_layout: dict,
    ) -> Optional[str]:
        """
        Returns the zone_id that contains the bbox centroid, or None.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        zones = self._get_zones(camera_id, store_layout)

        for zone in zones:
            if self._point_in_polygon(cx, cy, zone["polygon"]):
                return zone["zone_id"]

        return None

    def get_sku_zone(
        self,
        zone_id: str,
        camera_id: str,
        store_layout: dict,
    ) -> Optional[str]:
        """
        Returns the sku_zone label for a zone_id (used in event metadata).
        """
        zones = self._get_zones(camera_id, store_layout)
        for z in zones:
            if z["zone_id"] == zone_id:
                return z.get("sku_zone")
        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_zones(self, camera_id: str, store_layout: dict) -> list[dict]:
        if camera_id in self._zone_cache:
            return self._zone_cache[camera_id]

        zones = self._extract_zones(camera_id, store_layout)
        self._zone_cache[camera_id] = zones
        return zones

    def _extract_zones(
        self, camera_id: str, store_layout: dict
    ) -> list[dict]:
        """
        Parse zone definitions from store_layout.json.
        Falls back to grid partition if layout lacks polygon data.
        """
        # Traverse store_layout structure
        # Support two possible layouts:
        #   (a) { stores: { STORE_ID: { cameras: { CAM_ID: { zones: [...] } } } } }
        #   (b) Flat: { cameras: { CAM_ID: { zones: [...] } } }
        #   (c) Flat list: { zones: [...], camera_coverage: { CAM_ID: [zone_ids] } }

        try:
            # Try nested store → cameras → zones
            for store_id, store_data in store_layout.get("stores", {}).items():
                cameras = store_data.get("cameras", {})
                if camera_id in cameras:
                    raw_zones = cameras[camera_id].get("zones", [])
                    return self._normalise_zones(raw_zones)

            # Try flat cameras
            cameras = store_layout.get("cameras", {})
            if camera_id in cameras:
                raw_zones = cameras[camera_id].get("zones", [])
                return self._normalise_zones(raw_zones)

            # Try flat zone list with camera_coverage mapping
            coverage = store_layout.get("camera_coverage", {})
            if camera_id in coverage:
                zone_ids = coverage[camera_id]
                all_zones = store_layout.get("zones", [])
                raw_zones = [z for z in all_zones if z.get("zone_id") in zone_ids]
                return self._normalise_zones(raw_zones)

        except Exception as exc:
            logger.warning(
                "Failed to parse zones for camera %s: %s. Using grid fallback.",
                camera_id, exc,
            )

        return self._grid_fallback(camera_id)

    @staticmethod
    def _normalise_zones(raw_zones: list[dict]) -> list[dict]:
        """
        Ensure every zone has a 'polygon' key as list of [x,y] pairs.
        Converts bbox format [x1, y1, x2, y2] to polygon if necessary.
        """
        normalised = []
        for z in raw_zones:
            zone = dict(z)
            if "polygon" not in zone and "bbox" in zone:
                x1, y1, x2, y2 = zone["bbox"]
                zone["polygon"] = [
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ]
            if "polygon" not in zone:
                logger.debug("Zone %s has no polygon; skipping", zone.get("zone_id"))
                continue
            normalised.append(zone)
        return normalised

    @staticmethod
    def _grid_fallback(camera_id: str, frame_w: int = 1920, frame_h: int = 1080) -> list[dict]:
        """
        Divide frame into a 3×2 grid and assign generic zone names.
        Used when store_layout.json lacks polygon data for this camera.

        Grid layout:
          LEFT_FRONT | CENTER_FRONT | RIGHT_FRONT
          LEFT_BACK  | CENTER_BACK  | RIGHT_BACK
        """
        logger.info("Using grid fallback for camera %s", camera_id)
        col_w = frame_w // 3
        row_h = frame_h // 2
        names = [
            "LEFT_FRONT", "CENTER_FRONT", "RIGHT_FRONT",
            "LEFT_BACK",  "CENTER_BACK",  "RIGHT_BACK",
        ]
        zones = []
        for i, name in enumerate(names):
            col = i % 3
            row = i // 3
            x1 = col * col_w
            y1 = row * row_h
            x2 = x1 + col_w
            y2 = y1 + row_h
            zones.append({
                "zone_id": name,
                "sku_zone": name,
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            })
        return zones

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
        """
        Ray-casting algorithm for point-in-polygon test.
        polygon: list of [x, y] pairs (any length ≥ 3)
        """
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside
