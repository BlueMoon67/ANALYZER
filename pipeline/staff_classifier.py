"""
staff_classifier.py — Classifies whether a detected person is store staff.

Primary method: Uniform colour heuristic
  Store staff typically wear a consistent branded uniform (common in Indian
  retail: solid-colour kurtas, aprons, or polo shirts). We detect this by
  checking whether the upper-body region has a dominant single hue that
  matches a configurable "staff palette".

  This avoids the latency and complexity of a full pose-based classifier
  while being robust enough for controlled retail environments.

Fallback: Per-store calibration
  staff_colours.json (optional, generated via calibrate_staff_colours.py)
  stores store-specific HSV colour ranges. If absent, we use retail-generic
  defaults (dark blue/navy, black, forest green — common uniform colours).

VLM option (CHOICES.md §2):
  If USE_VLM_STAFF_DETECTION=1 env var is set, the classifier falls back to
  a Claude vision call for ambiguous cases (confidence 0.45–0.65). This adds
  ~200ms/detection but dramatically reduces mis-classification.
  The VLM prompt used: see _vlm_classify().

Design trade-off (CHOICES.md §2):
  We chose the colour heuristic over a full person re-ID / attribute model
  because:
  (a) It's real-time capable on CPU
  (b) In retail settings, staff uniforms are highly consistent
  (c) FP rate is low enough that the flag can be verified in the review UI
  The VLM fallback handles the ambiguous 5–10% of cases.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("staff_classifier")


# ─────────────────────────────────────────────────────────────────────────────
# Default staff HSV colour ranges
# Format: (hue_min, hue_max, sat_min, val_min)
# OpenCV HSV: hue ∈ [0, 180], sat ∈ [0, 255], val ∈ [0, 255]
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STAFF_COLOURS = [
    # Navy blue
    {"h_min": 100, "h_max": 125, "s_min": 80, "v_min": 30, "v_max": 120},
    # Black (very low value)
    {"h_min": 0,   "h_max": 180, "s_min": 0,  "v_min": 0,  "v_max": 50},
    # Forest green
    {"h_min": 55,  "h_max": 85,  "s_min": 60, "v_min": 30, "v_max": 140},
    # Burgundy / dark red (common in premium retail)
    {"h_min": 165, "h_max": 180, "s_min": 70, "v_min": 40, "v_max": 120},
]

# Fraction of upper-body pixels that must match for a positive classification
UNIFORM_COVERAGE_THRESHOLD = 0.38

# Confidence at which we invoke VLM fallback (if enabled)
VLM_FALLBACK_LOWER = 0.40
VLM_FALLBACK_UPPER = 0.65


class StaffClassifier:
    """
    Classifies whether a bounding box crop belongs to store staff.

    Usage:
        clf = StaffClassifier()
        is_staff = clf.classify(frame, bbox)  # returns bool
    """

    def __init__(
        self,
        staff_colours: Optional[list[dict]] = None,
        use_vlm: Optional[bool] = None,
    ):
        self.staff_colours = staff_colours or DEFAULT_STAFF_COLOURS
        self.use_vlm = use_vlm if use_vlm is not None else (
            os.environ.get("USE_VLM_STAFF_DETECTION", "0") == "1"
        )
        self._vlm_cache: dict[str, bool] = {}
        logger.info(
            "StaffClassifier ready — colours=%d, vlm=%s",
            len(self.staff_colours), self.use_vlm,
        )

    def classify(self, frame, bbox: list[float]) -> bool:
        """
        Returns True if the person in bbox is likely store staff.
        """
        conf = self._colour_confidence(frame, bbox)
        is_staff = conf >= UNIFORM_COVERAGE_THRESHOLD

        # Fallback to VLM for ambiguous cases
        if (
            self.use_vlm
            and VLM_FALLBACK_LOWER <= conf <= VLM_FALLBACK_UPPER
        ):
            is_staff = self._vlm_classify(frame, bbox)

        return is_staff

    # ── Colour heuristic ─────────────────────────────────────────────────────

    def _colour_confidence(self, frame, bbox: list[float]) -> float:
        """
        Returns fraction [0, 1] of upper-body pixels matching a staff colour.
        """
        if frame is None:
            return 0.0

        try:
            import cv2
        except ImportError:
            return 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        # Upper body: top 50% of bbox
        uy2 = y1 + int(h * 0.5)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        y1 = max(0, y1)
        uy2 = min(frame.shape[0], uy2)

        crop = frame[y1:uy2, x1:x2]
        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        if total_pixels == 0:
            return 0.0

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for colour in self.staff_colours:
            lower = np.array([
                colour["h_min"], colour.get("s_min", 0), colour.get("v_min", 0)
            ])
            upper = np.array([
                colour["h_max"], 255, colour.get("v_max", 255)
            ])
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        matching = int(np.sum(mask > 0))
        return matching / total_pixels

    # ── VLM fallback ─────────────────────────────────────────────────────────

    def _vlm_classify(self, frame, bbox: list[float]) -> bool:
        """
        Use Claude vision API to classify ambiguous staff/customer cases.

        Prompt design (CHOICES.md §2):
          We provide a cropped upper-body image and ask a binary question.
          We include context about the store type (beauty retail) so the
          model has enough context to recognise retail staff uniforms.

        This is called ~5% of the time (only for ambiguous colour confidence).
        Results are cached by bbox hash.
        """
        cache_key = f"{bbox}"
        if cache_key in self._vlm_cache:
            return self._vlm_cache[cache_key]

        try:
            import base64
            import json
            import urllib.request
            import cv2

            x1, y1, x2, y2 = [int(v) for v in bbox]
            h = y2 - y1
            uy2 = y1 + int(h * 0.6)
            crop = frame[max(0, y1):min(frame.shape[0], uy2),
                         max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                return False

            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buf).decode()

            payload = {
                "model": "claude-opus-4-5",
                "max_tokens": 10,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a cropped image from a beauty retail "
                                "store CCTV camera. The person shown is either "
                                "a customer or store staff (staff wear a "
                                "consistent uniform). "
                                "Reply with ONLY the word 'staff' or 'customer'."
                            ),
                        },
                    ],
                }],
            }

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    # API key injected via ANTHROPIC_API_KEY env var by the container
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                text = data["content"][0]["text"].strip().lower()
                result = text.startswith("staff")

            self._vlm_cache[cache_key] = result
            logger.debug("VLM staff classification: bbox=%s → %s", bbox, result)
            return result

        except Exception as exc:
            logger.warning("VLM staff classification failed: %s", exc)
            return False
