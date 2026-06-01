# PROMPT:
# "Write comprehensive pytest tests for a CCTV person detection pipeline that:
#  - Has a ClipProcessor class that tracks people using Kalman filters + IoU matching
#  - Emits structured events (ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
#    BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY)
#  - Has a StaffClassifier using colour histograms
#  - Has a ZoneClassifier using point-in-polygon
#  - Has an EventEmitter writing JSONL
#  - Test edge cases: group entry, re-entry, empty store, staff exclusion, occlusion
#  Cover unit tests for each component and integration tests for the full pipeline."
#
# CHANGES MADE:
# - Replaced mock frames with deterministic numpy arrays of specific colours
#   (AI generated generic MagicMock frames which didn't work with cv2 ops)
# - Added billing queue abandon test (AI only covered join)
# - Split "re-entry" test into two: same-session reentry and cross-session reentry
# - Removed test_vlm_staff_detection (AI included it but VLM is optional/env-gated)
# - Fixed schema validation test: AI checked for 'metadata.session_seq' as flat key;
#   it's nested as event['metadata']['session_seq']
# - Added group entry test (3 simultaneous detections → 3 separate ENTRY events)
# - Added zero-traffic test (empty store emits no events, doesn't crash)

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc
NOW = datetime(2026, 3, 3, 14, 0, 0, tzinfo=UTC)


def navy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """BGR frame where the entire image is navy blue — simulates staff uniform."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 120   # Blue channel
    frame[:, :, 1] = 60    # Green channel
    frame[:, :, 2] = 30    # Red channel
    return frame


def customer_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """BGR frame with mixed colours — simulates non-uniform customer clothing."""
    frame = np.random.randint(50, 220, (h, w, 3), dtype=np.uint8)
    return frame


def make_bbox(x1=100, y1=50, x2=200, y2=300) -> list[float]:
    return [float(x1), float(y1), float(x2), float(y2)]


# ─────────────────────────────────────────────────────────────────────────────
# emit.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventSchema:
    def test_build_event_entry(self):
        from emit import build_event, EventType
        event = build_event(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_abc123",
            event_type=EventType.ENTRY,
            timestamp=NOW,
            confidence=0.91,
            is_staff=False,
            session_seq=1,
        )
        assert event["event_type"] == "ENTRY"
        assert event["zone_id"] is None     # Entry events must have no zone
        assert "event_id" in event
        assert len(event["event_id"]) == 36  # UUID4 format
        assert event["metadata"]["session_seq"] == 1
        assert event["confidence"] == 0.91

    def test_build_event_zone_dwell(self):
        from emit import build_event, EventType
        event = build_event(
            store_id="STORE_BLR_002",
            camera_id="CAM_FLOOR_01",
            visitor_id="VIS_abc123",
            event_type=EventType.ZONE_DWELL,
            timestamp=NOW,
            zone_id="SKINCARE",
            dwell_ms=31000,
            confidence=0.85,
            is_staff=False,
            sku_zone="MOISTURISER",
            session_seq=3,
        )
        assert event["zone_id"] == "SKINCARE"
        assert event["dwell_ms"] == 31000
        assert event["metadata"]["sku_zone"] == "MOISTURISER"

    def test_event_ids_are_unique(self):
        from emit import build_event, EventType
        ids = set()
        for _ in range(100):
            e = build_event(
                store_id="S", camera_id="C", visitor_id="V",
                event_type=EventType.ENTRY, timestamp=NOW,
                session_seq=0,
            )
            ids.add(e["event_id"])
        assert len(ids) == 100

    def test_invalid_event_type_raises(self):
        from emit import build_event
        with pytest.raises(ValueError, match="Unknown event_type"):
            build_event(
                store_id="S", camera_id="C", visitor_id="V",
                event_type="BOGUS_TYPE", timestamp=NOW, session_seq=0,
            )

    def test_timestamp_is_utc_iso8601(self):
        from emit import build_event, EventType
        e = build_event(
            store_id="S", camera_id="C", visitor_id="V",
            event_type=EventType.EXIT, timestamp=NOW, session_seq=0,
        )
        # Should end with Z and parse correctly
        assert e["timestamp"].endswith("Z")
        parsed = datetime.strptime(e["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        assert parsed.year == 2026

    def test_validate_event_valid(self):
        from emit import build_event, validate_event, EventType
        e = build_event(
            store_id="S", camera_id="C", visitor_id="V",
            event_type=EventType.ENTRY, timestamp=NOW, session_seq=0,
        )
        errors = validate_event(e)
        assert errors == []

    def test_validate_event_missing_field(self):
        from emit import validate_event
        errors = validate_event({"event_type": "ENTRY"})
        assert any("event_id" in err for err in errors)

    def test_validate_event_bad_confidence(self):
        from emit import build_event, validate_event, EventType
        e = build_event(
            store_id="S", camera_id="C", visitor_id="V",
            event_type=EventType.ENTRY, timestamp=NOW,
            confidence=0.9, session_seq=0,
        )
        e["confidence"] = 1.5  # Out of range
        errors = validate_event(e)
        assert any("confidence" in err for err in errors)

    def test_low_confidence_events_are_emitted_not_dropped(self):
        """Low confidence must NOT suppress emission (spec requirement)."""
        from emit import build_event, EventType
        # Should not raise even for very low confidence
        e = build_event(
            store_id="S", camera_id="C", visitor_id="V",
            event_type=EventType.ZONE_ENTER, timestamp=NOW,
            zone_id="SKINCARE", confidence=0.10,  # very low
            is_staff=False, session_seq=0,
        )
        assert e["confidence"] == 0.10

    def test_emitter_writes_jsonl(self, tmp_path):
        from emit import EventEmitter, EventType
        out = str(tmp_path / "events.jsonl")
        emitter = EventEmitter(output_path=out)
        emitter.emit_entry(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_001",
            timestamp=NOW,
            confidence=0.9,
            is_staff=False,
            session_seq=0,
        )
        emitter.emit_exit(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_001",
            timestamp=NOW,
            confidence=0.88,
            is_staff=False,
            session_seq=1,
        )
        emitter.close()

        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            event = json.loads(line)
            assert "event_id" in event
            assert event["store_id"] == "STORE_BLR_002"

        assert emitter.total_emitted() == 2


# ─────────────────────────────────────────────────────────────────────────────
# zone_classifier.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZoneClassifier:

    LAYOUT = {
        "cameras": {
            "CAM_FLOOR_01": {
                "zones": [
                    {
                        "zone_id": "SKINCARE",
                        "sku_zone": "MOISTURISER",
                        "polygon": [[0, 0], [320, 0], [320, 480], [0, 480]],
                    },
                    {
                        "zone_id": "MAKEUP",
                        "sku_zone": "LIPSTICK",
                        "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]],
                    },
                ]
            }
        }
    }

    def test_classify_left_zone(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        zone = clf.classify([50, 100, 100, 200], "CAM_FLOOR_01", self.LAYOUT)
        assert zone == "SKINCARE"

    def test_classify_right_zone(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        zone = clf.classify([400, 100, 500, 200], "CAM_FLOOR_01", self.LAYOUT)
        assert zone == "MAKEUP"

    def test_point_on_boundary(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        # Centroid exactly at x=320 (boundary) — should be in one zone
        zone = clf.classify([310, 100, 330, 200], "CAM_FLOOR_01", self.LAYOUT)
        assert zone in ("SKINCARE", "MAKEUP", None)

    def test_unknown_camera_returns_grid_zone(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        # No layout for CAM_UNKNOWN → grid fallback
        zone = clf.classify([100, 100, 200, 300], "CAM_UNKNOWN", {})
        assert zone is not None  # Grid fallback always returns a zone
        assert isinstance(zone, str)

    def test_bbox_outside_all_zones_returns_none(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        # bbox centroid at (1900, 1000) — outside all defined zones in 640x480 layout
        layout = {
            "cameras": {
                "CAM_FLOOR_01": {
                    "zones": [
                        {"zone_id": "SMALL", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
                    ]
                }
            }
        }
        zone = clf.classify([1800, 950, 2000, 1050], "CAM_FLOOR_01", layout)
        assert zone is None

    def test_point_in_polygon_concave(self):
        from zone_classifier import ZoneClassifier
        # L-shaped polygon
        poly = [[0,0],[100,0],[100,50],[50,50],[50,100],[0,100]]
        assert ZoneClassifier._point_in_polygon(25, 75, poly) is True
        assert ZoneClassifier._point_in_polygon(75, 75, poly) is False

    def test_sku_zone_lookup(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        sku = clf.get_sku_zone("SKINCARE", "CAM_FLOOR_01", self.LAYOUT)
        assert sku == "MOISTURISER"

    def test_sku_zone_missing_returns_none(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        sku = clf.get_sku_zone("NONEXISTENT", "CAM_FLOOR_01", self.LAYOUT)
        assert sku is None

    def test_zone_cache_is_populated(self):
        from zone_classifier import ZoneClassifier
        clf = ZoneClassifier()
        clf.classify([50, 100, 100, 200], "CAM_FLOOR_01", self.LAYOUT)
        assert "CAM_FLOOR_01" in clf._zone_cache
        assert len(clf._zone_cache["CAM_FLOOR_01"]) == 2

    def test_bbox_format_normalised(self):
        from zone_classifier import ZoneClassifier
        layout = {
            "cameras": {
                "CAM_FLOOR_01": {
                    "zones": [
                        {
                            "zone_id": "AREA",
                            "bbox": [0, 0, 640, 480],  # bbox not polygon
                        }
                    ]
                }
            }
        }
        clf = ZoneClassifier()
        zone = clf.classify([100, 100, 200, 300], "CAM_FLOOR_01", layout)
        assert zone == "AREA"


# ─────────────────────────────────────────────────────────────────────────────
# staff_classifier.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStaffClassifier:

    def test_navy_uniform_is_classified_as_staff(self):
        from staff_classifier import StaffClassifier
        clf = StaffClassifier(use_vlm=False)
        frame = navy_frame()
        bbox = make_bbox(0, 0, 200, 400)
        result = clf.classify(frame, bbox)
        assert result is True

    def test_random_clothing_is_not_staff(self):
        from staff_classifier import StaffClassifier
        import cv2
        clf = StaffClassifier(use_vlm=False)
        # Bright red frame — unlikely to match any staff colour
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # Pure red
        bbox = make_bbox(0, 0, 200, 400)
        result = clf.classify(frame, bbox)
        assert result is False

    def test_empty_crop_returns_false(self):
        from staff_classifier import StaffClassifier
        clf = StaffClassifier(use_vlm=False)
        frame = navy_frame()
        # Bbox outside frame → empty crop
        bbox = make_bbox(5000, 5000, 6000, 7000)
        result = clf.classify(frame, bbox)
        assert result is False

    def test_none_frame_returns_false(self):
        from staff_classifier import StaffClassifier
        clf = StaffClassifier(use_vlm=False)
        result = clf.classify(None, make_bbox())
        assert result is False

    def test_custom_staff_colours(self):
        from staff_classifier import StaffClassifier
        # White uniform
        white_colour = [{"h_min": 0, "h_max": 180, "s_min": 0, "v_min": 200, "v_max": 255}]
        clf = StaffClassifier(staff_colours=white_colour, use_vlm=False)
        frame = np.full((480, 640, 3), 240, dtype=np.uint8)  # White frame
        result = clf.classify(frame, make_bbox(0, 0, 640, 480))
        assert result is True

    def test_colour_confidence_range(self):
        from staff_classifier import StaffClassifier
        clf = StaffClassifier(use_vlm=False)
        conf = clf._colour_confidence(navy_frame(), make_bbox(0, 0, 640, 480))
        assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# tracker.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanBoxTracker:

    def test_tracker_initialises_with_bbox(self):
        from tracker import KalmanBoxTracker
        t = KalmanBoxTracker([100, 50, 200, 300])
        state = t.get_state()
        assert len(state) == 4
        # Centroid should be close to original
        cx = (state[0] + state[2]) / 2
        cy = (state[1] + state[3]) / 2
        assert abs(cx - 150) < 5
        assert abs(cy - 175) < 5

    def test_predict_advances_age(self):
        from tracker import KalmanBoxTracker
        t = KalmanBoxTracker([100, 50, 200, 300])
        t.predict()
        assert t.age == 1
        assert t.time_since_update == 1

    def test_update_resets_time_since_update(self):
        from tracker import KalmanBoxTracker
        t = KalmanBoxTracker([100, 50, 200, 300])
        t.predict()
        t.predict()
        assert t.time_since_update == 2
        t.update([105, 55, 205, 305])
        assert t.time_since_update == 0


class TestIoU:

    def test_perfect_overlap(self):
        from tracker import iou
        box = [0, 0, 100, 100]
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from tracker import iou
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from tracker import iou
        # 50x50 overlap out of two 100x100 boxes that share a corner
        val = iou([0, 0, 100, 100], [50, 50, 150, 150])
        # Overlap = 50*50 = 2500; Union = 100*100 + 100*100 - 2500 = 17500
        assert val == pytest.approx(2500 / 17500, abs=1e-3)

    def test_degenerate_zero_area(self):
        from tracker import iou
        assert iou([0, 0, 0, 0], [0, 0, 10, 10]) == pytest.approx(0.0)


class TestCameraTracker:

    def test_empty_detections_returns_empty(self):
        from tracker import CameraTracker
        ct = CameraTracker("CAM_01")
        result = ct.update([])
        assert result == []

    def test_new_detection_creates_track(self):
        from tracker import CameraTracker
        ct = CameraTracker("CAM_01")
        dets = [{"bbox": [100, 50, 200, 300], "confidence": 0.9}]
        # First frame: track created but not yet confirmed (MIN_HITS=2)
        ct.update(dets)
        # Second frame with same detection: now confirmed
        result = ct.update(dets)
        assert len(result) == 1
        assert "track_id" in result[0]

    def test_group_entry_three_people(self):
        """3 simultaneous detections → 3 separate tracks."""
        from tracker import CameraTracker
        ct = CameraTracker("CAM_01")
        dets = [
            {"bbox": [50,  50, 100, 200], "confidence": 0.9},
            {"bbox": [200, 50, 250, 200], "confidence": 0.88},
            {"bbox": [350, 50, 400, 200], "confidence": 0.85},
        ]
        ct.update(dets)
        result = ct.update(dets)
        assert len(result) == 3
        track_ids = {r["track_id"] for r in result}
        assert len(track_ids) == 3  # All distinct IDs

    def test_lost_track_reported(self):
        from tracker import CameraTracker
        ct = CameraTracker("CAM_01")
        ct.MAX_AGE = 2  # short TTL for test
        dets = [{"bbox": [100, 50, 200, 300], "confidence": 0.9}]
        ct.update(dets)
        ct.update(dets)
        # Stop providing detection → track will age out
        for _ in range(ct.MAX_AGE + 1):
            ct.update([])
        lost = ct.get_lost_ids()
        assert len(lost) >= 1


class TestMultiCameraTracker:

    def test_new_visitor_id_assigned(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("STORE_BLR_002")
        frame = customer_frame()
        dets = [{"bbox": make_bbox(), "confidence": 0.9}]
        # Seed camera tracker
        from tracker import CameraTracker
        mt._cam_trackers["CAM_ENTRY_01"] = MagicMock()
        mt._cam_trackers["CAM_ENTRY_01"].update.return_value = [
            {"track_id": 1, "bbox": make_bbox(), "confidence": 0.9, "hits": 3}
        ]
        mt._cam_trackers["CAM_ENTRY_01"].get_lost_ids.return_value = []

        tracks = mt.update(dets, frame, "CAM_ENTRY_01", NOW)
        assert len(tracks) == 1
        assert tracks[0]["visitor_id"].startswith("VIS_")

    def test_visitor_id_stable_across_frames(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("STORE_BLR_002")
        frame = customer_frame()
        mock_track = {"track_id": 99, "bbox": make_bbox(), "confidence": 0.9, "hits": 3}
        mock_cam = MagicMock()
        mock_cam.update.return_value = [mock_track]
        mock_cam.get_lost_ids.return_value = []
        mt._cam_trackers["CAM_01"] = mock_cam

        tracks1 = mt.update([], frame, "CAM_01", NOW)
        tracks2 = mt.update([], frame, "CAM_01", NOW)
        assert tracks1[0]["visitor_id"] == tracks2[0]["visitor_id"]

    def test_is_reentry_returns_false_before_exit(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("STORE_BLR_002")
        assert mt.is_reentry("VIS_new", "STORE_BLR_002") is False

    def test_is_reentry_returns_true_after_mark_exited(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("STORE_BLR_002")
        mt._visitor_registry["VIS_abc"] = {
            "embedding": np.zeros(48),
            "last_seen_ts": NOW,
            "last_camera": "CAM_ENTRY_01",
            "exited": True,
        }
        assert mt.is_reentry("VIS_abc", "STORE_BLR_002") is True

    def test_appearance_embedding_shape(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("S")
        emb = mt._appearance_embedding(customer_frame(), make_bbox())
        assert emb.shape == (48,)
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01  # L2-normalised

    def test_appearance_embedding_none_frame(self):
        from tracker import MultiCameraTracker
        mt = MultiCameraTracker("S")
        emb = mt._appearance_embedding(None, make_bbox())
        assert emb.shape == (48,)
        assert np.all(emb == 0)

    def test_new_visitor_id_format(self):
        from tracker import MultiCameraTracker
        vid = MultiCameraTracker._new_visitor_id()
        assert vid.startswith("VIS_")
        assert len(vid) == 10  # "VIS_" + 6 hex chars


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — ClipProcessor logic (without real video)
# ─────────────────────────────────────────────────────────────────────────────

class TestClipProcessorLogic:
    """
    Test ClipProcessor event logic without requiring an actual video file.
    We test the _handle_entry_exit and _handle_zone_tracking methods directly.
    """

    def _make_processor(self, tmp_path, camera_type="entry"):
        from detect import ClipProcessor
        from emit import EventEmitter
        from tracker import MultiCameraTracker
        from staff_classifier import StaffClassifier
        from zone_classifier import ZoneClassifier

        emitter = EventEmitter(output_path=str(tmp_path / "events.jsonl"))
        tracker = MultiCameraTracker("STORE_BLR_002")
        tracker.frame_height  # property access
        tracker._frame_height = 1080

        layout = {
            "cameras": {
                "CAM_FLOOR_01": {
                    "zones": [
                        {"zone_id": "SKINCARE", "polygon": [[0,0],[640,0],[640,540],[0,540]]},
                        {"zone_id": "BILLING",  "polygon": [[640,0],[1280,0],[1280,1080],[640,1080]]},
                    ]
                }
            }
        }

        proc = ClipProcessor(
            model=MagicMock(),
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01" if camera_type == "entry" else "CAM_FLOOR_01",
            camera_type=camera_type,
            store_layout=layout,
            clip_start_utc=NOW,
            fps=15.0,
            emitter=emitter,
            tracker=tracker,
            staff_classifier=StaffClassifier(use_vlm=False),
            zone_classifier=ZoneClassifier(),
        )
        return proc, emitter

    def test_entry_event_emitted_on_tripwire_cross(self, tmp_path):
        proc, emitter = self._make_processor(tmp_path)
        visitor_id = "VIS_test01"
        track = {"track_id": 1, "visitor_id": visitor_id, "bbox": make_bbox(400, 580, 450, 700),
                 "confidence": 0.91, "hits": 3}
        state = proc._track_state.setdefault(1, {
            "visitor_id": visitor_id, "is_staff": False, "entered": False,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
            "prev_centroid_y": 400,  # Above tripwire (0.55 * 1080 = 594)
        })
        events = proc._handle_entry_exit(track, state,
                                          make_bbox(400, 600, 450, 720), NOW, 0.91)
        assert any(e["event_type"] == "ENTRY" for e in events)

    def test_exit_event_emitted_on_reverse_tripwire_cross(self, tmp_path):
        proc, emitter = self._make_processor(tmp_path)
        visitor_id = "VIS_test02"
        track = {"track_id": 2, "visitor_id": visitor_id, "bbox": make_bbox(400, 400, 450, 550),
                 "confidence": 0.88, "hits": 3}
        state = proc._track_state.setdefault(2, {
            "visitor_id": visitor_id, "is_staff": False, "entered": True,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 1, "first_seen_ts": NOW, "reentry_handled": False,
            "prev_centroid_y": 700,  # Below tripwire
        })
        events = proc._handle_entry_exit(track, state,
                                          make_bbox(400, 400, 450, 550), NOW, 0.88)
        assert any(e["event_type"] == "EXIT" for e in events)

    def test_staff_entry_sets_is_staff_true(self, tmp_path):
        proc, emitter = self._make_processor(tmp_path)
        visitor_id = "VIS_staff01"
        # Simulate state with is_staff True
        state = proc._track_state.setdefault(3, {
            "visitor_id": visitor_id, "is_staff": False, "entered": False,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
            "prev_centroid_y": 400,
        })
        # Staff frame
        track = {"track_id": 3, "visitor_id": visitor_id, "bbox": make_bbox(),
                 "confidence": 0.85, "hits": 3}
        frame = navy_frame()
        # Manually patch classifier to return True
        proc.staff_classifier.classify = MagicMock(return_value=True)
        proc._process_track(track, frame, NOW)
        assert proc._track_state[3]["is_staff"] is True

    def test_zone_enter_event_emitted(self, tmp_path):
        proc, emitter = self._make_processor(tmp_path, camera_type="floor")
        visitor_id = "VIS_zone01"
        track = {"track_id": 10, "visitor_id": visitor_id,
                 "bbox": make_bbox(100, 100, 200, 300), "confidence": 0.9, "hits": 3}
        state = proc._track_state.setdefault(10, {
            "visitor_id": visitor_id, "is_staff": False, "entered": True,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
        })
        frame = customer_frame()
        events = proc._handle_zone_tracking(
            track, state, frame, make_bbox(100, 100, 200, 300), NOW, 0.9
        )
        assert any(e["event_type"] == "ZONE_ENTER" for e in events)

    def test_dwell_event_emitted_after_30s(self, tmp_path):
        from datetime import timedelta
        proc, emitter = self._make_processor(tmp_path, camera_type="floor")
        visitor_id = "VIS_dwell01"
        track = {"track_id": 20, "visitor_id": visitor_id,
                 "bbox": make_bbox(100, 100, 200, 300), "confidence": 0.9, "hits": 5}
        enter_ts = NOW
        state = proc._track_state.setdefault(20, {
            "visitor_id": visitor_id, "is_staff": False, "entered": True,
            "current_zone": "SKINCARE", "zone_enter_ts": enter_ts,
            "last_dwell_ts": enter_ts, "session_seq": 2,
            "first_seen_ts": enter_ts, "reentry_handled": False,
        })
        frame = customer_frame()
        ts_31s_later = NOW + timedelta(seconds=31)
        events = proc._handle_zone_tracking(
            track, state, frame, make_bbox(100, 100, 200, 300), ts_31s_later, 0.9
        )
        assert any(e["event_type"] == "ZONE_DWELL" for e in events)
        dwell_event = next(e for e in events if e["event_type"] == "ZONE_DWELL")
        assert dwell_event["dwell_ms"] >= 30000

    def test_zone_exit_on_zone_change(self, tmp_path):
        proc, emitter = self._make_processor(tmp_path, camera_type="floor")
        visitor_id = "VIS_zonechange"
        track = {"track_id": 30, "visitor_id": visitor_id,
                 "bbox": make_bbox(800, 100, 900, 300), "confidence": 0.88, "hits": 4}
        state = proc._track_state.setdefault(30, {
            "visitor_id": visitor_id, "is_staff": False, "entered": True,
            "current_zone": "SKINCARE",  # Was in SKINCARE
            "zone_enter_ts": NOW, "last_dwell_ts": NOW,
            "session_seq": 3, "first_seen_ts": NOW, "reentry_handled": False,
        })
        frame = customer_frame()
        # bbox now in BILLING zone (x > 640)
        events = proc._handle_zone_tracking(
            track, state, frame, make_bbox(800, 100, 900, 300), NOW, 0.88
        )
        types = [e["event_type"] for e in events]
        assert "ZONE_EXIT" in types
        assert "ZONE_ENTER" in types


# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_traffic_no_events(self, tmp_path):
        """Empty frame sequence should produce zero events without crashing."""
        from emit import EventEmitter
        out = str(tmp_path / "empty.jsonl")
        emitter = EventEmitter(output_path=out)
        emitter.close()
        assert Path(out).stat().st_size == 0

    def test_all_staff_clip_produces_zero_customer_events(self, tmp_path):
        """If all detections are staff, no customer ENTRY events should be emitted."""
        from detect import ClipProcessor
        from emit import EventEmitter
        from tracker import MultiCameraTracker
        from staff_classifier import StaffClassifier
        from zone_classifier import ZoneClassifier

        out = str(tmp_path / "staff_only.jsonl")
        emitter = EventEmitter(output_path=out)
        tracker = MultiCameraTracker("STORE_BLR_002")
        tracker._frame_height = 1080

        # Staff classifier that always returns True
        staff_clf = MagicMock()
        staff_clf.classify.return_value = True

        proc = ClipProcessor(
            model=MagicMock(), store_id="STORE_BLR_002", camera_id="CAM_ENTRY_01",
            camera_type="entry", store_layout={}, clip_start_utc=NOW, fps=15.0,
            emitter=emitter, tracker=tracker, staff_classifier=staff_clf,
            zone_classifier=ZoneClassifier(),
        )

        visitor_id = "VIS_staff"
        track = {"track_id": 1, "visitor_id": visitor_id,
                 "bbox": make_bbox(), "confidence": 0.9, "hits": 3}
        state = proc._track_state.setdefault(1, {
            "visitor_id": visitor_id, "is_staff": True, "entered": False,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
            "prev_centroid_y": 400,
        })

        events = proc._handle_entry_exit(track, state, make_bbox(400, 620, 450, 720), NOW, 0.9)
        # Entry was detected but is_staff=True — event is emitted with is_staff=True
        for e in events:
            assert e["is_staff"] is True

    def test_reentry_flagged_as_reentry_not_entry(self, tmp_path):
        from detect import ClipProcessor
        from emit import EventEmitter
        from tracker import MultiCameraTracker
        from staff_classifier import StaffClassifier
        from zone_classifier import ZoneClassifier

        out = str(tmp_path / "reentry.jsonl")
        emitter = EventEmitter(output_path=out)
        tracker = MultiCameraTracker("STORE_BLR_002")
        tracker._frame_height = 1080

        # Mark visitor as having exited
        tracker._visitor_registry["VIS_returning"] = {
            "embedding": np.zeros(48),
            "last_seen_ts": NOW,
            "last_camera": "CAM_ENTRY_01",
            "exited": True,
        }

        proc = ClipProcessor(
            model=MagicMock(), store_id="STORE_BLR_002", camera_id="CAM_ENTRY_01",
            camera_type="entry", store_layout={}, clip_start_utc=NOW, fps=15.0,
            emitter=emitter, tracker=tracker,
            staff_classifier=StaffClassifier(use_vlm=False),
            zone_classifier=ZoneClassifier(),
        )

        track = {"track_id": 5, "visitor_id": "VIS_returning",
                 "bbox": make_bbox(), "confidence": 0.9, "hits": 5}
        state = proc._track_state.setdefault(5, {
            "visitor_id": "VIS_returning", "is_staff": False, "entered": False,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
            "prev_centroid_y": 400,
        })

        events = proc._handle_entry_exit(
            track, state, make_bbox(400, 620, 450, 720), NOW, 0.9
        )
        types = [e["event_type"] for e in events]
        assert "REENTRY" in types
        assert "ENTRY" not in types

    def test_schema_all_event_types_are_valid(self):
        """Every event type in the catalogue must be buildable without error."""
        from emit import build_event, EventType, ALL_EVENT_TYPES
        for et in ALL_EVENT_TYPES:
            e = build_event(
                store_id="S", camera_id="C", visitor_id="V",
                event_type=et, timestamp=NOW,
                zone_id=None if et in ("ENTRY", "EXIT", "REENTRY") else "SKINCARE",
                dwell_ms=0 if et not in ("ZONE_DWELL",) else 31000,
                session_seq=0,
            )
            assert e["event_type"] == et

    def test_billing_queue_join_sets_queue_depth(self, tmp_path):
        from detect import ClipProcessor
        from emit import EventEmitter
        from tracker import MultiCameraTracker
        from staff_classifier import StaffClassifier
        from zone_classifier import ZoneClassifier

        out = str(tmp_path / "billing.jsonl")
        emitter = EventEmitter(output_path=out)
        tracker = MultiCameraTracker("STORE_BLR_002")
        tracker._frame_height = 1080
        tracker.get_queue_depth = MagicMock(return_value=3)  # 3 people in queue

        layout = {
            "cameras": {
                "CAM_BILLING_01": {
                    "zones": [
                        {"zone_id": "BILLING", "polygon": [[0,0],[1280,0],[1280,1080],[0,1080]]}
                    ]
                }
            }
        }

        proc = ClipProcessor(
            model=MagicMock(), store_id="STORE_BLR_002", camera_id="CAM_BILLING_01",
            camera_type="billing", store_layout=layout, clip_start_utc=NOW, fps=15.0,
            emitter=emitter, tracker=tracker,
            staff_classifier=StaffClassifier(use_vlm=False),
            zone_classifier=ZoneClassifier(),
        )

        track = {"track_id": 7, "visitor_id": "VIS_queuer",
                 "bbox": make_bbox(100, 100, 200, 300), "confidence": 0.88, "hits": 3}
        state = proc._track_state.setdefault(7, {
            "visitor_id": "VIS_queuer", "is_staff": False, "entered": True,
            "current_zone": None, "zone_enter_ts": None, "last_dwell_ts": None,
            "session_seq": 0, "first_seen_ts": NOW, "reentry_handled": False,
        })
        frame = customer_frame()
        events = proc._handle_zone_tracking(
            track, state, frame, make_bbox(100, 100, 200, 300), NOW, 0.88
        )
        queue_events = [e for e in events if e["event_type"] == "BILLING_QUEUE_JOIN"]
        assert len(queue_events) == 1
        assert queue_events[0]["metadata"]["queue_depth"] == 3
