import pytest
import numpy as np
from backend.app.tracking.tracker import MultiObjectTracker, Track, calculate_iou
from backend.app.violations.association import RiderAssociator
from backend.app.violations.helmet_detector import HelmetDetector
from backend.app.violations.passenger_counter import PassengerCounter
from backend.app.violations.wrong_way import WrongWayDetector
from backend.app.violations.evidence_generator import EvidenceGenerator

def test_iou_calculation():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [0.0, 0.0, 10.0, 10.0]
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = [10.0, 10.0, 20.0, 20.0]
    assert calculate_iou(boxA, boxC) == 0.0

    boxD = [5.0, 0.0, 15.0, 10.0]
    assert pytest.approx(calculate_iou(boxA, boxD), 0.01) == 0.333

def test_multiobject_tracker():
    tracker = MultiObjectTracker(iou_threshold=0.20, max_age=5)

    det_frame1 = [{
        "frame_number": 0,
        "timestamp": 0.0,
        "object_class": "car",
        "confidence": 0.9,
        "bounding_box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    }]
    tracked1 = tracker.update(det_frame1)
    assert len(tracked1) == 1
    assert tracked1[0]["tracking_id"] == 1

    # Frame 2: slightly moved car
    det_frame2 = [{
        "frame_number": 1,
        "timestamp": 0.05,
        "object_class": "car",
        "confidence": 0.92,
        "bounding_box": {"x1": 105, "y1": 105, "x2": 205, "y2": 205}
    }]
    tracked2 = tracker.update(det_frame2)
    assert len(tracked2) == 1
    # Tracking ID must be preserved
    assert tracked2[0]["tracking_id"] == 1
    track = tracker.get_track_by_id(1)
    assert track is not None
    assert len(track.history) == 2

def test_rider_associator():
    associator = RiderAssociator()
    # Motorcycle bounding box
    moto_box = [200.0, 300.0, 300.0, 450.0]

    # Valid rider sitting on motorcycle
    rider_box = [210.0, 240.0, 290.0, 380.0]
    is_rider, score = associator.is_person_riding_motorcycle(rider_box, moto_box)
    assert is_rider is True
    assert score > 0.3

    # Distant pedestrian on the sidewalk
    pedestrian_box = [500.0, 250.0, 560.0, 390.0]
    is_ped, score_ped = associator.is_person_riding_motorcycle(pedestrian_box, moto_box)
    assert is_ped is False
    assert score_ped == 0.0

def test_head_region_extraction():
    associator = RiderAssociator()
    person_box = [100.0, 200.0, 200.0, 500.0] # w=100, h=300
    head_box = associator.extract_head_region(person_box, (720, 1280))
    hx1, hy1, hx2, hy2 = head_box
    assert hy1 == 200
    assert hy2 == 200 + int(300 * 0.30)
    assert hx1 > 100
    assert hx2 < 200

def test_helmet_detector_missing_weights_handling():
    """Verifies that missing weights are handled safely and transparently."""
    detector = HelmetDetector(weights_path="non_existent_weights.pt")
    assert detector.is_available is False

    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    head_box = {"x1": 50, "y1": 50, "x2": 100, "y2": 100}

    result = detector.analyze_head_region(dummy_frame, head_box)
    assert result["status"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["is_available"] is False
    assert "not found" in result["explanation"]

def test_passenger_counter():
    counter = PassengerCounter(buffer_size=5, min_consecutive_frames=3)

    m_track_id = 42
    riders_3 = [{"person_tracking_id": 1}, {"person_tracking_id": 2}, {"person_tracking_id": 3}]

    # Frame 1: 3 riders (first observation, not enough consecutive frames yet)
    viol1 = counter.update(m_track_id, riders_3, passenger_limit=2)
    assert viol1 is None

    # Frame 2: 3 riders
    viol2 = counter.update(m_track_id, riders_3, passenger_limit=2)
    assert viol2 is None

    # Frame 3: 3 riders (reaches min_consecutive_frames=3) -> VIOLATION!
    viol3 = counter.update(m_track_id, riders_3, passenger_limit=2)
    assert viol3 is not None
    assert viol3["violation_type"] == "multiple_riders"
    assert viol3["occupant_count"] == 3

    # Frame 4: deduplication should prevent duplicate alert for same track
    viol4 = counter.update(m_track_id, riders_3, passenger_limit=2)
    assert viol4 is None

def test_wrong_way_detector():
    detector = WrongWayDetector(
        min_movement_pixels=20.0,
        consecutive_frames_required=3,
        cosine_threshold=-0.35,
        window_size=6
    )

    # Expected direction: Southbound (downwards, angle 90 deg, dx=0, dy=1)
    camera_dir = {"angle_deg": 90.0, "dx": 0.0, "dy": 1.0}

    # Simulate a legal car moving downwards
    legal_det = {
        "frame_number": 0,
        "timestamp": 0.0,
        "object_class": "car",
        "confidence": 0.9,
        "bounding_box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    }
    legal_track = Track(1, legal_det)
    for f in range(1, 8):
        legal_track.update({
            "frame_number": f,
            "timestamp": f * 0.05,
            "confidence": 0.9,
            "bounding_box": {"x1": 100, "y1": 100 + f * 15, "x2": 200, "y2": 200 + f * 15}
        })
    viol_legal = detector.evaluate_track(legal_track, camera_dir)
    assert viol_legal is None

    # Simulate a wrong-way car moving upwards against traffic (y decreasing)
    wrong_det = {
        "frame_number": 0,
        "timestamp": 0.0,
        "object_class": "car",
        "confidence": 0.9,
        "bounding_box": {"x1": 400, "y1": 500, "x2": 500, "y2": 600}
    }
    wrong_track = Track(2, wrong_det)
    # Feed multiple frames of upward motion
    for f in range(1, 10):
        wrong_track.update({
            "frame_number": f,
            "timestamp": f * 0.05,
            "confidence": 0.9,
            "bounding_box": {"x1": 400, "y1": 500 - f * 18, "x2": 500, "y2": 600 - f * 18}
        })
        viol_wrong = detector.evaluate_track(wrong_track, camera_dir)
        if viol_wrong is not None:
            break

    assert viol_wrong is not None
    assert viol_wrong["violation_type"] == "wrong_way"
    assert viol_wrong["tracking_id"] == 2
    assert viol_wrong["movement_vector"]["cosine_similarity"] < -0.8

