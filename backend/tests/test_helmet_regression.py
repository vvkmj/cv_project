from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.detection.pipeline import HelmetObservations, TrafficAnalysisPipeline
from backend.app.tracking.tracker import MultiObjectTracker
from backend.app.violations.association import RiderAssociator
from backend.app.violations.evidence_generator import EvidenceGenerator
from backend.app.violations.helmet_detector import head_match


def detection(label, track, box, frame=0):
    return dict(object_class=label, tracking_id=track, confidence=0.9, frame_number=frame,
                timestamp=frame / 30, bounding_box=dict(zip(('x1', 'y1', 'x2', 'y2'), box)))


def test_owner_is_unique_and_ambiguous_riders_are_not_assigned():
    left = detection('motorcycle', 1, [30, 70, 110, 140])
    right = detection('motorcycle', 2, [90, 70, 170, 140])
    driver = detection('person', 3, [42, 25, 68, 117])
    passenger = detection('person', 4, [60, 28, 85, 120])
    ambiguous = detection('person', 5, [85, 24, 115, 115])
    other_driver = detection('person', 6, [123, 25, 153, 116])
    for bikes in ([left, right], [right, left]):
        motorcycles, _ = RiderAssociator().associate_riders([*bikes, driver, passenger, ambiguous, other_driver], (180, 200))
        assert {bike['tracking_id']: {rider['person_tracking_id'] for rider in bike['associated_riders']}
                for bike in motorcycles} == {1: {3, 4}, 2: {6}}


def test_prediction_must_localize_to_the_head():
    person = {'x1': 10, 'y1': 10, 'x2': 40, 'y2': 38}
    neighbor = {'x1': 35, 'y1': 10, 'x2': 70, 'y2': 38}
    assert head_match((13, 11, 30, 30), person) > 0.8
    assert head_match((39, 11, 52, 30), person) == 0
    assert head_match((39, 11, 52, 30), neighbor) > 0.8
    assert head_match((14, 11, 33, 90), person) == 0


def test_consecutive_observations_reset_and_deduplicate():
    history = HelmetObservations()
    observations = [(7, 0, 'no_helmet'), (7, 1, 'unknown'), (7, 2, 'no_helmet'),
                    (7, 3, 'no_helmet'), (8, 4, 'no_helmet'), (8, 6, 'no_helmet'),
                    (8, 7, 'no_helmet'), (8, 8, 'no_helmet'), (8, 9, 'no_helmet')]
    assert [history.observe(3, bike, frame, frame / 30, status) for bike, frame, status in observations] == [
        False, False, False, False, False, False, False, True, False,
    ]
    history.prune(set())
    assert not history.reported and not history.streaks


def test_tracker_preserves_identity_through_brief_occlusion_and_bounds_history():
    tracker = MultiObjectTracker()
    for frame in range(3000):
        if frame == 2:
            tracker.update([])
        else:
            left = min(frame, 4) * 4
            tracks = tracker.update([detection('person', None, [left, 0, 20 + left, 40], frame)])
            assert tracks[0]['tracking_id'] == 1
    assert len(tracker.tracks[0].history) == 32


def test_original_and_crop_are_full_resolution_and_exclusive(tmp_path):
    generator = EvidenceGenerator(base_dir=str(tmp_path))
    original = np.full((120, 200, 3), 50, dtype=np.uint8)
    annotation = np.full_like(original, 110)
    crop = dict(x1=20, y1=30, x2=70, y2=80)
    paths = generator.save_evidence('VIOL-TEST', original, annotation, crop)
    assert cv2.imread(paths['evidence_path']).shape == original.shape
    assert cv2.imread(paths['crop_path']).shape == (50, 50, 3)
    initial = Path(paths['evidence_path']).read_bytes()
    with pytest.raises(FileExistsError):
        EvidenceGenerator(base_dir=str(tmp_path)).save_evidence('VIOL-TEST', annotation, annotation, crop)
    assert Path(paths['evidence_path']).read_bytes() == initial


def test_corrupt_video_is_not_successful(tmp_path):
    media = tmp_path / 'bad.mp4'
    media.write_bytes(b'invalid video')
    pipeline = TrafficAnalysisPipeline.__new__(TrafficAnalysisPipeline)
    with pytest.raises(ValueError, match='Could not open video'):
        pipeline.analyze_video(str(media), {}, 'BAD')


def test_video_captures_one_pending_incident_for_the_matching_passenger(tmp_path):
    media = tmp_path / 'two_riders.mp4'
    writer = cv2.VideoWriter(str(media), cv2.VideoWriter_fourcc(*'mp4v'), 30, (200, 180))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((180, 200, 3), 32, dtype=np.uint8))
    writer.release()

    def objects(_, frame_number, timestamp):
        return [detection('motorcycle', None, [30, 70, 110, 140], frame_number),
                detection('person', None, [42, 25, 68, 117], frame_number),
                detection('person', None, [75, 28, 100, 120], frame_number)]

    def known_head_status(image, heads, visible_heads):
        return [{'status': 'helmet' if head['x1'] < 50 else 'no_helmet', 'confidence': 0.9} for head in heads]

    pipeline = TrafficAnalysisPipeline.__new__(TrafficAnalysisPipeline)
    pipeline.evidence_generator = EvidenceGenerator(base_dir=str(tmp_path / 'evidence'))
    pipeline.rider_associator = RiderAssociator()
    pipeline.vehicle_detector = SimpleNamespace(detect=objects)
    pipeline.helmet_detector = SimpleNamespace(analyze_heads=known_head_status)
    persisted = []
    result = pipeline.analyze_video(str(media), {'enabled_violations': {'wrong_way': False}}, 'TEST',
                                    incident_callback=persisted.append, write_annotated_video=False)
    assert result['processed_frames'] == 8 and result['violations'] == persisted
    incident, = persisted
    assert incident['review_status'] == 'pending' and incident['frame_number'] == 2
    assert incident['tracking_id'] == 3 and incident['motorcycle_tracking_id'] == 1
    assert cv2.imread(incident['evidence_path']).shape == (180, 200, 3)
    assert cv2.imread(incident['crop_path']).shape == (112, 80, 3)
