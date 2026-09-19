import numpy as np
import pytest

from odin_vision import tracking
from odin_vision.vision import Detection


def scalar_pairs(tracks, detections):
    return sorted([(tracking.iou(t.box, d.box), t.id, i) for t in tracks
                   for i, d in enumerate(detections) if t.category == d.category], reverse=True)


@pytest.mark.parametrize("count,categories", [(0, 1), (1, 1), (15, 1), (16, 1), (100, 1), (100, 64)])
def test_candidates_equal_scalar_including_ties_and_boundary(count, categories):
    rng = np.random.default_rng(721)
    boxes = rng.uniform(0, .5, (count, 4))
    boxes[:, 2:] += boxes[:, :2]
    tracks = [tracking.Track(i+1, tuple(box), str(i % categories), .9, 1) for i, box in enumerate(boxes)]
    detections = [Detection(tuple(box), str(i % categories) if i % 3 else "car", .8)
                  for i, box in enumerate(boxes)]
    if count:
        tracks[0].box = (0., 0., 1., 1.)
        detections += [Detection((0., 0., edge, 1.), "0", .8)
                       for edge in [np.nextafter(.2, 0), .2, np.nextafter(.2, 1), .2]]
    expected = [pair for pair in scalar_pairs(tracks, detections) if pair[0] >= .2]
    assert tracking.candidate_pairs(tracks, detections) == expected


def test_vectorized_association_preserves_trajectories_and_expiry(monkeypatch):
    actual, reference = tracking.Tracker(), tracking.Tracker()
    rng = np.random.default_rng(972)
    for frame in range(1, 41):
        origin = rng.uniform(0, .7, (40, 2))
        boxes = np.concatenate([origin, origin + rng.uniform(.01, .3, (40, 2))], axis=1)
        detections = [Detection(tuple(box), ["person", "car"][i % 2], .9) for i, box in enumerate(boxes)]
        if frame % 5 == 0:
            detections = None
        elif frame % 7 == 0:
            detections = []
        observed = actual.update(detections, frame)
        with monkeypatch.context() as patch:
            patch.setattr(tracking, "candidate_pairs", scalar_pairs)
            expected = reference.update(detections, frame)
        assert [t.id for t in observed] == [t.id for t in expected]
        assert actual.next_id == reference.next_id
        assert actual.tracks.keys() == reference.tracks.keys()
        for tid, track in actual.tracks.items():
            other = reference.tracks[tid]
            assert track.box == other.box and track.anchor == other.anchor
            assert track.last_seen == other.last_seen and track.category == other.category
            np.testing.assert_array_equal(track.velocity, other.velocity)


def test_zero_area_and_fully_overlapping_tracks():
    boxes = [(0., 0., 1., 1.)] * 16 + [(0., 0., 0., 0.)]
    tracks = [tracking.Track(i+1, box, "person", .9, 1) for i, box in enumerate(boxes)]
    detections = [Detection(box, "person", .8) for box in boxes]
    assert tracking.candidate_pairs(tracks, detections) == [
        pair for pair in scalar_pairs(tracks, detections) if pair[0] >= .2]


@pytest.mark.parametrize("count", [0, 1, 15, 16, 100, 1500])
def test_prediction_batch_equals_individual_clipping(count):
    rng = np.random.default_rng(127)
    tracker = tracking.Tracker(max_age=15)
    for i in range(count):
        anchor = tuple(rng.uniform(0, 1, 4))
        tracker.tracks[i] = tracking.Track(i, anchor, "person", .9, 20-i % 16, anchor=anchor,
                                           velocity=rng.uniform(-.2, .2, 4))
    tracker.tracks[count] = tracking.Track(count, (0., 0., 1., 1.), "person", .9, 0,
                                           anchor=(0., 0., 1., 1.))
    expected = {tid: tuple(np.clip(np.asarray(t.anchor) + t.velocity*(20-t.last_seen), 0, 1))
                for tid, t in tracker.tracks.items() if 20-t.last_seen <= tracker.max_age}
    result = tracker.update(None, 20)
    assert {track.id: track.box for track in result} == expected
