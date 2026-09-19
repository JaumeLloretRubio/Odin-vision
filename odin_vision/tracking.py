"""Asociación IoU con velocidad constante; un tracker independiente por stream."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def candidate_pairs(tracks, detections):
    """IoU por categoría; los grupos pequeños evitan crear matrices temporales."""
    if len(tracks) * len(detections) < 256:
        return sorted(((overlap, t.id, i) for t in tracks for i, d in enumerate(detections)
                       if t.category == d.category and (overlap := iou(t.box, d.box)) >= 0.2), reverse=True)
    pairs = []
    grouped_tracks, grouped_indices = {}, {}
    for track in tracks:
        grouped_tracks.setdefault(track.category, []).append(track)
    for index, detection in enumerate(detections):
        grouped_indices.setdefault(detection.category, []).append(index)
    for category, group in grouped_tracks.items():
        indices = grouped_indices.get(category, [])
        if not indices:
            continue
        if len(group) * len(indices) < 64:
            pairs.extend((overlap, t.id, i) for t in group for i in indices
                         if (overlap := iou(t.box, detections[i].box)) >= 0.2)
            continue
        a = np.asarray([t.box for t in group], dtype=np.float64)
        b = np.asarray([detections[i].box for i in indices], dtype=np.float64)
        extent = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                            np.maximum(a[:, None, :2], b[None, :, :2]))
        intersection = extent[:, :, 0] * extent[:, :, 1]
        area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
        area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
        union = area_a[:, None] + area_b[None, :] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        rows, columns = np.nonzero(overlap >= 0.2)
        pairs.extend((float(overlap[row, column]), group[row].id, indices[column])
                     for row, column in zip(rows, columns))
    # Conservar el desempate histórico por ID e índice de detección.
    return sorted(pairs, reverse=True)


@dataclass
class Track:
    id: int
    box: tuple
    category: str
    confidence: float
    last_seen: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    anchor: tuple = ()
    label: str | None = None
    similarity: float | None = None
    description: str | None = None
    recognized_at: int = -10000
    revision: int = -1
    crops: deque = field(default_factory=lambda: deque(maxlen=8))
    last_event: float = 0
    announced: bool = False


class Tracker:
    def __init__(self, max_age=15):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, detections, frame):
        self.tracks = {k: t for k, t in self.tracks.items() if frame-t.last_seen <= self.max_age}
        tracks = list(self.tracks.values())
        if len(tracks) < 16:
            for track in tracks:
                track.box = tuple(np.clip(np.asarray(track.anchor) + track.velocity*(frame-track.last_seen), 0, 1))
        else:
            anchors = np.asarray([track.anchor for track in tracks])
            velocities = np.asarray([track.velocity for track in tracks])
            elapsed = np.asarray([frame-track.last_seen for track in tracks])[:, None]
            boxes = np.clip(anchors + velocities*elapsed, 0, 1)
            for track, box in zip(tracks, boxes):
                track.box = tuple(box)
        if detections is None:
            return tracks
        pairs = candidate_pairs(tracks, detections)
        matched_tracks, matched_detections = set(), set()
        for overlap, tid, index in pairs:
            if overlap < 0.2 or tid in matched_tracks or index in matched_detections:
                continue
            track, detection = self.tracks[tid], detections[index]
            elapsed = max(1, frame-track.last_seen)
            track.velocity = (np.asarray(detection.box)-np.asarray(track.anchor))/elapsed
            track.box = track.anchor = detection.box
            track.confidence = detection.confidence
            track.last_seen = frame
            matched_tracks.add(tid)
            matched_detections.add(index)
        for index, detection in enumerate(detections):
            if index not in matched_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, detection.box, detection.category, detection.confidence,
                                         frame, anchor=detection.box)
                matched_tracks.add(tid)
        # No publicar cajas desaparecidas en un frame que sí tiene detección.
        return [t for t in self.tracks.values() if t.id in matched_tracks]
