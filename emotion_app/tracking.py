"""Short-lived face tracks. These are screen IDs, not people's identities."""

from dataclasses import dataclass
import math
import numpy as np

from .vision import LABELS


def iou(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[:2] + a[2:4], b[:2] + b[2:4])
    intersection = float(np.prod(np.maximum(0, right - left)))
    union = float(a[2] * a[3] + b[2] * b[3] - intersection)
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    last_seen: float
    scores: np.ndarray | None = None
    last_prediction: float | None = None
    label_index: int | None = None
    candidate: int | None = None
    candidate_since: float = 0.0
    samples: int = 0
    reason: str | None = None

    @property
    def label(self):
        if self.reason:
            return self.reason
        return LABELS[self.label_index] if self.label_index is not None else "Uncertain"

    def observe(self, scores, now, threshold=0.55, margin=0.12, smoothing=0.22):
        if scores is None:
            # Don't keep showing a happy label after the face turns away.
            self.scores = None
            self.last_prediction = None
            self.label_index = self.candidate = None
            self.samples = 0
            return
        if self.scores is None:
            self.scores = scores.copy()
        else:
            elapsed = max(0.001, now - self.last_prediction)
            weight = 1 - math.exp(-elapsed / smoothing)
            self.scores = (1 - weight) * self.scores + weight * scores
        self.last_prediction = now
        self.samples += 1
        order = np.argsort(self.scores)
        best, runner_up = int(order[-1]), int(order[-2])
        confident = self.scores[best] >= threshold and self.scores[best] - self.scores[runner_up] >= margin
        if not confident:
            self.label_index = self.candidate = None
            return
        if self.candidate != best:
            self.candidate, self.candidate_since = best, now
        if best != self.label_index:
            self.label_index = None
        # A brief hold avoids swapping labels on every blink. It uses time,
        # so the behavior stays similar on a slow laptop and a fast one.
        if self.samples >= 3 and now - self.candidate_since >= 0.18:
            self.label_index = best


class FaceTracker:
    def __init__(self, timeout=0.45):
        self.timeout = timeout
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks.clear()

    def update(self, faces, now):
        self.tracks = [track for track in self.tracks if now - track.last_seen <= self.timeout]
        pairs = []
        for ti, track in enumerate(self.tracks):
            for fi, face in enumerate(faces):
                overlap = iou(track.box, face[:4])
                old_center = track.box[:2] + track.box[2:4] / 2
                new_center = face[:2] + face[2:4] / 2
                distance = np.linalg.norm(old_center - new_center) / max(1, np.linalg.norm(track.box[2:4]))
                ratio = face[2] * face[3] / max(1, track.box[2] * track.box[3])
                if 0.45 < ratio < 2.2 and (overlap > 0.15 or distance < 0.30):
                    pairs.append((overlap - 0.3 * distance, ti, fi))
        used_tracks, matched = set(), {}
        for _, ti, fi in sorted(pairs, reverse=True):
            if ti not in used_tracks and fi not in matched:
                used_tracks.add(ti)
                matched[fi] = self.tracks[ti]
        visible = []
        for fi, face in enumerate(faces):
            track = matched.get(fi)
            if track is None:
                track = Track(self.next_id, face[:4].copy(), now)
                self.next_id += 1
                self.tracks.append(track)
            elif now - track.last_seen > 0.2:
                track.observe(None, now)
            track.box, track.last_seen = face[:4].copy(), now
            visible.append(track)
        # Missing tracks can be matched briefly, but are never drawn as ghosts.
        return visible
