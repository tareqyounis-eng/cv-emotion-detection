"""One frame in, detected faces and smoothed estimates out."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .tracking import FaceTracker, Track
from .vision import ExpressionModel, FaceDetector, align_face, quality_reason


@dataclass
class Result:
    faces: list[Track]
    detected: int
    inference_ms: float


class Engine:
    def __init__(self, paths, max_faces=4, detector_size=640, threshold=0.55,
                 min_face=64, checkpoint=None):
        # Large thread pools can cost more than these little networks save.
        cv2.setNumThreads(2)
        self.detector = FaceDetector(paths["face"], detector_size)
        if checkpoint:
            from .training import CheckpointModel
            self.expression = CheckpointModel(checkpoint)
        else:
            self.expression = ExpressionModel(paths["expression"])
        self.tracker = FaceTracker()
        self.max_faces = max_faces
        self.threshold = threshold
        self.min_face = min_face

    def process(self, frame, now=None, still=False):
        started = time.perf_counter()
        now = time.monotonic() if now is None else now
        faces = self.detector.detect(frame)
        selected = faces[:self.max_faces]
        tracks = self.tracker.update(selected, now)
        for face, track in zip(selected, tracks):
            track.reason = quality_reason(frame, face, self.min_face)
            scores = None
            if track.reason is None:
                try:
                    crop = align_face(frame, face[4:14])
                    scores = self.expression.predict(crop)
                except (ValueError, np.linalg.LinAlgError):
                    track.reason = "Face the camera"
            track.observe(scores, now, threshold=self.threshold)
            if still and scores is not None:
                # A single photo has no temporal evidence to smooth.
                order = np.argsort(scores)
                if scores[order[-1]] >= self.threshold and scores[order[-1]] - scores[order[-2]] >= 0.12:
                    track.label_index = int(order[-1])
        return Result(tracks, len(faces), (time.perf_counter() - started) * 1000)
