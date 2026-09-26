import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from emotion_app.camera import LatestCamera
from emotion_app.display import card_position, render
from emotion_app.models import ensure_models
from emotion_app.pipeline import Engine, Result
from emotion_app.tracking import FaceTracker, Track
from emotion_app.vision import REFERENCE_POINTS, align_face, probabilities, quality_reason


def face(x=30, y=30, width=120):
    points = REFERENCE_POINTS * (width / 112) + (x, y)
    return np.r_[x, y, width, width, points.reshape(-1), 0.99].astype(np.float32)


def confident(index):
    scores = np.full(7, 0.025, dtype=np.float32)
    scores[index] = 0.85
    return scores


class TrackingTests(unittest.TestCase):
    def test_reordered_detections_keep_their_own_scores(self):
        tracker = FaceTracker()
        left, right = tracker.update([face(10), face(350)], 1.0)
        left.observe(confident(3), 1)
        right.observe(confident(5), 1)
        moved = tracker.update([face(345), face(15)], 1.04)
        self.assertEqual([t.id for t in moved], [right.id, left.id])
        self.assertEqual([int(t.scores.argmax()) for t in moved], [5, 3])

    def test_missing_face_is_not_drawn_and_expired_face_gets_new_id(self):
        tracker = FaceTracker()
        old = tracker.update([face()], 0)[0].id
        self.assertEqual(tracker.update([], 0.1), [])
        new = tracker.update([face()], 1)[0].id
        self.assertNotEqual(old, new)

    def test_brief_loss_clears_old_expression_before_reacquiring(self):
        tracker = FaceTracker()
        track = tracker.update([face()], 0)[0]
        track.observe(confident(3), 0)
        again = tracker.update([face()], 0.3)[0]
        self.assertEqual(again.id, track.id)
        self.assertIsNone(again.scores)

    def test_stable_evidence_needed_and_quality_loss_clears_label(self):
        track = Track(1, face()[:4], 0)
        for now in (0, 0.1):
            track.observe(confident(3), now)
            self.assertEqual(track.label, "Uncertain")
        track.observe(confident(3), 0.2)
        self.assertEqual(track.label, "Happy")
        track.reason = "More light needed"
        track.observe(None, 0.3)
        self.assertEqual(track.label, "More light needed")
        self.assertIsNone(track.scores)

    def test_uniform_scores_never_force_an_emotion(self):
        track = Track(1, face()[:4], 0)
        for now in np.arange(0, 2, 0.1):
            track.observe(np.ones(7) / 7, now)
        self.assertEqual(track.label, "Uncertain")

    def test_tracks_are_not_reused_for_a_distant_new_person(self):
        tracker = FaceTracker()
        first = tracker.update([face(10)], 0)[0]
        second = tracker.update([face(500)], 0.1)[0]
        self.assertNotEqual(first.id, second.id)

    def test_one_track_cannot_match_two_faces(self):
        tracker = FaceTracker()
        tracker.update([face()], 0)
        tracks = tracker.update([face(31), face(34)], 0.1)
        self.assertEqual(len({t.id for t in tracks}), 2)


class VisionTests(unittest.TestCase):
    def test_softmax_is_stable_with_large_logits(self):
        scores = probabilities([10000, 9999, 9998, 9997, 9996, 9995, 9994])
        self.assertAlmostEqual(float(scores.sum()), 1, places=6)
        self.assertEqual(int(scores.argmax()), 0)
        with self.assertRaises(RuntimeError):
            probabilities([float("nan")] * 7)

    def test_alignment_recovers_known_translation_and_scale(self):
        image = np.random.default_rng(8).integers(0, 255, (112, 112, 3), dtype=np.uint8)
        np.testing.assert_array_equal(align_face(image, REFERENCE_POINTS), image)
        large = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        shifted = np.zeros((280, 280, 3), np.uint8)
        shifted[20:244, 30:254] = large
        aligned = align_face(shifted, REFERENCE_POINTS * 2 + [30, 20])
        self.assertLess(float(np.abs(aligned.astype(float) - image).mean()), 1)

    def test_degenerate_landmarks_are_rejected(self):
        with self.assertRaises(ValueError):
            align_face(np.zeros((112, 112, 3), np.uint8), np.ones((5, 2)))

    def test_quality_gates_dark_tiny_and_clipped_faces(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        self.assertEqual(quality_reason(frame, face()), "More light needed")
        self.assertEqual(quality_reason(frame, face(width=30)), "Move closer")
        self.assertEqual(quality_reason(frame, face(x=-5)), "Keep face in view")

    def test_clear_textured_front_face_passes_quality_gate(self):
        frame = np.random.default_rng(4).integers(60, 220, (300, 400, 3), dtype=np.uint8)
        self.assertIsNone(quality_reason(frame, face()))


class ModelDownloadTests(unittest.TestCase):
    def test_bad_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("emotion_app.models.urlopen", return_value=io.BytesIO(b"not a model")):
                with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                    ensure_models(folder, download=True)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_offline_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "download-models"):
                ensure_models(folder)


class DisplayTests(unittest.TestCase):
    def test_edge_cards_stay_inside_camera(self):
        for box in ((0, 0, 150, 150), (850, 450, 960, 540), (390, 120, 570, 440)):
            x1, y1, x2, y2 = card_position(box, [], [box])
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 960)
            self.assertLessEqual(y2, 540)

    def test_no_face_and_portrait_frames_render_without_mutating_source(self):
        frame = np.full((640, 360, 3), 50, np.uint8)
        original = frame.copy()
        view = render(frame, Result([], 0, 10))
        self.assertEqual(view.shape, (664, 1280, 3))
        np.testing.assert_array_equal(frame, original)

    def test_multiple_faces_render_at_both_mirror_settings(self):
        tracks = [Track(i, face(x=x)[:4], 0) for i, x in enumerate((10, 170, 330), 1)]
        for track in tracks:
            track.scores = confident(3)
            track.label_index = 3
        for mirror in (True, False):
            self.assertEqual(render(np.zeros((360, 640, 3), np.uint8), Result(tracks, 3, 12),
                                    mirrored=mirror).shape, (664, 1280, 3))


class FakeCapture:
    def __init__(self, failure=False):
        self.value = 0
        self.released = False
        self.failure = failure

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        time.sleep(0.003)
        self.value += 1
        if self.failure:
            return False, None
        return True, np.full((4, 4, 3), self.value % 256, np.uint8)

    def release(self):
        self.released = True


class CameraTests(unittest.TestCase):
    def test_camera_drops_old_frames_and_releases_on_close(self):
        capture = FakeCapture()
        with patch("emotion_app.camera.cv2.VideoCapture", return_value=capture):
            with LatestCamera() as camera:
                first = camera.read_after(0, 1)
                time.sleep(0.04)
                latest = camera.read_after(first[0], 1)
                self.assertGreater(latest[0], first[0] + 1)
        self.assertTrue(capture.released)
        self.assertFalse(camera.thread.is_alive())

    def test_camera_failure_reaches_the_main_thread(self):
        capture = FakeCapture(failure=True)
        with patch("emotion_app.camera.cv2.VideoCapture", return_value=capture):
            with LatestCamera() as camera:
                with self.assertRaisesRegex(RuntimeError, "stopped returning"):
                    camera.read_after(0, 3)
        self.assertTrue(capture.released)


class ModelIntegrationTests(unittest.TestCase):
    def test_real_models_run_and_blank_image_produces_no_faces(self):
        try:
            paths = ensure_models()
        except RuntimeError:
            self.skipTest("Run python app.py --download-models for model integration tests")
        engine = Engine(paths)
        result = engine.process(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(result.detected, 0)
        scores = engine.expression.predict(np.full((112, 112, 3), 128, np.uint8))
        self.assertEqual(scores.shape, (7,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1, places=5)


if __name__ == "__main__":
    unittest.main()
