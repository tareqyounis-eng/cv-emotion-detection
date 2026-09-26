"""Face detection, alignment, and the expression model's exact input recipe."""

import cv2
import numpy as np

# This order comes from the pretrained model, not ImageFolder's directory order.
LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float64)


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 7 or not np.isfinite(values).all():
        raise RuntimeError("Expression model returned invalid scores.")
    values = np.exp(values - values.max())
    return (values / values.sum()).astype(np.float32)


def align_face(frame, landmarks):
    points = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    if not np.isfinite(points).all():
        raise ValueError("Face landmarks are not finite.")
    # Fit the inverse similarity transform used by OpenCV Zoo. No random RANSAC:
    # a still face should get the same crop every time.
    x, y = REFERENCE_POINTS.T
    ones, zeros = np.ones(5), np.zeros(5)
    design = np.vstack((np.column_stack((x, y, ones, zeros)),
                        np.column_stack((y, -x, zeros, ones))))
    a, b, tx, ty = np.linalg.lstsq(design, points.T.reshape(-1), rcond=None)[0]
    if a * a + b * b < 1e-8:
        raise ValueError("Face landmarks collapsed to a point.")
    inverse = np.array([[a, b, tx], [-b, a, ty], [0, 0, 1]])
    matrix = np.linalg.inv(inverse)[:2]
    return cv2.warpAffine(frame, matrix, (112, 112))


def quality_reason(frame, face, minimum_size=64):
    height, width = frame.shape[:2]
    x, y, w, h = face[:4]
    if min(w, h) < minimum_size:
        return "Move closer"
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return "Keep face in view"
    crop = frame[int(y):int(y + h), int(x):int(x + w)]
    if crop.size == 0:
        return "Keep face in view"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 38:
        return "More light needed"
    if brightness > 235:
        return "Too much glare"
    # Check blur at a fixed size; otherwise distance changes this measurement.
    small = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    if cv2.Laplacian(small, cv2.CV_64F).var() < 16:
        return "Hold still / focus"
    eyes = face[4:8].reshape(2, 2)
    nose = face[8:10]
    eye_width = float(np.linalg.norm(eyes[1] - eyes[0]))
    if eye_width < 0.18 * w or np.linalg.norm(nose - eyes.mean(axis=0)) > 0.60 * w:
        return "Face the camera"
    return None


class ExpressionModel:
    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, crop):
        # MobileFaceNet expects RGB in [-1, 1], not the old 48px grayscale data.
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob, "data")
        return probabilities(self.net.forward("label"))


class FaceDetector:
    def __init__(self, path, max_width=640, threshold=0.85):
        self.max_width = max_width
        self.detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), threshold, 0.3, 5000
        )

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / max(width, height))
        size = (max(32, round(width * scale)), max(32, round(height * scale)))
        small = cv2.resize(frame, size) if size != (width, height) else frame
        self.detector.setInputSize(size)
        _, faces = self.detector.detect(small)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        faces = faces.copy()
        # Rounding can give x and y slightly different scale factors.
        faces[:, [0, 2, 4, 6, 8, 10, 12]] *= width / size[0]
        faces[:, [1, 3, 5, 7, 9, 11, 13]] *= height / size[1]
        return faces[np.argsort(-(faces[:, 2] * faces[:, 3]))]
