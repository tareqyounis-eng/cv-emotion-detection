"""Keep only the newest camera frame, rather than building up video lag."""

import sys
import threading
import time

import cv2

CAMERA_HELP = (
    "Camera unavailable. Close other camera apps, then check System Settings > "
    "Privacy & Security > Camera for Terminal (or the app launching Python). "
    "Try --camera 1 if the built-in camera is not device 0."
)


class LatestCamera:
    def __init__(self, index=0, width=1280, height=720):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(CAMERA_HELP)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._read, name="camera", daemon=True)
        self.thread.start()

    def _read(self):
        failures = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 20:
                        raise RuntimeError("Camera stopped returning frames. " + CAMERA_HELP)
                    self.stop.wait(0.05)
                    continue
                failures = 0
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.timestamp = time.monotonic()
                    self.condition.notify_all()
        except Exception as error:
            self.error = error
            with self.condition:
                self.condition.notify_all()
        finally:
            self.capture.release()

    def read_after(self, sequence, timeout=0.1):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > sequence or self.error is not None or self.stop.is_set(),
                timeout=timeout,
            )
            if self.error:
                raise RuntimeError(str(self.error)) from self.error
            if self.sequence <= sequence:
                return None
            return self.sequence, self.timestamp, self.frame

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
