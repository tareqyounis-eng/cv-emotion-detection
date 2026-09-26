#!/usr/bin/env python3
"""Run the webcam app, check the setup, or try a photo/video without a camera."""

import argparse
from collections import deque
from contextlib import ExitStack
import json
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from emotion_app.camera import CAMERA_HELP, LatestCamera
from emotion_app.display import render
from emotion_app.models import MODEL_DIR, ensure_models
from emotion_app.pipeline import Engine

WINDOW = "Expression Live"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def score_threshold(value):
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam device number (default: 0)")
    source.add_argument("--video", type=Path, help="process a local video instead of the webcam")
    source.add_argument("--image", type=Path, help="process one photo")
    p.add_argument("--output", type=Path, help="save the annotated photo (requires --image)")
    p.add_argument("--download-models", action="store_true", help="download/verify models, then exit")
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--doctor", action="store_true", help="check model inference without opening the camera")
    p.add_argument("--camera-check", action="store_true", help="read one camera frame, print its size, then release it")
    p.add_argument("--headless", action="store_true", help="do not open a display window")
    p.add_argument("--max-frames", type=positive, help="stop after this many processed frames")
    p.add_argument("--report", type=Path, help="write aggregate performance numbers, without faces or images")
    p.add_argument("--max-faces", type=positive, default=4, help="classify up to this many faces (default: 4)")
    p.add_argument("--detector-size", type=positive, default=640, help="long edge used for face detection")
    p.add_argument("--min-face", type=positive, default=64, help="minimum face size in source pixels")
    p.add_argument("--threshold", type=score_threshold, default=0.55, help="minimum smoothed model score for a label")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="use your trained FER2013 checkpoint (requires torch)")
    return p


def doctor(paths, args):
    print(f"Python {platform.python_version()} / {platform.machine()}")
    print(f"OpenCV {cv2.__version__} / NumPy {np.__version__}")
    engine = Engine(paths, checkpoint=args.checkpoint)
    result = engine.process(np.zeros((480, 640, 3), np.uint8))
    scores = engine.expression.predict(np.full((112, 112, 3), 128, np.uint8))
    if len(result.faces) != 0 or scores.shape != (7,) or not np.isclose(scores.sum(), 1):
        raise RuntimeError("Model self-check failed.")
    print("PASS: checksums, face inference, expression inference, and seven finite scores.")
    print("Camera was not opened. Use --camera-check to test macOS camera access.")


def describe(result):
    return [{"id": t.id, "label": t.label,
             "scores": t.scores.tolist() if t.scores is not None else None}
            for t in result.faces]


def run(args, paths):
    engine = Engine(paths, args.max_faces, args.detector_size, args.threshold, args.min_face, args.checkpoint)
    mirrored, details, paused = not args.no_mirror, True, False
    model_name = "Your FER2013 CNN" if args.checkpoint else "MobileFaceNet / OpenCV"
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
        result = engine.process(frame, still=True)
        # Photos use their original orientation unless this is a live feed.
        view = render(frame, result, mirrored=False, model_name=model_name)
        print(json.dumps({"faces": describe(result), "inference_ms": result.inference_ms}, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), view):
                raise RuntimeError(f"Could not write {args.output}")
        if not args.headless:
            cv2.imshow(WINDOW, view)
            while cv2.waitKey(50) & 0xFF not in (27, ord("q")):
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        return

    count, sequence, face_frames = 0, 0, 0
    started = last_display = time.monotonic()
    last_camera_frame = started
    times, fps = deque(maxlen=3000), 0.0
    view = None
    with ExitStack() as stack:
        if args.video:
            capture = cv2.VideoCapture(str(args.video))
            stack.callback(capture.release)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {args.video}")
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            video_fps = video_fps if np.isfinite(video_fps) and video_fps > 0 else 30
        else:
            camera = stack.enter_context(LatestCamera(args.camera))
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 664)
        while True:
            if not paused:
                if args.video:
                    ok, frame = capture.read()
                    if not ok:
                        if count == 0:
                            raise RuntimeError("Video opened but contains no readable frames.")
                        break
                    timestamp = count / video_fps
                else:
                    packet = camera.read_after(sequence)
                    if packet is None:
                        if time.monotonic() - last_camera_frame > 5:
                            raise RuntimeError("No fresh camera frame for five seconds. " + CAMERA_HELP)
                        if not args.headless and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                        continue
                    sequence, timestamp, frame = packet
                    last_camera_frame = time.monotonic()
                result = engine.process(frame, timestamp)
                count += 1
                face_frames += bool(result.faces)
                times.append(result.inference_ms)
                current = time.monotonic()
                instantaneous = 1 / max(0.001, current - last_display)
                fps = instantaneous if count == 1 else 0.9 * fps + 0.1 * instantaneous
                last_display = current
            if not args.headless:
                view = render(frame, result, fps, mirrored, paused, details, model_name=model_name)
                cv2.imshow(WINDOW, view)
                # Local video playback follows source time; headless mode benchmarks full speed.
                spent_ms = (time.monotonic() - current) * 1000 + result.inference_ms
                wait_ms = max(1, round(1000 / video_fps - spent_ms)) if args.video and not paused else 1
                key = cv2.waitKey(min(wait_ms, 100)) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    paused = not paused
                    if not paused:
                        engine.tracker.reset()
                        last_camera_frame = time.monotonic()
                elif key == ord("m"):
                    mirrored = not mirrored
                elif key == ord("d"):
                    details = not details
                elif key == ord("r"):
                    engine.tracker.reset()
                if paused:
                    time.sleep(0.025)
            if args.max_frames and count >= args.max_frames:
                break
    elapsed = time.monotonic() - started
    report = {"frames": count, "frames_with_faces": face_frames,
              "elapsed_seconds": round(elapsed, 3), "average_fps": round(count / max(elapsed, 0.001), 2),
              "median_inference_ms": round(float(np.median(times)), 2) if times else None,
              "p95_inference_ms": round(float(np.percentile(times, 95)), 2) if times else None,
              "timing_sample_frames": len(times)}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = parser()
    args = p.parse_args()
    if args.output and not args.image:
        p.error("--output requires --image; webcam frames are not recorded")
    if args.report and args.image:
        p.error("--report is for webcam/video performance; image results are printed to the terminal")
    if args.max_faces > 12:
        p.error("--max-faces must be at most 12")
    if args.detector_size < 160 or args.detector_size > 1280:
        p.error("--detector-size must be between 160 and 1280")
    try:
        if args.camera_check:
            with LatestCamera(args.camera) as camera:
                packet = camera.read_after(0, timeout=5)
                if packet is None:
                    raise RuntimeError(CAMERA_HELP)
                print(f"Camera {args.camera}: {packet[2].shape[1]} x {packet[2].shape[0]}; frame received.")
            return 0
        paths = ensure_models(args.model_dir, download=args.download_models)
        if args.download_models:
            print("Both models verified. Ready to run.")
        elif args.doctor:
            doctor(paths, args)
        else:
            run(args, paths)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
