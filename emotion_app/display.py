"""A simple OpenCV dashboard. No browser, server, or camera uploads."""

import cv2
import numpy as np

from .vision import LABELS

BG = (22, 20, 18)
PANEL = (34, 31, 28)
TEXT = (237, 235, 225)
MUTED = (155, 153, 143)
ACCENT = (162, 220, 91)
COLORS = [(127, 144, 248), (184, 177, 151), (216, 174, 190),
          (121, 221, 184), (221, 197, 146), (237, 176, 133), (131, 213, 246)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(image, words, x, y, size=0.5, color=TEXT, weight=1):
    cv2.putText(image, str(words), (int(x), int(y)), FONT, size, color, weight, cv2.LINE_AA)


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def card_position(box, occupied, face_boxes, width=960, height=540):
    x1, y1, x2, y2 = box
    card_width, card_height = 208, 67
    candidates = [(x2 + 12, y1), (x1 - card_width - 12, y1),
                  (x1, y1 - card_height - 12), (x1, y2 + 12)]
    best, best_cost = None, float("inf")
    for x, y in candidates:
        x = int(np.clip(x, 8, width - card_width - 8))
        y = int(np.clip(y, 8, height - card_height - 8))
        rect = (x, y, x + card_width, y + card_height)
        cost = sum(overlap(rect, face) for face in face_boxes)
        cost += 2 * sum(overlap(rect, other) for other in occupied)
        if cost < best_cost:
            best, best_cost = rect, cost
    return best


def render(frame, result, fps=0.0, mirrored=True, paused=False, details=True, notice="", model_name="MobileFaceNet"):
    canvas = np.full((664, 1280, 3), BG, dtype=np.uint8)
    text(canvas, "EXPRESSION / LIVE", 24, 35, 0.76, TEXT, 2)
    text(canvas, "LOCAL CAMERA LAB", 25, 56, 0.34, MUTED)
    cv2.circle(canvas, (990, 29), 4, MUTED if paused else ACCENT, -1)
    text(canvas, "PAUSED" if paused else "ON DEVICE", 1003, 34, 0.45, ACCENT)
    text(canvas, "No recording", 1145, 34, 0.42, MUTED)

    h, w = frame.shape[:2]
    scale = min(960 / w, 540 / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
    if mirrored:
        resized = cv2.flip(resized, 1)
    camera = np.full((540, 960, 3), (12, 12, 12), np.uint8)
    ox, oy = (960 - resized.shape[1]) // 2, (540 - resized.shape[0]) // 2
    camera[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    boxes = []
    for track in result.faces:
        x, y, bw, bh = track.box * scale
        if mirrored:
            x = resized.shape[1] - x - bw
        boxes.append((int(x + ox), int(y + oy), int(x + ox + bw), int(y + oy + bh)))

    occupied = []
    for track, box in zip(result.faces, boxes):
        color = COLORS[track.label_index] if track.label_index is not None else MUTED
        x1, y1, x2, y2 = box
        # Corners leave the face itself unobstructed.
        for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(camera, (x, y), (x + 18 * dx, y), color, 2, cv2.LINE_AA)
            cv2.line(camera, (x, y), (x, y + 18 * dy), color, 2, cv2.LINE_AA)
        cx1, cy1, cx2, cy2 = card_position(box, occupied, boxes)
        occupied.append((cx1, cy1, cx2, cy2))
        cv2.line(camera, ((x1 + x2) // 2, y1), ((cx1 + cx2) // 2, cy1 + 33), color, 1, cv2.LINE_AA)
        cv2.rectangle(camera, (cx1, cy1), (cx2, cy2), PANEL, -1)
        cv2.rectangle(camera, (cx1, cy1), (cx1 + 3, cy2), color, -1)
        text(camera, f"FACE {track.id:02d} / ESTIMATE", cx1 + 12, cy1 + 17, 0.32, MUTED)
        text(camera, track.label, cx1 + 12, cy1 + 40, 0.52, color, 1)
        if track.scores is not None:
            score = float(track.scores.max())
            text(camera, f"Top model score {score:.0%}", cx1 + 12, cy1 + 56, 0.32, MUTED)
    if not result.faces:
        cv2.rectangle(camera, (260, 225), (700, 309), PANEL, -1)
        text(camera, "Bring a face into view", 314, 258, 0.70)
        text(camera, "Look toward the camera in even light.", 305, 287, 0.43, MUTED)
    if paused:
        cv2.rectangle(camera, (12, 12), (205, 45), PANEL, -1)
        text(camera, "PAUSED / SPACE to resume", 23, 34, 0.38, ACCENT)
    canvas[76:616, :960] = camera

    cv2.rectangle(canvas, (976, 76), (1263, 616), PANEL, -1)
    text(canvas, "SESSION", 996, 104, 0.37, MUTED)
    text(canvas, f"{fps:04.1f}", 996, 143, 0.92, TEXT, 2)
    text(canvas, "FPS", 1092, 141, 0.4, MUTED)
    text(canvas, f"{result.inference_ms:.0f} ms inference", 996, 167, 0.40, MUTED)
    text(canvas, f"{len(result.faces)} tracked / {result.detected} detected", 996, 189, 0.40, MUTED)
    cv2.line(canvas, (996, 207), (1242, 207), (63, 58, 52), 1)
    if result.faces:
        lead = result.faces[0]
        text(canvas, f"FACE {lead.id:02d} / LARGEST IN VIEW", 996, 234, 0.36, MUTED)
        text(canvas, lead.label, 996, 265, 0.62, ACCENT)
        if details and lead.scores is not None:
            for index, label in enumerate(LABELS):
                y = 301 + index * 31
                text(canvas, label, 996, y, 0.39, TEXT)
                text(canvas, f"{lead.scores[index]:.0%}", 1201, y, 0.36, MUTED)
                cv2.rectangle(canvas, (1076, y - 8), (1187, y - 2), (65, 59, 53), -1)
                cv2.rectangle(canvas, (1076, y - 8), (1076 + round(111 * float(lead.scores[index])), y - 2), COLORS[index], -1)
        else:
            text(canvas, "Scores hidden" if not details else "Waiting for a clear face", 996, 305, 0.41, MUTED)
    else:
        text(canvas, "WAITING FOR A FACE", 996, 241, 0.43, TEXT)
        text(canvas, "Front-facing works best.", 996, 279, 0.41, MUTED)
        text(canvas, "Try soft, even lighting.", 996, 303, 0.41, MUTED)
    text(canvas, model_name, 996, 545, 0.4, MUTED)
    text(canvas, "Scores are not calibrated", 996, 568, 0.36, MUTED)
    text(canvas, "probabilities of real feelings.", 996, 587, 0.36, MUTED)
    text(canvas, "SPACE pause   M mirror   D scores   R reset   Q / ESC quit", 24, 646, 0.44, MUTED)
    if notice:
        text(canvas, notice, 775, 646, 0.36, ACCENT)
    else:
        text(canvas, "Expressions are clues, not a readout of emotion.", 824, 646, 0.34, MUTED)
    return canvas
