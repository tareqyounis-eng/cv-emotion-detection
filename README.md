# EmotionDetection

A local webcam app for a MacBook: find faces, estimate their facial expressions, and
put a label next to each face. The original FER2013 exploration notebooks are still
here. The runnable app uses pretrained networks so you don't have to wait for a
training run before trying it.

## Start it

Double-click **Start EmotionDetection.command**. Allow **Terminal** to use the
camera when macOS asks. The first setup needs an internet connection for Python
packages and roughly 5 MB of model weights. After that, inference works offline.

From a terminal:

```bash
cd /Users/tareq/Documents/EmotionDetection
./setup.sh
.venv/bin/python app.py
```

The runtime uses its own `.venv`; your existing `visenv` and notebooks are left
alone. Python 3.11 or newer is required. The pinned packages were tested on this
Apple Silicon Mac with Python 3.11. On another machine, use a Python version and OS
supported by those wheels.

If the camera fails, go to **System Settings → Privacy & Security → Camera** and
enable the application that launched Python, usually Terminal. Close and reopen
that application after changing permissions. A Codex-launched process may have
different permission from Terminal. FaceTime or another camera app can also get
in the way. Try `.venv/bin/python app.py --camera 1` for a different camera.

## What you get

- Face boxes and expression cards beside up to four faces by default.
- Seven expression classes: angry, disgust, fear, happy, neutral, sad, surprise.
- Per-face temporal smoothing and a short hold before a new label appears.
- An `Uncertain` state when scores are weak or close together.
- Hints for a small, clipped, dark, overexposed, blurry, or strongly turned face.
- A live score panel for the largest face, frame rate, and inference timing.
- Latest-frame camera capture so processing doesn't build up a video queue.
- Photo/video inputs for debugging without a webcam.

The percentages are **model scores**, not calibrated probabilities of what someone
feels. A face does not establish a person's internal emotional state. Lighting,
pose, occlusion, demographics, individual expressiveness, and the training data
all affect the result. This is an expression-classification project; no accuracy
percentage has been established for your webcam. The quality checks are practical
heuristics, not a guarantee that every bad frame gets caught.

| Key | Action |
| --- | --- |
| Space | Pause/resume the displayed feed |
| M | Mirror the display; inference keeps its original orientation |
| D | Show/hide score bars |
| R | Reset temporary face tracks |
| Q or Escape | Quit and release the camera |

Pausing freezes the display; capture continues draining frames so resuming is
current. Quit to release the camera. No webcam images, video, face identities, or
emotion histories are recorded or uploaded. The optional `--report` contains only
aggregate performance numbers. Screen IDs expire quickly and are not biometric
identities; they can switch when people cross or occlude each other. Dense scenes
can still make label cards overlap.

## Useful commands

```bash
# Verify model checksums and execute both networks, without opening the camera.
.venv/bin/python app.py --doctor

# Test camera permissions, receive one frame, then release it without saving it.
.venv/bin/python app.py --camera-check

# More conservative labels, or more people at once.
.venv/bin/python app.py --threshold 0.65 --max-faces 6

# A smaller detector input can help on a slower machine.
.venv/bin/python app.py --detector-size 480

# A photo has no temporal smoothing. Saving requires an explicit --output.
.venv/bin/python app.py --image /path/to/photo.jpg --output /tmp/result.jpg --headless

# Process a clip at full speed without a window; write timing numbers only.
.venv/bin/python app.py --video /path/to/video.mp4 --headless --max-frames 300 --report /tmp/timing.json

# Runtime regression tests; model integration runs when the weights are present.
.venv/bin/python -m unittest discover -s tests -v
```

Timing statistics retain at most the most recent 3,000 processed frames, while the
frame counts and average FPS cover the whole session. End-to-end FPS includes
capture and rendering; inference timing covers detection, alignment, expression
classification, and tracking. `--headless` video numbers don't measure camera or
display performance.

## How it works

```text
Mac camera -> newest frame -> YuNet face boxes + five landmarks
                             -> aligned 112x112 RGB face crop
                             -> MobileFaceNet expression logits
                             -> softmax + per-face smoothing + uncertainty
                             -> face cards and score panel
```

The detector runs on a reduced copy of the image; crops use the original frame.
Alignment and normalization follow the model's published OpenCV Zoo recipe. Model
files are pinned to a repository revision and SHA-256 checked before use. Downloads
are written to temporary files and only installed after verification. Startup will
not silently use a corrupt model.

Main files:

- `app.py`: command-line entry point, UI loop, error handling, cleanup.
- `emotion_app/vision.py`: YuNet, alignment, quality checks, MobileFaceNet.
- `emotion_app/tracking.py`: short-lived geometric tracks and score smoothing.
- `emotion_app/camera.py`: a background reader with one latest frame.
- `emotion_app/display.py`: the OpenCV dashboard and label placement.
- `emotion_app/models.py`: pinned downloads and checksum verification.
- `train.py`, `evaluate.py`, `emotion_app/training.py`: optional custom FER2013 CNN.
- `tests/`: tracking, alignment, bad-frame handling, model loading, camera cleanup,
  rendering, dataset splitting, and checkpoint checks.

## Train your own FER2013 CNN

This is optional. The default pretrained model is already installed by setup.
The custom CNN is a separate model with **48x48 grayscale input**; it has not been
trained to completion as part of setup. A two-batch smoke run only checks the wiring.

Keep your locally downloaded FER2013 data in:

```text
archive/train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
archive/test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
```

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python train.py --smoke --output runs/check
.venv/bin/python train.py --epochs 35 --output runs/first-model
.venv/bin/python evaluate.py runs/first-model/best.pt --output runs/first-model/test-results.json
.venv/bin/python app.py --checkpoint runs/first-model/best.pt
```

`--device auto` chooses MPS when available, then CUDA, then CPU. `--device cpu`
works when accelerator access is unavailable. Training uses a seeded, stratified
80/20 split of **train only**, with separate transforms for validation. The test
directory is read only by `evaluate.py`, after you have selected your model.
Minority-class weighting, augmentation, gradient clipping, LR reduction, early
stopping, confusion matrices, class recall, and macro F1 are included. Actual
training time depends on the machine. Seeds make the split reproducible; exact
floating-point results can differ across devices.

Checkpoints include architecture, preprocessing, class order, and split indices.
A checkpoint from `--smoke` is deliberately refused by the live app and final
evaluation. A training run also refuses to overwrite an existing `best.pt`; choose
a new output directory. The pretrained RGB model and your grayscale CNN are never
silently mixed. The live custom-CNN path uses aligned camera crops, which still
have a domain difference from FER2013; evaluate it on your own lighting before
expecting pretrained-model performance.

## Model sources and notices

- [YuNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet):
  `face_detection_yunet_2023mar.onnx`, MIT license.
- [Progressive Teacher / MobileFaceNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/facial_expression_recognition):
  `facial_expression_recognition_mobilefacenet_2022july.onnx`, Apache 2.0 per its model README.
- [FER2013 data source](https://www.kaggle.com/datasets/msambare/fer2013): download
  separately; the dataset and your existing environment are excluded from Git.

License copies and attribution are in `licenses/`. See `VALIDATION.md` for the
checks actually performed and the remaining camera-permission limitation.
