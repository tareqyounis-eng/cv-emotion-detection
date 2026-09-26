# Validation — September 26, 2026

## Passed

- Runtime environment: Python 3.11.16, Apple Silicon, OpenCV 4.13.0.92,
  NumPy 2.4.4. `pip check` reports no broken requirements.
- Both model files downloaded from their pinned revision and passed SHA-256
  verification. `app.py --doctor` executed both neural networks successfully.
- 20 runtime tests passed. The three optional training tests were skipped in
  `.venv`, then all three passed separately in the existing `visenv` with
  PyTorch 2.11.0 and torchvision 0.26.0. Total: 23 passing tests across the
  runtime and training environments.
- Checks cover geometric tracking under reordered detections, missing faces,
  expiration, score isolation, uncertainty, quality rejection, alignment,
  invalid model downloads, camera failure propagation, camera release,
  newest-frame capture, rendering, deterministic dataset splits, validation
  transforms, and checkpoint compatibility.
- Actual photo inference found a face in OpenCV's `lena.jpg` sample. The generated
  dashboard image was visually inspected for labels, face brackets, score bars,
  spacing, and clipping. This is rendered-image verification, not a native-window
  interaction test.
- A synthetic 180-frame, 1280x720, 30 FPS clip made from that sample exercised
  movement, one/two faces, face departure, and empty frames. The headless app
  completed all 180 frames, detected faces in the expected 150 nonempty frames,
  and reported 64.99 FPS overall, 16.94 ms median inference, 19.12 ms p95 inference.
  These are local processing numbers on repeated sample imagery, **not measured
  webcam performance or expression accuracy**.
- Optional CNN training completed a two-batch CPU smoke run on local FER2013
  training data. The split contained 22,968 training images and 5,741 validation
  images. Test data was not read. The smoke checkpoint is stored outside the
  project in task scratch space and is not used by the app.
- Shell syntax and `git diff --check` passed. Existing notebook edits, the
  pre-existing deleted `face_detection.ipynb`, `visenv`, and the dataset were
  preserved during implementation.

## Live verification still needed

`app.py --camera-check` returned OpenCV's explicit macOS **camera access denied**
message in the Codex-launched process. No webcam frame was captured or saved.

A separate video test reached `cv2.namedWindow` and exited with SIGABRT (134).
The macOS crash report shows the abort in `_RegisterApplication` through
`NSApplication` and `cvNamedWindow`, before window rendering. This is consistent
with the restricted execution session lacking native GUI access; operation from
Terminal has not been verified. The UI automation tool also refuses Terminal
access, so it could not finish that check on the user's behalf.

On the Mac, double-click `Start EmotionDetection.command` from Finder. Allow
Terminal/Python camera access if prompted. Confirm that:

1. The window shows your webcam and a card beside your face.
2. The card settles over several frames, and disappears when you leave view.
3. A second face gets its own card; Space, M, D, R, and Q work.
4. Quitting releases the camera (the camera indicator turns off).

If access was denied previously: System Settings → Privacy & Security → Camera,
then enable the application running Python and restart it. Nothing in this project
resets macOS privacy settings or tries to bypass them.

## Model limits

The default application uses pretrained OpenCV Zoo models. The optional custom
CNN is a complete training/evaluation path, but was only smoke-tested, not trained
to useful accuracy. No new accuracy claim has been established for this webcam or
its users. Smoothing and quality filters improve display behavior; they do not
validate an inference about internal feelings. Track IDs can switch at crossings.

Model sources, licenses, usage commands, and tuning options are in `README.md`.
