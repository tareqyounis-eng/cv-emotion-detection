"""Small, pinned models so a fresh checkout can actually run."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODELS = {
    "face": (
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "expression": (
        "facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx",
        "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9",
    ),
}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def ensure_models(directory=MODEL_DIR, download=False):
    directory = Path(directory)
    paths = {}
    for key, (relative, expected) in MODELS.items():
        path = directory / Path(relative).name
        if not path.is_file() or digest(path) != expected:
            if not download:
                raise RuntimeError(
                    f"Missing or damaged model: {path.name}. "
                    "Run python app.py --download-models first."
                )
            directory.mkdir(parents=True, exist_ok=True)
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/{relative}"
            print(f"Downloading {path.name} ...", flush=True)
            temporary = None
            try:
                # Never leave a half-downloaded file with the real model's name.
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as target:
                    temporary = Path(target.name)
                    with urlopen(url, timeout=40) as response:
                        total = 0
                        while chunk := response.read(1024 * 1024):
                            total += len(chunk)
                            if total > 20 * 1024 * 1024:
                                raise RuntimeError("Model download was unexpectedly large.")
                            target.write(chunk)
                if digest(temporary) != expected:
                    raise RuntimeError(f"Checksum mismatch for {path.name}; download rejected.")
                os.replace(temporary, path)
            except (OSError, TimeoutError) as error:
                raise RuntimeError(f"Could not download {path.name}: {error}") from error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        paths[key] = path
    return paths
