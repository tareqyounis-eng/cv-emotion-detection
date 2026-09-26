"""Optional FER2013 training. The webcam works without PyTorch installed."""

from pathlib import Path
import random

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
except ImportError as error:
    raise RuntimeError("Training/checkpoints need: python -m pip install -r requirements-training.txt") from error

from .vision import LABELS

CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        channels = 1
        for width in (32, 64, 128):
            blocks.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.15),
            ])
            channels = width
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                                        nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(128, 7))

    def forward(self, images):
        return self.classifier(self.features(images))


def split_indices(targets, fraction=0.2, seed=42):
    generator = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, validation = [], []
    for label in sorted(set(targets.tolist())):
        indices = np.flatnonzero(targets == label)
        if len(indices) < 2:
            raise ValueError("Each class needs at least two training images.")
        generator.shuffle(indices)
        n_val = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:n_val].tolist())
        train.extend(indices[n_val:].tolist())
    generator.shuffle(train)
    return train, validation


def make_datasets(root, seed=42):
    root = Path(root)
    common = [transforms.Grayscale(1), transforms.Resize((48, 48))]
    tensor = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    train_transform = transforms.Compose(common + [transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.05, 0.05))] + tensor)
    eval_transform = transforms.Compose(common + tensor)
    training = ImageFolder(root / "train", transform=train_transform)
    validation = ImageFolder(root / "train", transform=eval_transform)
    if tuple(training.classes) != CLASSES:
        raise ValueError(f"Expected these seven folders in archive/train: {CLASSES}; got {training.classes}")
    train_indices, val_indices = split_indices(training.targets, seed=seed)
    # Separate ImageFolder objects are deliberate. A Subset alone still shares
    # its parent's random transforms, which made the old validation set jitter.
    return Subset(training, train_indices), Subset(validation, val_indices), eval_transform


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this Python process. Use --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu.")
    return torch.device(requested)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size=64, workers=0, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)


def pass_epoch(model, loader, device, criterion, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, correct = 0.0, 0, 0
    confusion = np.zeros((7, 7), dtype=np.int64)
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite; checkpoint was not saved.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            predicted = logits.argmax(1)
            batch = targets.size(0)
            count += batch
            correct += (predicted == targets).sum().item()
            loss_sum += loss.item() * batch
            np.add.at(confusion, (targets.cpu().numpy(), predicted.cpu().numpy()), 1)
    if not count:
        raise RuntimeError("No images were loaded.")
    recalls = np.divide(confusion.diagonal(), confusion.sum(1), out=np.zeros(7), where=confusion.sum(1) > 0)
    precisions = np.divide(confusion.diagonal(), confusion.sum(0), out=np.zeros(7), where=confusion.sum(0) > 0)
    f1 = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros(7), where=precisions + recalls > 0)
    return {"loss": loss_sum / count, "accuracy": correct / count, "macro_f1": float(f1.mean()),
            "per_class_recall": dict(zip(CLASSES, recalls.tolist())),
            "confusion_matrix": confusion.tolist(), "images": count}


def read_checkpoint(path, device="cpu"):
    # weights_only avoids executing arbitrary pickle code from a checkpoint.
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("architecture") != "emotion-cnn-v1" or tuple(data.get("classes", ())) != CLASSES:
        raise ValueError("Checkpoint architecture or class order does not match this project.")
    if data.get("preprocessing") != "gray48-minus1-plus1":
        raise ValueError("Checkpoint preprocessing is missing or incompatible.")
    model = EmotionCNN().to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


class CheckpointModel:
    def __init__(self, path):
        torch.set_num_threads(2)
        self.model, self.metadata = read_checkpoint(path)
        if self.metadata.get("smoke_test"):
            raise ValueError("This is a smoke-test checkpoint, not a trained classifier. Train without --smoke first.")
        if tuple(name.lower() for name in LABELS) != CLASSES:
            raise ValueError("Runtime labels no longer match the training labels.")

    def predict(self, aligned):
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1)[None, None]
        with torch.inference_mode():
            return self.model(tensor).softmax(dim=1)[0].numpy()
