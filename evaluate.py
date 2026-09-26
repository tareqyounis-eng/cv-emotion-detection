#!/usr/bin/env python3
"""Evaluate a trained CNN on FER2013's held-out test images."""

import argparse
import json
from pathlib import Path

from emotion_app.models import ROOT
from emotion_app.training import (CLASSES, ImageFolder, choose_device, make_loader,
    nn, pass_epoch, read_checkpoint, torch, transforms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "archive" / "test")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, metadata = read_checkpoint(args.checkpoint, device)
    if metadata.get("smoke_test"):
        p.error("A smoke-test checkpoint is not ready for a test-set evaluation.")
    transform = transforms.Compose([transforms.Grayscale(1), transforms.Resize((48, 48)),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = ImageFolder(args.data, transform=transform)
    if tuple(dataset.classes) != CLASSES:
        p.error("Test directory class folders do not match the checkpoint.")
    result = pass_epoch(model, make_loader(dataset), device, nn.CrossEntropyLoss())
    result["classes"] = list(CLASSES)
    result["checkpoint"] = str(args.checkpoint)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
