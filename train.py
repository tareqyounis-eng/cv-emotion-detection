#!/usr/bin/env python3
"""Train the optional CNN on archive/train. The test directory stays untouched."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

from emotion_app.models import ROOT
from emotion_app.training import (CLASSES, EmotionCNN, choose_device, make_datasets,
                                  make_loader, pass_epoch, torch, nn)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "archive")
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "fer2013")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="two batches for wiring checks; NOT an accuracy run")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        p.error("epochs, batch-size and patience must be positive; workers cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    train, validation, _ = make_datasets(args.data, args.seed)
    train_loader = make_loader(train, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(validation, args.batch_size, args.workers, seed=args.seed)
    # Weight the minority classes gently. Full inverse-frequency weighting can
    # let a handful of noisy disgust images dominate the whole training run.
    counts = np.bincount(np.array(train.dataset.targets)[train.indices], minlength=7)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss()
    model = EmotionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    # A quick wiring check must never overwrite a real trained model.
    output = args.output / "smoke" if args.smoke else args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        p.error(f"{output / 'best.pt'} already exists. Choose a new --output directory.")
    history, best, stale = [], -1, 0
    print(f"Device: {device}; train: {len(train)}; validation: {len(validation)}; test: untouched", flush=True)
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        limit = 2 if args.smoke else None
        training = pass_epoch(model, train_loader, device, criterion, optimizer, limit)
        val = pass_epoch(model, val_loader, device, eval_criterion, max_batches=limit)
        scheduler.step(val["macro_f1"])
        history.append({"epoch": epoch, "train": training, "validation": val})
        print(f"Epoch {epoch:02d}: train {training['accuracy']:.1%}, validation {val['accuracy']:.1%}, macro F1 {val['macro_f1']:.3f}", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            checkpoint = {"architecture": "emotion-cnn-v1", "classes": list(CLASSES),
                          "preprocessing": "gray48-minus1-plus1", "state_dict": state,
                          "epoch": epoch, "seed": args.seed, "validation": val,
                          "smoke_test": args.smoke, "train_indices": train.indices,
                          "validation_indices": validation.indices}
            temporary = output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("Validation stopped improving; keeping the best checkpoint.")
                break
    print(f"Saved {output / 'best.pt'}" + (" (SMOKE TEST ONLY)" if args.smoke else ""))


if __name__ == "__main__":
    main()
