"""Train the ResNet-18 + LSTM spatiotemporal anomaly detection model.

Usage:
    uv run python scripts/train.py                          # defaults
    uv run python scripts/train.py --epochs 50 --lr 5e-5    # override params
    uv run python scripts/train.py --device cuda            # GPU training
    uv run python scripts/train.py --load checkpoints/best.pt  # resume
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from inmate_anomaly_detection.preprocessing import process_dataset
from inmate_anomaly_detection.dataloader import build_loaders
from inmate_anomaly_detection.model import SpatiotemporalModel, count_parameters
from inmate_anomaly_detection.train_utils import (
    train_one_epoch, validate, save_checkpoint,
    load_checkpoint, EarlyStopping,
)
from inmate_anomaly_detection.config import (
    CLIP_LENGTH, STRIDE, BATCH_SIZE,
    DATASET_PATHS, CHECKPOINT_DIR,
    LR, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR,
    EARLY_STOP_PATIENCE, GRAD_CLIP,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    FREEZE_BACKBONE, UNFREEZE_LAYERS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train anomaly detection model")
    p.add_argument("--epochs", type=int, default=30, help="Epochs [TUNE]")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate [TUNE]")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size [TUNE]")
    p.add_argument("--lstm-hidden", type=int, default=LSTM_HIDDEN, help="LSTM hidden size [TUNE]")
    p.add_argument("--lstm-layers", type=int, default=LSTM_LAYERS, help="LSTM stacked layers [TUNE]")
    p.add_argument("--unfreeze-layers", type=int, default=UNFREEZE_LAYERS, help="ResNet layers to unfreeze [TUNE]")
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=FREEZE_BACKBONE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 penalty [TUNE]")
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP, help="Max grad norm, 0=off [TUNE]")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--load", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--max-frames", type=int, default=200, help="Max frames per video for memory")
    p.add_argument("--dataset", type=str, default="ucf_crime", help="Dataset key in config")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    train_path = DATASET_PATHS[args.dataset] / "Train"
    test_path = DATASET_PATHS[args.dataset] / "Test"

    if not train_path.exists():
        print(f"Train path not found: {train_path}")
        print(f"Available: {list(DATASET_PATHS.keys())} -> {[str(DATASET_PATHS[k]) for k in DATASET_PATHS]}")
        return

    print(f"Loading training data from: {train_path}")

    train_clips, train_labels = [], []
    for clip, label in process_dataset(train_path, augment=True, max_frames=args.max_frames):
        train_clips.append(clip)
        train_labels.append(label)

    print(f"Train: {len(train_clips)} clips from {len(set(train_labels))} categories")

    # Split: train on NormalVideos only (reconstruction model learns normality),
    # validate on both normal + anomalous
    normal_indices = [i for i, lbl in enumerate(train_labels) if lbl == "NormalVideos"]
    anomalous_indices = [i for i, lbl in enumerate(train_labels) if lbl != "NormalVideos"]

    if not normal_indices:
        print("WARNING: No NormalVideos found in dataset. Using all data for training.")
        normal_indices = list(range(len(train_clips)))

    # Train/val split on normal data: 80/20
    split = int(len(normal_indices) * 0.8)
    train_idx = normal_indices[:split]
    val_normal_idx = normal_indices[split:]

    train_c = [train_clips[i] for i in train_idx]
    train_l = [train_labels[i] for i in train_idx]

    val_c = [train_clips[i] for i in val_normal_idx] + [train_clips[i] for i in anomalous_indices]
    val_l = [train_labels[i] for i in val_normal_idx] + [train_labels[i] for i in anomalous_indices]

    print(f"Train split: {len(train_c)} clips (normal only)")
    print(f"Val split:   {len(val_c)} clips ({len(val_normal_idx)} normal, {len(anomalous_indices)} anomalous)")

    train_loader, val_loader = build_loaders(train_c, train_l, val_c, val_l, batch_size=args.batch_size)

    # ---- Build model ----
    model = SpatiotemporalModel(
        freeze_backbone=args.freeze_backbone,
        unfreeze_layers=args.unfreeze_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=LSTM_DROPOUT,
    ).to(device)

    trainable, total = count_parameters(model)
    print(f"\nModel: {trainable:,} trainable / {total:,} total parameters")
    print(f"  ResNet unfrozen layers: {args.unfreeze_layers}")
    print(f"  LSTM: {args.lstm_layers} layer(s), hidden={args.lstm_hidden}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    start_epoch = 1
    best_loss = float("inf")
    best_path = CHECKPOINT_DIR / "best.pt"

    if args.load:
        state = load_checkpoint(model, optimizer, Path(args.load), str(device))
        start_epoch = state.get("epoch", 1) + 1
        best_loss = state.get("loss", float("inf"))
        print(f"Resumed from {args.load} at epoch {start_epoch}")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, LR={args.lr}, batch={args.batch_size}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.6f} | "
              f"val_loss={val_metrics['loss']:.6f} | "
              f"lr={current_lr:.2e}")

        # Show per-label breakdown every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            print("  Val loss by label:")
            for lbl, stats in sorted(val_metrics["per_label"].items()):
                print(f"    {lbl:20s}: {stats['mean']:.6f} (n={stats['count']})")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, best_loss, best_path,
                            extra={"epoch": epoch, "config": vars(args)})
            print(f"  -> Saved best checkpoint (loss={best_loss:.6f})")

        if early_stop(val_metrics["loss"]):
            print(f"\nEarly stopped at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val loss: {best_loss:.6f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
