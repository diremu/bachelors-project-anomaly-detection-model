"""Training utilities: loop, checkpointing, early stopping, LR scheduling."""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from collections import defaultdict

from .config import (
    LR, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR,
    EARLY_STOP_PATIENCE, GRAD_CLIP, CHECKPOINT_DIR,
)


class AverageMeter:
    """Tracks mean and current value for scalars."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-6, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    extra: Optional[dict] = None,
):
    """Save model and optimizer state."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    map_location: str = "cpu",
) -> dict:
    """Load model (and optionally optimizer) from checkpoint. Returns checkpoint metadata."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = GRAD_CLIP,
) -> dict:
    """Single training epoch. Returns dict of averaged metrics."""
    model.train()
    loss_meter = AverageMeter()
    batch_times = AverageMeter()

    for batch_idx, (clips, labels) in enumerate(dataloader):
        t0 = time.time()
        clips = clips.to(device)  # (B, T, C, H, W)

        # Forward: get reconstructed features
        reconstructed = model(clips)

        # Build ground-truth features
        B, T, C, H, W = clips.shape
        clips_flat = clips.view(B * T, C, H, W)
        with torch.no_grad():
            original_feats = model.cnn_encoder(clips_flat).view(B, T, -1)

        loss = criterion(reconstructed, original_feats)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        loss_meter.update(loss.item(), B)
        batch_times.update(time.time() - t0)

    return {"loss": loss_meter.avg, "batch_time_ms": batch_times.avg * 1000}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validation pass. Returns dict of averaged metrics."""
    model.eval()
    loss_meter = AverageMeter()
    per_label_loss = defaultdict(list)

    for clips, labels in dataloader:
        clips = clips.to(device)
        B, T, C, H, W = clips.shape

        reconstructed = model(clips)

        clips_flat = clips.view(B * T, C, H, W)
        original_feats = model.cnn_encoder(clips_flat).view(B, T, -1)

        loss = criterion(reconstructed, original_feats)
        loss_meter.update(loss.item(), B)

        # Per-sample losses for breakdown
        sample_losses = model.lstm_ae.compute_reconstruction_error(original_feats, reconstructed)
        for lbl, sl in zip(labels, sample_losses.cpu().tolist()):
            per_label_loss[lbl].append(sl)

    # Aggregate per-label
    label_stats = {lbl: {"mean": sum(v) / len(v), "count": len(v)} for lbl, v in per_label_loss.items()}

    return {"loss": loss_meter.avg, "per_label": label_stats}
