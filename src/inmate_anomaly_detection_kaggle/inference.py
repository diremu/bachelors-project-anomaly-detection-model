"""
inference.py — Inference and scoring pipeline for frontend integration.

Handles:
  - Loading a trained model from checkpoint
  - Processing a directory of frames into scored clips
  - Producing ranked JSON event logs for the web platform
  - Exporting to TorchScript for deployment
  - Temporal anomaly trend generation for visualisation
"""
import json, time, gc
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch

import config as cfg
from model import SpatiotemporalModel, NormalisedSpatiotemporalModel
import preprocessing as prep


# ──────────────────────────────────────────────────────────────
#  1.  Model loading
# ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path=None, device=None):
    """Load a trained model from checkpoint.

    Returns (model, hyperparameters) where model is a
    NormalisedSpatiotemporalModel ready for inference.
    """
    checkpoint_path = checkpoint_path or cfg.BEST_MODEL_FILE
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    hp = state.get("hyperparameters", {})

    base = SpatiotemporalModel(
        lstm_hidden=hp.get("lstm_hidden", 128),
        lstm_layers=hp.get("lstm_layers", 2),
        unfreeze_layers=hp.get("unfreeze_layers", 3),
        dropout=cfg.DROPOUT,
    )
    base.load_state_dict(state["model_state_dict"])

    model = NormalisedSpatiotemporalModel(base).to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path} (epoch {state.get('epoch', '?')})")
    return model, hp


# ──────────────────────────────────────────────────────────────
#  2.  Score a directory of frames
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def score_directory(model, frame_dir, device=None, fps=30.0,
                    max_frames=None, transform=None):
    """Process a frame directory into scored clips.

    Parameters
    ----------
    model      : trained NormalisedSpatiotemporalModel (or SpatiotemporalModel)
    frame_dir  : path to directory containing frame images (flat or subdir)
    fps        : assumed frame rate for timestamp calculation
    max_frames : cap on frames loaded per video (None = all)

    Returns
    -------
    list of dicts, each:
        {clip_id, video_id, start_frame, end_frame,
         start_time_s, end_time_s, anomaly_score, frames_used}
    """
    device = device or next(model.parameters()).device
    if transform is None:
        transform = prep.get_transforms(augment=False)
    frame_dir = Path(frame_dir)
    model.eval()

    # Discover and group frames
    groups = prep.get_groups(frame_dir)
    results = []
    clip_counter = 0

    if groups:
        # Flat layout (UCF-Crime style)
        for video_id in sorted(groups.keys()):
            paths = sorted(groups[video_id],
                           key=lambda p: int(p.stem.split("_")[-1]))
            if max_frames:
                paths = paths[:max_frames]
            results.extend(
                _score_frame_list(model, paths, video_id, fps,
                                  transform, device, clip_counter)
            )
            clip_counter += len([r for r in results if r["video_id"] == video_id])
    else:
        # Subdir layout (Avenue/ShanghaiTech style)
        subdirs = sorted(d for d in frame_dir.iterdir() if d.is_dir())
        for vd in subdirs:
            paths = sorted(
                (f for f in vd.iterdir()
                 if f.suffix.lower() in cfg.IMAGE_EXTENSIONS),
                key=lambda p: p.stem,
            )
            if max_frames:
                paths = paths[:max_frames]
            results.extend(
                _score_frame_list(model, paths, vd.name, fps,
                                  transform, device, clip_counter)
            )
            clip_counter += len([r for r in results if r["video_id"] == vd.name])

    return results


def _score_frame_list(model, frame_paths, video_id, fps,
                      transform, device, start_clip_id=0):
    """Score clips from a single video's frame list."""
    if len(frame_paths) < cfg.CLIP_LENGTH:
        return []

    tensors = [prep.preprocess_frame(prep.load_image(p), transform)
               for p in frame_paths]
    clips = prep.group_into_clips(tensors, cfg.CLIP_LENGTH, cfg.STRIDE)

    if not clips:
        return []

    results = []
    batch = torch.stack(clips).to(device)

    # Process in sub-batches to avoid OOM
    bs = cfg.BATCH_SIZE
    all_scores = []
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        scores = model.anomaly_score(chunk) if hasattr(model, "anomaly_score") \
                 else model.base.anomaly_score(chunk)
        all_scores.append(scores.cpu().numpy())
    all_scores = np.concatenate(all_scores)

    for ci, score in enumerate(all_scores):
        start_frame = ci * cfg.STRIDE
        end_frame = start_frame + cfg.CLIP_LENGTH - 1
        results.append({
            "clip_id":       f"clip_{start_clip_id + ci:05d}",
            "video_id":      video_id,
            "start_frame":   start_frame,
            "end_frame":     end_frame,
            "start_time_s":  round(start_frame / fps, 2),
            "end_time_s":    round(end_frame / fps, 2),
            "anomaly_score": round(float(score), 6),
        })

    del batch, clips, tensors
    gc.collect()
    return results


# ──────────────────────────────────────────────────────────────
#  3.  Rank and export as JSON for the web platform
# ──────────────────────────────────────────────────────────────
def rank_events(scored_clips, threshold=None):
    """Sort clips by anomaly score descending, add rank and flag.

    Parameters
    ----------
    scored_clips : list of dicts from score_directory()
    threshold    : anomaly score threshold. Clips above are flagged.

    Returns
    -------
    list of dicts with added keys: rank, flagged, severity
    """
    ranked = sorted(scored_clips, key=lambda c: c["anomaly_score"], reverse=True)

    if threshold is None and ranked:
        scores = [c["anomaly_score"] for c in ranked]
        threshold = float(np.percentile(scores, 90))  # top 10% flagged

    for i, clip in enumerate(ranked, 1):
        clip["rank"] = i
        clip["flagged"] = clip["anomaly_score"] >= threshold
        # Severity levels: critical (top 5%), high (top 15%), medium (top 30%), low
        pct = i / len(ranked)
        if pct <= 0.05:
            clip["severity"] = "critical"
        elif pct <= 0.15:
            clip["severity"] = "high"
        elif pct <= 0.30:
            clip["severity"] = "medium"
        else:
            clip["severity"] = "low"

    return ranked, threshold


def export_event_log(ranked_clips, output_path=None, threshold=None,
                     metadata=None):
    """Export ranked event log as JSON for the web platform.

    The output format matches the web platform's Incident Reports
    schema (§4.9.3 of the documentation).
    """
    output_path = output_path or (cfg.OUTPUT_DIR / "event_log.json")

    event_log = {
        "generated_at":  datetime.now().isoformat(),
        "model":         "ResNet18-LSTM Autoencoder",
        "threshold":     threshold,
        "total_clips":   len(ranked_clips),
        "flagged_clips": sum(1 for c in ranked_clips if c.get("flagged")),
        "metadata":      metadata or {},
        "events":        ranked_clips,
    }

    with open(str(output_path), "w") as f:
        json.dump(event_log, f, indent=2, default=str)

    print(f"Event log exported → {output_path}")
    print(f"  Total clips:   {event_log['total_clips']}")
    print(f"  Flagged clips: {event_log['flagged_clips']}")
    return event_log


# ──────────────────────────────────────────────────────────────
#  4.  Temporal anomaly trend for a single video
# ──────────────────────────────────────────────────────────────
def compute_temporal_trend(scored_clips, video_id=None, smooth_window=5):
    """Extract a temporal anomaly score sequence for one video.

    If video_id is None, uses the first video found.
    Returns (times, raw_scores, smoothed_scores, video_id).
    """
    if video_id is None:
        video_id = scored_clips[0]["video_id"]

    video_clips = [c for c in scored_clips if c["video_id"] == video_id]
    video_clips.sort(key=lambda c: c["start_frame"])

    if not video_clips:
        return None, None, None, video_id

    times = np.array([c["start_time_s"] for c in video_clips])
    scores = np.array([c["anomaly_score"] for c in video_clips])

    # Smoothing
    if len(scores) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        padded = np.pad(scores, smooth_window // 2, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:len(scores)]
    else:
        smoothed = scores.copy()

    return times, scores, smoothed, video_id


# ──────────────────────────────────────────────────────────────
#  5.  Model export for deployment
# ──────────────────────────────────────────────────────────────
def export_torchscript(model, output_path=None):
    """Export the base SpatiotemporalModel as TorchScript.

    The exported model accepts (B, T, C, H, W) clips and returns
    (reconstructed, features) tensors.
    """
    output_path = output_path or (cfg.OUTPUT_DIR / "model_scripted.pt")

    base = model.base if hasattr(model, "base") else model
    base.eval()
    base_cpu = base.cpu()

    example = torch.randn(1, cfg.CLIP_LENGTH, 3, cfg.FRAME_SIZE, cfg.FRAME_SIZE)
    try:
        scripted = torch.jit.trace(base_cpu, example)
        scripted.save(str(output_path))
        print(f"TorchScript model exported → {output_path}")
        print(f"  Input:  (B, {cfg.CLIP_LENGTH}, 3, {cfg.FRAME_SIZE}, {cfg.FRAME_SIZE})")
        print(f"  Output: (reconstructed, features) both (B, {cfg.CLIP_LENGTH}, 512)")
        return True
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        print("Falling back to state_dict export only.")
        return False


def export_scoring_config(hyperparams, threshold, score_mean=None,
                          score_std=None, output_path=None):
    """Export a JSON config that the web platform needs to run inference.

    Contains model hyperparameters, normalisation statistics,
    and the operating threshold.
    """
    output_path = output_path or (cfg.OUTPUT_DIR / "scoring_config.json")

    config = {
        "model_type":     "ResNet18-LSTM Autoencoder",
        "hyperparameters": hyperparams,
        "preprocessing": {
            "frame_size":     cfg.FRAME_SIZE,
            "clip_length":    cfg.CLIP_LENGTH,
            "stride":         cfg.STRIDE,
            "imagenet_mean":  cfg.IMAGENET_MEAN,
            "imagenet_std":   cfg.IMAGENET_STD,
        },
        "scoring": {
            "threshold":      threshold,
            "score_mean":     score_mean,
            "score_std":      score_std,
            "method":         "reconstruction_error_mse",
        },
        "severity_bands": {
            "critical": "top 5%",
            "high":     "top 15%",
            "medium":   "top 30%",
            "low":      "bottom 70%",
        },
    }

    with open(str(output_path), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Scoring config exported → {output_path}")
    return config