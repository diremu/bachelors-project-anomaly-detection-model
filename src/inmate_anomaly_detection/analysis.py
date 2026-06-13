import warnings
import numpy as np
import torch
from typing import List, Optional


def compute_frame_differences(clip: torch.Tensor) -> np.ndarray:
    """
    Compute absolute frame-to-frame differences for a clip.

    Args:
        clip: Tensor of shape (clip_length, 3, 224, 224).
            Values are ImageNet-normalized (roughly in [-2.5, +2.5] per channel).

    Returns:
        Differences array of shape (clip_length - 1, 224, 224).
    """
    # Detach from computation graph before converting to numpy
    frames = clip.detach().cpu().numpy()
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1])
        diff_gray = np.mean(diff, axis=0)   # average across RGB channels
        diffs.append(diff_gray)
    return np.stack(diffs)


def compute_motion_energy(diffs: np.ndarray) -> float:
    """
    Compute total motion energy as the mean of all frame differences.

    Args:
        diffs: Pre-computed frame differences of shape (clip_length-1, H, W).

    Higher values indicate more motion, which may correlate with
    anomalous or violent activity in correctional facility footage.
    """
    return float(np.mean(diffs))


def compute_motion_entropy(diffs: np.ndarray, bins: int = 32) -> float:
    """
    Compute motion entropy across frame differences.

    Args:
        diffs: Pre-computed frame differences of shape (clip_length-1, H, W).

    High entropy = chaotic/unpredictable motion (potential anomaly).
    Low entropy = structured/predictable motion (likely normal).
    """
    diff_flat = diffs.flatten()
    if diff_flat.max() == diff_flat.min():
        return 0.0

    hist, _ = np.histogram(diff_flat, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compute_anomaly_score(
    diffs: np.ndarray,
    energy: Optional[float] = None,
    entropy: Optional[float] = None,
    w_energy: float = 0.4,
    w_entropy: float = 0.6,
    energy_scale: float = 0.5,
    entropy_scale: float = 5.0,
) -> float:
    """
    Combined anomaly score from motion energy and entropy.

    Both components are normalized by their expected maximum scale
    before weighting so neither dominates the combined score.

    Args:
        diffs:        Pre-computed frame differences.
        energy:       Pre-computed motion energy (avoids recomputing).
        entropy:      Pre-computed motion entropy (avoids recomputing).
        w_energy:     Weight for motion energy component (default 0.4).
        w_entropy:    Weight for motion entropy component (default 0.6).
        energy_scale: Expected max energy for normalization.
                      FIX: default raised from 1.0 → 0.5. ImageNet
                      normalization maps pixels to roughly [-2.5, +2.5], so
                      inter-frame differences can easily reach ~0.5 even for
                      moderate motion. A scale of 1.0 caused norm_energy to
                      saturate at 1.0 for nearly every clip, making the energy
                      component useless. Tune this on your validation set.
        entropy_scale: Expected max entropy in bits — log2(32 bins) = 5.0.

    Returns:
        Anomaly score in [0, 1] range (higher = more likely anomalous).
    """
    if energy is None:
        energy = compute_motion_energy(diffs)
    if entropy is None:
        entropy = compute_motion_entropy(diffs)

    # Normalize each component to [0, 1] before weighting
    norm_energy = np.clip(energy / energy_scale, 0.0, 1.0)
    norm_entropy = np.clip(entropy / entropy_scale, 0.0, 1.0)

    return float(w_energy * norm_energy + w_entropy * norm_entropy)


def analyze_batch(
    batch: torch.Tensor,
    labels: List[int],
    categories: List[str],
    threshold: Optional[float] = None,
) -> List[dict]:
    """
    Analyze a batch of clips and return per-clip results.

    Args:
        batch:      Tensor of shape (batch_size, clip_length, 3, 224, 224).
        labels:     Integer labels per clip — 0 = normal, 1 = anomalous.
        categories: Category name per clip (e.g. 'Fighting', 'Normal_Videos').
        threshold:  Anomaly threshold for binary prediction. Should be
                    calibrated on the validation set. If None, the batch
                    median is used as a fallback with a warning.

    Returns:
        List of result dicts with motion stats, score, and prediction.
    """
    results = []
    scores = []

    for i in range(batch.shape[0]):
        clip = batch[i]

        # Compute frame differences once and reuse for all metrics
        diffs = compute_frame_differences(clip)
        energy = compute_motion_energy(diffs)
        entropy = compute_motion_entropy(diffs)
        score = compute_anomaly_score(diffs, energy=energy, entropy=entropy)

        scores.append(score)
        results.append({
            "label":          labels[i],           # int: 0 or 1
            "category":       categories[i],        # str: e.g. 'Fighting'
            "motion_energy":  round(energy, 6),
            "motion_entropy": round(entropy, 6),
            "anomaly_score":  round(score, 6),
        })

    # Threshold resolution
    if threshold is None:
        warnings.warn(
            "No threshold provided to analyze_batch(). "
            "Using batch median as fallback — results will not be "
            "consistent across batches. Calibrate threshold on the "
            "validation set and pass it explicitly.",
            UserWarning,
            stacklevel=2,
        )
        threshold = float(np.median(scores))

    for r in results:
        r["threshold"] = round(threshold, 6)
        r["predicted_label"] = 1 if r["anomaly_score"] > threshold else 0
        r["prediction"] = "anomalous" if r["predicted_label"] == 1 else "normal"

    return results