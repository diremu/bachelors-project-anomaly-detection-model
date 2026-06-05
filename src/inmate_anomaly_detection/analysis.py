import numpy as np
import torch
from typing import List, Tuple


def compute_frame_differences(clip: torch.Tensor) -> np.ndarray:
    """
    Compute absolute frame-to-frame differences for a clip.

    Args:
        clip: Tensor of shape (clip_length, 3, 224, 224).

    Returns:
        Differences array of shape (clip_length - 1, 224, 224).
    """
    frames = clip.numpy()
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1])
        diff_gray = np.mean(diff, axis=0)
        diffs.append(diff_gray)
    return np.stack(diffs)


def compute_motion_energy(clip: torch.Tensor) -> float:
    """
    Compute total motion energy as the mean of all frame differences.

    Higher values indicate more motion, which may correlate with
    anomalous/violent activity.
    """
    diffs = compute_frame_differences(clip)
    return float(np.mean(diffs))


def compute_motion_entropy(clip: torch.Tensor, bins: int = 32) -> float:
    """
    Compute motion entropy across frame differences.

    High entropy = chaotic/unpredictable motion (potential anomaly).
    Low entropy = structured/predictable motion (normal).
    """
    diffs = compute_frame_differences(clip)
    diff_flat = diffs.flatten()
    if diff_flat.max() == diff_flat.min():
        return 0.0

    hist, _ = np.histogram(diff_flat, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compute_anomaly_score(clip: torch.Tensor, w_energy: float = 0.4, w_entropy: float = 0.6) -> float:
    """
    Combined anomaly score from motion energy and entropy.

    Args:
        clip: Tensor of shape (clip_length, 3, 224, 224).
        w_energy: Weight for motion energy component.
        w_entropy: Weight for motion entropy component.

    Returns:
        Anomaly score (higher = more likely anomalous).
    """
    energy = compute_motion_energy(clip)
    entropy = compute_motion_entropy(clip)
    return w_energy * energy + w_entropy * entropy


def analyze_batch(
    batch: torch.Tensor,
    labels: List[str],
    threshold: float = None,
) -> List[dict]:
    """
    Analyze a batch of clips and return per-clip results.

    Args:
        batch: Tensor of shape (batch_size, clip_length, 3, 224, 224).
        labels: Corresponding labels for each clip.
        threshold: Optional anomaly threshold for binary classification.

    Returns:
        List of result dicts with score and prediction.
    """
    results = []
    scores = []

    for i in range(len(batch)):
        clip = batch[i]
        score = compute_anomaly_score(clip)
        scores.append(score)
        results.append({
            "label": labels[i],
            "motion_energy": compute_motion_energy(clip),
            "motion_entropy": compute_motion_entropy(clip),
            "anomaly_score": score,
        })

    if threshold is None:
        threshold = np.median(scores)

    for r in results:
        r["threshold"] = float(threshold)
        r["prediction"] = "anomalous" if r["anomaly_score"] > threshold else "normal"

    return results
