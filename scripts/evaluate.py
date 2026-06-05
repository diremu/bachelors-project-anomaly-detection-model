"""Evaluate the trained model and compute per-category anomaly scores.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --device cuda
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from inmate_anomaly_detection.preprocessing import process_dataset
from inmate_anomaly_detection.dataloader import build_loaders
from inmate_anomaly_detection.model import SpatiotemporalModel
from inmate_anomaly_detection.train_utils import load_checkpoint, validate
from inmate_anomaly_detection.config import (
    DATASET_PATHS, CHECKPOINT_DIR, BATCH_SIZE, ANOMALY_PERCENTILE,
)

sns.set_style("whitegrid")
sns.set_palette("Set2")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dataset", type=str, default="ucf_crime")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--percentile", type=int, default=ANOMALY_PERCENTILE,
                   help="Score percentile for anomaly threshold [TUNE]")
    return p.parse_args()


@torch.no_grad()
def compute_anomaly_scores(model, dataloader, device):
    """Compute reconstruction error per clip. Returns dict[label] = list[scores]."""
    model.eval()
    label_scores = defaultdict(list)

    for clips, labels in dataloader:
        clips = clips.to(device)
        scores = model.anomaly_score(clips)  # (B,)
        for lbl, s in zip(labels, scores.cpu().tolist()):
            label_scores[lbl].append(s)

    return dict(label_scores)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading model from: {ckpt_path}")
    model = SpatiotemporalModel().to(device)
    state = load_checkpoint(model, None, ckpt_path, str(device))
    print(f"  Trained for {state.get('epoch', '?')} epochs, loss={state.get('loss', '?'):.6f}")

    # Load test data
    test_path = DATASET_PATHS[args.dataset] / "Test"
    if not test_path.exists():
        test_path = DATASET_PATHS[args.dataset] / "Train"
        print(f"Test set not found, falling back to: {test_path}")

    print(f"\nLoading evaluation data from: {test_path}")

    test_clips, test_labels = [], []
    for clip, label in process_dataset(test_path, augment=False, max_frames=args.max_frames):
        test_clips.append(clip)
        test_labels.append(label)

    print(f"Evaluation: {len(test_clips)} clips from {len(set(test_labels))} categories")
    _, eval_loader = build_loaders(test_clips, test_labels, None, None, batch_size=BATCH_SIZE)

    # Compute scores
    label_scores = compute_anomaly_scores(model, eval_loader, device)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Box plot by category
    cat_names = sorted(label_scores.keys(), key=lambda c: np.mean(label_scores[c]))
    score_data = [label_scores[c] for c in cat_names]
    means = [np.mean(label_scores[c]) for c in cat_names]

    bp = axes[0].boxplot(score_data, vert=False, patch_artist=True, widths=0.6)
    for patch, name in zip(bp['boxes'], cat_names):
        color = '#2ecc71' if name == 'NormalVideos' else '#e74c3c'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_yticks(range(1, len(cat_names) + 1))
    axes[0].set_yticklabels(cat_names, fontsize=8)
    axes[0].set_xlabel("Reconstruction Error (MSE)")
    axes[0].set_title("Anomaly Scores by Category (Reconstruction Error)")

    # 2. Histogram: normal vs anomalous
    normal_scores = label_scores.get("NormalVideos", [])
    anomalous_scores = []
    for cat, scores in label_scores.items():
        if cat != "NormalVideos":
            anomalous_scores.extend(scores)

    all_scores = normal_scores + anomalous_scores
    threshold = np.percentile(normal_scores, args.percentile) if normal_scores else np.median(all_scores)

    axes[1].hist(normal_scores, bins=40, alpha=0.6, label=f"Normal ({len(normal_scores)})", color='#2ecc71')
    axes[1].hist(anomalous_scores, bins=40, alpha=0.6, label=f"Anomalous ({len(anomalous_scores)})", color='#e74c3c')
    axes[1].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                    label=f"Threshold (p{args.percentile})={threshold:.4f}")
    axes[1].set_xlabel("Reconstruction Error")
    axes[1].set_title("Reconstruction Error Distribution")
    axes[1].legend(fontsize=8)

    # 3. Bar chart of mean scores
    colors = ['#2ecc71' if n == 'NormalVideos' else '#e74c3c' for n in cat_names]
    axes[2].barh(cat_names, means, color=colors, alpha=0.7)
    axes[2].axvline(x=threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_xlabel("Mean Reconstruction Error")
    axes[2].set_title("Mean Anomaly Score per Category")
    axes[2].invert_yaxis()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "evaluation_results.png")
    plt.show()

    # ---- Console report ----
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"Threshold (p{args.percentile} of NormalVideos): {threshold:.6f}")
    print(f"{'='*60}")
    print(f"{'Category':20s} {'N':>6s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} {'Flagged%':>8s}")
    print("-" * 60)

    for cat in cat_names:
        s = np.array(label_scores[cat])
        flagged = np.mean(s > threshold) * 100
        print(f"{cat:20s} {len(s):6d} {s.mean():10.6f} {np.median(s):10.6f} {s.std():10.6f} {flagged:7.1f}%")

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
