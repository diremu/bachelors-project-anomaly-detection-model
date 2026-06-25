"""
eval_utils.py — Comprehensive evaluation: AUC-ROC, EER, DET curve,
confusion matrix, precision-recall, per-category breakdown,
fixed operating-point metrics, score smoothing, and visualisations.
"""
import json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, det_curve,
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
)

import config as cfg

matplotlib.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150,
                            "savefig.bbox": "tight"})


# ──────────────────────────────────────────────────────────────
#  1.  Anomaly Scoring
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_scores(model, loader, device):
    """Run inference on *loader*, return (scores, labels, categories)."""
    model.eval()
    y_true, y_score, y_cats = [], [], []
    for clips, labels, categories in loader:
        clips = clips.to(device)
        scores = model.anomaly_score(clips)
        y_true.extend(labels if isinstance(labels, list) else
                      labels.cpu().numpy().tolist())
        y_score.extend(scores.cpu().numpy().tolist())
        y_cats.extend(categories)
    return np.array(y_score), np.array(y_true), np.array(y_cats)


def smooth_scores(scores, window=5):
    """Centered moving-average smoothing."""
    kernel = np.ones(window) / window
    padded = np.pad(scores, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(scores)]


# ──────────────────────────────────────────────────────────────
#  2.  Threshold selection
# ──────────────────────────────────────────────────────────────
def find_threshold_f1(y_true, y_score):
    """Threshold that maximises F1."""
    _, _, thresholds = roc_curve(y_true, y_score)
    best_f1, best_t = 0.0, thresholds[0]
    for t in thresholds:
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)

def find_threshold_youden(y_true, y_score):
    """Threshold via Youden's J statistic (argmax TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


# ──────────────────────────────────────────────────────────────
#  3.  Equal Error Rate (EER)
# ──────────────────────────────────────────────────────────────
def compute_eer(y_true, y_score):
    """EER = point where FPR == FNR."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    try:
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, fpr.min(), fpr.max())
    except Exception:
        idx = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2)
    return float(eer)


# ──────────────────────────────────────────────────────────────
#  4.  Full metrics computation
# ──────────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_score, threshold=None):
    if threshold is None:
        threshold = find_threshold_f1(y_true, y_score)
    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    m = {
        "auc_roc":        round(roc_auc_score(y_true, y_score), 4),
        "avg_precision":  round(average_precision_score(y_true, y_score), 4),
        "eer":            round(compute_eer(y_true, y_score), 4),
        "threshold":      round(threshold, 6),
        "accuracy":       round(accuracy_score(y_true, preds), 4),
        "balanced_acc":   round(balanced_accuracy_score(y_true, preds), 4),
        "precision":      round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":         round(recall_score(y_true, preds, zero_division=0), 4),
        "f1":             round(f1_score(y_true, preds, zero_division=0), 4),
        "specificity":    round(tn / (tn + fp + 1e-8), 4),
        "npv":            round(tn / (tn + fn + 1e-8), 4),
        "mcc":            round(matthews_corrcoef(y_true, preds), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_normal":       int((y_true == 0).sum()),
        "n_anomaly":      int((y_true == 1).sum()),
    }
    return m


def print_metrics(m):
    print("\n" + "=" * 55)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 55)
    for key in ["auc_roc", "avg_precision", "eer", "threshold",
                "accuracy", "balanced_acc", "precision", "recall",
                "f1", "specificity", "npv", "mcc"]:
        print(f"  {key:20s}: {m[key]}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  Samples: {m['n_normal']} normal / {m['n_anomaly']} anomaly")
    print("=" * 55 + "\n")


def save_metrics(m, path=None):
    path = path or cfg.METRICS_FILE
    with open(str(path), "w") as f:
        json.dump(m, f, indent=2)
    print(f"Metrics saved → {path}")


# ──────────────────────────────────────────────────────────────
#  5.  ROC Curve
# ──────────────────────────────────────────────────────────────
def plot_roc(y_true, y_score, y_score_smooth=None, save=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_raw = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#3498db",
            label=f"Raw (AUC={auc_raw:.4f})")
    if y_score_smooth is not None:
        fpr_s, tpr_s, _ = roc_curve(y_true, y_score_smooth)
        auc_s = auc(fpr_s, tpr_s)
        ax.plot(fpr_s, tpr_s, lw=2, color="#e74c3c",
                label=f"Smoothed (AUC={auc_s:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#3498db")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — ResNet-18 + LSTM Autoencoder")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  6.  Confusion Matrix
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, preds, threshold=None, save=None):
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    title = "Confusion Matrix"
    if threshold is not None:
        title += f"  (threshold={threshold:.4f})"
    ax.set_title(title)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  7.  Precision-Recall Curve
# ──────────────────────────────────────────────────────────────
def plot_pr_curve(y_true, y_score, y_score_smooth=None, save=None):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, color="#3498db", label=f"Raw (AP={ap:.4f})")
    if y_score_smooth is not None:
        p2, r2, _ = precision_recall_curve(y_true, y_score_smooth)
        ap2 = average_precision_score(y_true, y_score_smooth)
        ax.plot(r2, p2, lw=2, color="#e74c3c", label=f"Smoothed (AP={ap2:.4f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", ls="--", lw=1,
               label=f"Random (AP={baseline:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  8.  DET Curve
# ──────────────────────────────────────────────────────────────
def plot_det(y_true, y_score, y_score_smooth=None, save=None):
    fpr_d, fnr_d, _ = det_curve(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_d, fnr_d, lw=2, color="#3498db",
            label=f"Raw (EER={eer*100:.2f}%)")
    if y_score_smooth is not None:
        f2, n2, _ = det_curve(y_true, y_score_smooth)
        eer2 = compute_eer(y_true, y_score_smooth)
        ax.plot(f2, n2, lw=2, color="#e74c3c",
                label=f"Smoothed (EER={eer2*100:.2f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="Miss Rate (FNR)",
           title="Detection Error Tradeoff (DET) Curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  9.  EER on ROC
# ──────────────────────────────────────────────────────────────
def plot_eer(y_true, y_score, save=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    eer = compute_eer(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#3498db", label=f"ROC (AUC={auc_val:.4f})")
    ax.plot(fpr, 1 - fpr, "k--", lw=1, label="FPR = FNR")
    ax.scatter([eer], [1 - eer], color="red", s=80, zorder=5,
               label=f"EER = {eer*100:.2f}%")
    ax.set(xlabel="FPR", ylabel="TPR",
           title="ROC Curve with Equal Error Rate")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  10.  Score Distribution
# ──────────────────────────────────────────────────────────────
def plot_score_distribution(y_true, y_score, threshold=None, save=None):
    norm_s = y_score[y_true == 0]
    anom_s = y_score[y_true == 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(y_score.min(), y_score.max(), 50)
    ax1.hist(norm_s, bins=bins, alpha=0.6, color="#2ecc71", density=True,
             label=f"Normal (n={len(norm_s)})")
    ax1.hist(anom_s, bins=bins, alpha=0.6, color="#e74c3c", density=True,
             label=f"Anomaly (n={len(anom_s)})")
    if threshold is not None:
        ax1.axvline(threshold, color="black", ls="--", lw=1.5,
                    label=f"Threshold={threshold:.4f}")
    ax1.set(xlabel="Anomaly Score", ylabel="Density",
            title="Score Distribution")
    ax1.legend()
    ax2.boxplot([norm_s, anom_s], labels=["Normal", "Anomalous"],
                patch_artist=True,
                boxprops=dict(facecolor="#ecf0f1"),
                medianprops=dict(color="black", lw=2))
    if threshold is not None:
        ax2.axhline(threshold, color="black", ls="--", lw=1.5)
    ax2.set(ylabel="Anomaly Score", title="Score Boxplot")
    fig.suptitle("Anomaly Score Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()

    sep = (anom_s.mean() - norm_s.mean()) / ((norm_s.std() + anom_s.std()) / 2 + 1e-8)
    print(f"Normal    — mean={norm_s.mean():.4f}  std={norm_s.std():.4f}  "
          f"median={np.median(norm_s):.4f}")
    print(f"Anomalous — mean={anom_s.mean():.4f}  std={anom_s.std():.4f}  "
          f"median={np.median(anom_s):.4f}")
    print(f"Cohen's d : {sep:.4f}")
    return fig


# ──────────────────────────────────────────────────────────────
#  11.  Training Curves
# ──────────────────────────────────────────────────────────────
def plot_training_curves(history, save=None):
    eps = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(eps, [h["train_loss"] for h in history], "o-", color="#3498db",
            label="Train")
    a1.plot(eps, [h["val_loss"] for h in history], "s-", color="#e74c3c",
            label="Val")
    a1.set(xlabel="Epoch", ylabel="Reconstruction Loss",
           title="Training & Validation Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(eps, [h["val_auc"] for h in history], "D-", color="#27ae60",
            label="Val AUC")
    a2.set(xlabel="Epoch", ylabel="AUC-ROC",
           title="Validation AUC-ROC", ylim=(0, 1)); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle("Training Progress", fontsize=13); fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  12.  Grid Search bar chart
# ──────────────────────────────────────────────────────────────
def plot_grid_results(results, save=None):
    ranked = sorted(results, key=lambda x: x["auc"], reverse=True)
    labels = [f"h{r['lstm_hidden']}_L{r['lstm_layers']}_"
              f"u{r['unfreeze_layers']}_lr{r['lr']}" for r in ranked]
    means = [r["auc"] for r in ranked]
    stds  = [r.get("auc_std", 0) for r in ranked]

    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.4)))
    colors = ["#2563eb" if i == 0 else "#93c5fd" for i in range(len(labels))]
    ax.barh(range(len(labels)), means, xerr=stds, height=0.6,
            color=colors, edgecolor="white", capsize=3)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Mean AUC-ROC", title="Grid Search Results (ranked)")
    ax.grid(axis="x", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.003, i, f"{m:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    if save: fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  13.  Per-category breakdown
# ──────────────────────────────────────────────────────────────
def per_category_report(y_true, y_score, y_cats, threshold):
    preds = (y_score >= threshold).astype(int)
    print(f"\n{'Category':28s} {'N':>5} {'Label':>9} {'MeanScore':>10} "
          f"{'AUC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 85)
    for cat in sorted(set(y_cats)):
        mask = y_cats == cat
        cs, cl, cp = y_score[mask], y_true[mask], preds[mask]
        lbl = "normal" if cl[0] == 0 else "anomalous"
        ms = cs.mean()
        cat_auc = roc_auc_score(cl, cs) if len(np.unique(cl)) > 1 else float("nan")
        p = precision_score(cl, cp, zero_division=0)
        r = recall_score(cl, cp, zero_division=0)
        f = f1_score(cl, cp, zero_division=0)
        auc_s = f"{cat_auc:.4f}" if not np.isnan(cat_auc) else "  N/A "
        print(f"{cat:28s} {mask.sum():5d} {lbl:>9} {ms:10.6f} "
              f"{auc_s:>7} {p:7.4f} {r:7.4f} {f:7.4f}")
    print("-" * 85)


# ──────────────────────────────────────────────────────────────
#  14.  Fixed operating-point metrics (TPR @ FAR)
# ──────────────────────────────────────────────────────────────
def operating_point_metrics(y_true, y_score, threshold):
    fpr_op, tpr_op, thresh_op = roc_curve(y_true, y_score)
    print("\nTPR at fixed FAR operating points:")
    for far in [0.01, 0.05, 0.10]:
        idx = min(np.searchsorted(fpr_op, far, side="right") - 1,
                  len(tpr_op) - 1)
        idx = max(0, idx)
        print(f"  FAR={far*100:.0f}%  →  TPR = {tpr_op[idx]*100:.2f}%  "
              f"(threshold={thresh_op[idx]:.4f})")


# ──────────────────────────────────────────────────────────────
#  15.  Full evaluation pipeline
# ──────────────────────────────────────────────────────────────
def full_evaluation(model, test_loader, device, save_dir=None):
    """Run complete evaluation: scores → metrics → all plots.

    Returns (metrics, y_score, y_true, y_cats).
    """
    save_dir = Path(save_dir or cfg.OUTPUT_DIR)
    print("Collecting anomaly scores …")
    y_score, y_true, y_cats = collect_scores(model, test_loader, device)
    y_smooth = smooth_scores(y_score, window=5)

    print("Computing threshold & metrics …")
    threshold = find_threshold_f1(y_true, y_score)
    youden_t  = find_threshold_youden(y_true, y_score)
    metrics   = compute_all_metrics(y_true, y_score, threshold)
    print_metrics(metrics)
    save_metrics(metrics, save_dir / "final_metrics.json")

    preds = (y_score >= threshold).astype(int)

    print("Generating visualisations …\n")
    plot_roc(y_true, y_score, y_smooth, save=save_dir / "roc_curve.png")
    plot_confusion_matrix(y_true, preds, threshold,
                          save=save_dir / "confusion_matrix.png")
    plot_pr_curve(y_true, y_score, y_smooth,
                  save=save_dir / "pr_curve.png")
    plot_eer(y_true, y_score, save=save_dir / "eer_curve.png")
    plot_det(y_true, y_score, y_smooth, save=save_dir / "det_curve.png")
    plot_score_distribution(y_true, y_score, threshold,
                            save=save_dir / "score_distributions.png")

    print(f"\n{classification_report(y_true, preds, target_names=['Normal', 'Anomalous'])}")

    per_category_report(y_true, y_score, y_cats, threshold)
    operating_point_metrics(y_true, y_score, threshold)

    # Smoothed AUC
    auc_smooth = roc_auc_score(y_true, y_smooth)
    print(f"\nAUC (raw):      {metrics['auc_roc']:.4f}")
    print(f"AUC (smoothed): {auc_smooth:.4f}")

    return metrics, y_score, y_true, y_cats


# ──────────────────────────────────────────────────────────────
#  16.  Temporal anomaly trend plot (§3.8 requirement)
# ──────────────────────────────────────────────────────────────
def plot_temporal_trend(times, raw_scores, smoothed_scores,
                        video_id="", threshold=None, save=None):
    """Plot anomaly score over time for a single video sequence.

    Demonstrates that the LSTM produces smooth, interpretable temporal
    trends rather than noisy frame-level scores (doc §3.8).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, raw_scores, alpha=0.35, color="#94a3b8", lw=1,
            label="Raw scores")
    ax.plot(times, smoothed_scores, color="#2563eb", lw=2,
            label="Smoothed trend")
    if threshold is not None:
        ax.axhline(threshold, color="#ef4444", ls="--", lw=1.5,
                   label=f"Threshold = {threshold:.4f}")
        # Shade anomalous regions
        above = smoothed_scores >= threshold
        ax.fill_between(times, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        where=above, alpha=0.08, color="#ef4444")
    ax.set(xlabel="Time (seconds)", ylabel="Anomaly Score",
           title=f"Temporal Anomaly Trend — {video_id}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(str(save))
    plt.show(); plt.close()
    return fig


# ──────────────────────────────────────────────────────────────
#  17.  Per-dataset contribution chart
# ──────────────────────────────────────────────────────────────
def plot_dataset_contribution(categories_list, title="Training Data Sources",
                              save=None):
    """Show how many clips each dataset/category contributed."""
    from collections import Counter
    counts = Counter(categories_list)
    names = sorted(counts.keys())
    values = [counts[n] for n in names]
    colors = []
    for n in names:
        if "Normal" in n:
            colors.append("#2ecc71")
        elif n in cfg.UCF_ANOMALY_CATS:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.5)))
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.set(xlabel="Number of Clips", title=title)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(str(save))
    plt.show(); plt.close()
    return fig