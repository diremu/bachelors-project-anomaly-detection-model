"""
train_utils.py — Training loops, early stopping, warmup scheduling,
grid search with 3-run averaging and persistent checkpointing.
"""
import os, gc, json, time, random
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config as cfg
from model import SpatiotemporalModel, count_parameters

# ──────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
#  Early stopping
# ──────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.triggered = False

    def __call__(self, value):
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        self.triggered = self.counter >= self.patience
        return self.triggered


# ──────────────────────────────────────────────────────────────
#  Loss: reconstruction + temporal smoothness + variance penalty
# ──────────────────────────────────────────────────────────────
_mse = nn.MSELoss()

def compute_loss(recon, feats,
                 temporal_w=None, variance_w=None):
    """Combined training loss.

    - Reconstruction MSE between original and reconstructed features
    - Temporal smoothness: MSE on inter-frame deltas
    - Variance penalty: prevents decoder from collapsing to constant output
    """
    temporal_w = temporal_w if temporal_w is not None else cfg.TEMPORAL_LOSS_W
    variance_w = variance_w if variance_w is not None else cfg.VARIANCE_PENALTY_W

    recon_loss = _mse(recon, feats)

    orig_d  = feats[:, 1:, :] - feats[:, :-1, :]
    recon_d = recon[:, 1:, :] - recon[:, :-1, :]
    temporal_loss = _mse(recon_d, orig_d)

    recon_var = recon.var(dim=1).mean()
    var_penalty = torch.clamp(0.1 - recon_var, min=0.0)

    total = recon_loss + temporal_w * temporal_loss + variance_w * var_penalty
    return total, recon_loss.item(), temporal_loss.item()


# ──────────────────────────────────────────────────────────────
#  AMP compatibility
# ──────────────────────────────────────────────────────────────
try:
    from torch.amp import autocast as _autocast_cls, GradScaler as _GradScaler
    def _amp_ctx(dev, enabled):
        return _autocast_cls(device_type=dev.type, enabled=enabled)
    def _make_scaler(dev, enabled):
        return _GradScaler(dev.type, enabled=enabled)
except (ImportError, TypeError):
    from torch.cuda.amp import autocast as _autocast_cls, GradScaler as _GradScaler
    def _amp_ctx(dev, enabled):
        return _autocast_cls(enabled=enabled)
    def _make_scaler(dev, enabled):
        return _GradScaler(enabled=enabled)


# ──────────────────────────────────────────────────────────────
#  Full training run
# ──────────────────────────────────────────────────────────────
def train_model(model, train_loader, eval_loader, device,
                lr=1e-3, cnn_lr=None, epochs=25,
                warmup_epochs=None, es_patience=None,
                verbose=True):
    """Train for *epochs* with warmup, early stopping, and teacher forcing.

    Returns (best_auc, history, best_state_dict).
    """
    cnn_lr        = cnn_lr or cfg.CNN_LR
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.WARMUP_EPOCHS
    es_patience   = es_patience if es_patience is not None else cfg.ES_PATIENCE
    use_amp       = device.type == "cuda"

    optimizer = torch.optim.Adam([
        {"params": filter(lambda p: p.requires_grad,
                          model.cnn_encoder.parameters()), "lr": cnn_lr},
        {"params": model.lstm_ae.parameters(), "lr": lr},
    ], weight_decay=cfg.WEIGHT_DECAY)

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=cfg.SCHEDULER_PATIENCE, factor=cfg.SCHEDULER_FACTOR,
    )
    early_stop = EarlyStopping(patience=es_patience, mode="min")
    scaler = _make_scaler(device, use_amp)

    best_auc, best_val_loss = 0.0, float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tf_ratio = max(0.1, 1.0 - epoch / epochs)

        # ── Train ──
        model.train()
        train_loss, grad_norms = 0.0, []

        for clips, labels, categories in train_loader:
            clips = clips.to(device)
            optimizer.zero_grad(set_to_none=True)

            with _amp_ctx(device, use_amp):
                recon, feats = model(clips, teacher_forcing_ratio=tf_ratio)
                loss, rl, tl = compute_loss(recon, feats)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            grad_norms.append(gn.item())
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)

        # ── Eval ──
        model.eval()
        val_loss = 0.0
        y_true, y_score = [], []

        with torch.no_grad():
            for clips, labels, categories in eval_loader:
                clips = clips.to(device)
                recon, feats = model(clips, teacher_forcing_ratio=0.0)
                loss, _, _ = compute_loss(recon, feats)
                val_loss += loss.item()
                scores = model.lstm_ae.compute_reconstruction_error(feats, recon)
                y_true.extend(labels if isinstance(labels, list) else
                              labels.cpu().numpy().tolist())
                y_score.extend(scores.cpu().numpy().tolist())

        avg_val = val_loss / max(len(eval_loader), 1)
        y_arr = np.array(y_true)
        auc = roc_auc_score(y_arr, np.array(y_score)) \
              if len(np.unique(y_arr)) >= 2 else 0.5

        # ── Scheduling ──
        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(avg_val)

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
            "val_auc": auc, "lr": optimizer.param_groups[1]["lr"],
            "grad_norm": np.mean(grad_norms) if grad_norms else 0,
            "tf_ratio": tf_ratio, "time_s": round(elapsed, 1),
        }
        history.append(record)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if auc > best_auc:
            best_auc = auc

        if verbose:
            print(f"  Ep {epoch:02d}/{epochs} | train {avg_train:.6f} | "
                  f"val {avg_val:.6f} | AUC {auc:.4f} | "
                  f"grad {record['grad_norm']:.3f} | tf {tf_ratio:.2f} | "
                  f"{elapsed:.0f}s")

        if early_stop(avg_val):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_auc, history, best_state


# ──────────────────────────────────────────────────────────────
#  Grid Search with checkpoint-resume and multi-run averaging
# ──────────────────────────────────────────────────────────────
def _load_grid_results(path=None):
    path = path or cfg.GRID_RESULTS_FILE
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return []

def _save_grid_results(results, path=None):
    path = path or cfg.GRID_RESULTS_FILE
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def run_grid_search(train_loader, eval_loader, device,
                    n_runs=None, epochs=None):
    """Exhaustive grid search, 3 runs per config, checkpoint-resumable.

    Uses the SAME dataloaders for all runs (loaders are already shuffled
    at construction). Each run uses a different random seed.
    """
    n_runs = n_runs or cfg.GRID_RUNS_PER_CFG
    epochs = epochs or cfg.GRID_EPOCHS

    keys   = list(cfg.GRID_SPACE.keys())
    combos = list(product(*[cfg.GRID_SPACE[k] for k in keys]))
    total  = len(combos)

    results   = _load_grid_results()
    completed = {
        (r["lstm_hidden"], r["lstm_layers"], r["unfreeze_layers"], r["lr"])
        for r in results
    }

    print(f"\n{'='*60}")
    print(f"  Grid Search: {total} configs × {n_runs} runs = {total * n_runs} total")
    print(f"{'='*60}\n")

    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        key = (params["lstm_hidden"], params["lstm_layers"],
               params["unfreeze_layers"], params["lr"])

        if key in completed:
            print(f"[{ci}/{total}] {params} — already done, skipping.")
            continue

        print(f"\n[{ci}/{total}] {params}")
        run_aucs = []

        for run in range(n_runs):
            set_seed(run * 10 + ci)

            model = SpatiotemporalModel(
                lstm_hidden=params["lstm_hidden"],
                lstm_layers=params["lstm_layers"],
                unfreeze_layers=params["unfreeze_layers"],
                dropout=cfg.DROPOUT,
            ).to(device)

            best_auc, _, _ = train_model(
                model, train_loader, eval_loader, device,
                lr=params["lr"], epochs=epochs,
                warmup_epochs=cfg.WARMUP_EPOCHS,
                es_patience=cfg.ES_PATIENCE,
                verbose=True,
            )
            run_aucs.append(best_auc)
            print(f"  Run {run+1}/{n_runs} → AUC {best_auc:.4f}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append({
            **params,
            "auc":      float(np.mean(run_aucs)),
            "auc_std":  float(np.std(run_aucs)),
            "run_aucs": run_aucs,
        })
        _save_grid_results(results)
        print(f"  Mean AUC: {np.mean(run_aucs):.4f} ± {np.std(run_aucs):.4f}")

    # Summary
    results_sorted = sorted(results, key=lambda x: x["auc"], reverse=True)
    print(f"\n{'='*60}")
    print("  Grid Search Complete — Top 5")
    print(f"{'='*60}")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"  {i}. h={r['lstm_hidden']} L={r['lstm_layers']} "
              f"u={r['unfreeze_layers']} lr={r['lr']} → "
              f"AUC {r['auc']:.4f} ± {r['auc_std']:.4f}")

    _save_grid_results(results)
    return results


def best_config_from_grid(results=None):
    """Return params dict for the highest mean-AUC config."""
    if results is None:
        results = _load_grid_results()
    best = max(results, key=lambda x: x["auc"])
    print(f"Best: h={best['lstm_hidden']} L={best['lstm_layers']} "
          f"u={best['unfreeze_layers']} lr={best['lr']} "
          f"(AUC {best['auc']:.4f})")
    return {
        "lstm_hidden":     best["lstm_hidden"],
        "lstm_layers":     best["lstm_layers"],
        "unfreeze_layers": best["unfreeze_layers"],
        "lr":              best["lr"],
    }