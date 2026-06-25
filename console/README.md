# Inmate Anomaly Detection

Computer vision pipeline for predicting inmate violence in Nigerian correctional facilities via anomaly detection in surveillance footage.

## Overview

This project processes CCTV footage to detect anomalous/violent behavior using a **ResNet-18 + LSTM** architecture. The model learns to reconstruct normal activity patterns — clips with high reconstruction error are flagged as anomalous.

**Datasets**: [UCF-Crime](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset), [Avenue](https://www.kaggle.com/datasets/janeshvarsivakumar/avenue-dataset), [ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech)

## Pipeline

```
Video frames → Resize 224×224 → BGR→RGB → ImageNet normalize
    → Group into 16-frame clips (stride 8)
    → ResNet-18 spatial encoder → LSTM temporal model
    → Reconstruction error → Anomaly score
```

## Setup

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
# Clone and enter project
cd E:\Data

# Create venv and install dependencies
uv venv
uv pip install -r requirements.txt

# Or use uv sync (if lockfile exists)
uv sync
```

If `uv sync` is unavailable, install manually:

```bash
uv pip install torch torchvision opencv-python pillow numpy albumentations \
    scikit-learn matplotlib seaborn jupyter tqdm
```

### Dataset

Download the datasets from Kaggle and place them under `dataset/`:

```
dataset/
├── archive (1)/          # UCF-Crime (Train/Test with category subdirs)
├── Avenue Dataset/       # Avenue (training_videos/, testing_videos/)
└── archive (3)/          # ShanghaiTech (part_A/, part_B/)
```

## Usage

### Quick Test (baseline motion analysis)

```bash
uv run python scripts/test_pipeline.py
```

### Jupyter Notebook

```bash
uv run jupyter notebook notebooks/anomaly_detection_showcase.ipynb
```

### Train the ResNet+LSTM model

```bash
uv run python scripts/train.py
```

Key tunable parameters (edit `src/inmate_anomaly_detection/config.py` or pass `--help`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LR` | 1e-4 | Learning rate — lower = stabler, higher = faster |
| `EPOCHS` | 20 | More epochs may improve reconstruction |
| `LSTM_HIDDEN` | 512 | Hidden size — larger captures more temporal nuance |
| `LSTM_LAYERS` | 2 | Stacked LSTM depth |
| `FREEZE_BACKBONE` | True | Freeze early ResNet layers for transfer learning |
| `UNFREEZE_LAYERS` | 3 | Number of ResNet layer blocks to unfreeze (1-4) |

### Export to ONNX (for browser inference)

```
uv run python scripts/export_onnx.py \
    --checkpoint checkpoints/best.pt \
    --output console/public \
    --lstm-hidden 256 --lstm-layers 2 --unfreeze-layers 3
```

This creates `encoder.onnx` and `temporal.onnx` in `console/public/`. The web console loads them automatically — when present, the Live Feed page runs real model inference instead of simulated scores.

### Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

## Project Structure

```
├── src/inmate_anomaly_detection/
│   ├── config.py           # All hyperparameters and paths
│   ├── preprocessing.py    # Frame loading, transforms, clip grouping
│   ├── dataloader.py       # PyTorch Dataset + DataLoader
│   ├── analysis.py         # Baseline: frame differencing + motion entropy
│   ├── model.py            # ResNet-18 encoder + LSTM decoder (reconstruction)
│   └── train_utils.py      # Training loop, checkpointing, metrics
├── scripts/
│   ├── test_pipeline.py    # End-to-end validation
│   ├── train.py            # Model training entry point
│   ├── evaluate.py         # Evaluation and anomaly scoring
│   └── export_onnx.py      # Export trained model to ONNX for browser
├── notebooks/
│   └── anomaly_detection_showcase.ipynb
└── dataset/                # Downloaded datasets (gitignored)
```

## Model Architecture

```
Input: (B, 16, 3, 224, 224)
  │
  ├─ ResNet-18 (frozen early layers, unfrozen layers 3-4)
  │     └─ Feature vector: 512-dim per frame
  │
  ├─ LSTM (2 layers, hidden=512, unidirectional)
  │     └─ Encoded sequence representation
  │
  └─ Decoder (Linear layers)
        └─ Reconstructs frame features → MSE loss
```

**Anomaly scoring**: Reconstruction error of the LSTM decoder. Higher MSE = more anomalous.

## Improving Performance

Areas to experiment with, in priority order:

1. **Hyperparameter tuning** — LSTM hidden size (256–1024), learning rate (1e-5–1e-3), batch size (4–16)
2. **Unfreeze more layers** — Set `UNFREEZE_LAYERS=4` to fine-tune full ResNet
3. **Sequence length** — Try `CLIP_LENGTH=32` with `STRIDE=16` for longer temporal context
4. **Data augmentation** — Enable more aggressive augmentations during training
5. **Loss function** — Try cosine similarity loss or contrastive approaches instead of MSE
6. **Model backbone** — Swap ResNet-18 for ResNet-34 or EfficientNet-B0
7. **Attention mechanism** — Add self-attention between LSTM and decoder
8. **Mixed precision training** — Enable `torch.cuda.amp` for faster training on GPU