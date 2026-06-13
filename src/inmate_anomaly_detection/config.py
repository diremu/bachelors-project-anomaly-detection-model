from pathlib import Path

# ---------------------------------------------------------------------------
# Image / clip dimensions
# ---------------------------------------------------------------------------
FRAME_SIZE = 224
CLIP_LENGTH = 16          # [TUNE] Try 8, 32 with matching STRIDE
STRIDE = 8                # [TUNE] Try 4 (denser) or 16 (sparser)
BATCH_SIZE = 8            # [TUNE] Try 4 (less memory) or 16 (faster if GPU fits)

# ---------------------------------------------------------------------------
# Normalization (ImageNet stats — do not change with pretrained backbone)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LSTM_HIDDEN   = 512      # [TUNE] 256-1024. Larger = more expressive, risk of overfit
LSTM_LAYERS   = 2        # [TUNE] 1-4 stacked layers
LSTM_DROPOUT  = 0.3      # [TUNE] 0.0-0.5. Regularisation between LSTM layers
BIDIRECTIONAL = False     # False per design doc; True doubles params
CNN_FEATURE_DIM = 512     # ResNet-18 output dim (fixed)

# ---------------------------------------------------------------------------
# Partial fine-tuning
# ---------------------------------------------------------------------------
FREEZE_BACKBONE = True     # When True, freeze early ResNet layers
UNFREEZE_LAYERS  = 3       # [TUNE] 1-4. Number of ResNet layer blocks to unfreeze
                            # 1 = only layer4, 4 = all layers trainable

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS       = 30          # [TUNE] More epochs may improve reconstruction
LR           = 1e-4        # [TUNE] 1e-5 to 3e-4
WEIGHT_DECAY = 1e-5        # [TUNE] 0-1e-3 for L2 regularisation
LR_PATIENCE  = 5           # Epochs without improvement before reducing LR
LR_FACTOR    = 0.5         # Multiplicative LR reduction factor
EARLY_STOP_PATIENCE = 10   # Epochs without improvement before stopping
GRAD_CLIP    = 1.0         # [TUNE] Max gradient norm. 0 = no clipping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(__file__).parent.parent.parent / "dataset"

DATASET_PATHS = {
    "avenue":       DATASET_ROOT / "Avenue Dataset",
    "ucf_crime":    DATASET_ROOT / "archive (1)",
    "shanghaitech": DATASET_ROOT / "archive (3)" / "ShanghaiTech",
}

PROCESSED_DIR  = DATASET_ROOT.parent / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"

# FIX: mkdir calls removed from module level. Running them at import time
# creates directories in any environment that imports config — including tests
# and CI — even when those paths don't exist or shouldn't be created.
# Callers that need the directories should create them explicitly, e.g.:
#   PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#   CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
# save_checkpoint() in train_utils.py already does this for CHECKPOINT_DIR.

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
ANOMALY_PERCENTILE = 95    # [TUNE] Percentile threshold for anomaly flagging.
                            # Lower = more sensitive (more false positives)