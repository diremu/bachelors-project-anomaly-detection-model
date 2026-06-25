"""
config.py — Central configuration for the Correctional Facility
Video Anomaly Detection pipeline (ResNet18-LSTM Autoencoder).

Paths match the Kaggle dataset mounts for pre-extracted frame directories.
"""
import os
from pathlib import Path

# ============================================================
#  Dataset Paths (Kaggle mounts — pre-extracted frames)
# ============================================================
UCF_TRAIN       = Path("/kaggle/input/datasets/odins0n/ucf-crime-dataset/Train")
UCF_TEST        = Path("/kaggle/input/datasets/odins0n/ucf-crime-dataset/Test")
AVE_TRAIN       = Path("/kaggle/input/datasets/vibhavvasudevan/avenue/avenue/training/frames")
AVE_TEST        = Path("/kaggle/input/datasets/vibhavvasudevan/avenue/avenue/testing/frames")
SHAN_TRAIN      = Path("/kaggle/input/datasets/nikanvasei/shanghaitech-campus-dataset/SHANGHAI/SHANGHAI_TRAIN/frames")
SHAN_TEST       = Path("/kaggle/input/datasets/nikanvasei/shanghaitech-campus-dataset-test/SHANGHAI/SHANGHAI_Test/frames")

# UCF-Crime categories relevant to correctional environments
UCF_ANOMALY_CATS  = ["Abuse", "Arrest", "Assault", "Fighting", "Shooting"]
UCF_NORMAL_CAT    = "NormalVideos"
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ============================================================
#  Preprocessing
# ============================================================
FRAME_SIZE        = 224
CLIP_LENGTH       = 16        # frames per clip
STRIDE            = 8         # 50 % overlap
BATCH_SIZE        = 4         # kept small for T4 VRAM
NUM_WORKERS       = 2

# ImageNet channel statistics
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]

# ============================================================
#  Grid Search
# ============================================================
GRID_SPACE = {
    "lstm_hidden":     [64, 128],
    "lstm_layers":     [1, 2],
    "unfreeze_layers": [2, 3],
    "lr":              [1e-3, 5e-4],
}
GRID_RUNS_PER_CFG = 3
GRID_EPOCHS       = 25
GRID_MAX_TRAIN_VIDEOS = 15    # per-source cap during grid search
GRID_MAX_EVAL_VIDEOS  = 3

# ============================================================
#  Final Training
# ============================================================
FINAL_EPOCHS       = 30
CNN_LR             = 1e-4
WEIGHT_DECAY       = 1e-5
DROPOUT            = 0.2
TEMPORAL_LOSS_W    = 0.1      # temporal smoothness weight
VARIANCE_PENALTY_W = 0.05     # variance collapse penalty
GRAD_CLIP          = 1.0
WARMUP_EPOCHS      = 3
ES_PATIENCE        = 7        # early stopping
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR   = 0.5
TRAIN_RATIO        = 0.80     # video-level train/eval split (doc §4.3: 80/20)

MAX_TRAIN_VIDEOS   = 30       # per normal source
MAX_TRAIN_FRAMES   = 300      # frames per video
MAX_EVAL_VIDEOS    = 3
MAX_EVAL_FRAMES    = 150
MAX_TEST_VIDEOS    = 2
MAX_TEST_FRAMES    = 150

# ============================================================
#  Outputs
# ============================================================
OUTPUT_DIR         = Path("/kaggle/working")
CHECKPOINT_DIR     = OUTPUT_DIR / "checkpoints"
GRID_RESULTS_FILE  = OUTPUT_DIR / "grid_search_results.json"
BEST_MODEL_FILE    = OUTPUT_DIR / "best_anomaly_model.pt"
METRICS_FILE       = OUTPUT_DIR / "final_metrics.json"
SCAN_CACHE_FILE    = OUTPUT_DIR / "_frame_scan_cache.json"