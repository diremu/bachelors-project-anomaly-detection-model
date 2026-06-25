"""
preprocessing.py — Frame discovery, clip construction, transforms,
and data collection for pre-extracted frame datasets.

Handles two layouts:
  - UCF-Crime flat: {Category}/{VideoPrefix}_{frameIdx}.jpg
  - Avenue / ShanghaiTech subdir: {video_dir}/{frame}.jpg
"""
import os, json, gc, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config as cfg

# ──────────────────────────────────────────────────────────────
#  1.  Frame scan with persistent caching
# ──────────────────────────────────────────────────────────────
_SCAN_CACHE = {}

def _load_disk_cache():
    if cfg.SCAN_CACHE_FILE.exists():
        try:
            with open(cfg.SCAN_CACHE_FILE) as f:
                return {k: [Path(p) for p in v] for k, v in json.load(f).items()}
        except Exception:
            return {}
    return {}

def _save_disk_cache():
    try:
        ser = {k: [str(p) for p in v] for k, v in _SCAN_CACHE.items()}
        tmp = cfg.SCAN_CACHE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(ser, f)
        tmp.replace(cfg.SCAN_CACHE_FILE)
    except Exception:
        pass

_SCAN_CACHE.update(_load_disk_cache())


def discover_frame_paths(root, use_cache=True):
    """Find all image frame files under *root* (single-level iterdir)."""
    root = Path(root)
    key = str(root)
    if use_cache and key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    frames = []
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix.lower() in cfg.IMAGE_EXTENSIONS:
                frames.append(entry)
    frames.sort()
    if use_cache:
        _SCAN_CACHE[key] = frames
        _save_disk_cache()
    return frames


# ──────────────────────────────────────────────────────────────
#  2.  Grouping helpers
# ──────────────────────────────────────────────────────────────
def extract_video_id(path):
    name = Path(path).stem
    parts = name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else name

def extract_frame_index(path):
    return int(Path(path).stem.split("_")[-1])

def group_frames_by_video_prefix(paths):
    """Group frame paths by their video prefix, sorted chronologically."""
    groups = defaultdict(list)
    for p in paths:
        groups[extract_video_id(p)].append(p)
    return {
        vid: sorted(frames, key=extract_frame_index)
        for vid, frames in groups.items()
    }

def get_groups(directory, use_cache=True):
    """Return {video_prefix: [sorted frame paths]} for a directory."""
    frames = discover_frame_paths(directory, use_cache=use_cache)
    return group_frames_by_video_prefix(frames)


# ──────────────────────────────────────────────────────────────
#  3.  Image loading and transforms
# ──────────────────────────────────────────────────────────────
def load_image(path):
    """Load image as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_frame(img, transform):
    """Apply torchvision transform to an RGB numpy image → tensor."""
    return transform(img)

def get_transforms(augment=False, size=None):
    size = size or cfg.FRAME_SIZE
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────
#  4.  Clip generation
# ──────────────────────────────────────────────────────────────
def group_into_clips(tensors, clip_length=None, stride=None):
    """Sliding-window clip generation from a list of frame tensors."""
    cl = clip_length or cfg.CLIP_LENGTH
    st = stride or cfg.STRIDE
    clips = []
    for i in range(0, len(tensors) - cl + 1, st):
        clips.append(torch.stack(tensors[i:i + cl]))
    return clips


# ──────────────────────────────────────────────────────────────
#  5.  Clip collection from a directory
# ──────────────────────────────────────────────────────────────
def collect_clips_from_dir(directory, label, category_name,
                           max_videos=None, max_frames=200,
                           prefixes=None, transform=None,
                           _precomputed_groups=None):
    """Collect clips from a frame directory.

    Frames are sorted chronologically within each video.
    The returned clip list is SHUFFLED to prevent memorisation.
    Supports both flat (UCF) and subdir (Avenue/ShanghaiTech) layouts.
    """
    directory = Path(directory)
    if transform is None:
        transform = get_transforms(augment=False)
    clips, labels, categories = [], [], []

    # ── Flat / prefix layout (UCF-Crime) ──
    groups = _precomputed_groups or get_groups(directory)
    if groups:
        if prefixes is None:
            prefixes = sorted(groups.keys())
            if max_videos is not None:
                prefixes = prefixes[:max_videos]
        for prefix in prefixes:
            if prefix not in groups:
                continue
            paths = sorted(groups[prefix],
                           key=lambda p: int(p.stem.split("_")[-1]))[:max_frames]
            if len(paths) >= cfg.CLIP_LENGTH:
                tensors = [preprocess_frame(load_image(p), transform) for p in paths]
                for clip in group_into_clips(tensors):
                    clips.append(clip)
                    labels.append(label)
                    categories.append(category_name)
        combined = list(zip(clips, labels, categories))
        random.shuffle(combined)
        if combined:
            clips, labels, categories = map(list, zip(*combined))
        return clips, labels, categories

    # ── Subdir layout (Avenue / ShanghaiTech) ──
    all_subdirs = sorted(d for d in directory.iterdir() if d.is_dir())
    if prefixes is not None:
        name_map = {d.name: d for d in all_subdirs}
        selected = [name_map[p] for p in prefixes if p in name_map]
    else:
        selected = all_subdirs
        if max_videos is not None:
            selected = selected[:max_videos]
    for vd in selected:
        paths = sorted(
            (f for f in vd.iterdir()
             if f.is_file() and f.suffix.lower() in cfg.IMAGE_EXTENSIONS),
            key=lambda p: p.stem
        )[:max_frames]
        if len(paths) >= cfg.CLIP_LENGTH:
            tensors = [preprocess_frame(load_image(p), transform) for p in paths]
            for clip in group_into_clips(tensors):
                clips.append(clip)
                labels.append(label)
                categories.append(category_name)
    combined = list(zip(clips, labels, categories))
    random.shuffle(combined)
    if combined:
        clips, labels, categories = map(list, zip(*combined))
    return clips, labels, categories


# ──────────────────────────────────────────────────────────────
#  6.  Train / eval prefix splitting
# ──────────────────────────────────────────────────────────────
def split_prefixes(directory, train_ratio=None, seed=42,
                   _precomputed_groups=None):
    """Deterministic train/eval split by video prefix or subdir name."""
    train_ratio = train_ratio or cfg.TRAIN_RATIO
    directory = Path(directory)
    rng = random.Random(seed)
    groups = _precomputed_groups or get_groups(directory)
    if groups:
        prefixes = sorted(groups.keys())
        rng.shuffle(prefixes)
        cut = max(1, int(len(prefixes) * train_ratio))
        return prefixes[:cut], prefixes[cut:]
    subdirs = [d.name for d in sorted(directory.iterdir()) if d.is_dir()]
    rng.shuffle(subdirs)
    cut = max(1, int(len(subdirs) * train_ratio))
    return subdirs[:cut], subdirs[cut:]


# ──────────────────────────────────────────────────────────────
#  7.  PyTorch Dataset & DataLoader
# ──────────────────────────────────────────────────────────────
class AnomalyClipDataset(Dataset):
    """Dataset of pre-loaded clip tensors with labels and categories."""
    def __init__(self, clips, labels, categories, shuffle=True):
        assert len(clips) == len(labels) == len(categories)
        indices = list(range(len(clips)))
        if shuffle:
            random.shuffle(indices)
        self.clips      = [clips[i] for i in indices]
        self.labels     = [labels[i] for i in indices]
        self.categories = [categories[i] for i in indices]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx], self.labels[idx], self.categories[idx]


def create_dataloader(clips, labels, categories,
                      batch_size=None, shuffle=False, drop_last=False):
    ds = AnomalyClipDataset(clips, labels, categories, shuffle=shuffle)
    return DataLoader(ds, batch_size=batch_size or cfg.BATCH_SIZE,
                      shuffle=False, num_workers=0, drop_last=drop_last)

def build_loaders(train_clips, train_labels, train_categories,
                  eval_clips, eval_labels, eval_categories,
                  batch_size=None):
    """Build train (shuffled, drop_last) and eval DataLoaders."""
    bs = batch_size or cfg.BATCH_SIZE
    train_loader = create_dataloader(train_clips, train_labels, train_categories,
                                     batch_size=bs, shuffle=True, drop_last=True)
    eval_loader  = create_dataloader(eval_clips, eval_labels, eval_categories,
                                     batch_size=bs, shuffle=False, drop_last=False)
    return train_loader, eval_loader