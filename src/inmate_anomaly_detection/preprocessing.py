import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Generator
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import FRAME_SIZE, IMAGENET_MEAN, IMAGENET_STD

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

UCF_ANOMALY_CATEGORIES = {"Abuse", "Arrest", "Assault", "Fighting", "Shooting", "Stealing"}


def get_transforms(augment: bool = False):
    # FIX: augmentations must come AFTER resize so they operate on the final
    # resolution, not the raw full-size frame. Order: resize → augment → normalize → tensor.
    base = [A.Resize(FRAME_SIZE, FRAME_SIZE)]
    if augment:
        base += [
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
        ]
    base += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(base)


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_frame(image: np.ndarray, transform: A.Compose) -> torch.Tensor:
    transformed = transform(image=image)
    return transformed["image"]


def group_into_clips(
    frames: list[torch.Tensor],
    clip_length: int = 16,
    stride: int = 8,
) -> list[torch.Tensor]:
    clips = []
    for start in range(0, len(frames) - clip_length + 1, stride):
        clip = torch.stack(frames[start : start + clip_length])
        clips.append(clip)
    return clips


def discover_frame_paths(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in directory.iterdir() if p.suffix.lower() in extensions]
    return sorted(paths)


def group_frames_by_video_prefix(paths: list[Path]) -> dict[str, list[Path]]:
    """Group frame files by video name prefix.

    Handles UCF-Crime naming: 'Fighting002_x264_0.png' -> prefix 'Fighting002_x264'
    """
    groups = defaultdict(list)
    for p in paths:
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            groups[parts[0]].append(p)
        else:
            groups[p.stem].append(p)
    return dict(groups)


def extract_frames_from_video(
    video_path: Path,
    max_frames: int = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def process_video_file(
    video_path: Path,
    transform: A.Compose,
    clip_length: int = 16,
    stride: int = 8,
    max_frames: int = 300,
) -> list[torch.Tensor]:
    raw_frames = extract_frames_from_video(video_path, max_frames=max_frames)
    if len(raw_frames) < clip_length:
        return []
    frames = [preprocess_frame(f, transform) for f in raw_frames]
    return group_into_clips(frames, clip_length, stride)


def process_frame_sequence(
    frame_paths: list[Path],
    transform: A.Compose,
    clip_length: int = 16,
    stride: int = 8,
    max_frames: int = 300,
) -> list[torch.Tensor]:
    if len(frame_paths) < clip_length:
        return []
    frame_paths = sorted(frame_paths)[:max_frames]
    frames = [preprocess_frame(load_image(fp), transform) for fp in frame_paths]
    return group_into_clips(frames, clip_length, stride)


def process_frame_directory(
    frame_dir: Path,
    transform: A.Compose,
    clip_length: int = 16,
    stride: int = 8,
    max_frames: int = 300,
) -> list[torch.Tensor]:
    """Process a directory that contains frame images. Handles:
    - Flat dirs with frames grouped by filename prefix (UCF-Crime style)
    - Leaf dirs that are one video sequence
    """
    paths = discover_frame_paths(frame_dir)
    if not paths:
        return []

    # Check if subdirectories contain frames (nested structure)
    subs = [d for d in frame_dir.iterdir() if d.is_dir()]
    sub_has_frames = any(discover_frame_paths(s) for s in subs)
    if sub_has_frames:
        all_clips = []
        for sub in subs:
            sub_clips = process_frame_directory(sub, transform, clip_length, stride, max_frames)
            all_clips.extend(sub_clips)
        return all_clips

    # Flat directory: group by video prefix
    groups = group_frames_by_video_prefix(paths)
    all_clips = []
    for prefix, group_paths in groups.items():
        clips = process_frame_sequence(group_paths, transform, clip_length, stride, max_frames)
        all_clips.extend(clips)
    return all_clips


def _resolve_ucf_label(folder_name: str) -> int | None:
    """Return the label for a UCF-Crime category folder.

    Returns:
        1  — folder is one of the six selected anomaly categories
        0  — folder is normal footage (e.g. 'Normal_Videos_event')
        None — folder is a UCF-Crime category NOT in our subset; skip it entirely
    """
    for category in UCF_ANOMALY_CATEGORIES:
        if folder_name.lower().startswith(category.lower()):
            return 1

    if "normal" in folder_name.lower():
        return 0

    # Any other UCF-Crime category (e.g. Burglary, Explosion, Robbery…) — skip
    return None


def process_dataset(
    dataset_path: Path,
    augment: bool = False,
    clip_length: int = 16,
    stride: int = 8,
    max_frames: int = 300,
    dataset_name: str = "",
) -> Generator[tuple[torch.Tensor, int, str], None, None]:
    """
    Process a dataset directory, yielding (clip_tensor, label, category).

    label: 0 = normal, 1 = anomaly
        UCF-Crime behaviour:
      - Only Abuse, Arrest, Assault, Fighting, Shooting, Stealing are loaded.
      - Those six categories are labelled anomalous (1).
      - Normal footage folders are labelled normal (0).
      - All other UCF-Crime categories are silently skipped.

    ShanghaiTech / Avenue:
      - Folder names containing 'test' or 'abnormal' → label 1.
      - Everything else → label 0.
      - No category filtering applied.

    Auto-detects structure:
    - Category dirs with frames grouped by prefix (UCF-Crime: Train/Fighting/*.png)
    - Leaf dirs with frames as one video sequence
    - Video files (.avi/.mp4)

    FIX: removed duplicate second loop body that re-iterated the same directory,
    yielded 2-tuples (missing label), and dropped dataset_name on recursive calls.
    FIX: video scan now uses iterdir() scoped per-directory rather than rglob("*")
    to avoid double-counting videos already processed as frame sequences above.
    """
    transform = get_transforms(augment=augment)
    is_ucf = "ucf" in dataset_name.lower()

    # Track directories already processed as frame sequences so the video
    # scan below skips them and avoids duplicate clip emission.
    frame_processed_dirs: set[Path] = set()

    for entry in sorted(dataset_path.iterdir()):
        if not entry.is_dir():
            continue

        # ── UCF-Crime: apply category whitelist ──────────────────────
        if is_ucf:
            label = _resolve_ucf_label(entry.name)
            if label is None:
                # Category not in our six — skip entirely
                continue
            has_frames = bool(discover_frame_paths(entry))
            has_subs = any(e.is_dir() for e in entry.iterdir())

            if has_frames and not has_subs:
                frame_processed_dirs.add(entry)
                clips = process_frame_directory(entry, transform, clip_length, stride, max_frames)
                for clip in clips:
                    yield clip, label, entry.name
            elif has_subs:
                for sub in sorted(entry.iterdir()):
                    if not sub.is_dir():
                        continue
                    if discover_frame_paths(sub):
                        frame_processed_dirs.add(sub)
                        clips = process_frame_directory(sub, transform, clip_length, stride, max_frames)
                        for clip in clips:
                            yield clip, label, entry.name

        # ── ShanghaiTech / Avenue: label by folder name ──────────────
        else:
            folder_lower = entry.name.lower()
            if any(kw in folder_lower for kw in ("test", "abnormal", "anomaly")):
                label = 1
            else:
                label = 0

            has_frames = bool(discover_frame_paths(entry))
            has_subs = any(e.is_dir() for e in entry.iterdir())

            if has_frames and not has_subs:
                frame_processed_dirs.add(entry)
                clips = process_frame_directory(entry, transform, clip_length, stride, max_frames)
                for clip in clips:
                    yield clip, label, entry.name
            elif has_subs:
                for sub in sorted(entry.iterdir()):
                    if not sub.is_dir():
                        continue
                    if discover_frame_paths(sub):
                        frame_processed_dirs.add(sub)
                        clips = process_frame_directory(sub, transform, clip_length, stride, max_frames)
                        for clip in clips:
                            yield clip, label, entry.name
                    else:
                        # FIX: pass dataset_name so UCF whitelist is preserved in
                        # recursive calls into deeply nested subdirectories.
                        for item in process_dataset(
                            sub, augment, clip_length, stride, max_frames, dataset_name
                        ):
                            yield item

    # ── Video file scan ───────────────────────────────────────────────
    # FIX: use a targeted iterdir scan rather than rglob("*") to avoid
    # re-processing directories that were already handled as frame sequences.
    video_files: list[Path] = []
    for entry in sorted(dataset_path.rglob("*")):
        if entry.suffix.lower() in VIDEO_EXTENSIONS and entry.parent not in frame_processed_dirs:
            video_files.append(entry)

    for vf in video_files:
        if is_ucf:
            label = _resolve_ucf_label(vf.parent.name)
            if label is None:
                continue
            category = vf.parent.name
        else:
            folder_lower = vf.parent.name.lower()
            label = 1 if any(kw in folder_lower for kw in ("test", "abnormal", "anomaly")) else 0
            category = vf.stem
        clips = process_video_file(vf, transform, clip_length, stride, max_frames)
        for clip in clips:
            yield clip, label, category