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


def get_transforms(augment: bool = False):
    transforms = [
        A.Resize(FRAME_SIZE, FRAME_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if augment:
        augs = [
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
        ]
        transforms = augs + transforms
    return A.Compose(transforms)


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


def process_dataset(
    dataset_path: Path,
    augment: bool = False,
    clip_length: int = 16,
    stride: int = 8,
    max_frames: int = 300,
) -> Generator[tuple[torch.Tensor, str], None, None]:
    """
    Process a dataset directory, yielding (clip_tensor, label).

    Auto-detects structure:
    - Category dirs with frames grouped by prefix (UCF-Crime: Train/Fighting/*.png)
    - Leaf dirs with frames as one video sequence
    - Video files (.avi/.mp4)
    """
    transform = get_transforms(augment=augment)

    for entry in sorted(dataset_path.iterdir()):
        if not entry.is_dir():
            continue

        # Check if this entry contains frames directly
        has_frames = bool(discover_frame_paths(entry))
        has_subs = any(e.is_dir() for e in entry.iterdir())

        if has_frames and not has_subs:
            # Category directory with flat frames (UCF-Crime style)
            clips = process_frame_directory(entry, transform, clip_length, stride, max_frames)
            for clip in clips:
                yield clip, entry.name
        elif has_subs:
            # Nested: recurse into subdirectories
            for sub in sorted(entry.iterdir()):
                if not sub.is_dir():
                    continue
                if discover_frame_paths(sub):
                    clips = process_frame_directory(sub, transform, clip_length, stride, max_frames)
                    for clip in clips:
                        yield clip, entry.name  # label = parent category
                else:
                    # Deeper nesting
                    for item in process_dataset(sub, augment, clip_length, stride, max_frames):
                        yield item

    # Also check for video files
    video_files = sorted(dataset_path.rglob("*"))
    video_files = [vf for vf in video_files if vf.suffix.lower() in VIDEO_EXTENSIONS]
    for vf in video_files:
        clips = process_video_file(vf, transform, clip_length, stride, max_frames)
        for clip in clips:
            yield clip, vf.stem
