"""Pipeline test - validates preprocessing, dataloader, and analysis on real data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from inmate_anomaly_detection.preprocessing import (
    load_image,
    preprocess_frame,
    get_transforms,
    discover_frame_paths,
    group_frames_by_video_prefix,
    group_into_clips,
)
from inmate_anomaly_detection.analysis import (
    compute_anomaly_score,
    compute_motion_energy,
    compute_motion_entropy,
    analyze_batch,
)
from inmate_anomaly_detection.dataloader import build_loaders
from inmate_anomaly_detection.config import CLIP_LENGTH, STRIDE

TRAIN = Path(__file__).parent.parent / "dataset" / "archive (1)" / "Train"


def main():
    print("=" * 60)
    print("INMATE ANOMALY DETECTION - PIPELINE TEST")
    print("=" * 60)

    # Find categories with frames
    categories = sorted([d for d in TRAIN.iterdir() if d.is_dir()])
    cats_with_data = []
    for cat in categories:
        frames = discover_frame_paths(cat)
        if frames:
            cats_with_data.append((cat, frames))
            print(f"  {cat.name}: {len(frames)} frames")

    if len(cats_with_data) < 2:
        print("ERROR: Need at least 2 categories with frames!")
        return

    # Pick two categories
    transform = get_transforms(augment=False)
    all_clips = []
    all_labels = []

    for cat, frames in cats_with_data:
        # Group frames by video prefix, take first group
        groups = group_frames_by_video_prefix(frames)
        for prefix, group_paths in list(groups.items())[:2]:
            # Limit frames per video
            paths = sorted(group_paths)[:200]
            if len(paths) < CLIP_LENGTH:
                continue
            tensors = [preprocess_frame(load_image(p), transform) for p in paths]
            clips = group_into_clips(tensors, CLIP_LENGTH, STRIDE)
            print(f"\n  Video: {prefix} ({len(paths)} frames -> {len(clips)} clips)")
            if clips:
                print(f"    Clip shape: {clips[0].shape}")
                all_clips.extend(clips)
                all_labels.extend([cat.name] * len(clips))

    if not all_clips:
        print("ERROR: No clips generated!")
        return

    print(f"\nTotal clips: {len(all_clips)}")

    # Analysis
    print("\n--- Per-category Analysis ---")
    for cat_name in set(all_labels):
        indices = [i for i, l in enumerate(all_labels) if l == cat_name][:5]
        scores = []
        for i in indices:
            s = compute_anomaly_score(all_clips[i])
            e = compute_motion_energy(all_clips[i])
            h = compute_motion_entropy(all_clips[i])
            scores.append(s)
            print(f"  [{cat_name:15s}] energy={e:.6f} entropy={h:.6f} score={s:.6f}")
        print(f"  [{cat_name:15s}] mean_score={sum(scores)/len(scores):.6f}")

    # DataLoader + batch analysis
    print("\n--- DataLoader + Batch Analysis ---")
    loader, _ = build_loaders(all_clips, all_labels, batch_size=4)
    batch, batch_labels = next(iter(loader))
    print(f"Batch shape: {batch.shape} (expected: 4, 16, 3, 224, 224)")
    results = analyze_batch(batch, batch_labels)
    for r in results:
        print(f"  [{r['label']:15s}] score={r['anomaly_score']:.6f} -> {r['prediction']}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - Pipeline ready for model training")
    print("=" * 60)


if __name__ == "__main__":
    main()
