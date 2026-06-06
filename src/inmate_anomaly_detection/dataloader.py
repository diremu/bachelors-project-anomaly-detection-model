import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional

from .config import BATCH_SIZE, CLIP_LENGTH, STRIDE


class ClipDataset(Dataset):
    """PyTorch Dataset that stores precomputed clips, integer labels, and category names."""

    def __init__(
        self,
        clips: List[torch.Tensor],
        labels: List[int],
        categories: List[str],
    ):
        assert len(clips) == len(labels) == len(categories), \
            "clips, labels, and categories must have the same length."
        self.clips = clips
        self.labels = labels
        self.categories = categories

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        return self.clips[idx], self.labels[idx], self.categories[idx]


def collate_clips(
    batch: List[Tuple[torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate: stack clips into a batch tensor of shape
    (batch_size, clip_length, 3, 224, 224).
    Labels are stacked into a long tensor for loss computation.
    Categories are kept as a list of strings for evaluation logging.
    """
    clips, labels, categories = zip(*batch)
    clips_tensor = torch.stack(clips)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return clips_tensor, labels_tensor, list(categories)


def create_dataloader(
    clips: List[torch.Tensor],
    labels: List[int],
    categories: List[str],
    shuffle: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from preprocessed clips, integer labels, and category names.

    Args:
        clips:      List of clip tensors, each (clip_length, 3, 224, 224).
        labels:     Integer labels — 0 for normal, 1 for anomalous.
        categories: Category name per clip (e.g. 'Fighting', 'Normal_Videos').
        shuffle:    Whether to shuffle each epoch. True for training.
        batch_size: Number of clips per batch.
        num_workers: Number of subprocesses for data loading.
        drop_last:  Drop the final incomplete batch. True for training.
    """
    dataset = ClipDataset(clips, labels, categories)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_clips,
        pin_memory=torch.cuda.is_available(),  # only pin memory if GPU is present
        drop_last=drop_last,
    )
    return loader


def build_loaders(
    train_clips: List[torch.Tensor],
    train_labels: List[int],
    train_categories: List[str],
    eval_clips: Optional[List[torch.Tensor]] = None,
    eval_labels: Optional[List[int]] = None,
    eval_categories: Optional[List[str]] = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build training and evaluation DataLoaders.

    Train loader:
        - Shuffled randomly each epoch so the model never sees
          clips in the same order twice.
        - drop_last=True to keep batch sizes consistent for
          batch normalisation in the ResNet backbone.

    Eval loader:
        - Unshuffled for consistent, reproducible evaluation.
        - Includes both normal and anomalous clips.
        - drop_last=False so every clip is scored.
    """
    train_loader = create_dataloader(
        train_clips,
        train_labels,
        train_categories,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    eval_loader = None
    if eval_clips is not None and eval_labels is not None and eval_categories is not None:
        eval_loader = create_dataloader(
            eval_clips,
            eval_labels,
            eval_categories,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
        )
    return train_loader, eval_loader