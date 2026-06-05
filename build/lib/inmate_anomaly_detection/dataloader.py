import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

from .config import BATCH_SIZE, CLIP_LENGTH, STRIDE


class ClipDataset(Dataset):
    """PyTorch Dataset that stores precomputed clips and labels."""

    def __init__(
        self,
        clips: List[torch.Tensor],
        labels: List[str],
    ):
        self.clips = clips
        self.labels = labels

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.clips[idx], self.labels[idx]


def collate_clips(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    """
    Custom collate: stack clips into batch tensor of shape
    (batch_size, clip_length, 3, 224, 224).
    """
    clips, labels = zip(*batch)
    clips_tensor = torch.stack(clips)
    return clips_tensor, list(labels)


def create_dataloader(
    clips: List[torch.Tensor],
    labels: List[str],
    shuffle: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from preprocessed clips and labels.

    Args:
        clips: List of clip tensors, each (clip_length, 3, 224, 224).
        labels: Corresponding labels (video directory names).
        shuffle: Whether to shuffle each epoch (True for training).
        batch_size: Number of clips per batch.
        num_workers: Number of subprocesses for data loading.
    """
    dataset = ClipDataset(clips, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_clips,
        pin_memory=True,
    )
    return loader


def build_loaders(
    train_clips: List[torch.Tensor],
    train_labels: List[str],
    eval_clips: List[torch.Tensor] = None,
    eval_labels: List[str] = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Build training and evaluation DataLoaders.

    Train loader: shuffled, serves clips in normal order per batch.
    Eval loader: unshuffled, no augmentation assumed.
    """
    train_loader = create_dataloader(
        train_clips, train_labels, shuffle=True, batch_size=batch_size
    )
    eval_loader = None
    if eval_clips is not None and eval_labels is not None:
        eval_loader = create_dataloader(
            eval_clips, eval_labels, shuffle=False, batch_size=batch_size
        )
    return train_loader, eval_loader
