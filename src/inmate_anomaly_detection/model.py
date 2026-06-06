"""ResNet-18 spatial encoder + LSTM temporal model with reconstruction decoder."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .config import (
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, BIDIRECTIONAL,
    CNN_FEATURE_DIM, FREEZE_BACKBONE, UNFREEZE_LAYERS, FRAME_SIZE,
)


class ResNet18Encoder(nn.Module):
    """ResNet-18 without the final classification layer.
    Outputs a 512-dim feature vector per frame.
    """

    def __init__(self, freeze_backbone: bool = FREEZE_BACKBONE, unfreeze_layers: int = UNFREEZE_LAYERS):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,   # 64-dim, H/4
            backbone.layer2,   # 128-dim, H/8
            backbone.layer3,   # 256-dim, H/16
            backbone.layer4,   # 512-dim, H/32
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._configure_finetune(freeze_backbone, unfreeze_layers)

    def _configure_finetune(self, freeze: bool, unfreeze: int):
        """Freeze early layers, unfreeze the last N layer blocks."""
        if not freeze:
            return

        # Always freeze the stem
        for i in range(3):  # conv1 + bn1 + relu
            for p in self.features[i].parameters():
                p.requires_grad = False

        # Layer blocks: indices 4=layer1, 5=layer2, 6=layer3, 7=layer4
        layer_indices = [4, 5, 6, 7]  # layer1 through layer4
        frozen_count = 4 - unfreeze

        for idx in layer_indices[:frozen_count]:
            for p in self.features[idx].parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, 512)"""
        feats = self.features(x)
        feats = self.pool(feats)
        return feats.view(feats.size(0), -1)


class LSTMAutoencoder(nn.Module):
    """LSTM sequence autoencoder for temporal reconstruction.
    Encodes a sequence of frame features, then decodes back to the original.
    Reconstruction error serves as the anomaly score.
    """

    def __init__(
        self,
        feature_dim: int = CNN_FEATURE_DIM,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size * self.num_directions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, feature_dim) encoded frame features
        Returns:
            reconstructed: (B, T, feature_dim)
            encoded: (B, T, hidden * num_directions)  last-layer hidden states
        """
        # Encode
        encoded, (h_n, c_n) = self.encoder_lstm(x)

        # Decode — repeat last hidden state across time steps then feed through decoder LSTM
        batch_size, seq_len = x.size(0), x.size(1)
        decoder_input = encoded[:, -1:, :].repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input)

        # Project back to feature space
        reconstructed = self.proj(decoded)
        return reconstructed, encoded

    def compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE across time and feature dimensions. Shape: (B,)"""
        return torch.mean((original - reconstructed) ** 2, dim=(1, 2))


class SpatiotemporalModel(nn.Module):
    """Full model: ResNet-18 encoder + LSTM autoencoder.

    Input:  (B, T, 3, H, W)
    Output: (B, T, feature_dim)  reconstructed frame features
    """

    def __init__(
        self,
        freeze_backbone: bool = FREEZE_BACKBONE,
        unfreeze_layers: int = UNFREEZE_LAYERS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.cnn_encoder = ResNet18Encoder(
            freeze_backbone=freeze_backbone,
            unfreeze_layers=unfreeze_layers,
        )
        self.lstm_ae = LSTMAutoencoder(
            feature_dim=CNN_FEATURE_DIM,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, 224, 224)
        Returns:
            reconstructed features: (B, T, CNN_FEATURE_DIM)
        """
        B, T, C, H, W = x.shape
        # Process each frame through CNN
        x_flat = x.view(B * T, C, H, W)
        features = self.cnn_encoder(x_flat)          # (B*T, 512)
        features = features.view(B, T, -1)            # (B, T, 512)

        reconstructed, _ = self.lstm_ae(features)
        return reconstructed

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-clip anomaly scores. Higher = more anomalous."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        original_feats = self.cnn_encoder(x_flat).view(B, T, -1)  # run CNN once
        reconstructed, _ = self.lstm_ae(original_feats)            # reconstruct
        return self.lstm_ae.compute_reconstruction_error(original_feats, reconstructed)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
