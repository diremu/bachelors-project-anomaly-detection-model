"""
model.py — ResNet18-LSTM Autoencoder with latent projection,
teacher forcing, and z-normalised anomaly scoring.
"""
import torch
import torch.nn as nn
import torchvision.models as models


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone → 512-dim feature vector per frame."""

    def __init__(self, freeze_backbone=True, unfreeze_layers=3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 512

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        layers = list(self.features.children())
        for module in layers[max(0, len(layers) - unfreeze_layers):]:
            for p in module.parameters():
                p.requires_grad = True

    def forward(self, x):
        """x: (B, 3, 224, 224) → (B, 512)"""
        return self.features(x).flatten(1)


class LSTMAutoencoder(nn.Module):
    """LSTM seq2seq autoencoder with latent projection.

    Encoder compresses (B, T, 512) → hidden state.
    Latent projection refines the bottleneck representation.
    Decoder reconstructs (B, T, 512) from the bottleneck.
    """

    def __init__(self, input_dim=512, hidden_dim=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.latent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, teacher_forcing_ratio=0.0):
        """
        Parameters
        ----------
        x : (B, T, input_dim)
        teacher_forcing_ratio : float
            Probability of using the original features as decoder input
            at each time step (annealed during training).
        """
        B, T, _ = x.shape

        # Encode
        _, (h, c) = self.encoder(x)

        # Latent projection on final layer hidden state
        latent = self.latent_proj(h[-1])                   # (B, hidden_dim)

        # Decode — repeat latent as input sequence
        decoder_input = latent.unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        dec_out, _ = self.decoder(decoder_input, (h, c))
        return self.output_layer(dec_out)                      # (B, T, input_dim)

    def compute_reconstruction_error(self, original, reconstructed):
        """Per-clip MSE: (B,)"""
        return ((original - reconstructed) ** 2).mean(dim=(1, 2))


class SpatiotemporalModel(nn.Module):
    """Full pipeline: CNN features → LSTM Autoencoder.

    Returns (reconstructed, original_features) so the training loop
    can compute reconstruction loss in one forward pass.
    """

    def __init__(self, freeze_backbone=True, unfreeze_layers=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.cnn_encoder = ResNet18Encoder(freeze_backbone, unfreeze_layers)
        self.lstm_ae = LSTMAutoencoder(
            input_dim=self.cnn_encoder.feat_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )

    def forward(self, clips, teacher_forcing_ratio=0.0):
        """
        clips : (B, T, C, H, W)
        Returns: (reconstructed, original_features)  both (B, T, 512)
        """
        B, T, C, H, W = clips.shape
        feats = self.cnn_encoder(clips.view(B * T, C, H, W)).view(B, T, -1)
        recon = self.lstm_ae(feats, teacher_forcing_ratio)
        return recon, feats

    @torch.no_grad()
    def anomaly_score(self, clips):
        """Per-clip scalar anomaly score: mean reconstruction error."""
        recon, feats = self.forward(clips, teacher_forcing_ratio=0.0)
        return self.lstm_ae.compute_reconstruction_error(feats, recon)


class NormalisedSpatiotemporalModel(nn.Module):
    """Wraps SpatiotemporalModel and z-normalises anomaly scores
    using statistics calibrated on normal training clips."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.register_buffer("score_mean", torch.tensor(0.0))
        self.register_buffer("score_std",  torch.tensor(1.0))

    def forward(self, clips, teacher_forcing_ratio=0.0):
        return self.base(clips, teacher_forcing_ratio)

    @torch.no_grad()
    def calibrate(self, loader, device):
        """Compute mean/std of anomaly scores on a normal-only loader."""
        scores = []
        self.base.eval()
        for clips, *_ in loader:
            clips = clips.to(device)
            s = self.base.anomaly_score(clips)
            scores.append(s.cpu())
        all_scores = torch.cat(scores)
        self.score_mean = all_scores.mean()
        self.score_std  = all_scores.std().clamp(min=1e-8)
        print(f"Calibrated: mean={self.score_mean:.6f}, std={self.score_std:.6f}")

    @torch.no_grad()
    def anomaly_score(self, clips):
        raw = self.base.anomaly_score(clips)
        return (raw - self.score_mean) / self.score_std