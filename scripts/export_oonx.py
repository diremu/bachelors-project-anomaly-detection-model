"""Export trained SpatiotemporalModel to ONNX format for browser inference.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output console/public/model.onnx

The export splits the model into two ONNX files:
  1. encoder.onnx  — ResNet18 spatial encoder (per-frame)
  2. temporal.onnx  — LSTM autoencoder + anomaly scoring (per-clip)

Splitting avoids running the full B*T*3*224*224 tensor through a single
graph, which would exceed browser memory. The console runs the encoder
frame-by-frame, stacks the features, then runs the temporal model once.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from inmate_anomaly_detection.model import SpatiotemporalModel
from inmate_anomaly_detection.config import CNN_FEATURE_DIM


class TemporalScorer(nn.Module):
    """Wraps the LSTM autoencoder to output a scalar anomaly score."""

    def __init__(self, lstm_ae):
        super().__init__()
        self.lstm_ae = lstm_ae

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (1, T, 512) — stacked CNN features for one clip
        Returns:
            score: (1,) — reconstruction error (anomaly score)
        """
        reconstructed, _ = self.lstm_ae(features)
        error = torch.mean((features - reconstructed) ** 2, dim=(1, 2))
        return error


def export(checkpoint_path: str, output_dir: str,
           lstm_hidden: int = 256, lstm_layers: int = 2,
           unfreeze_layers: int = 3):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = SpatiotemporalModel(
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        unfreeze_layers=unfreeze_layers,
    )

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    # ── Export 1: CNN Encoder ──────────────────────────────
    encoder = model.cnn_encoder
    encoder.eval()

    dummy_frame = torch.randn(1, 3, 224, 224)
    encoder_path = out / "encoder.onnx"

    torch.onnx.export(
        encoder,
        dummy_frame,
        str(encoder_path),
        opset_version=17,
        input_names=["frame"],
        output_names=["features"],
        dynamic_axes={
            "frame": {0: "batch"},
            "features": {0: "batch"},
        },
    )
    print(f"Exported encoder → {encoder_path}  ({encoder_path.stat().st_size / 1e6:.1f} MB)")

    # ── Export 2: Temporal Scorer ──────────────────────────
    scorer = TemporalScorer(model.lstm_ae)
    scorer.eval()

    clip_len = 16
    dummy_features = torch.randn(1, clip_len, CNN_FEATURE_DIM)
    temporal_path = out / "temporal.onnx"

    torch.onnx.export(
        scorer,
        dummy_features,
        str(temporal_path),
        opset_version=17,
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "score": {0: "batch"},
        },
    )
    print(f"Exported temporal → {temporal_path}  ({temporal_path.stat().st_size / 1e6:.1f} MB)")

    # ── Verify ────────────────────────────────────────────
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(str(encoder_path))
    tmp_sess = ort.InferenceSession(str(temporal_path))

    # Run encoder on 16 frames
    frames = dummy_frame.numpy()
    feats_list = []
    for _ in range(clip_len):
        out_feat = enc_sess.run(None, {"frame": frames})[0]
        feats_list.append(out_feat)

    import numpy as np
    feats = np.stack(feats_list, axis=1)  # (1, 16, 512)
    score = tmp_sess.run(None, {"features": feats})[0]
    print(f"Verification — score shape: {score.shape}, value: {score[0]:.6f}")

    # Compare with PyTorch
    with torch.no_grad():
        pt_score = model.anomaly_score(
            torch.randn(1, clip_len, 3, 224, 224)
        )
    print(f"PyTorch ref score: {pt_score[0].item():.6f}")
    print("\nDone. Place encoder.onnx and temporal.onnx in console/public/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="console/public", help="Output directory")
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--unfreeze-layers", type=int, default=3)
    args = parser.parse_args()

    export(args.checkpoint, args.output,
           args.lstm_hidden, args.lstm_layers, args.unfreeze_layers)