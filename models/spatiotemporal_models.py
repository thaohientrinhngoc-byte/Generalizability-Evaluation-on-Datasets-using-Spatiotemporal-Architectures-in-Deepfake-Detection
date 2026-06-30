"""
Spatiotemporal Architectures for Deepfake Detection.

Configurations (Table I from paper):
  - M_CNN_F (M_CNN-F): EfficientNet-B0 (Fully Frozen)             -> GAP        -> Linear Classifier
  - M_CNN_T (M_CNN-T): EfficientNet-B0 (Tuned Blocks 7 & 8)        -> GAP        -> Linear Classifier
  - M_CL_F  (M_CL-F) : EfficientNet-B0 (Fully Frozen)             -> 3-L LSTM   -> Frame Logits Mean
  - M_CL_T  (M_CL-T) : EfficientNet-B0 (Tuned Blocks 7 & 8)        -> 3-L LSTM   -> Frame Logits Mean
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SpatiotemporalDeepfakeDetector(nn.Module):
    """
    Spatiotemporal Deepfake Detector based on EfficientNet-B0 backbone and optional LSTM temporal head.

    Args:
        model_type (str): Model configuration identifier. Supported values:
            - 'M_CNN_F' or 'M_CNN-F' (Frozen CNN + GAP)
            - 'M_CNN_T' or 'M_CNN-T' (Tuned CNN + GAP)
            - 'M_CL_F'  or 'M_CL-F'  (Frozen CNN + 3-Layer LSTM)
            - 'M_CL_T'  or 'M_CL-T'  (Tuned CNN + 3-Layer LSTM)
        hidden_dim (int): Hidden dimension for LSTM temporal module (default: 256).
        num_layers (int): Number of LSTM layers (default: 3).
    """

    def __init__(self, model_type: str, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()

        # Normalize model type string
        mt = model_type.upper().replace("-", "_").strip()
        # Aliases mapping
        alias_map = {
            "MODEL_A": "M_CNN_T",
            "MODEL_B": "M_CL_F",
            "MODEL_C": "M_CL_T",
            "MODEL_D": "M_CNN_F",
        }
        if mt in alias_map:
            mt = alias_map[mt]

        valid_types = {"M_CNN_F", "M_CNN_T", "M_CL_F", "M_CL_T"}
        if mt not in valid_types:
            raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {valid_types}.")

        self.model_type = mt

        # ── Backbone Setup ──────────────────────────────────────────────────
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280

        # Register ImageNet normalization constants as non-trainable GPU buffers
        self.register_buffer(
            "norm_mean",
            torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1),
        )

        self._apply_freezing_logic()

        # ── Temporal Head Setup ──────────────────────────────────────────────
        if self.model_type in ("M_CL_F", "M_CL_T"):
            self.lstm = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0,
            )
            self.classifier = nn.Linear(hidden_dim, 1)
        elif self.model_type in ("M_CNN_F", "M_CNN_T"):
            self.classifier = nn.Linear(self.feature_dim, 1)

    def _apply_freezing_logic(self):
        """Applies parameter freezing or selective unfreezing based on Paper Table I."""
        if self.model_type in ("M_CNN_T", "M_CL_T"):
            # Fine-tuned top blocks (Blocks 7 and 8)
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.features[7].parameters():
                param.requires_grad = True
            for param in self.backbone.features[8].parameters():
                param.requires_grad = True

        elif self.model_type in ("M_CNN_F", "M_CL_F"):
            # Fully frozen CNN backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input float tensor of shape [B, T, C, H, W] in range [0.0, 1.0].

        Returns:
            video_logits: Output float tensor [B] containing raw logits (no sigmoid applied).
        """
        B, T, C, H, W = x.shape

        # Flatten batch + time dimensions for 2D CNN backbone
        x_flat = x.view(B * T, C, H, W)

        # Apply ImageNet normalization directly on GPU tensor
        x_norm = (x_flat - self.norm_mean) / self.norm_std

        features = self.backbone(x_norm)            # [B*T, feature_dim]
        features = features.view(B, T, -1)          # [B, T, feature_dim]

        if self.model_type in ("M_CL_F", "M_CL_T"):
            lstm_out, _ = self.lstm(features)       # [B, T, hidden_dim]
            per_frame_logits = self.classifier(lstm_out).squeeze(-1)  # [B, T]
            return per_frame_logits.mean(dim=1)      # [B]

        elif self.model_type in ("M_CNN_F", "M_CNN_T"):
            features_mean = features.mean(dim=1)    # [B, feature_dim]
            return self.classifier(features_mean).squeeze(-1)  # [B]
