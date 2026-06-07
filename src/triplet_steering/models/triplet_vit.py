from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn


class TripletViTMetaRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        meta_dim: int = 3,
        fusion_dim: int = 384,
        dropout: float = 0.1,
        meta_shap_weights=None,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        image_feature_dim = self.backbone.num_features

        self.img_proj = nn.Sequential(
            nn.Linear(image_feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        if meta_shap_weights is None:
            meta_shap_weights = np.ones((meta_dim,), dtype=np.float32)

        meta_shap_weights = np.asarray(meta_shap_weights, dtype=np.float32)
        if meta_shap_weights.shape[0] != meta_dim:
            raise ValueError("meta_shap_weights size must match meta_dim")

        self.register_buffer("meta_shap_weights", torch.tensor(meta_shap_weights, dtype=torch.float32))

        concat_dim = fusion_dim * 4

        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def encode_image(self, inputs):
        features = self.backbone(inputs)
        return self.img_proj(features)

    def encode_metadata(self, metadata):
        weighted_metadata = metadata * self.meta_shap_weights.unsqueeze(0)
        return self.meta_proj(weighted_metadata)

    def forward(self, left, center, right, metadata):
        left_features = self.encode_image(left)
        center_features = self.encode_image(center)
        right_features = self.encode_image(right)
        metadata_features = self.encode_metadata(metadata)

        fused_features = torch.cat([left_features, center_features, right_features, metadata_features], dim=1)
        outputs = self.head(fused_features)
        return torch.tanh(outputs).squeeze(1)
