from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead
from .simplenet import SimpleNetHead


class SALADHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        structural_config = config.get("structural_stream", {})
        logical_config = config.get("logical_stream", {})
        fusion_config = config.get("fusion", {})

        self.structural_stream = SimpleNetHead(structural_config)
        self.logical_stream = ComponentAwareHead(logical_config)

        self.fusion_method = fusion_config.get("method", "weighted_sum")
        self.structural_weight = fusion_config.get("structural_weight", 0.5)
        self.logical_weight = fusion_config.get("logical_weight", 0.5)

    def fit(self, features: list[torch.Tensor]) -> None:
        self.structural_stream.fit(features)
        self.logical_stream.fit(features)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        structural_out = self.structural_stream(features)
        logical_out = self.logical_stream(features)

        structural_score = structural_out["anomaly_score"]
        logical_score = logical_out["anomaly_score"]

        if self.fusion_method == "weighted_sum":
            anomaly_score = (
                self.structural_weight * structural_score
                + self.logical_weight * logical_score
            )
        elif self.fusion_method == "max":
            anomaly_score = torch.max(structural_score, logical_score)
        else:
            anomaly_score = (structural_score + logical_score) / 2

        structural_map = structural_out.get(
            "anomaly_map", structural_score.unsqueeze(-1).unsqueeze(-1)
        )
        logical_map = logical_out.get(
            "anomaly_map", logical_score.unsqueeze(-1).unsqueeze(-1)
        )

        if structural_map.shape != logical_map.shape:
            target_size = structural_map.shape[-2:]
            logical_map = F.interpolate(
                logical_map.unsqueeze(1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        anomaly_map = (
            self.structural_weight * structural_map + self.logical_weight * logical_map
        )

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "structural_score": structural_score,
            "logical_score": logical_score,
        }


class ComponentAwareHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.num_components = config.get("num_components", 10)

        self.encoder: nn.Module | None = None
        self.component_embeddings: nn.Parameter | None = None

    def fit(self, features: list[torch.Tensor]) -> None:
        in_channels = sum(f.shape[1] for f in features)

        layers = []
        prev_dim = in_channels
        for dim in self.hidden_dims:
            layers.extend([nn.Conv2d(prev_dim, dim, 1), nn.ReLU(inplace=True)])
            prev_dim = dim
        self.encoder = nn.Sequential(*layers)

        self.component_embeddings = nn.Parameter(
            torch.randn(self.num_components, prev_dim)
        )

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.encoder is None or self.component_embeddings is None:
            raise RuntimeError("Model not fitted")

        target_size = features[0].shape[-2:]
        resized = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            resized.append(feat)

        aggregated = torch.cat(resized, dim=1)
        encoded = self.encoder(aggregated)

        batch_size, channels, h, w = encoded.shape
        flat = encoded.permute(0, 2, 3, 1).reshape(-1, channels)

        embeddings = F.normalize(self.component_embeddings, dim=1)
        flat_norm = F.normalize(flat, dim=1)

        similarity = torch.mm(flat_norm, embeddings.t())
        max_sim, _ = similarity.max(dim=1)
        anomaly_values = 1 - max_sim

        anomaly_map = anomaly_values.reshape(batch_size, h, w)
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}
