from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class SimpleNetHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.feature_aggregation = config.get("feature_aggregation", "concat")
        self.projection_dim = config.get("projection_dim", 256)

        disc_config = config.get("discriminator", {})
        self.num_layers = disc_config.get("num_layers", 3)
        self.activation = disc_config.get("activation", "leaky_relu")
        self.dropout = disc_config.get("dropout", 0.1)

        self.projection: nn.Module | None = None
        self.discriminator: nn.Module | None = None

    def _build_discriminator(self, in_dim: int) -> nn.Module:
        layers = []
        prev_dim = in_dim

        for i in range(self.num_layers - 1):
            out_dim = (
                self.hidden_dims[i]
                if i < len(self.hidden_dims)
                else self.hidden_dims[-1]
            )
            layers.append(nn.Linear(prev_dim, out_dim))
            if self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            prev_dim = out_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        in_channels = aggregated.shape[1]

        self.projection = nn.Conv2d(in_channels, self.projection_dim, 1)
        self.discriminator = self._build_discriminator(self.projection_dim)

    def _aggregate_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) == 1:
            return features[0]

        target_size = features[0].shape[-2:]
        resized = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            resized.append(feat)

        if self.feature_aggregation == "concat":
            return torch.cat(resized, dim=1)
        return torch.stack(resized).mean(dim=0)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.projection is None or self.discriminator is None:
            raise RuntimeError("Model not fitted")

        aggregated = self._aggregate_features(features)
        batch_size, _, h, w = aggregated.shape

        projected = self.projection(aggregated)
        flat = projected.permute(0, 2, 3, 1).reshape(-1, self.projection_dim)

        scores = self.discriminator(flat).reshape(batch_size, h, w)
        anomaly_map = torch.sigmoid(scores)
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}
