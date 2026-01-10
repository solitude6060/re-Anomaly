from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class LinearHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.projection_dim = config.get("projection_dim", 256)
        self.num_layers = config.get("num_layers", 2)
        self.activation = config.get("activation", "gelu")
        self.dropout = config.get("dropout", 0.1)
        self.feature_aggregation = config.get("feature_aggregation", "concat")
        self.normalize_features = config.get("normalize_features", True)

        self.encoder: nn.Module | None = None
        self.decoder: nn.Module | None = None

    def _build_mlp(self, in_dim: int, out_dim: int) -> nn.Module:
        layers = []
        prev_dim = in_dim

        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(prev_dim, self.projection_dim))
            if self.activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            prev_dim = self.projection_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        in_channels = aggregated.shape[1]

        self.encoder = self._build_mlp(in_channels, self.projection_dim)
        self.decoder = self._build_mlp(self.projection_dim, in_channels)

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
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Model not fitted")

        aggregated = self._aggregate_features(features)
        batch_size, channels, h, w = aggregated.shape

        flat = aggregated.permute(0, 2, 3, 1).reshape(-1, channels)

        if self.normalize_features:
            flat = F.normalize(flat, p=2, dim=1)

        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)

        reconstruction_error = ((flat - decoded) ** 2).mean(dim=1)
        anomaly_map = reconstruction_error.reshape(batch_size, h, w)
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        score_config = self.config.get("anomaly_score", {})
        if score_config.get("normalize", True):
            anomaly_score = (anomaly_score - anomaly_score.min()) / (
                anomaly_score.max() - anomaly_score.min() + 1e-8
            )

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}
