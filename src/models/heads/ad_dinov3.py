from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import BaseHead


class ADDINOv3Head(BaseHead):
    """AD-DINOv3 inspired calibration head for DINOv3 features."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.feature_aggregation = config.get("feature_aggregation", "concat")
        self.normalize_features = config.get("normalize_features", True)

        self.prototype: torch.Tensor | None = None

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        flat = aggregated.flatten(2).mean(dim=2)
        if self.normalize_features:
            flat = F.normalize(flat, p=2, dim=1)
        self.prototype = flat.mean(dim=0)
        if self.normalize_features:
            self.prototype = F.normalize(self.prototype, p=2, dim=0)

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
        if self.feature_aggregation == "mean":
            return torch.stack(resized).mean(dim=0)
        return torch.stack(resized).max(dim=0)[0]

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.prototype is None:
            raise RuntimeError("Model not fitted")

        aggregated = self._aggregate_features(features)
        bsz, _, h, w = aggregated.shape
        flat = aggregated.flatten(2).transpose(1, 2)

        if self.normalize_features:
            flat = F.normalize(flat, p=2, dim=-1)
            prototype = F.normalize(self.prototype, p=2, dim=0)
        else:
            prototype = self.prototype

        distances = torch.cdist(flat, prototype.unsqueeze(0))
        anomaly_map = distances.squeeze(-1).view(bsz, h, w)
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}
