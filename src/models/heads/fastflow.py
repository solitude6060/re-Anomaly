from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class FastFlowHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        flow_config = config.get("flow", {})
        self.flow_type = flow_config.get("type", "realnvp")
        self.num_blocks = flow_config.get("num_blocks", 8)
        self.hidden_dims = flow_config.get("hidden_dims", [256, 256])
        self.clamp = flow_config.get("clamp", 2.0)

        feature_config = config.get("feature_processing", {})
        self.reduce_dim = feature_config.get("reduce_dim", True)
        self.projection_dim = feature_config.get("projection_dim", 256)
        self.normalize = feature_config.get("normalize", True)

        self.projection: nn.Module | None = None
        self.flow: nn.Module | None = None

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        in_channels = aggregated.shape[1]

        if self.reduce_dim:
            self.projection = nn.Conv2d(in_channels, self.projection_dim, 1)
            in_channels = self.projection_dim

        self.flow = FastFlow2D(
            in_channels, self.num_blocks, self.hidden_dims, self.clamp
        )

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

        return torch.cat(resized, dim=1)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.flow is None:
            raise RuntimeError("Flow not fitted")

        aggregated = self._aggregate_features(features)

        if self.reduce_dim and self.projection is not None:
            aggregated = self.projection(aggregated)

        if self.normalize:
            aggregated = F.normalize(aggregated, p=2, dim=1)

        log_prob = self.flow.log_prob(aggregated)
        anomaly_map = -log_prob

        anomaly_score = anomaly_map.amax(dim=(1, 2))

        score_config = self.config.get("anomaly_score", {})
        if score_config.get("normalize", True):
            anomaly_score = (anomaly_score - anomaly_score.min()) / (
                anomaly_score.max() - anomaly_score.min() + 1e-8
            )

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}


class FastFlow2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        hidden_dims: list[int],
        clamp: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.clamp = clamp
        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.blocks.append(AffineCouplingBlock2D(in_channels, hidden_dims, clamp))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            x, log_det = block(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z**2).sum(dim=1)
        return log_pz + log_det.unsqueeze(-1).unsqueeze(-1) / (
            z.shape[-1] * z.shape[-2]
        )


class AffineCouplingBlock2D(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int], clamp: float) -> None:
        super().__init__()
        self.clamp = clamp
        self.split_dim = in_channels // 2

        layers = []
        prev_dim = self.split_dim
        for dim in hidden_dims:
            layers.extend(
                [nn.Conv2d(prev_dim, dim, 3, padding=1), nn.ReLU(inplace=True)]
            )
            prev_dim = dim
        layers.append(nn.Conv2d(prev_dim, self.split_dim * 2, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        params = self.net(x1)
        shift, log_scale = params.chunk(2, dim=1)
        log_scale = self.clamp * torch.tanh(log_scale / self.clamp)
        y2 = x2 * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=(1, 2, 3))
        return torch.cat([x1, y2], dim=1), log_det
