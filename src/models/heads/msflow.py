from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class MSFlowHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        flow_config = config.get("flow", {})
        self.flow_type = flow_config.get("type", "realnvp")
        self.num_blocks = flow_config.get("num_blocks", 8)
        self.hidden_dims = flow_config.get("hidden_dims", [256, 256])

        self.scales = config.get("scales", [])
        self.flows: nn.ModuleList = nn.ModuleList()

        feature_config = config.get("feature_processing", {})
        self.reduce_dim = feature_config.get("reduce_dim", True)
        self.projection_dim = feature_config.get("projection_dim", 256)

        self.projections: nn.ModuleList = nn.ModuleList()

    def _build_flow(self, in_channels: int) -> nn.Module:
        return RealNVPFlow(in_channels, self.num_blocks, self.hidden_dims)

    def fit(self, features: list[torch.Tensor]) -> None:
        self.flows.clear()
        self.projections.clear()

        for i, feat in enumerate(features):
            in_channels = feat.shape[1]

            if self.reduce_dim:
                proj = nn.Conv2d(in_channels, self.projection_dim, 1)
                self.projections.append(proj)
                in_channels = self.projection_dim

            flow = self._build_flow(in_channels)
            self.flows.append(flow)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(self.flows) == 0:
            raise RuntimeError("Flows not fitted")

        scale_scores = []
        scale_maps = []

        for i, (feat, flow) in enumerate(zip(features, self.flows)):
            if self.reduce_dim and i < len(self.projections):
                feat = self.projections[i](feat)

            log_prob = flow.log_prob(feat)
            anomaly_map = -log_prob

            scale_maps.append(anomaly_map)
            scale_scores.append(anomaly_map.amax(dim=(1, 2)))

        score_config = self.config.get("anomaly_score", {})
        weights = score_config.get("scale_weights", [1.0] * len(scale_scores))

        weighted_scores = sum(w * s for w, s in zip(weights, scale_scores))
        total_weight = sum(weights)
        anomaly_score = weighted_scores / total_weight

        target_size = scale_maps[0].shape[-2:]
        resized_maps = []
        for m in scale_maps:
            if m.shape[-2:] != target_size:
                m = F.interpolate(
                    m.unsqueeze(1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            resized_maps.append(m)

        anomaly_map = torch.stack(resized_maps).max(dim=0)[0]

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}


class RealNVPFlow(nn.Module):
    def __init__(
        self, in_channels: int, num_blocks: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.blocks.append(CouplingBlock(in_channels, hidden_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            x, log_det = block(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z**2).sum(dim=1)
        return (log_pz + log_det).mean(dim=(-2, -1))


class CouplingBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int]) -> None:
        super().__init__()
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
        log_scale = torch.tanh(log_scale)
        y2 = x2 * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=(1, 2, 3))
        return torch.cat([x1, y2], dim=1), log_det
