from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class RectFlowHead(BaseHead):
    """
    Rectified Flow based anomaly detection head.
    Learns straight ODE paths from data to noise, enabling single-step inference.
    Anomaly score is based on reconstruction error after flow transport.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.projection_dim = config.get("projection_dim", 256)
        self.num_timesteps = config.get("num_timesteps", 1000)
        self.inference_steps = config.get("inference_steps", 1)

        feature_config = config.get("feature_processing", {})
        self.reduce_dim = feature_config.get("reduce_dim", True)
        self.normalize = feature_config.get("normalize", True)

        self.projection: nn.Module | None = None
        self.velocity_net: nn.Module | None = None

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        in_channels = aggregated.shape[1]

        if self.reduce_dim:
            self.projection = nn.Conv2d(in_channels, self.projection_dim, 1)
            in_channels = self.projection_dim

        self.velocity_net = VelocityNetwork(
            in_channels, self.hidden_dims, self.projection_dim
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
        if self.velocity_net is None:
            raise RuntimeError("Model not fitted")

        aggregated = self._aggregate_features(features)

        if self.reduce_dim and self.projection is not None:
            aggregated = self.projection(aggregated)

        if self.normalize:
            aggregated = F.normalize(aggregated, p=2, dim=1)

        reconstructed = self._rectified_flow_transport(aggregated)

        reconstruction_error = ((aggregated - reconstructed) ** 2).mean(dim=1)
        anomaly_map = reconstruction_error
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        score_config = self.config.get("anomaly_score", {})
        if score_config.get("normalize", True):
            anomaly_score = (anomaly_score - anomaly_score.min()) / (
                anomaly_score.max() - anomaly_score.min() + 1e-8
            )

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}

    def _rectified_flow_transport(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(x)

        dt = 1.0 / self.inference_steps
        x_t = z

        for step in range(self.inference_steps):
            t = torch.ones(x.shape[0], device=x.device) * (step * dt)
            v = self.velocity_net(x_t, t)
            x_t = x_t + v * dt

        return x_t

    def compute_training_loss(self, features: list[torch.Tensor]) -> torch.Tensor:
        aggregated = self._aggregate_features(features)

        if self.reduce_dim and self.projection is not None:
            aggregated = self.projection(aggregated)

        if self.normalize:
            aggregated = F.normalize(aggregated, p=2, dim=1)

        x_1 = aggregated
        x_0 = torch.randn_like(x_1)

        t = torch.rand(x_1.shape[0], device=x_1.device)
        t_expanded = t.view(-1, 1, 1, 1)

        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

        target_velocity = x_1 - x_0
        predicted_velocity = self.velocity_net(x_t, t)

        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss


class VelocityNetwork(nn.Module):
    def __init__(
        self, in_channels: int, hidden_dims: list[int], out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = 64

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        layers = []
        prev_dim = in_channels + self.time_embed_dim

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(prev_dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = dim

        layers.append(nn.Conv2d(prev_dim, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_mlp(t.unsqueeze(-1))
        t_embed = t_embed.view(t_embed.shape[0], -1, 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])

        x_with_t = torch.cat([x, t_embed], dim=1)
        return self.net(x_with_t)
