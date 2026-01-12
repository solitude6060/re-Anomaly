from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedAnomalyDetector(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.backbone_type = config.get("backbone", "dinov3_vitl16")
        self.use_sdg = config.get("use_sdg", True)
        self.use_attention_prior = config.get("use_attention_prior", False)

        self.projection_dim = config.get("projection_dim", 256)
        self.flow_blocks = config.get("flow_blocks", 8)
        self.discriminator_hidden = config.get("discriminator_hidden", [256, 128])

        self.backbone: nn.Module | None = None
        self.projection: nn.Module | None = None
        self.flow_head: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.memory_bank: torch.Tensor | None = None

        self._initialized = False
        self._feature_dim: int | None = None

    def _build_projection(self, in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, self.projection_dim, 1),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.projection_dim, self.projection_dim, 1),
        )

    def _build_flow(self, in_channels: int) -> nn.Module:
        from src.models.heads.fastflow import FastFlow2D

        return FastFlow2D(
            in_channels=in_channels,
            num_blocks=self.flow_blocks,
            hidden_dims=[256, 256],
            clamp=2.0,
        )

    def _build_discriminator(self, in_dim: int) -> nn.Module:
        layers = []
        prev_dim = in_dim
        for hidden_dim in self.discriminator_hidden:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def initialize(self, features: list[torch.Tensor]) -> None:
        if self._initialized:
            return

        aggregated = self._aggregate_features(features)
        in_channels = aggregated.shape[1]
        self._feature_dim = in_channels

        self.projection = self._build_projection(in_channels)
        self.flow_head = self._build_flow(self.projection_dim)
        self.discriminator = self._build_discriminator(self.projection_dim)

        self._initialized = True

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

    def _get_flow_score(self, projected: torch.Tensor) -> torch.Tensor:
        assert self.flow_head is not None
        z, log_det = self.flow_head(projected)
        log_pz = -0.5 * (z**2).sum(dim=1)
        log_prob = log_pz + log_det.unsqueeze(-1).unsqueeze(-1) / (
            z.shape[-1] * z.shape[-2]
        )
        return -log_prob

    def _get_discriminator_score(self, projected: torch.Tensor) -> torch.Tensor:
        assert self.discriminator is not None
        b, c, h, w = projected.shape
        flat = projected.permute(0, 2, 3, 1).reshape(b * h * w, c)
        scores = self.discriminator(flat).view(b, h, w)
        return torch.sigmoid(scores)

    def _get_memory_score(self, projected: torch.Tensor) -> torch.Tensor:
        if self.memory_bank is None:
            return torch.zeros(
                projected.shape[0],
                projected.shape[2],
                projected.shape[3],
                device=projected.device,
            )

        b, c, h, w = projected.shape
        flat = projected.permute(0, 2, 3, 1).reshape(b * h * w, c)

        flat_norm = F.normalize(flat, p=2, dim=1)
        bank_norm = F.normalize(self.memory_bank, p=2, dim=1)

        dists = torch.cdist(flat_norm, bank_norm)
        k = min(9, dists.shape[1])
        knn_dists, _ = dists.topk(k, dim=1, largest=False)
        scores = knn_dists.mean(dim=1)

        return scores.view(b, h, w)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self._initialized:
            self.initialize(features)
            self.to(features[0].device)

        aggregated = self._aggregate_features(features)

        assert self.projection is not None
        projected = self.projection(aggregated)
        projected = F.normalize(projected, p=2, dim=1)

        flow_map = self._get_flow_score(projected)
        disc_map = self._get_discriminator_score(projected)
        memory_map = self._get_memory_score(projected)

        flow_weight = self.config.get("flow_weight", 0.5)
        disc_weight = self.config.get("disc_weight", 0.3)
        memory_weight = self.config.get("memory_weight", 0.2)

        flow_norm = (flow_map - flow_map.min()) / (
            flow_map.max() - flow_map.min() + 1e-8
        )
        memory_norm = (memory_map - memory_map.min()) / (
            memory_map.max() - memory_map.min() + 1e-8
        )

        combined_map = (
            flow_weight * flow_norm
            + disc_weight * disc_map
            + memory_weight * memory_norm
        )

        anomaly_score = combined_map.amax(dim=(1, 2))

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": combined_map,
            "flow_map": flow_map,
            "disc_map": disc_map,
            "memory_map": memory_map,
        }

    def fit(self, features: list[torch.Tensor]) -> None:
        self.initialize(features)

    def fit_memory_bank(
        self, all_features: list[torch.Tensor], max_samples: int = 10000
    ) -> None:
        if not self._initialized:
            self.initialize(all_features[:1] if all_features else [])

        all_patches = []
        for features in all_features:
            aggregated = self._aggregate_features(features)
            assert self.projection is not None
            projected = self.projection(aggregated)
            b, c, h, w = projected.shape
            patches = projected.permute(0, 2, 3, 1).reshape(-1, c)
            all_patches.append(patches)

        all_patches = torch.cat(all_patches, dim=0)

        if all_patches.shape[0] > max_samples:
            indices = torch.randperm(all_patches.shape[0])[:max_samples]
            all_patches = all_patches[indices]

        self.memory_bank = all_patches.detach()

    def compute_training_loss(
        self,
        features: list[torch.Tensor],
        aug_features: list[torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self._initialized:
            self.initialize(features)
            self.to(features[0].device)

        aggregated = self._aggregate_features(features)
        assert self.projection is not None
        projected = self.projection(aggregated)
        projected_norm = F.normalize(projected, p=2, dim=1)

        assert self.flow_head is not None
        z, log_det = self.flow_head(projected_norm)
        log_pz = -0.5 * (z**2).sum(dim=1)
        num_spatial = z.shape[-1] * z.shape[-2]
        log_det_per_loc = log_det.unsqueeze(-1).unsqueeze(-1) / num_spatial
        log_prob = log_pz + log_det_per_loc
        flow_loss = -log_prob.mean()

        disc_loss = torch.tensor(0.0, device=features[0].device)
        if labels is not None and aug_features is not None:
            aug_aggregated = self._aggregate_features(aug_features)
            aug_projected = self.projection(aug_aggregated)

            b, c, h, w = projected.shape
            normal_flat = projected.permute(0, 2, 3, 1).reshape(-1, c)
            aug_flat = aug_projected.permute(0, 2, 3, 1).reshape(-1, c)

            assert self.discriminator is not None
            normal_scores = self.discriminator(normal_flat).squeeze()
            aug_scores = self.discriminator(aug_flat).squeeze()

            normal_mask = (
                (labels == 0).unsqueeze(-1).unsqueeze(-1).expand(-1, h, w).reshape(-1)
            )
            aug_mask = (
                (labels == 1).unsqueeze(-1).unsqueeze(-1).expand(-1, h, w).reshape(-1)
            )

            if normal_mask.any():
                disc_loss = disc_loss + F.binary_cross_entropy_with_logits(
                    normal_scores[normal_mask],
                    torch.zeros_like(normal_scores[normal_mask]),
                )
            if aug_mask.any():
                disc_loss = disc_loss + F.binary_cross_entropy_with_logits(
                    aug_scores[aug_mask], torch.ones_like(aug_scores[aug_mask])
                )

        total_loss = flow_loss + 0.5 * disc_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "disc_loss": disc_loss,
        }
