from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class PatchCoreHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.k_nearest = config.get("k_nearest", 9)
        self.coreset_ratio = config.get("coreset_sampling_ratio", 0.01)
        self.coreset_method = config.get("coreset_method", "greedy")
        self.feature_aggregation = config.get("feature_aggregation", "concat")

        memory_config = config.get("memory_bank", {})
        self.max_size = memory_config.get("max_size", 100000)
        self.normalize = memory_config.get("normalize", True)
        self.use_faiss = memory_config.get("use_faiss", True)

        self.memory_bank: torch.Tensor | None = None
        self.faiss_index: Any = None

    def fit(self, features: list[torch.Tensor]) -> None:
        aggregated = self._aggregate_features(features)
        batch_size, channels, h, w = aggregated.shape
        patch_features = aggregated.permute(0, 2, 3, 1).reshape(-1, channels)

        if self.normalize:
            patch_features = F.normalize(patch_features, p=2, dim=1)

        if self.coreset_ratio < 1.0:
            patch_features = self._coreset_sampling(patch_features)

        self.memory_bank = patch_features

        if self.use_faiss:
            self._build_faiss_index()

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
        elif self.feature_aggregation == "mean":
            return torch.stack(resized).mean(dim=0)
        else:
            return torch.stack(resized).max(dim=0)[0]

    def _coreset_sampling(self, features: torch.Tensor) -> torch.Tensor:
        num_samples = int(features.shape[0] * self.coreset_ratio)
        num_samples = max(1, min(num_samples, self.max_size))

        if self.coreset_method == "random":
            indices = torch.randperm(features.shape[0])[:num_samples]
            return features[indices]

        selected_indices = [torch.randint(features.shape[0], (1,)).item()]

        for _ in range(num_samples - 1):
            selected = features[selected_indices]
            distances = torch.cdist(features, selected).min(dim=1)[0]
            new_idx = distances.argmax().item()
            selected_indices.append(new_idx)

        return features[selected_indices]

    def _build_faiss_index(self) -> None:
        if self.memory_bank is None:
            return

        try:
            import faiss

            dim = self.memory_bank.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(self.memory_bank.cpu().numpy())
        except ImportError:
            self.use_faiss = False

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not fitted")

        aggregated = self._aggregate_features(features)
        batch_size, channels, h, w = aggregated.shape
        patch_features = aggregated.permute(0, 2, 3, 1).reshape(-1, channels)

        if self.normalize:
            patch_features = F.normalize(patch_features, p=2, dim=1)

        distances = self._compute_distances(patch_features)
        anomaly_map = distances.reshape(batch_size, h, w)
        anomaly_score = anomaly_map.amax(dim=(1, 2))

        score_config = self.config.get("anomaly_score", {})
        if score_config.get("normalize", True):
            anomaly_score = (anomaly_score - anomaly_score.min()) / (
                anomaly_score.max() - anomaly_score.min() + 1e-8
            )

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}

    def _compute_distances(self, features: torch.Tensor) -> torch.Tensor:
        if self.use_faiss and self.faiss_index is not None:
            distances, _ = self.faiss_index.search(
                features.cpu().numpy(), self.k_nearest
            )
            return torch.from_numpy(distances).mean(dim=1).to(features.device)

        distances = torch.cdist(features, self.memory_bank.to(features.device))
        topk_distances, _ = distances.topk(self.k_nearest, dim=1, largest=False)
        return topk_distances.mean(dim=1)
