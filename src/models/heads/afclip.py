from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class AFCLIPHead(BaseHead):
    """AF-CLIP style anomaly-focused CLIP head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.embed_dim = config.get("embed_dim", 768)
        self.category = config.get("category", "object")
        self.image_size = config.get("image_size", 224)
        self.temperature = config.get("temperature", 1.0)
        self.use_scale_weights = config.get("use_scale_weights", True)
        self.attn_hidden_dim = config.get("attn_hidden_dim", 256)

        self.normal_prompts = config.get(
            "normal_prompts",
            [
                "a photo of a normal {}.",
                "a good photo of a {}.",
                "a flawless {}.",
                "a perfect {}.",
                "a {} without defects.",
            ],
        )
        self.abnormal_prompts = config.get(
            "abnormal_prompts",
            [
                "a photo of a damaged {}.",
                "a photo of a defective {}.",
                "a {} with defects.",
                "a {} with anomalies.",
                "a broken {}.",
            ],
        )

        self.visual_adapter = nn.Linear(self.embed_dim, self.embed_dim)
        self.anomaly_adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.attn_hidden_dim),
            nn.GELU(),
            nn.Linear(self.attn_hidden_dim, 1),
        )
        self.scale_weights: nn.Parameter | None = None

        self._visual_to_text_proj: nn.Linear | None = None
        self._normal_embed: torch.Tensor | None = None
        self._abnormal_embed: torch.Tensor | None = None
        self._backbone: Any = None

    def set_backbone(self, backbone: Any) -> None:
        self._backbone = backbone
        text_dim = getattr(backbone, "embed_dim", self.embed_dim)
        if text_dim != self.embed_dim:
            self.embed_dim = text_dim
            self.visual_adapter = nn.Linear(self.embed_dim, self.embed_dim)
            self.anomaly_adapter = nn.Sequential(
                nn.Linear(self.embed_dim, self.attn_hidden_dim),
                nn.GELU(),
                nn.Linear(self.attn_hidden_dim, 1),
            )
            self._normal_embed = None
            self._abnormal_embed = None

        visual_dim = self.embed_dim
        if hasattr(backbone, "model") and hasattr(backbone.model, "visual"):
            visual = backbone.model.visual
            proj = getattr(visual, "proj", None)
            if proj is not None:
                if hasattr(proj, "weight"):
                    visual_dim = proj.weight.shape[0]
                elif hasattr(proj, "data"):
                    visual_dim = proj.data.shape[0]
                elif hasattr(proj, "shape"):
                    visual_dim = proj.shape[0]

        if visual_dim != self.embed_dim:
            self._visual_to_text_proj = nn.Linear(visual_dim, self.embed_dim)
            if list(self.parameters()):
                self._visual_to_text_proj.to(next(self.parameters()).device)
        else:
            self._visual_to_text_proj = None

    def set_category(self, category: str) -> None:
        self.category = category
        self._normal_embed = None
        self._abnormal_embed = None

    def fit(self, features: list[torch.Tensor]) -> None:
        if self.use_scale_weights and self.scale_weights is None:
            self.scale_weights = nn.Parameter(torch.zeros(len(features)))

    def _ensure_text_embeddings(self) -> None:
        if self._backbone is None:
            raise RuntimeError("Backbone not set. Call set_backbone() first.")

        device = next(self.parameters()).device

        if self._normal_embed is None:
            normal_texts = [p.format(self.category) for p in self.normal_prompts]
            with torch.no_grad():
                normal_embeds: torch.Tensor = self._backbone.encode_text(normal_texts)
                normal_embeds = normal_embeds / normal_embeds.norm(dim=-1, keepdim=True)
                normal_embed: torch.Tensor = normal_embeds.mean(dim=0)
                if normal_embed.dim() > 1:
                    normal_embed = normal_embed.mean(dim=0)
                normal_embed = normal_embed / normal_embed.norm()
                self._normal_embed = normal_embed.to(device)

        if self._abnormal_embed is None:
            abnormal_texts = [p.format(self.category) for p in self.abnormal_prompts]
            with torch.no_grad():
                abnormal_embeds: torch.Tensor = self._backbone.encode_text(
                    abnormal_texts
                )
                abnormal_embeds = abnormal_embeds / abnormal_embeds.norm(
                    dim=-1, keepdim=True
                )
                abnormal_embed: torch.Tensor = abnormal_embeds.mean(dim=0)
                if abnormal_embed.dim() > 1:
                    abnormal_embed = abnormal_embed.mean(dim=0)
                abnormal_embed = abnormal_embed / abnormal_embed.norm()
                self._abnormal_embed = abnormal_embed.to(device)

    def _aggregate_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) == 1:
            return features[0]

        target_h, target_w = features[-1].shape[2:]
        resized = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            resized.append(feat)

        if self.use_scale_weights and self.scale_weights is not None:
            weights = torch.softmax(self.scale_weights, dim=0)
            stacked = torch.stack(resized, dim=0)
            return (weights[:, None, None, None, None] * stacked).sum(dim=0)

        return torch.stack(resized, dim=0).mean(dim=0)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        self._ensure_text_embeddings()
        normal_embed_ref: torch.Tensor = cast(torch.Tensor, self._normal_embed)
        abnormal_embed_ref: torch.Tensor = cast(torch.Tensor, self._abnormal_embed)

        feat = self._aggregate_features(features)
        bsz, _, h, w = feat.shape

        visual_feat = feat.permute(0, 2, 3, 1).reshape(bsz, h * w, -1)
        if self._visual_to_text_proj is not None:
            visual_feat = self._visual_to_text_proj(visual_feat)
        visual_feat = self.visual_adapter(visual_feat)
        visual_feat = F.normalize(visual_feat, dim=-1)

        normal_embed: torch.Tensor = F.normalize(normal_embed_ref, dim=-1)
        abnormal_embed: torch.Tensor = F.normalize(abnormal_embed_ref, dim=-1)

        sim_normal = torch.einsum("bnd,d->bn", visual_feat, normal_embed)
        sim_abnormal = torch.einsum("bnd,d->bn", visual_feat, abnormal_embed)
        sim_stack = torch.stack([sim_normal, sim_abnormal], dim=-1)
        probs = F.softmax(sim_stack / self.temperature, dim=-1)
        anomaly_probs = probs[..., 1]

        attn = torch.sigmoid(self.anomaly_adapter(visual_feat)).squeeze(-1)
        weighted_probs = anomaly_probs * attn

        anomaly_map = weighted_probs.view(bsz, h, w)
        image_score = weighted_probs.mean(dim=1)

        anomaly_map_upsampled = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return {
            "anomaly_score": image_score,
            "anomaly_map": anomaly_map_upsampled,
            "anomaly_map_raw": anomaly_map,
        }
