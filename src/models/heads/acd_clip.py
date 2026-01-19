from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class ACDCLIPHead(BaseHead):
    """ACD-CLIP style zero-shot head with Conv-LoRA-like adaptation."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.embed_dim = config.get("embed_dim", 768)
        self.category = config.get("category", "object")
        self.image_size = config.get("image_size", 224)
        self.temperature = config.get("temperature", 1.0)
        self.lora_rank = config.get("lora_rank", 8)
        self.lora_scale = config.get("lora_scale", 1.0)

        self.normal_prompts = config.get(
            "normal_prompts",
            [
                "a photo of a normal {}.",
                "a good photo of a {}.",
                "a flawless {}.",
                "a perfect {}.",
            ],
        )
        self.abnormal_prompts = config.get(
            "abnormal_prompts",
            [
                "a photo of a damaged {}.",
                "a photo of a defective {}.",
                "a {} with defects.",
                "a {} with anomalies.",
            ],
        )

        self.visual_adapter = nn.Linear(self.embed_dim, self.embed_dim)
        self.lora_a = nn.Linear(self.embed_dim, self.lora_rank, bias=False)
        self.lora_b = nn.Linear(self.lora_rank, self.embed_dim, bias=False)

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
            self.lora_a = nn.Linear(self.embed_dim, self.lora_rank, bias=False)
            self.lora_b = nn.Linear(self.lora_rank, self.embed_dim, bias=False)
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
        pass

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

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        self._ensure_text_embeddings()
        assert self._normal_embed is not None
        assert self._abnormal_embed is not None
        normal_embed_ref = self._normal_embed
        abnormal_embed_ref = self._abnormal_embed

        if len(features) == 1:
            feat = features[0]
        else:
            target_h, target_w = features[-1].shape[2:]
            resized = []
            for f in features:
                if f.shape[2:] != (target_h, target_w):
                    f = F.interpolate(
                        f,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                resized.append(f)
            feat = torch.stack(resized, dim=0).mean(dim=0)

        bsz, _, h, w = feat.shape
        visual_feat = feat.permute(0, 2, 3, 1).reshape(bsz, h * w, -1)

        if self._visual_to_text_proj is not None:
            visual_feat = self._visual_to_text_proj(visual_feat)

        lora_update = self.lora_b(self.lora_a(visual_feat)) * self.lora_scale
        visual_feat = self.visual_adapter(visual_feat) + lora_update
        visual_feat = F.normalize(visual_feat, dim=-1)

        normal_embed = F.normalize(normal_embed_ref, dim=-1)
        abnormal_embed = F.normalize(abnormal_embed_ref, dim=-1)

        sim_normal = torch.einsum("bnd,d->bn", visual_feat, normal_embed)
        sim_abnormal = torch.einsum("bnd,d->bn", visual_feat, abnormal_embed)
        sim_stack = torch.stack([sim_normal, sim_abnormal], dim=-1)
        probs = F.softmax(sim_stack / self.temperature, dim=-1)
        anomaly_probs = probs[..., 1]

        anomaly_map = anomaly_probs.view(bsz, h, w)
        image_score = anomaly_probs.mean(dim=1)

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
