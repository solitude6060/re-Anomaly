from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class DINOv2Backbone(BaseBackbone):
    VARIANTS = {
        "dinov2_vits14": ("facebook/dinov2-small", 384, 12),
        "dinov2_vitb14": ("facebook/dinov2-base", 768, 12),
        "dinov2_vitl14": ("facebook/dinov2-large", 1024, 24),
        "dinov2_vitg14": ("facebook/dinov2-giant", 1536, 40),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.variant = config.get("variant", "dinov2_vitb14")
        self.model_id = config.get("model_id")
        self.image_size = config.get("image_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.interpolate_pos = config.get("interpolate_pos_encoding", True)

        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")

        model_id, self.embed_dim, self.num_layers = self.VARIANTS[self.variant]
        self.model_id = self.model_id or model_id

        self.model: nn.Module | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._features: dict[int, torch.Tensor] = {}

        if config.get("pretrained", True):
            self.load_pretrained(config.get("pretrained_weights"))

        if config.get("freeze_backbone", True):
            self.freeze()

    def load_pretrained(self, path: str | None = None) -> None:
        if path:
            self.model = torch.load(path, weights_only=False)
        else:
            try:
                from transformers import Dinov2Model

                self.model = Dinov2Model.from_pretrained(self.model_id)
            except ImportError:
                self.model = torch.hub.load("facebookresearch/dinov2", self.variant)

        self._register_hooks()

    def _register_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()

        if self.model is None:
            return

        encoder = getattr(self.model, "encoder", self.model)
        layers = getattr(encoder, "layer", getattr(encoder, "blocks", None))

        if layers is None:
            return

        for layer_idx in self.output_layers:
            if layer_idx < len(layers):

                def hook_fn(
                    module: nn.Module, input: Any, output: Any, idx: int = layer_idx
                ) -> None:
                    if isinstance(output, tuple):
                        output = output[0]
                    self._features[idx] = output

                handle = layers[layer_idx].register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._features.clear()
        batch_size = x.shape[0]

        if self.interpolate_pos and x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        _ = self.model(x)

        features = []
        h = w = self.image_size // self.patch_size

        for layer_idx in sorted(self.output_layers):
            feat = self._features.get(layer_idx)
            if feat is not None:
                if feat.dim() == 3:
                    feat = feat[:, 1:, :]
                feat = feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                features.append(feat)

        return features
