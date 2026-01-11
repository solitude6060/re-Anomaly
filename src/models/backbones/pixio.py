"""
Pixio Backbone for re-Anomaly.

Pixio (released Dec 2024) uses:
- MAE-based pretraining (not DINO)
- patch_size=16
- 8 class tokens (vs DINOv2's 1 CLS token)

HuggingFace models:
- facebook/pixio-vitb16 (86M params)
- facebook/pixio-vitl16 (304M params)

Note: Non-commercial license (FAIR Noncommercial Research License).
Requires HuggingFace login and license agreement.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class PixIOBackbone(BaseBackbone):
    VARIANTS = {
        "pixio_vitb16": ("facebook/pixio-vitb16", 768, 12, 16),
        "pixio_vitl16": ("facebook/pixio-vitl16", 1024, 24, 16),
        "pixio_vith16": ("facebook/pixio-vith16", 1280, 32, 16),
        "pixio_vit1b16": ("facebook/pixio-vit1b16", 1536, 40, 16),
        "pixio_vit5b16": ("facebook/pixio-vit5b16", 2560, 48, 16),
    }

    NUM_CLASS_TOKENS = 8

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.variant = config.get("variant", "pixio_vitl16")
        self.model_id = config.get("model_id")
        self.image_size = config.get("image_size", 256)
        self.interpolate_pos = config.get("interpolate_pos_encoding", True)
        self.num_class_tokens = config.get("num_class_tokens", self.NUM_CLASS_TOKENS)
        self._using_fallback = False

        if self.variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {self.variant}. "
                f"Available: {list(self.VARIANTS.keys())}"
            )

        model_id, self.embed_dim, self.num_layers, self.patch_size = self.VARIANTS[
            self.variant
        ]
        self.model_id = self.model_id or model_id

        self.model: nn.Module | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._features: dict[int, torch.Tensor] = {}

        mae_config = config.get("mae_pretrain", {})
        self.mae_enabled = mae_config.get("enabled", False)
        self.mae_checkpoint = mae_config.get("checkpoint_path")

        if config.get("pretrained", True):
            self.load_pretrained(config.get("pretrained_weights"))

        if config.get("freeze_backbone", True):
            self.freeze()

    def load_pretrained(self, path: str | None = None) -> None:
        checkpoint_path = path or self.mae_checkpoint

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self._build_model_from_state_dict(state_dict)
        else:
            self._load_from_huggingface()

        self._register_hooks()

    def _load_from_huggingface(self) -> None:
        try:
            from transformers import AutoModel

            if self.model_id is None:
                raise ValueError("model_id is not set")
            loaded_model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            if isinstance(loaded_model, nn.Module):
                self.model = loaded_model
            else:
                raise TypeError(f"Expected nn.Module, got {type(loaded_model)}")
        except Exception as e:
            print(f"WARNING: Failed to load Pixio from HuggingFace: {e}")
            print("Attempting fallback to ViT architecture...")
            self._using_fallback = True
            self._build_default_model()

    def _build_default_model(self) -> None:
        from transformers import ViTConfig, ViTModel

        config = ViTConfig(
            hidden_size=self.embed_dim,
            num_hidden_layers=self.num_layers,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_attention_heads=self.embed_dim // 64,
        )
        self.model = ViTModel(config)

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._build_default_model()
        if self.model is not None:
            self.model.load_state_dict(state_dict, strict=False)

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
        skip_tokens = 1 if self._using_fallback else self.num_class_tokens

        for layer_idx in sorted(self.output_layers):
            feat = self._features.get(layer_idx)
            if feat is not None:
                if feat.dim() == 3:
                    feat = feat[:, skip_tokens:, :]
                feat = feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                features.append(feat)

        return features
