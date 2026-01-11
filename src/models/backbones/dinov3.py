"""
DINOv3 Backbone for re-Anomaly.

DINOv3 (released Aug 2025) uses:
- patch_size=16 (vs DINOv2's 14)
- RoPE position encoding (native resolution support)
- 4 register tokens
- Gram anchoring for better dense features

HuggingFace models:
- facebook/dinov3-vitb16-pretrain-lvd1689m (86M params)
- facebook/dinov3-vitl16-pretrain-lvd1689m (303M params)

Note: These are gated models requiring HuggingFace login.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class DINOv3Backbone(BaseBackbone):
    """DINOv3 backbone with RoPE and register tokens."""

    VARIANTS = {
        "dinov3_vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384, 12, 16),
        "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768, 12, 16),
        "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024, 24, 16),
        "dinov3_vith16": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", 1280, 32, 16),
        "dinov3_vit7b16": ("facebook/dinov3-vit7b16-pretrain-lvd1689m", 4096, 32, 16),
        "dinov3_convnext_tiny": (
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            768,
            12,
            32,
        ),
        "dinov3_convnext_small": (
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            768,
            12,
            32,
        ),
        "dinov3_convnext_base": (
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            1024,
            12,
            32,
        ),
        "dinov3_convnext_large": (
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            1536,
            12,
            32,
        ),
    }

    # Number of register tokens in DINOv3
    NUM_REGISTER_TOKENS = 4

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.variant = config.get("variant", "dinov3_vitl16")
        self.model_id = config.get("model_id")
        self.image_size = config.get("image_size", 224)
        self.interpolate_pos = config.get("interpolate_pos_encoding", True)
        self.use_gram_anchoring = config.get("use_gram_anchoring", True)
        self._using_fallback = False  # Track if using DINOv2 fallback

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

        # DAPT (Domain-Adaptive Pre-Training) config
        dapt_config = config.get("dapt", {})
        self.dapt_enabled = dapt_config.get("enabled", False)
        self.dapt_checkpoint = dapt_config.get("checkpoint_path")

        if config.get("pretrained", True):
            self.load_pretrained(config.get("pretrained_weights"))

        if config.get("freeze_backbone", True):
            self.freeze()

    def load_pretrained(self, path: str | None = None) -> None:
        checkpoint_path = path or self.dapt_checkpoint

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
            # Fallback: try to use DINOv2 architecture as base
            print(f"WARNING: Failed to load DINOv3 from HuggingFace: {e}")
            print("Attempting fallback to DINOv2 architecture...")
            try:
                from transformers import Dinov2Model

                # Map DINOv3 variant to closest DINOv2
                fallback_map = {
                    "dinov3_vitb16": "facebook/dinov2-base",
                    "dinov3_vitl16": "facebook/dinov2-large",
                }
                fallback_id = fallback_map.get(self.variant, "facebook/dinov2-base")
                self.model = Dinov2Model.from_pretrained(fallback_id)
                # Adjust patch_size for DINOv2 fallback
                self.patch_size = 14
                self._using_fallback = True
                print(f"Loaded fallback model: {fallback_id} (patch_size=14)")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load DINOv3 model. "
                    f"Original error: {e}. Fallback error: {e2}"
                )

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(
            hidden_size=self.embed_dim,
            num_hidden_layers=self.num_layers,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        self.model = Dinov2Model(config)
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
        """
        Extract multi-scale features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at different layers, each of shape (B, D, h, w)
        """
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

        # DINOv3 output: [batch, 1 + num_register + num_patches, embed_dim]
        # 1 CLS token + 4 register tokens + patch tokens
        # DINOv2 fallback: only 1 CLS token
        skip_tokens = 1 if self._using_fallback else (1 + self.NUM_REGISTER_TOKENS)

        for layer_idx in sorted(self.output_layers):
            feat = self._features.get(layer_idx)
            if feat is not None:
                if feat.dim() == 3:
                    # Skip CLS and register tokens, keep only patch tokens
                    feat = feat[:, skip_tokens:, :]
                feat = feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                features.append(feat)

        return features
