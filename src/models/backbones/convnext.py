"""
ConvNeXt Backbone for re-Anomaly.

ConvNeXt (CVPR 2022) provides strong CNN-based features through:
- Modernized architecture with depthwise convolutions
- LayerScale and stochastic depth
- Patchify stem with larger kernel sizes

Reference: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
arXiv: 2201.03545

Supported variants:
- ConvNeXt-Tiny (28M params)
- ConvNeXt-Base (88M params)
- ConvNeXt-Large (197M params)
- ConvNeXt-XLarge (350M params)
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class ConvNeXtBackbone(BaseBackbone):
    """
    ConvNeXt backbone using timm library.

    ConvNeXt modernizes CNN architectures with:
    - Depthwise separable convolutions (like Swin)
    - Inverted bottleneck blocks (like MobileNet)
    - LayerScale for training stability
    - Stochastic depth regularization
    """

    # Variant configurations: (model_name, pretrained_tag, embed_dim, out_indices)
    # embed_dim is the final stage feature dimension
    VARIANTS = {
        "convnext_tiny": ("convnext_tiny", "in22k", 768, [0, 1, 2, 3]),
        "convnext_base": ("convnext_base", "in22k", 1024, [0, 1, 2, 3]),
        "convnext_large": ("convnext_large", "in22k", 1536, [0, 1, 2, 3]),
        "convnext_xlarge": ("convnext_xlarge", "in22k", 2048, [0, 1, 2, 3]),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.variant = config.get("variant", "convnext_tiny")
        self.pretrained = config.get("pretrained", True)
        self.freeze_backbone = config.get("freeze_backbone", True)

        if self.variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {self.variant}. "
                f"Available: {list(self.VARIANTS.keys())}"
            )

        (
            self.model_name,
            self.pretrained_tag,
            self.embed_dim,
            self.output_layers,
        ) = self.VARIANTS[self.variant]

        self.model: nn.Module | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._features: dict[int, torch.Tensor] = {}

        if self.pretrained:
            self.load_pretrained(config.get("pretrained_weights"))

        if self.freeze_backbone:
            self.freeze()

    def load_pretrained(self, path: str | None = None) -> None:
        """Load pretrained ConvNeXt model."""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ConvNeXt backbone. Install with: pip install timm"
            )

        if path:
            # Load from custom checkpoint
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=0,  # Remove classifier head
                global_pool="",
            )
            state_dict = torch.load(path, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Load from ImageNet pretrained
            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,  # Remove classifier head
                global_pool="",
            )

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on specified stages to extract features."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()

        if self.model is None:
            return

        # ConvNeXt structure: model.stages (Sequential of 4 ConvNeXtStage)
        # Each ConvNeXtStage has a 'blocks' attribute with multiple ConvNeXtBlock
        if hasattr(self.model, "stages"):
            stages = self.model.stages
            for layer_idx in self.output_layers:
                if layer_idx < len(stages):
                    stage = stages[layer_idx]
                    blocks = getattr(stage, "blocks", None)

                    if blocks is not None and len(blocks) > 0:
                        last_block = blocks[-1]

                        def hook_fn(
                            module: nn.Module,
                            input: Any,
                            output: torch.Tensor,
                            idx: int = layer_idx,
                        ) -> None:
                            self._features[idx] = output

                        handle = last_block.register_forward_hook(hook_fn)
                        self._hooks.append(handle)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W), normalized to [0, 1]

        Returns:
            List of feature tensors at different stages
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._features.clear()

        # Resize if needed
        if x.shape[-1] != 224:
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        # Forward through the model
        _ = self.model(x)

        features = []
        for layer_idx in sorted(self.output_layers):
            feat = self._features.get(layer_idx)
            if feat is not None:
                # ConvNeXt outputs (B, C, H, W) directly
                features.append(feat)

        return features

    def get_feature_dim(self) -> int:
        """Return the feature dimension of this backbone."""
        return self.embed_dim


class ConvNeXtTinyBackbone(ConvNeXtBackbone):
    """Convenience class for ConvNeXt-Tiny."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = {}
        config["variant"] = "convnext_tiny"
        super().__init__(config)


class ConvNeXtBaseBackbone(ConvNeXtBackbone):
    """Convenience class for ConvNeXt-Base."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = {}
        config["variant"] = "convnext_base"
        super().__init__(config)
