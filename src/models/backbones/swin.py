from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class SwinBackbone(BaseBackbone):
    VARIANTS = {
        "swin_tiny": ("microsoft/swin-tiny-patch4-window7-224", 768, [2, 2, 6, 2]),
        "swin_small": ("microsoft/swin-small-patch4-window7-224", 768, [2, 2, 18, 2]),
        "swin_base": ("microsoft/swin-base-patch4-window7-224", 1024, [2, 2, 18, 2]),
        "swin_large": ("microsoft/swin-large-patch4-window7-224", 1536, [2, 2, 18, 2]),
        "swinv2_tiny": ("microsoft/swinv2-tiny-patch4-window8-256", 768, [2, 2, 6, 2]),
        "swinv2_small": (
            "microsoft/swinv2-small-patch4-window8-256",
            768,
            [2, 2, 18, 2],
        ),
        "swinv2_base": (
            "microsoft/swinv2-base-patch4-window8-256",
            1024,
            [2, 2, 18, 2],
        ),
    }

    STAGE_DIMS = {
        "swin_tiny": [96, 192, 384, 768],
        "swin_small": [96, 192, 384, 768],
        "swin_base": [128, 256, 512, 1024],
        "swin_large": [192, 384, 768, 1536],
        "swinv2_tiny": [96, 192, 384, 768],
        "swinv2_small": [96, 192, 384, 768],
        "swinv2_base": [128, 256, 512, 1024],
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.variant = config.get("variant", "swin_base")
        self.model_id = config.get("model_id")
        self.image_size = config.get("image_size", 224)
        self.patch_size = config.get("patch_size", 4)
        self.interpolate_pos = config.get("interpolate_pos_encoding", True)

        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")

        model_id, self.embed_dim, self.depths = self.VARIANTS[self.variant]
        self.model_id = self.model_id or model_id
        self.stage_dims = self.STAGE_DIMS[self.variant]

        if not self.output_layers:
            self.output_layers = [0, 1, 2, 3]

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
                from transformers import SwinModel

                self.model = SwinModel.from_pretrained(self.model_id)
            except ImportError:
                import timm

                timm_name = self._get_timm_name()
                self.model = timm.create_model(timm_name, pretrained=True)

        self._register_hooks()

    def _get_timm_name(self) -> str:
        timm_mapping = {
            "swin_tiny": "swin_tiny_patch4_window7_224",
            "swin_small": "swin_small_patch4_window7_224",
            "swin_base": "swin_base_patch4_window7_224",
            "swin_large": "swin_large_patch4_window7_224",
            "swinv2_tiny": "swinv2_tiny_window8_256",
            "swinv2_small": "swinv2_small_window8_256",
            "swinv2_base": "swinv2_base_window8_256",
        }
        return timm_mapping.get(self.variant, "swin_base_patch4_window7_224")

    def _register_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()

        if self.model is None:
            return

        encoder = getattr(self.model, "encoder", self.model)
        stages = getattr(encoder, "layers", getattr(encoder, "stages", None))

        if stages is None:
            stages = getattr(self.model, "layers", None)

        if stages is None:
            return

        for stage_idx in self.output_layers:
            if stage_idx < len(stages):

                def hook_fn(
                    module: nn.Module, input: Any, output: Any, idx: int = stage_idx
                ) -> None:
                    if isinstance(output, tuple):
                        output = output[0]
                    self._features[idx] = output

                handle = stages[stage_idx].register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._features.clear()
        batch_size = x.shape[0]

        target_size = 256 if "v2" in self.variant else 224
        if self.interpolate_pos and x.shape[-1] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        _ = self.model(x)

        features = []

        for stage_idx in sorted(self.output_layers):
            feat = self._features.get(stage_idx)
            if feat is not None:
                if feat.dim() == 3:
                    h = w = int(feat.shape[1] ** 0.5)
                    if h * w != feat.shape[1]:
                        downsample = 2 ** (stage_idx + 2)
                        h = w = target_size // downsample
                    feat = feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                elif feat.dim() == 4 and feat.shape[1] != self.stage_dims[stage_idx]:
                    feat = feat.permute(0, 3, 1, 2)
                features.append(feat)

        return features
