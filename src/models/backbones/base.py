from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.name = config.get("name", "base")
        self.output_layers = config.get("output_layers", [])
        self.feature_dims = config.get("feature_dims", [])

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def load_pretrained(self, path: str | None = None) -> None:
        pass

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _move_inner_model(self, *args, **kwargs) -> None:
        """Move self.model if it exists (HuggingFace models loaded separately)."""
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to(*args, **kwargs)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self._move_inner_model(*args, **kwargs)
        return result

    def cuda(self, device=None):
        result = super().cuda(device)
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.cuda(device)
        return result

    def cpu(self):
        result = super().cpu()
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.cpu()
        return result
