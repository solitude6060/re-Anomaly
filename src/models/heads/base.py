from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseHead(ABC, nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.name = config.get("name", "base")

    @abstractmethod
    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def fit(self, features: list[torch.Tensor]) -> None:
        pass

    def get_anomaly_score(self, features: list[torch.Tensor]) -> torch.Tensor:
        output = self.forward(features)
        return output["anomaly_score"]

    def get_anomaly_map(self, features: list[torch.Tensor]) -> torch.Tensor:
        output = self.forward(features)
        return output.get("anomaly_map", output["anomaly_score"])
