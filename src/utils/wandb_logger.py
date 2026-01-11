"""
Weights & Biases Logger for re-Anomaly experiments.

This module provides a unified interface for logging experiments to W&B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WandbConfig:
    """Configuration for W&B logging."""

    project: str = "re-anomaly"
    entity: str | None = None
    group: str | None = None
    job_type: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    name: str | None = None
    mode: str = "online"  # "online", "offline", "disabled"
    save_code: bool = True
    resume: str = "allow"
    log_model: bool = False
    log_images: bool = True
    log_anomaly_maps: bool = True


class WandbLogger:
    """Weights & Biases logger for experiment tracking."""

    def __init__(
        self,
        config: WandbConfig | dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        """
        Initialize W&B logger.

        Args:
            config: W&B configuration (WandbConfig or dict)
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self._run = None
        self._wandb = None

        if isinstance(config, dict):
            self.config = WandbConfig(**config)
        elif config is None:
            self.config = WandbConfig()
        else:
            self.config = config

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            self._wandb = wandb

            # Check if already initialized
            if wandb.run is not None:
                self._run = wandb.run
                return

            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                group=self.config.group,
                job_type=self.config.job_type,
                tags=self.config.tags,
                notes=self.config.notes,
                name=self.config.name,
                mode=self.config.mode,
                save_code=self.config.save_code,
                resume=self.config.resume,
            )
        except ImportError:
            print("WARNING: wandb not installed. Logging disabled.")
            self.enabled = False
        except Exception as e:
            print(f"WARNING: Failed to initialize wandb: {e}")
            self.enabled = False

    @property
    def run(self):
        """Get the current W&B run."""
        return self._run

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics to W&B.

        Args:
            data: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or self._run is None:
            return

        if step is not None:
            self._run.log(data, step=step)
        else:
            self._run.log(data)

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        if not self.enabled or self._run is None:
            return

        self._run.config.update(config)

    def log_summary(self, data: dict[str, Any]) -> None:
        """
        Log summary metrics (final results).

        Args:
            data: Dictionary of summary metrics
        """
        if not self.enabled or self._run is None:
            return

        for key, value in data.items():
            self._run.summary[key] = value

    def log_image(
        self,
        key: str,
        image: np.ndarray | Any,
        caption: str | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log an image to W&B.

        Args:
            key: Image key/name
            image: Image array (H, W, C) or PIL Image
            caption: Optional caption
            step: Optional step number
        """
        if not self.enabled or self._run is None or self._wandb is None:
            return

        wandb_image = self._wandb.Image(image, caption=caption)
        self.log({key: wandb_image}, step=step)

    def log_anomaly_map(
        self,
        key: str,
        original: np.ndarray,
        anomaly_map: np.ndarray,
        caption: str | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log anomaly map visualization.

        Args:
            key: Image key/name
            original: Original image (H, W, C)
            anomaly_map: Anomaly heatmap (H, W)
            caption: Optional caption
            step: Optional step number
        """
        if not self.enabled or self._run is None or self._wandb is None:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm

            # Normalize anomaly map
            if anomaly_map.max() > anomaly_map.min():
                norm_map = (anomaly_map - anomaly_map.min()) / (
                    anomaly_map.max() - anomaly_map.min()
                )
            else:
                norm_map = np.zeros_like(anomaly_map)

            # Apply colormap
            heatmap = cm.jet(norm_map)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)

            # Blend with original
            if original.max() <= 1.0:
                original = (original * 255).astype(np.uint8)

            overlay = (0.6 * original + 0.4 * heatmap).astype(np.uint8)

            # Log as image
            self.log_image(key, overlay, caption=caption, step=step)
        except Exception as e:
            print(f"WARNING: Failed to log anomaly map: {e}")

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """
        Log a table to W&B.

        Args:
            key: Table key/name
            columns: Column names
            data: Table data (list of rows)
        """
        if not self.enabled or self._run is None or self._wandb is None:
            return

        table = self._wandb.Table(columns=columns, data=data)
        self.log({key: table})

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an artifact (file or directory) to W&B.

        Args:
            name: Artifact name
            artifact_type: Type of artifact (e.g., "model", "dataset")
            path: Path to file or directory
            metadata: Optional metadata dictionary
        """
        if not self.enabled or self._run is None or self._wandb is None:
            return

        artifact = self._wandb.Artifact(name, type=artifact_type, metadata=metadata)
        path = Path(path)

        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))

        self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


# Convenience functions for simpler usage


def init_wandb(
    project: str = "re-anomaly",
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str = "online",
    enabled: bool = True,
) -> WandbLogger:
    """
    Initialize W&B logging with simple parameters.

    Args:
        project: W&B project name
        name: Run name
        config: Experiment configuration to log
        tags: Tags for the run
        group: Group name for related runs
        job_type: Type of job
        mode: W&B mode ("online", "offline", "disabled")
        enabled: Whether logging is enabled

    Returns:
        WandbLogger instance
    """
    wandb_config = WandbConfig(
        project=project,
        name=name,
        tags=tags or [],
        group=group,
        job_type=job_type,
        mode=mode,
    )

    logger = WandbLogger(config=wandb_config, enabled=enabled)

    if config is not None:
        logger.log_config(config)

    return logger


def log_metrics(
    metrics: dict[str, float],
    step: int | None = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to the current W&B run.

    Args:
        metrics: Dictionary of metrics
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    try:
        import wandb

        if wandb.run is None:
            return

        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_anomaly_result(
    category: str,
    image_auroc: float,
    pixel_auroc: float | None = None,
    logical_auroc: float | None = None,
    structural_auroc: float | None = None,
    precision_at_100_recall: float | None = None,
    step: int | None = None,
) -> None:
    """
    Log anomaly detection results for a category.

    Args:
        category: Category name
        image_auroc: Image-level AUROC
        pixel_auroc: Pixel-level AUROC (optional)
        logical_auroc: Logical anomaly AUROC (optional, for LOCO)
        structural_auroc: Structural anomaly AUROC (optional, for LOCO)
        precision_at_100_recall: Precision at 100% recall (optional)
        step: Optional step number
    """
    metrics = {
        f"{category}/image_auroc": image_auroc,
    }

    if pixel_auroc is not None:
        metrics[f"{category}/pixel_auroc"] = pixel_auroc

    if logical_auroc is not None:
        metrics[f"{category}/logical_auroc"] = logical_auroc

    if structural_auroc is not None:
        metrics[f"{category}/structural_auroc"] = structural_auroc

    if precision_at_100_recall is not None:
        metrics[f"{category}/precision_at_100_recall"] = precision_at_100_recall

    log_metrics(metrics, step=step)
