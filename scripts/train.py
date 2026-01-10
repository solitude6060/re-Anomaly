import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.backbones import DINOv2Backbone, DINOv3Backbone, PixIOBackbone
from src.models.heads import (
    FastFlowHead,
    LinearHead,
    MSFlowHead,
    PatchCoreHead,
    SALADHead,
    SimpleNetHead,
)

log = logging.getLogger(__name__)

BACKBONE_REGISTRY = {
    "dinov2": DINOv2Backbone,
    "dinov3": DINOv3Backbone,
    "pixio": PixIOBackbone,
}

HEAD_REGISTRY = {
    "patchcore": PatchCoreHead,
    "msflow": MSFlowHead,
    "fastflow": FastFlowHead,
    "simplenet": SimpleNetHead,
    "salad": SALADHead,
    "linear": LinearHead,
}


def build_backbone(cfg: DictConfig) -> torch.nn.Module:
    backbone_type = cfg.backbone.name
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    return BACKBONE_REGISTRY[backbone_type](OmegaConf.to_container(cfg.backbone))


def build_head(cfg: DictConfig) -> torch.nn.Module:
    head_type = cfg.head.name
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head_type}")
    return HEAD_REGISTRY[head_type](OmegaConf.to_container(cfg.head))


def build_dataloader(cfg: DictConfig, split: str) -> DataLoader:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = Path(cfg.dataset.root_path) / split
    dataset = ImageFolder(str(data_path), transform=transform)

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.get("num_workers", 4),
        pin_memory=True,
    )


def train_epoch(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    head.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            features = backbone(images)

        optimizer.zero_grad()
        output = head(features)
        loss = output.get("loss", torch.tensor(0.0, device=device))

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            log.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}"
            )

    return total_loss / len(dataloader)


def fit_memory_bank(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    log.info("Fitting memory bank...")
    backbone.eval()

    all_features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = backbone(images)
            all_features.append([f.cpu() for f in features])

    merged = [
        torch.cat([f[i] for f in all_features], dim=0)
        for i in range(len(all_features[0]))
    ]
    head.fit([f.to(device) for f in merged])
    log.info("Memory bank fitted.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    backbone = build_backbone(cfg)
    backbone.load_pretrained(cfg.backbone.get("checkpoint"))
    backbone.freeze()
    backbone = backbone.to(device)
    backbone.eval()

    head = build_head(cfg)
    head = head.to(device)

    train_loader = build_dataloader(cfg, "train")

    head_type = cfg.head.name
    if head_type in ("patchcore",):
        fit_memory_bank(backbone, head, train_loader, device)
    else:
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
        )

        for epoch in range(1, cfg.training.epochs + 1):
            avg_loss = train_epoch(
                backbone, head, train_loader, optimizer, device, epoch
            )
            log.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            if (
                cfg.output.save_checkpoints
                and epoch % cfg.output.checkpoint_interval == 0
            ):
                ckpt_path = (
                    Path(cfg.output.root_dir)
                    / cfg.output.experiment_name
                    / f"epoch_{epoch:04d}.pth"
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"head": head.state_dict(), "epoch": epoch}, ckpt_path)
                log.info(f"Saved checkpoint: {ckpt_path}")

    final_path = Path(cfg.output.root_dir) / cfg.output.experiment_name / "final.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict()}, final_path)
    log.info(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
