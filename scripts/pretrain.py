import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.backbones import DINOv2Backbone, DINOv3Backbone, PixIOBackbone
from src.pretrain import DINODAPTTrainer, MAEPretrainer

log = logging.getLogger(__name__)

BACKBONE_REGISTRY = {
    "dinov2": DINOv2Backbone,
    "dinov3": DINOv3Backbone,
    "pixio": PixIOBackbone,
}


def build_pretrain_dataloader(cfg: DictConfig) -> DataLoader:
    data_cfg = cfg.get("data", {})
    image_size = data_cfg.get("image_size", 224)

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = Path(data_cfg.get("root_path", "data/smt_pretrain"))
    dataset = ImageFolder(str(data_path), transform=transform)

    training_cfg = cfg.get("training", {})
    return DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 16),
        pin_memory=True,
        drop_last=True,
    )


def build_dino_dataloader(cfg: DictConfig) -> DataLoader:
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})
    image_size = data_cfg.get("image_size", 224)

    global_crops_scale = aug_cfg.get("global_crops_scale", [0.4, 1.0])
    local_crops_scale = aug_cfg.get("local_crops_scale", [0.05, 0.4])
    local_crops_number = aug_cfg.get("local_crops_number", 8)

    class MultiCropTransform:
        def __init__(self) -> None:
            self.global_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=tuple(global_crops_scale)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.local_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(96, scale=tuple(local_crops_scale)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.local_crops_number = local_crops_number

        def __call__(self, img: Any) -> list[torch.Tensor]:
            crops = [self.global_transform(img), self.global_transform(img)]
            for _ in range(self.local_crops_number):
                crops.append(self.local_transform(img))
            return crops

    def collate_fn(batch: list) -> tuple[list[torch.Tensor], torch.Tensor]:
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])

        num_crops = len(images[0])
        stacked = [torch.stack([img[i] for img in images]) for i in range(num_crops)]
        return stacked, labels

    data_path = Path(data_cfg.get("root_path", "data/smt_pretrain"))
    dataset = ImageFolder(str(data_path), transform=MultiCropTransform())

    training_cfg = cfg.get("training", {})
    return DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 128),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 16),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


@hydra.main(version_base=None, config_path="../configs/pretrain", config_name="mae")
def main(cfg: DictConfig) -> None:
    log.info(f"Pretrain Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pretrain_type = cfg.get("pretrain_type", "mae")

    if pretrain_type == "mae":
        log.info("Starting MAE pretraining...")
        dataloader = build_pretrain_dataloader(cfg)
        trainer = MAEPretrainer(OmegaConf.to_container(cfg))
        trainer.train(dataloader)

    elif pretrain_type == "dino_dapt":
        log.info("Starting DINO DAPT pretraining...")

        backbone_cfg = cfg.get("backbone", {"name": "dinov3"})
        backbone_type = backbone_cfg.get("name", "dinov3")

        if backbone_type not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone for DAPT: {backbone_type}")

        backbone = BACKBONE_REGISTRY[backbone_type](backbone_cfg)
        backbone.load_pretrained(backbone_cfg.get("checkpoint"))

        dataloader = build_dino_dataloader(cfg)
        trainer = DINODAPTTrainer(OmegaConf.to_container(cfg))
        trainer.train(dataloader, backbone)

    else:
        raise ValueError(
            f"Unknown pretrain type: {pretrain_type}. Use 'mae' or 'dino_dapt'"
        )

    log.info("Pretraining complete!")


if __name__ == "__main__":
    main()
