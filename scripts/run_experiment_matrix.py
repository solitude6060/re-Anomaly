#!/usr/bin/env python3
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_HF_CACHE / "hub")

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mvtec import MVTecADDataset, MVTEC_AD_CATEGORIES
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.backbones.swin import SwinBackbone
from src.models.heads.patchcore import PatchCoreHead
from src.models.heads.fastflow import FastFlowHead
from src.models.heads.simplenet import SimpleNetHead
from src.models.heads.msflow import MSFlowHead
from src.models.heads.rectflow import RectFlowHead
from src.models.heads.dinomaly import DinomalyHead


@dataclass
class ExperimentConfig:
    backbone_name: str
    head_name: str
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4


BACKBONE_REGISTRY = {
    "dinov2_vitb14": {
        "class": DINOv2Backbone,
        "config": {
            "variant": "dinov2_vitb14",
            "model_id": "facebook/dinov2-base",
            "patch_size": 14,
            "output_layers": [3, 6, 9, 11],
        },
    },
    "dinov2_vitl14": {
        "class": DINOv2Backbone,
        "config": {
            "variant": "dinov2_vitl14",
            "model_id": "facebook/dinov2-large",
            "patch_size": 14,
            "output_layers": [8, 11, 17, 23],
        },
    },
    "dinov3_vitl16": {
        "class": DINOv3Backbone,
        "config": {
            "variant": "dinov3_vitl16",
            "patch_size": 16,
            "output_layers": [8, 11, 17, 23],
        },
    },
    "swin_base": {
        "class": SwinBackbone,
        "config": {
            "variant": "swin_base",
            "patch_size": 4,
            "output_layers": [0, 1, 2, 3],
        },
    },
    "swin_large": {
        "class": SwinBackbone,
        "config": {
            "variant": "swin_large",
            "patch_size": 4,
            "output_layers": [0, 1, 2, 3],
        },
    },
    "swinv2_base": {
        "class": SwinBackbone,
        "config": {
            "variant": "swinv2_base",
            "patch_size": 4,
            "output_layers": [0, 1, 2, 3],
        },
    },
}

HEAD_REGISTRY = {
    "patchcore": {
        "class": PatchCoreHead,
        "config": {
            "k_nearest": 9,
            "coreset_sampling_ratio": 0.1,
            "coreset_method": "greedy",
            "feature_aggregation": "concat",
            "memory_bank": {"max_size": 100000, "normalize": True, "use_faiss": True},
            "anomaly_score": {"normalize": True},
        },
        "trainable": False,
    },
    "fastflow": {
        "class": FastFlowHead,
        "config": {
            "flow": {
                "type": "realnvp",
                "num_blocks": 8,
                "hidden_dims": [256, 256],
                "clamp": 2.0,
            },
            "feature_processing": {
                "reduce_dim": True,
                "projection_dim": 256,
                "normalize": True,
            },
            "anomaly_score": {"normalize": True},
        },
        "trainable": True,
    },
    "simplenet": {
        "class": SimpleNetHead,
        "config": {
            "hidden_dims": [256, 256],
            "feature_aggregation": "concat",
            "projection_dim": 256,
            "discriminator": {
                "num_layers": 3,
                "activation": "leaky_relu",
                "dropout": 0.1,
            },
        },
        "trainable": True,
    },
    "msflow": {
        "class": MSFlowHead,
        "config": {
            "flow": {"type": "realnvp", "num_blocks": 8, "hidden_dims": [256, 256]},
            "scales": ["low", "mid", "high"],
            "feature_processing": {"reduce_dim": True, "projection_dim": 256},
            "anomaly_score": {"scale_weights": [0.2, 0.3, 0.5]},
        },
        "trainable": True,
    },
    "rectflow": {
        "class": RectFlowHead,
        "config": {
            "hidden_dims": [256, 256],
            "projection_dim": 256,
            "num_timesteps": 1000,
            "inference_steps": 1,
            "feature_processing": {"reduce_dim": True, "normalize": True},
            "anomaly_score": {"normalize": True},
        },
        "trainable": True,
    },
    "dinomaly": {
        "class": DinomalyHead,
        "config": {
            "embed_dim": 1024,
            "num_heads": 16,
            "decoder_depth": 8,
            "mlp_ratio": 4.0,
            "bottleneck_drop": 0.2,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "normalize_features": True,
            "num_register_tokens": 4,
            "fuse_layer_encoder": [[0, 1], [2, 3]],
            "fuse_layer_decoder": [[0, 1, 2, 3], [4, 5, 6, 7]],
        },
        "trainable": True,
    },
}


def create_backbone(name: str, image_size: int, device: torch.device) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}")

    cfg = BACKBONE_REGISTRY[name]
    backbone_config = {
        **cfg["config"],
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": image_size,
        "interpolate_pos_encoding": True,
    }
    backbone = cfg["class"](backbone_config)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def create_head(name: str) -> tuple[nn.Module, dict, bool]:
    if name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {name}")

    cfg = HEAD_REGISTRY[name]
    head = cfg["class"](cfg["config"])
    return head, cfg["config"], cfg["trainable"]


def train_head(
    backbone: nn.Module,
    head: nn.Module,
    head_name: str,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    print("  Initializing head...")
    backbone.eval()

    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch["image"].to(device)
        sample_features = backbone(sample_images)
        head.fit(sample_features)

    head = head.to(device)
    head.train()

    params = list(head.parameters())
    if not params:
        return

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Training {head_name} for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            with torch.no_grad():
                features = backbone(images)

            loss = compute_head_loss(head, head_name, features, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def compute_head_loss(
    head: nn.Module, head_name: str, features: list[torch.Tensor], device: torch.device
) -> torch.Tensor:
    if head_name == "fastflow":
        aggregated = head._aggregate_features(features)
        if head.reduce_dim and head.projection is not None:
            aggregated = head.projection(aggregated)
        if head.normalize:
            aggregated = torch.nn.functional.normalize(aggregated, p=2, dim=1)
        z, log_det = head.flow(aggregated)
        log_pz = -0.5 * (z**2).sum(dim=1)
        log_prob = log_pz + log_det.unsqueeze(-1).unsqueeze(-1) / (
            z.shape[-1] * z.shape[-2]
        )
        return -log_prob.mean()

    elif head_name == "simplenet":
        aggregated = head._aggregate_features(features)
        projected = head.projection(aggregated)
        flat_normal = projected.permute(0, 2, 3, 1).reshape(-1, head.projection_dim)
        noise = torch.randn_like(flat_normal) * 0.015
        flat_anomaly = flat_normal + noise
        normal_scores = head.discriminator(flat_normal).squeeze(-1)
        anomaly_scores = head.discriminator(flat_anomaly).squeeze(-1)
        scores = torch.cat([normal_scores, anomaly_scores])
        labels = torch.cat(
            [torch.zeros_like(normal_scores), torch.ones_like(anomaly_scores)]
        )
        return nn.functional.binary_cross_entropy_with_logits(scores, labels)

    elif head_name == "msflow":
        loss = torch.tensor(0.0, device=device)
        for i, (feat, flow) in enumerate(zip(features, head.flows)):
            if head.reduce_dim and i < len(head.projections):
                feat = head.projections[i](feat)
            z, log_det = flow(feat)
            log_pz = -0.5 * (z**2).sum(dim=1)
            log_prob = (log_pz + log_det).mean()
            loss = loss - log_prob
        return loss

    elif head_name == "rectflow":
        return head.compute_training_loss(features)

    elif head_name == "dinomaly":
        return head.compute_training_loss(features)

    return torch.tensor(0.0, device=device)


def fit_patchcore(
    backbone: nn.Module,
    head: PatchCoreHead,
    train_dataset: MVTecADDataset,
    device: torch.device,
) -> None:
    print(f"  Fitting memory bank with {len(train_dataset)} samples...")
    all_patches = []
    backbone.eval()

    with torch.no_grad():
        for i in tqdm(
            range(len(train_dataset)), desc="  Extracting features", leave=False
        ):
            sample = train_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            features = backbone(image)
            patches = head.extract_patches(features)
            all_patches.append(patches)

    all_patches = torch.cat(all_patches, dim=0)
    print(f"  Total patches: {all_patches.shape[0]}, applying coreset...")
    head.fit_from_patches(all_patches.to(device))


def evaluate_category(
    backbone: nn.Module,
    head: nn.Module,
    head_name: str,
    category: str,
    data_root: str,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    train_dataset = MVTecADDataset(
        root=data_root, category=category, split="train", image_size=image_size
    )
    test_dataset = MVTecADDataset(
        root=data_root, category=category, split="test", image_size=image_size
    )

    is_trainable = HEAD_REGISTRY[head_name]["trainable"]

    if head_name == "patchcore":
        fit_patchcore(backbone, head, train_dataset, device)
    elif is_trainable:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        train_head(backbone, head, head_name, train_loader, epochs, lr, device)

    head = head.to(device)
    head.eval()

    print(f"  Evaluating on {len(test_dataset)} test samples...")
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="  Testing", leave=False):
            sample = test_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"]

            features = backbone(image)
            output = head(features)

            score = output["anomaly_score"].cpu().item()
            all_scores.append(score)
            all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    image_auroc = roc_auc_score(all_labels, all_scores)

    anomaly_scores = all_scores[all_labels == 1]
    threshold_100_recall = anomaly_scores.min() if len(anomaly_scores) > 0 else 0
    predictions_100 = (all_scores >= threshold_100_recall).astype(int)
    tp = ((predictions_100 == 1) & (all_labels == 1)).sum()
    fp = ((predictions_100 == 1) & (all_labels == 0)).sum()
    precision_100 = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "category": category,
        "image_auroc": float(image_auroc),
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_anomalies": int(all_labels.sum()),
        "n_normal": int(len(all_labels) - all_labels.sum()),
        "precision_at_100_recall": float(precision_100),
    }


def run_experiment(
    exp_config: ExperimentConfig,
    data_root: str,
    output_dir: Path,
    image_size: int,
    categories: list[str],
    device: torch.device,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_config.backbone_name} + {exp_config.head_name}")
    print("=" * 70)

    backbone = create_backbone(exp_config.backbone_name, image_size, device)

    all_results = []

    for category in categories:
        print(f"\n  Category: {category}")

        head, _, _ = create_head(exp_config.head_name)

        start_time = time.time()
        result = evaluate_category(
            backbone=backbone,
            head=head,
            head_name=exp_config.head_name,
            category=category,
            data_root=data_root,
            image_size=image_size,
            epochs=exp_config.epochs,
            batch_size=exp_config.batch_size,
            lr=exp_config.lr,
            device=device,
        )
        result["eval_time_seconds"] = time.time() - start_time

        all_results.append(result)
        print(
            f"    AUROC: {result['image_auroc']:.4f}, Time: {result['eval_time_seconds']:.1f}s"
        )

        del head
        torch.cuda.empty_cache()

    avg_auroc = np.mean([r["image_auroc"] for r in all_results])
    avg_precision = np.mean([r["precision_at_100_recall"] for r in all_results])

    summary = {
        "experiment": f"{exp_config.backbone_name} + {exp_config.head_name}",
        "backbone": exp_config.backbone_name,
        "head": exp_config.head_name,
        "epochs": exp_config.epochs,
        "image_size": image_size,
        "avg_image_auroc": float(avg_auroc),
        "avg_precision_at_100_recall": float(avg_precision),
        "per_category_results": all_results,
    }

    exp_dir = output_dir / f"{exp_config.backbone_name}_{exp_config.head_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Average AUROC: {avg_auroc:.4f}")

    del backbone
    torch.cuda.empty_cache()

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run full experiment matrix")
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output_dir", type=str, default="results/experiment_matrix")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--backbones", type=str, nargs="+", default=None)
    parser.add_argument("--heads", type=str, nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or MVTEC_AD_CATEGORIES
    backbones = args.backbones or ["dinov3_vitl16", "swin_base"]
    heads = args.heads or ["patchcore", "fastflow", "simplenet", "rectflow"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbones: {backbones}")
    print(f"Heads: {heads}")
    print(f"Categories: {len(categories)}")

    all_experiments = []

    for backbone_name in backbones:
        for head_name in heads:
            exp_config = ExperimentConfig(
                backbone_name=backbone_name,
                head_name=head_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            summary = run_experiment(
                exp_config=exp_config,
                data_root=args.data_root,
                output_dir=output_dir,
                image_size=args.image_size,
                categories=categories,
                device=device,
            )
            all_experiments.append(summary)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Backbone':<20} {'Head':<15} {'Avg AUROC':<12}")
    print("-" * 50)
    for exp in sorted(
        all_experiments, key=lambda x: x["avg_image_auroc"], reverse=True
    ):
        print(f"{exp['backbone']:<20} {exp['head']:<15} {exp['avg_image_auroc']:.4f}")

    with open(output_dir / "all_experiments.json", "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
