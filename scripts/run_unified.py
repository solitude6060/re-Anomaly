#!/usr/bin/env python3
"""
Run Unified Anomaly Detector experiments on MVTec AD.

The Unified model combines:
- Multi-scale feature aggregation
- FastFlow (normalizing flow) for density estimation
- Discriminator for normal/anomaly classification
- Memory bank for k-NN scoring
- SDG (Synthetic Defect Generation) for training augmentation
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_HF_CACHE / "hub")

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mvtec import MVTecADDataset, MVTEC_AD_CATEGORIES
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.backbones.swin import SwinBackbone
from src.models.unified import UnifiedAnomalyDetector


@dataclass
class UnifiedConfig:
    backbone_name: str
    projection_dim: int = 256
    flow_blocks: int = 8
    discriminator_hidden: tuple = (256, 128)
    use_sdg: bool = True
    flow_weight: float = 0.5
    disc_weight: float = 0.3
    memory_weight: float = 0.2
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
}


def create_backbone(
    name: str, image_size: int, device: torch.device
) -> torch.nn.Module:
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


def create_unified_model(config: UnifiedConfig) -> UnifiedAnomalyDetector:
    model_config = {
        "backbone": config.backbone_name,
        "projection_dim": config.projection_dim,
        "flow_blocks": config.flow_blocks,
        "discriminator_hidden": list(config.discriminator_hidden),
        "use_sdg": config.use_sdg,
        "flow_weight": config.flow_weight,
        "disc_weight": config.disc_weight,
        "memory_weight": config.memory_weight,
    }
    return UnifiedAnomalyDetector(model_config)


def train_unified(
    backbone: torch.nn.Module,
    model: UnifiedAnomalyDetector,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_sdg: bool = True,
) -> None:
    """Train the unified model with optional SDG augmentation."""
    print("  Initializing unified model...")
    backbone.eval()

    # Initialize model with sample features
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch["image"].to(device)
        sample_features = backbone(sample_images)
        model.fit(sample_features)

    model = model.to(device)
    model.train()

    params = list(model.parameters())
    if not params:
        print("  No trainable parameters found!")
        return

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Import SDG if using augmentation
    sdg = None
    if use_sdg:
        try:
            from src.augmentation.sdg import SyntheticDefectGenerator

            sdg_config = {
                "defect_types": ["cutout", "noise", "blur", "scratch", "stain"],
                "defect_size_range": (0.1, 0.3),
                "num_defects_range": (1, 3),
            }
            sdg = SyntheticDefectGenerator(sdg_config)
            print("  SDG augmentation enabled")
        except ImportError:
            print("  SDG not available, training without augmentation")

    print(f"  Training unified model for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        flow_loss_sum = 0.0
        disc_loss_sum = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            with torch.no_grad():
                features = backbone(images)

            # Generate augmented features if SDG is available
            aug_features = None
            labels = None
            if sdg is not None:
                # Apply SDG to create synthetic anomalies
                aug_images, masks, batch_labels = sdg.generate_batch(images)
                with torch.no_grad():
                    aug_features = backbone(aug_images)
                labels = batch_labels

            losses = model.compute_training_loss(
                features=features,
                aug_features=aug_features,
                labels=labels,
            )

            loss = losses["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            flow_loss_sum += losses["flow_loss"].item()
            disc_loss_sum += losses["disc_loss"].item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            avg_flow = flow_loss_sum / num_batches
            avg_disc = disc_loss_sum / num_batches
            print(
                f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f} "
                f"(flow: {avg_flow:.4f}, disc: {avg_disc:.4f})"
            )


def fit_memory_bank(
    backbone: torch.nn.Module,
    model: UnifiedAnomalyDetector,
    train_dataset: MVTecADDataset,
    device: torch.device,
    max_samples: int = 10000,
) -> None:
    """Fit memory bank from training features."""
    print(f"  Fitting memory bank from {len(train_dataset)} samples...")
    backbone.eval()

    all_features = []
    with torch.no_grad():
        for i in tqdm(
            range(len(train_dataset)), desc="  Extracting features", leave=False
        ):
            sample = train_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            features = backbone(image)
            all_features.append([f.cpu() for f in features])

    # Reconstruct features for memory bank fitting
    print("  Building memory bank...")
    model.fit_memory_bank(
        [[f.to(device) for f in feats] for feats in all_features],
        max_samples=max_samples,
    )


def evaluate_category(
    backbone: torch.nn.Module,
    model: UnifiedAnomalyDetector,
    category: str,
    data_root: str,
    image_size: int,
    config: UnifiedConfig,
    device: torch.device,
) -> dict:
    """Train and evaluate on a single MVTec category."""
    train_dataset = MVTecADDataset(
        root=data_root, category=category, split="train", image_size=image_size
    )
    test_dataset = MVTecADDataset(
        root=data_root, category=category, split="test", image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Train the model
    train_unified(
        backbone=backbone,
        model=model,
        train_loader=train_loader,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        use_sdg=config.use_sdg,
    )

    # Fit memory bank after training
    fit_memory_bank(backbone, model, train_dataset, device)

    # Evaluate
    model = model.to(device)
    model.eval()

    print(f"  Evaluating on {len(test_dataset)} test samples...")
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="  Testing", leave=False):
            sample = test_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"]

            features = backbone(image)
            output = model(features)

            score = output["anomaly_score"].cpu().item()
            all_scores.append(score)
            all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    image_auroc = roc_auc_score(all_labels, all_scores)

    # Compute precision at 100% recall
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


def run_unified_experiment(
    config: UnifiedConfig,
    data_root: str,
    output_dir: Path,
    image_size: int,
    categories: list[str],
    device: torch.device,
) -> dict:
    """Run unified model experiment across all categories."""
    print(f"\n{'=' * 70}")
    print(f"UNIFIED MODEL EXPERIMENT: {config.backbone_name}")
    print(
        f"  Flow weight: {config.flow_weight}, Disc weight: {config.disc_weight}, "
        f"Memory weight: {config.memory_weight}"
    )
    print(f"  SDG augmentation: {config.use_sdg}")
    print("=" * 70)

    backbone = create_backbone(config.backbone_name, image_size, device)

    all_results = []

    for category in categories:
        print(f"\n  Category: {category}")

        # Create fresh model for each category
        model = create_unified_model(config)

        start_time = time.time()
        result = evaluate_category(
            backbone=backbone,
            model=model,
            category=category,
            data_root=data_root,
            image_size=image_size,
            config=config,
            device=device,
        )
        result["eval_time_seconds"] = time.time() - start_time

        all_results.append(result)
        print(
            f"    AUROC: {result['image_auroc']:.4f}, Time: {result['eval_time_seconds']:.1f}s"
        )

        del model
        torch.cuda.empty_cache()

    avg_auroc = np.mean([r["image_auroc"] for r in all_results])
    avg_precision = np.mean([r["precision_at_100_recall"] for r in all_results])

    summary = {
        "experiment": f"unified_{config.backbone_name}",
        "backbone": config.backbone_name,
        "config": {
            "projection_dim": config.projection_dim,
            "flow_blocks": config.flow_blocks,
            "flow_weight": config.flow_weight,
            "disc_weight": config.disc_weight,
            "memory_weight": config.memory_weight,
            "use_sdg": config.use_sdg,
            "epochs": config.epochs,
        },
        "image_size": image_size,
        "avg_image_auroc": float(avg_auroc),
        "avg_precision_at_100_recall": float(avg_precision),
        "per_category_results": all_results,
    }

    exp_dir = output_dir / f"unified_{config.backbone_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Average AUROC: {avg_auroc:.4f}")
    print(f"  Average Precision@100%Recall: {avg_precision:.4f}")

    del backbone
    torch.cuda.empty_cache()

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Unified Anomaly Detector experiments"
    )
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output_dir", type=str, default="results/unified")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vitl16",
        choices=list(BACKBONE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--no_sdg", action="store_true", help="Disable SDG augmentation"
    )
    parser.add_argument("--flow_weight", type=float, default=0.5)
    parser.add_argument("--disc_weight", type=float, default=0.3)
    parser.add_argument("--memory_weight", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or MVTEC_AD_CATEGORIES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Categories: {len(categories)}")
    print(f"SDG augmentation: {not args.no_sdg}")

    config = UnifiedConfig(
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_sdg=not args.no_sdg,
        flow_weight=args.flow_weight,
        disc_weight=args.disc_weight,
        memory_weight=args.memory_weight,
    )

    summary = run_unified_experiment(
        config=config,
        data_root=args.data_root,
        output_dir=output_dir,
        image_size=args.image_size,
        categories=categories,
        device=device,
    )

    # Save final summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
