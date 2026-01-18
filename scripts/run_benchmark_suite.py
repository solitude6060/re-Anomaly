#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for re-Anomaly.

This script runs systematic experiments across all backbones and heads
on MVTec AD and MVTec LOCO datasets.

Usage:
    # Run all experiments (takes ~hours)
    python scripts/run_benchmark_suite.py --all

    # Run specific subset
    python scripts/run_benchmark_suite.py --dataset mvtec_ad --categories bottle,cable

    # Run quick smoke test
    python scripts/run_benchmark_suite.py --smoke-test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")

from src.data.mvtec import (
    MVTecADDataset,
    MVTecLOCODataset,
    MVTEC_AD_CATEGORIES,
    MVTEC_LOCO_CATEGORIES,
)
from src.models.backbones.convnext import ConvNeXtBackbone
from src.models.backbones.clip import CLIPBackbone
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.backbones.pixio import PixIOBackbone
from src.models.backbones.swin import SwinBackbone
from src.models.heads.afrclip import AFRCLIPHead
from src.models.heads.dinomaly import DinomalyHead
from src.models.heads.fastflow import FastFlowHead
from src.models.heads.mambaad import MambaADHead
from src.models.heads.msflow import MSFlowHead
from src.models.heads.patchcore import PatchCoreHead
from src.models.heads.rectflow import RectFlowHead
from src.models.heads.simplenet import SimpleNetHead
from sklearn.metrics import roc_auc_score


# =============================================================================
# Configuration
# =============================================================================

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
    "clip_vitl14": {
        "class": CLIPBackbone,
        "config": {
            "variant": "clip_vitl14",
            "output_layers": [6, 12, 18, 23],
        },
    },
    "convnext_tiny": {
        "class": ConvNeXtBackbone,
        "config": {
            "variant": "convnext_tiny",
            "output_layers": [0, 1, 2, 3],
        },
    },
    "convnext_base": {
        "class": ConvNeXtBackbone,
        "config": {
            "variant": "convnext_base",
            "output_layers": [0, 1, 2, 3],
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
}

HEAD_REGISTRY = {
    "patchcore": {
        "class": PatchCoreHead,
        "config": {
            "k_nearest": 9,
            "coreset_sampling_ratio": 0.1,
            "coreset_method": "greedy",
            "feature_aggregation": "concat",
            "memory_bank": {"max_size": 100000, "normalize": True, "use_faiss": False},
            "anomaly_score": {"normalize": True},
        },
        "trainable": False,
    },
    "dinomaly": {
        "class": DinomalyHead,
        "config": {
            "embed_dim": 1024,
            "decoder_hidden_dim": 512,
            "num_layers": 3,
            "dropout": 0.3,
        },
        "trainable": True,
    },
    "fastflow": {
        "class": FastFlowHead,
        "config": {
            "in_channels": 1024,
            "hidden_channels": 512,
            "num_scales": 3,
            "scale_factor": 2,
            "use_batch_norm": True,
        },
        "trainable": True,
    },
    "mambaad": {
        "class": MambaADHead,
        "config": {
            "embed_dim": 1024,
            "image_size": 224,
        },
        "trainable": True,
    },
    "simplenet": {
        "class": SimpleNetHead,
        "config": {
            "input_dim": 1024,
            "hidden_dims": [512, 256],
            "margin": 0.1,
            "lr": 1e-3,
        },
        "trainable": True,
    },
}

# =============================================================================
# Core Functions
# =============================================================================


def create_backbone(name: str, image_size: int, device: torch.device):
    """Create backbone instance."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}")

    entry = BACKBONE_REGISTRY[name]
    cls = entry["class"]
    config = entry["config"].copy()
    config["name"] = name

    # Adjust for input size if needed
    if "image_size" not in config:
        config["image_size"] = image_size

    backbone = cls(config)
    backbone = backbone.to(device)
    backbone.eval()

    return backbone


def create_head(name: str, features: list[torch.Tensor], device: torch.device):
    """Create and fit head instance."""
    if name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {name}")

    entry = HEAD_REGISTRY[name]
    cls = entry["class"]
    config = entry["config"].copy()
    config["name"] = name

    # Adjust embed_dim based on feature channels
    if "embed_dim" in config:
        # Auto-detect from features
        pass

    head = cls(config)
    head = head.to(device)

    # Fit head if needed
    if hasattr(head, "fit"):
        with torch.no_grad():
            head.fit(features)

    return head


def train_head(
    head,
    head_name: str,
    features: list[torch.Tensor],
    train_loader: DataLoader,
    backbone,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
):
    """Train trainable heads."""
    if not HEAD_REGISTRY[head_name].get("trainable", False):
        return head

    head.train()
    params = list(head.parameters())
    if not params:
        return head

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        for batch in train_loader:
            images = batch["image"].to(device)

            with torch.no_grad():
                features = backbone(images)

            # Compute loss based on head type
            if head_name == "dinomaly":
                loss = head.compute_training_loss(features)
            elif head_name == "fastflow":
                aggregated = head._aggregate_features(features)
                z, log_det = head.flow(aggregated)
                log_pz = -0.5 * (z**2).sum(dim=1)
                loss = -(log_pz + log_det).mean()
            elif head_name == "mambaad":
                output = head(features)
                loss = head.compute_loss(features, output["decoder_features"])
            elif head_name == "simplenet":
                loss = head.compute_loss(features)
            else:
                # Default: simple MSE reconstruction
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    head.eval()
    return head


def evaluate(head, head_name: str, features: list[torch.Tensor]):
    """Evaluate head and return anomaly scores."""
    head.eval()

    with torch.no_grad():
        if head_name == "patchcore":
            output = head(features)
            score = output["anomaly_score"]
        elif head_name == "dinomaly":
            output = head(features)
            score = output["anomaly_score"]
        elif head_name == "afrclip":
            output = head(features)
            score = output["anomaly_score"]
        elif head_name == "fastflow":
            aggregated = head._aggregate_features(features)
            z, log_det = head.flow(aggregated)
            log_pz = -0.5 * (z**2).sum(dim=1)
            score = -(log_pz + log_det)
        elif head_name == "mambaad":
            output = head(features)
            score = output["anomaly_score"]
        elif head_name == "simplenet":
            output = head(features)
            score = output["anomaly_score"]
        else:
            # Default
            output = head(features)
            score = output.get(
                "anomaly_score", output.get("score", torch.zeros(len(features[0])))
            )

    return score.cpu().numpy()


def run_single_experiment(
    backbone_name: str,
    head_name: str,
    category: str,
    dataset: str,
    image_size: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run a single experiment and return results."""

    exp_id = f"{backbone_name}_{head_name}_{category}_{dataset}"
    exp_output_dir = output_dir / exp_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {backbone_name} + {head_name}")
    print(f"Category: {category}, Dataset: {dataset}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Load dataset
        if dataset == "mvtec_ad":
            train_dataset = MVTecADDataset(
                root=PROJECT_ROOT / "data" / "mvtec_ad",
                category=category,
                split="train",
                image_size=image_size,
            )
            test_dataset = MVTecADDataset(
                root=PROJECT_ROOT / "data" / "mvtec_ad",
                category=category,
                split="test",
                image_size=image_size,
            )
        elif dataset == "mvtec_loco":
            train_dataset = MVTecLOCODataset(
                root=PROJECT_ROOT / "data" / "mvtec_loco",
                category=category,
                split="train",
                image_size=image_size,
            )
            test_dataset = MVTecLOCODataset(
                root=PROJECT_ROOT / "data" / "mvtec_loco",
                category=category,
                split="test",
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # Create backbone
        backbone = create_backbone(backbone_name, image_size, device)
        backbone.eval()

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Fit head on training data
        print("Fitting head on training data...")

        # Get features from a few samples to fit head
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_features = backbone(sample_batch["image"].to(device))

        head = create_head(head_name, sample_features, device)

        # Train if needed
        if HEAD_REGISTRY[head_name].get("trainable", False):
            print(f"Training head for {epochs} epochs...")
            head = train_head(
                head, head_name, sample_features, train_loader, backbone, device, epochs
            )

        # Extract features from all test samples
        print("Extracting features and evaluating...")
        all_scores = []
        all_labels = []

        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            labels = batch.get("label", torch.zeros(len(images)))

            with torch.no_grad():
                features = backbone(images)
                scores = evaluate(head, head_name, features)

            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

        # Compute AUROC
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_scores)
        else:
            auroc = 0.5  # Default if only one class

        elapsed_time = time.time() - start_time

        result = {
            "experiment": f"{backbone_name} + {head_name}",
            "backbone": backbone_name,
            "head": head_name,
            "category": category,
            "dataset": dataset,
            "epochs": epochs,
            "image_size": image_size,
            "auroc": float(auroc),
            "elapsed_time": elapsed_time,
            "n_train": len(train_dataset),
            "n_test": len(test_dataset),
        }

        # Save result
        with open(exp_output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  AUROC: {auroc:.4f}")
        print(f"  Time: {elapsed_time:.1f}s")

        return result

    except Exception as e:
        print(f"\n  ERROR: {e}")
        elapsed_time = time.time() - start_time
        return {
            "experiment": f"{backbone_name} + {head_name}",
            "backbone": backbone_name,
            "head": head_name,
            "category": category,
            "dataset": dataset,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "auroc": 0.0,
        }


def _to_numpy(x):
    """Safe numpy conversion."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark suite")

    # Experiment settings
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run quick smoke test"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mvtec_ad", "mvtec_loco"],
        default="mvtec_ad",
        help="Dataset to use",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="bottle",
        help="Comma-separated list of categories",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="dinov3_vitl16,convnext_tiny",
        help="Comma-separated list of backbones",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="patchcore,dinomaly",
        help="Comma-separated list of heads",
    )

    # Training settings
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_suite",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine categories
    if args.all:
        if args.dataset == "mvtec_ad":
            categories = MVTEC_AD_CATEGORIES
        else:
            categories = MVTEC_LOCO_CATEGORIES
    else:
        categories = args.categories.split(",")

    # Determine backbones
    if args.all:
        backbones = list(BACKBONE_REGISTRY.keys())
    else:
        backbones = args.backbones.split(",")

    # Determine heads
    if args.all:
        heads = list(HEAD_REGISTRY.keys())
    else:
        heads = args.heads.split(",")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run smoke test first if requested
    if args.smoke_test:
        print("\n=== SMOKE TEST ===")
        result = run_single_experiment(
            backbone_name="dinov3_vitl16",
            head_name="patchcore",
            category="bottle",
            dataset=args.dataset,
            image_size=args.image_size,
            epochs=1,
            batch_size=args.batch_size,
            device=device,
            output_dir=output_dir,
        )
        print(f"\nSmoke test result: AUROC = {result.get('auroc', 'N/A'):.4f}")
        return

    # Run experiments
    all_results = []
    total_experiments = len(backbones) * len(heads) * len(categories)
    current = 0

    print(f"\n=== BENCHMARK SUITE ===")
    print(f"Backbones: {backbones}")
    print(f"Heads: {heads}")
    print(f"Categories: {categories}")
    print(f"Total experiments: {total_experiments}")

    for backbone_name in backbones:
        for head_name in heads:
            for category in categories:
                current += 1
                print(f"\n[{current}/{total_experiments}]")

                result = run_single_experiment(
                    backbone_name=backbone_name,
                    head_name=head_name,
                    category=category,
                    dataset=args.dataset,
                    image_size=args.image_size,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=output_dir,
                )
                all_results.append(result)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Group by backbone+head
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['backbone']} + {r['head']}"
        if "auroc" in r:
            grouped[key].append(r["auroc"])

    for key, aurocs in sorted(grouped.items(), key=lambda x: -np.mean(x[1])):
        mean_auroc = np.mean(aurocs)
        print(f"  {key}: {mean_auroc:.4f} (n={len(aurocs)})")

    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
