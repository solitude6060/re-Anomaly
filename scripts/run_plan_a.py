"""
Simple evaluation script for Plan A (DINOv2 + PatchCore) on MVTec AD.

This script runs the full evaluation across all categories and reports metrics.
Supports Weights & Biases logging for experiment tracking.
"""

import os
from pathlib import Path

# Set HuggingFace cache to project local directory BEFORE importing transformers
_PROJECT_ROOT = Path(__file__).parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_HF_CACHE / "hub")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.mvtec import MVTecADDataset, MVTEC_AD_CATEGORIES
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.backbones.pixio import PixIOBackbone
from src.models.heads.patchcore import PatchCoreHead
from src.utils.wandb_logger import WandbLogger, WandbConfig


BACKBONE_CONFIGS = {
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
    "dinov3_convnext_large": {
        "class": DINOv3Backbone,
        "config": {
            "variant": "dinov3_convnext_large",
            "patch_size": 32,
            "output_layers": [4, 8, 11],
        },
    },
    "pixio_vitl16": {
        "class": PixIOBackbone,
        "config": {
            "variant": "pixio_vitl16",
            "patch_size": 16,
            "output_layers": [8, 11, 17, 23],
            "image_size": 224,
        },
    },
}


def evaluate_category(
    backbone: torch.nn.Module,
    head: PatchCoreHead,
    category: str,
    data_root: str,
    image_size: int = 224,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Evaluate PatchCore on a single category."""

    # Load train dataset
    train_dataset = MVTecADDataset(
        root=data_root,
        category=category,
        split="train",
        image_size=image_size,
    )

    # Load test dataset
    test_dataset = MVTecADDataset(
        root=data_root,
        category=category,
        split="test",
        image_size=image_size,
    )

    # Extract features from training set and fit memory bank
    print(f"  Fitting memory bank with {len(train_dataset)} training samples...")
    all_patches = []
    backbone.eval()

    with torch.no_grad():
        for i in tqdm(
            range(len(train_dataset)), desc="  Extracting train features", leave=False
        ):
            sample = train_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            features = backbone(image)
            patches = head.extract_patches(features)
            all_patches.append(patches)
            del features

    all_patches = torch.cat(all_patches, dim=0)
    print(f"  Total patches: {all_patches.shape[0]}, applying coreset sampling...")
    head.fit_from_patches(all_patches.to(device))
    del all_patches
    torch.cuda.empty_cache()

    # Evaluate on test set
    print(f"  Evaluating on {len(test_dataset)} test samples...")
    all_scores = []
    all_labels = []

    head.eval()
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

    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    image_auroc = roc_auc_score(all_labels, all_scores)

    # Compute recall at different thresholds
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = all_labels[sorted_indices]

    n_anomalies = all_labels.sum()
    n_normal = len(all_labels) - n_anomalies

    # Find threshold for 100% recall
    anomaly_scores = all_scores[all_labels == 1]
    threshold_100_recall = anomaly_scores.min() if len(anomaly_scores) > 0 else 0

    # Compute precision at 100% recall
    predictions_100 = (all_scores >= threshold_100_recall).astype(int)
    tp = ((predictions_100 == 1) & (all_labels == 1)).sum()
    fp = ((predictions_100 == 1) & (all_labels == 0)).sum()
    precision_100 = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "category": category,
        "image_auroc": float(image_auroc),
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_anomalies": int(n_anomalies),
        "n_normal": int(n_normal),
        "precision_at_100_recall": float(precision_100),
        "threshold_100_recall": float(threshold_100_recall),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Plan A evaluation on MVTec AD")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/mvtec_ad",
        help="Path to MVTec AD dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plan_a_greedy",
        help="Output directory for results",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to evaluate (default: all)",
    )
    # W&B arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitl14",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone to use (default: dinov2_vitl14)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="re-anomaly",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B tags",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    data_root = args.data_root
    image_size = args.image_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Categories to evaluate
    categories = args.categories or MVTEC_AD_CATEGORIES

    backbone_name = args.backbone
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE_CONFIGS.keys())}"
        )
    backbone_cfg = BACKBONE_CONFIGS[backbone_name]

    wandb_logger = None
    if args.wandb:
        backbone_family = backbone_name.split("_")[0]
        wandb_config = WandbConfig(
            project=args.wandb_project,
            name=args.wandb_name or f"plan_a_{backbone_name}_patchcore",
            tags=args.wandb_tags
            or ["plan_a", backbone_family, backbone_name, "patchcore", "mvtec_ad"],
            group="plan_a",
            job_type="eval",
        )
        wandb_logger = WandbLogger(config=wandb_config, enabled=True)

        wandb_logger.log_config(
            {
                "experiment": "Plan A",
                "backbone": backbone_name,
                "backbone_config": backbone_cfg["config"],
                "head": "patchcore",
                "dataset": "mvtec_ad",
                "image_size": image_size,
                "data_root": data_root,
                "categories": categories,
            }
        )
    backbone_cfg = BACKBONE_CONFIGS[backbone_name]

    # Initialize W&B logger
    wandb_logger = None
    if args.wandb:
        # Extract backbone family for tags
        backbone_family = backbone_name.split("_")[0]  # dinov2, dinov3, pixio
        wandb_config = WandbConfig(
            project=args.wandb_project,
            name=args.wandb_name or f"plan_a_{backbone_name}_patchcore",
            tags=args.wandb_tags
            or ["plan_a", backbone_family, backbone_name, "patchcore", "mvtec_ad"],
            group="plan_a",
            job_type="eval",
        )
        wandb_logger = WandbLogger(config=wandb_config, enabled=True)

        # Log experiment config
        wandb_logger.log_config(
            {
                "experiment": "Plan A",
                "backbone": backbone_name,
                "backbone_config": backbone_cfg["config"],
                "head": "patchcore",
                "dataset": "mvtec_ad",
                "image_size": image_size,
                "data_root": data_root,
                "categories": categories,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {backbone_name} backbone...")
    BackboneClass = backbone_cfg["class"]
    backbone_config = {
        **backbone_cfg["config"],
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": image_size,
        "interpolate_pos_encoding": True,
    }
    backbone = BackboneClass(backbone_config)
    backbone = backbone.to(device)
    backbone.eval()
    print("Backbone loaded.")

    # Results
    all_results = []

    # Evaluate each category
    for idx, category in enumerate(categories):
        print(f"\nEvaluating category: {category}")

        # Create fresh head for each category
        head_config = {
            "k_nearest": 9,
            "coreset_sampling_ratio": 0.1,
            "coreset_method": "greedy",
            "feature_aggregation": "concat",
            "memory_bank": {
                "max_size": 100000,
                "normalize": True,
                "use_faiss": True,
            },
            "anomaly_score": {
                "normalize": True,
            },
        }
        head = PatchCoreHead(head_config)
        head = head.to(device)

        start_time = time.time()
        result = evaluate_category(
            backbone=backbone,
            head=head,
            category=category,
            data_root=data_root,
            image_size=image_size,
            device=device,
        )
        result["eval_time_seconds"] = time.time() - start_time

        all_results.append(result)

        print(f"  Image AUROC: {result['image_auroc']:.4f}")
        print(f"  Precision@100%Recall: {result['precision_at_100_recall']:.4f}")
        print(f"  Time: {result['eval_time_seconds']:.1f}s")

        # Log to W&B
        if wandb_logger:
            wandb_logger.log(
                {
                    f"{category}/image_auroc": result["image_auroc"],
                    f"{category}/precision_at_100_recall": result[
                        "precision_at_100_recall"
                    ],
                    f"{category}/eval_time_seconds": result["eval_time_seconds"],
                    f"{category}/n_train": result["n_train"],
                    f"{category}/n_test": result["n_test"],
                },
                step=idx,
            )

    # Compute averages
    avg_auroc = np.mean([r["image_auroc"] for r in all_results])
    avg_precision = np.mean([r["precision_at_100_recall"] for r in all_results])

    summary = {
        "experiment": f"Plan A - {backbone_name} + PatchCore",
        "backbone": backbone_name,
        "head": "patchcore",
        "image_size": image_size,
        "avg_image_auroc": float(avg_auroc),
        "avg_precision_at_100_recall": float(avg_precision),
        "per_category_results": all_results,
    }

    # Log summary to W&B
    if wandb_logger:
        wandb_logger.log_summary(
            {
                "avg_image_auroc": avg_auroc,
                "avg_precision_at_100_recall": avg_precision,
                "total_categories": len(all_results),
            }
        )

        # Log results table
        columns = [
            "category",
            "image_auroc",
            "precision_at_100_recall",
            "n_train",
            "n_test",
            "eval_time",
        ]
        data = [
            [
                r["category"],
                r["image_auroc"],
                r["precision_at_100_recall"],
                r["n_train"],
                r["n_test"],
                r["eval_time_seconds"],
            ]
            for r in all_results
        ]
        wandb_logger.log_table("results_table", columns, data)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nBackbone: {backbone_name}")
    print(f"Head: PatchCore (k=9, coreset=10%)")
    print(f"\nPer-category Image AUROC:")
    print("-" * 40)
    for r in all_results:
        print(f"  {r['category']:<15} {r['image_auroc']:.4f}")
    print("-" * 40)
    print(f"  {'AVERAGE':<15} {avg_auroc:.4f}")
    print(f"\nResults saved to: {results_path}")

    # Finish W&B run
    if wandb_logger:
        wandb_logger.finish()
        print("W&B logging complete.")


if __name__ == "__main__":
    main()
