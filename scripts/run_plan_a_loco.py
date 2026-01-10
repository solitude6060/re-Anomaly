"""
Simple evaluation script for Plan A (DINOv2 + PatchCore) on MVTec LOCO AD.

This script runs the full evaluation across all categories and reports metrics.
LOCO AD has logical and structural anomalies - we track performance on both.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.mvtec import MVTecLOCODataset, MVTEC_LOCO_CATEGORIES
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.heads.patchcore import PatchCoreHead


def evaluate_category(
    backbone: torch.nn.Module,
    head: PatchCoreHead,
    category: str,
    data_root: str,
    image_size: int = 224,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Evaluate PatchCore on a single LOCO category."""

    # Load train dataset
    train_dataset = MVTecLOCODataset(
        root=data_root,
        category=category,
        split="train",
        image_size=image_size,
    )

    # Load test dataset
    test_dataset = MVTecLOCODataset(
        root=data_root,
        category=category,
        split="test",
        image_size=image_size,
    )

    # Extract features from training set and fit memory bank
    print(f"  Fitting memory bank with {len(train_dataset)} training samples...")
    all_features = []
    backbone.eval()

    with torch.no_grad():
        for i in tqdm(
            range(len(train_dataset)), desc="  Extracting train features", leave=False
        ):
            sample = train_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            features = backbone(image)
            all_features.append([f.cpu() for f in features])

    # Merge features from all samples
    merged_features = [
        torch.cat([f[i] for f in all_features], dim=0)
        for i in range(len(all_features[0]))
    ]

    # Fit the memory bank
    head.fit([f.to(device) for f in merged_features])

    # Evaluate on test set
    print(f"  Evaluating on {len(test_dataset)} test samples...")
    all_scores = []
    all_labels = []
    all_anomaly_types = []

    head.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="  Testing", leave=False):
            sample = test_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"]
            anomaly_type = sample["anomaly_type"]

            features = backbone(image)
            output = head(features)

            score = output["anomaly_score"].cpu().item()
            all_scores.append(score)
            all_labels.append(label)
            all_anomaly_types.append(anomaly_type)

    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Overall AUROC
    image_auroc = roc_auc_score(all_labels, all_scores)

    # Separate by anomaly type
    logical_mask = np.array([t == "logical" for t in all_anomaly_types])
    structural_mask = np.array([t == "structural" for t in all_anomaly_types])
    good_mask = np.array([t is None for t in all_anomaly_types])

    # Logical AUROC (logical anomalies vs good)
    logical_indices = logical_mask | good_mask
    if logical_mask.sum() > 0:
        logical_auroc = roc_auc_score(
            all_labels[logical_indices], all_scores[logical_indices]
        )
    else:
        logical_auroc = None

    # Structural AUROC (structural anomalies vs good)
    structural_indices = structural_mask | good_mask
    if structural_mask.sum() > 0:
        structural_auroc = roc_auc_score(
            all_labels[structural_indices], all_scores[structural_indices]
        )
    else:
        structural_auroc = None

    # Count anomaly types
    n_logical = int(logical_mask.sum())
    n_structural = int(structural_mask.sum())
    n_good = int(good_mask.sum())

    # Compute recall metrics
    n_anomalies = all_labels.sum()

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
        "logical_auroc": float(logical_auroc) if logical_auroc is not None else None,
        "structural_auroc": float(structural_auroc)
        if structural_auroc is not None
        else None,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_logical": n_logical,
        "n_structural": n_structural,
        "n_good": n_good,
        "precision_at_100_recall": float(precision_100),
        "threshold_100_recall": float(threshold_100_recall),
    }


def main():
    # Configuration
    data_root = "data/mvtec_loco"
    image_size = 224
    output_dir = Path("outputs/plan_a_loco_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize backbone
    print("Loading DINOv2 backbone...")
    backbone_config = {
        "variant": "dinov2_vitb14",
        "model_id": "facebook/dinov2-base",
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": image_size,
        "patch_size": 14,
        "output_layers": [4, 8, 11],
        "interpolate_pos_encoding": True,
    }
    backbone = DINOv2Backbone(backbone_config)
    backbone = backbone.to(device)
    backbone.eval()
    print("Backbone loaded.")

    # Results
    all_results = []

    # Evaluate each category
    for category in MVTEC_LOCO_CATEGORIES:
        print(f"\nEvaluating category: {category}")

        # Create fresh head for each category
        head_config = {
            "k_nearest": 9,
            "coreset_sampling_ratio": 0.1,
            "coreset_method": "random",
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
        if result["logical_auroc"] is not None:
            print(f"  Logical AUROC: {result['logical_auroc']:.4f}")
        if result["structural_auroc"] is not None:
            print(f"  Structural AUROC: {result['structural_auroc']:.4f}")
        print(f"  Precision@100%Recall: {result['precision_at_100_recall']:.4f}")
        print(f"  Time: {result['eval_time_seconds']:.1f}s")

    # Compute averages
    avg_auroc = np.mean([r["image_auroc"] for r in all_results])
    avg_precision = np.mean([r["precision_at_100_recall"] for r in all_results])

    logical_aurocs = [
        r["logical_auroc"] for r in all_results if r["logical_auroc"] is not None
    ]
    structural_aurocs = [
        r["structural_auroc"] for r in all_results if r["structural_auroc"] is not None
    ]

    avg_logical = np.mean(logical_aurocs) if logical_aurocs else None
    avg_structural = np.mean(structural_aurocs) if structural_aurocs else None

    summary = {
        "experiment": "Plan A - DINOv2 + PatchCore on MVTec LOCO",
        "backbone": "dinov2_vitb14",
        "head": "patchcore",
        "image_size": image_size,
        "avg_image_auroc": float(avg_auroc),
        "avg_logical_auroc": float(avg_logical) if avg_logical is not None else None,
        "avg_structural_auroc": float(avg_structural)
        if avg_structural is not None
        else None,
        "avg_precision_at_100_recall": float(avg_precision),
        "per_category_results": all_results,
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - MVTec LOCO AD")
    print("=" * 60)
    print(f"\nBackbone: DINOv2-ViT-B/14")
    print(f"Head: PatchCore (k=9, coreset=10%)")
    print(f"\nPer-category Results:")
    print("-" * 60)
    print(f"  {'Category':<20} {'Image':>10} {'Logical':>10} {'Structural':>10}")
    print("-" * 60)
    for r in all_results:
        logical_str = (
            f"{r['logical_auroc']:.4f}" if r["logical_auroc"] is not None else "N/A"
        )
        structural_str = (
            f"{r['structural_auroc']:.4f}"
            if r["structural_auroc"] is not None
            else "N/A"
        )
        print(
            f"  {r['category']:<20} {r['image_auroc']:>10.4f} {logical_str:>10} {structural_str:>10}"
        )
    print("-" * 60)
    avg_logical_str = f"{avg_logical:.4f}" if avg_logical is not None else "N/A"
    avg_structural_str = (
        f"{avg_structural:.4f}" if avg_structural is not None else "N/A"
    )
    print(
        f"  {'AVERAGE':<20} {avg_auroc:>10.4f} {avg_logical_str:>10} {avg_structural_str:>10}"
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
