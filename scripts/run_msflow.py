"""
MSFlow (Multi-Scale Flow) evaluation script for MVTec AD.
Uses multi-scale normalizing flows. Only OK/NG labels needed.
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_HF_CACHE / "hub")

import argparse
import json
import time

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
from src.models.heads.msflow import MSFlowHead


BACKBONE_CONFIGS = {
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
}


def train_msflow(
    backbone: nn.Module,
    head: MSFlowHead,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda"),
) -> None:
    print("  Initializing MSFlow...")
    backbone.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch["image"].to(device)
        sample_features = backbone(sample_images)
        head.fit(sample_features)

    head = head.to(device)
    head.train()

    params = list(head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Training MSFlow for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            with torch.no_grad():
                features = backbone(images)

            loss = torch.tensor(0.0, device=device)

            for i, (feat, flow) in enumerate(zip(features, head.flows)):
                if head.reduce_dim and i < len(head.projections):
                    feat = head.projections[i](feat)

                z, log_det = flow(feat)
                log_pz = -0.5 * (z**2).sum(dim=1)
                log_prob = (log_pz + log_det).mean()
                loss = loss - log_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_category(
    backbone: nn.Module,
    head: MSFlowHead,
    category: str,
    data_root: str,
    image_size: int = 224,
    epochs: int = 100,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> dict:
    train_dataset = MVTecADDataset(
        root=data_root,
        category=category,
        split="train",
        image_size=image_size,
    )

    test_dataset = MVTecADDataset(
        root=data_root,
        category=category,
        split="test",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    train_msflow(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        epochs=epochs,
        device=device,
    )

    print(f"  Evaluating on {len(test_dataset)} test samples...")
    head.eval()
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

    n_anomalies = all_labels.sum()
    n_normal = len(all_labels) - n_anomalies

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
    parser = argparse.ArgumentParser(description="Run MSFlow evaluation on MVTec AD")
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output_dir", type=str, default="results/msflow")
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vitl16",
        choices=list(BACKBONE_CONFIGS.keys()),
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or MVTEC_AD_CATEGORIES
    backbone_name = args.backbone
    backbone_cfg = BACKBONE_CONFIGS[backbone_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {backbone_name} backbone...")
    BackboneClass = backbone_cfg["class"]
    backbone_config = {
        **backbone_cfg["config"],
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": args.image_size,
        "interpolate_pos_encoding": True,
    }
    backbone = BackboneClass(backbone_config)
    backbone = backbone.to(device)
    backbone.eval()
    print("Backbone loaded.")

    all_results = []

    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print("=" * 60)

        head_config = {
            "flow": {
                "type": "realnvp",
                "num_blocks": 8,
                "hidden_dims": [256, 256],
            },
            "scales": ["low", "mid", "high"],
            "feature_processing": {
                "reduce_dim": True,
                "projection_dim": 256,
            },
            "anomaly_score": {
                "scale_weights": [0.2, 0.3, 0.5],
            },
        }
        head = MSFlowHead(head_config)

        start_time = time.time()
        result = evaluate_category(
            backbone=backbone,
            head=head,
            category=category,
            data_root=args.data_root,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        result["eval_time_seconds"] = time.time() - start_time

        all_results.append(result)

        print(f"  Image AUROC: {result['image_auroc']:.4f}")
        print(f"  Precision@100%Recall: {result['precision_at_100_recall']:.4f}")
        print(f"  Time: {result['eval_time_seconds']:.1f}s")

        del head
        torch.cuda.empty_cache()

    avg_auroc = np.mean([r["image_auroc"] for r in all_results])
    avg_precision = np.mean([r["precision_at_100_recall"] for r in all_results])

    summary = {
        "experiment": f"MSFlow - {backbone_name}",
        "backbone": backbone_name,
        "head": "msflow",
        "image_size": args.image_size,
        "epochs": args.epochs,
        "avg_image_auroc": float(avg_auroc),
        "avg_precision_at_100_recall": float(avg_precision),
        "per_category_results": all_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - MSFlow")
    print("=" * 60)
    print(f"\nBackbone: {backbone_name}")
    print(f"Training epochs: {args.epochs}")
    print(f"\nPer-category Image AUROC:")
    print("-" * 40)
    for r in all_results:
        print(f"  {r['category']:<20} {r['image_auroc']:.4f}")
    print("-" * 40)
    print(f"  {'AVERAGE':<20} {avg_auroc:.4f}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
