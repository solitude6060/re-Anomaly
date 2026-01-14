#!/usr/bin/env python3
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

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mvtec import MVTecADDataset
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.heads.patchcore import PatchCoreHead
from src.models.heads.fastflow import FastFlowHead


WEAK_CATEGORIES = ["screw", "cable", "wood", "capsule", "toothbrush"]


def create_backbone(image_size: int, device: torch.device) -> torch.nn.Module:
    config = {
        "variant": "dinov3_vitl16",
        "patch_size": 16,
        "output_layers": [8, 11, 17, 23],
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": image_size,
        "interpolate_pos_encoding": True,
    }
    backbone = DINOv3Backbone(config)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def run_optimization_experiment(
    category: str,
    data_root: str,
    image_size: int,
    device: torch.device,
    backbone: torch.nn.Module,
    config_variations: list[dict],
) -> list[dict]:
    results = []

    train_dataset = MVTecADDataset(
        root=data_root, category=category, split="train", image_size=image_size
    )
    test_dataset = MVTecADDataset(
        root=data_root, category=category, split="test", image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    all_train_features_by_layer: list[list[torch.Tensor]] = []
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(
            train_loader, desc=f"Extracting {category} features", leave=False
        ):
            images = batch["image"].to(device)
            features = backbone(images)
            if not all_train_features_by_layer:
                all_train_features_by_layer = [[] for _ in features]
            for layer_idx, layer_feat in enumerate(features):
                all_train_features_by_layer[layer_idx].append(layer_feat.cpu())

    all_train_features = [
        torch.cat(layer_feats, dim=0) for layer_feats in all_train_features_by_layer
    ]

    for config in config_variations:
        print(f"    Config: {config['name']}")

        head: torch.nn.Module
        if config["head_type"] == "patchcore":
            head = PatchCoreHead(
                {
                    "feature_dim": 1024,
                    "projection_dim": config.get("projection_dim", 256),
                    "num_neighbors": config.get("num_neighbors", 9),
                    "coreset_ratio": config.get("coreset_ratio", 0.1),
                }
            ).to(device)

            features_gpu = [f.to(device) for f in all_train_features]
            head.fit(features_gpu)

        else:
            head = FastFlowHead(
                {
                    "flow": {
                        "num_blocks": config.get("flow_steps", 8),
                        "hidden_dims": [256, 256],
                        "clamp": config.get("clamp", 2.0),
                    },
                    "feature_processing": {
                        "reduce_dim": True,
                        "projection_dim": 256,
                        "normalize": True,
                    },
                }
            ).to(device)

            features_gpu = [f.to(device) for f in all_train_features]
            head.fit(features_gpu)
            head.to(device)

            head.train()
            optimizer = torch.optim.Adam(head.parameters(), lr=config.get("lr", 1e-4))
            epochs = config.get("epochs", 100)

            for epoch in range(epochs):
                for batch in train_loader:
                    images = batch["image"].to(device)
                    with torch.no_grad():
                        features = backbone(images)

                    output = head(features)
                    loss = -output.get("anomaly_score", torch.tensor(0.0)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            head.eval()

        all_scores = []
        all_labels = []
        head.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"]

                features = backbone(images)
                output = head(features)

                all_scores.extend(output["anomaly_score"].cpu().numpy())
                all_labels.extend(labels.numpy())

        auroc = roc_auc_score(np.array(all_labels), np.array(all_scores))

        results.append(
            {
                "config_name": config["name"],
                "config": config,
                "auroc": float(auroc),
            }
        )

        print(f"      AUROC: {auroc:.4f}")

        del head
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize weak categories")
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output_dir", type=str, default="results/weak_optimization")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or WEAK_CATEGORIES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Optimizing categories: {categories}")

    config_variations = [
        {
            "name": "patchcore_baseline",
            "head_type": "patchcore",
            "num_neighbors": 9,
            "coreset_ratio": 0.1,
        },
        {
            "name": "patchcore_k3",
            "head_type": "patchcore",
            "num_neighbors": 3,
            "coreset_ratio": 0.1,
        },
        {
            "name": "patchcore_k15",
            "head_type": "patchcore",
            "num_neighbors": 15,
            "coreset_ratio": 0.1,
        },
        {
            "name": "patchcore_coreset_25",
            "head_type": "patchcore",
            "num_neighbors": 9,
            "coreset_ratio": 0.25,
        },
        {
            "name": "fastflow_baseline",
            "head_type": "fastflow",
            "flow_steps": 8,
            "epochs": 100,
            "lr": 1e-4,
        },
        {
            "name": "fastflow_steps_12",
            "head_type": "fastflow",
            "flow_steps": 12,
            "epochs": 100,
            "lr": 1e-4,
        },
        {
            "name": "fastflow_epochs_200",
            "head_type": "fastflow",
            "flow_steps": 8,
            "epochs": 200,
            "lr": 1e-4,
        },
        {
            "name": "fastflow_lr_5e5",
            "head_type": "fastflow",
            "flow_steps": 8,
            "epochs": 100,
            "lr": 5e-5,
        },
    ]

    backbone = create_backbone(args.image_size, device)
    all_results = {}

    for category in categories:
        print(f"\n  Category: {category}")
        results = run_optimization_experiment(
            category=category,
            data_root=args.data_root,
            image_size=args.image_size,
            device=device,
            backbone=backbone,
            config_variations=config_variations,
        )
        all_results[category] = results

        best_result = max(results, key=lambda x: x["auroc"])
        print(f"    Best: {best_result['config_name']} ({best_result['auroc']:.4f})")

    summary = {
        "experiment": "weak_category_optimization",
        "categories": categories,
        "config_variations": [c["name"] for c in config_variations],
        "results": all_results,
        "best_per_category": {
            cat: max(results, key=lambda x: x["auroc"])
            for cat, results in all_results.items()
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    for cat, best in summary["best_per_category"].items():
        print(f"  {cat}: {best['config_name']} ({best['auroc']:.4f})")


if __name__ == "__main__":
    main()
