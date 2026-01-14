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

from src.data.mvtec import MVTecADDataset, MVTEC_AD_CATEGORIES
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.heads.patchcore import PatchCoreHead
from src.models.heads.fastflow import FastFlowHead
from src.models.heads.rectflow import RectFlowHead
from src.models.ensemble import EnsembleAnomalyDetector


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


def create_heads(device: torch.device) -> dict[str, torch.nn.Module]:
    patchcore_config = {
        "feature_dim": 1024,
        "projection_dim": 256,
        "num_neighbors": 9,
        "coreset_ratio": 0.1,
    }
    patchcore = PatchCoreHead(patchcore_config).to(device)

    fastflow_config = {
        "feature_dim": 1024,
        "flow_steps": 8,
        "hidden_ratio": 1.0,
        "clamp": 2.0,
    }
    fastflow = FastFlowHead(fastflow_config).to(device)

    rectflow_config = {
        "feature_dim": 1024,
        "hidden_dim": 512,
        "num_blocks": 4,
        "inference_steps": 20,
    }
    rectflow = RectFlowHead(rectflow_config).to(device)

    return {
        "patchcore": patchcore,
        "fastflow": fastflow,
        "rectflow": rectflow,
    }


def train_heads(
    backbone: torch.nn.Module,
    heads: dict[str, torch.nn.Module],
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
) -> None:
    # Collect features per layer (not per sample)
    all_features_by_layer: list[list[torch.Tensor]] = []
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting features", leave=False):
            images = batch["image"].to(device)
            features = backbone(images)  # List of [B, C, H, W] for each layer

            # Initialize layer lists on first batch
            if not all_features_by_layer:
                all_features_by_layer = [[] for _ in range(len(features))]

            # Append each layer's features
            for layer_idx, feat in enumerate(features):
                all_features_by_layer[layer_idx].append(feat.cpu())

    # Concatenate each layer's features along batch dimension
    all_features = [
        torch.cat(layer_feats, dim=0) for layer_feats in all_features_by_layer
    ]

    for name, head in heads.items():
        print(f"  Training {name}...")

        if name == "patchcore":
            features_gpu = [f.to(device) for f in all_features]
            head.fit(features_gpu)
        else:
            features_gpu = [f.to(device) for f in all_features]
            head.fit(features_gpu)
            head.to(device)
            head.train()
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)

            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    images = batch["image"].to(device)
                    with torch.no_grad():
                        features = backbone(images)

                    output = head(features)
                    loss = output.get("loss", output["anomaly_score"].mean())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 20 == 0:
                    print(
                        f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}"
                    )

            head.eval()


def evaluate_ensemble(
    backbone: torch.nn.Module,
    ensemble: EnsembleAnomalyDetector,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    all_scores = []
    all_labels = []
    individual_scores: dict[str, list] = {name: [] for name in ensemble.heads.keys()}

    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"]

            features = backbone(images)
            output = ensemble.predict(features)

            all_scores.extend(output["anomaly_score"].cpu().numpy())
            all_labels.extend(labels.numpy())

            for name, scores in output["individual_scores"].items():
                individual_scores[name].extend(scores.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    ensemble_auroc = roc_auc_score(all_labels, all_scores)

    individual_aurocs = {}
    for name, scores in individual_scores.items():
        individual_aurocs[name] = roc_auc_score(all_labels, np.array(scores))

    return {
        "ensemble_auroc": float(ensemble_auroc),
        "individual_aurocs": individual_aurocs,
    }


def run_ensemble_experiment(
    data_root: str,
    output_dir: Path,
    image_size: int,
    categories: list[str],
    device: torch.device,
    epochs: int,
    weights: dict[str, float],
) -> dict:
    print(f"\n{'=' * 70}")
    print("ENSEMBLE EXPERIMENT: PatchCore + FastFlow + RectFlow")
    print(f"  Weights: {weights}")
    print("=" * 70)

    backbone = create_backbone(image_size, device)
    all_results = []

    for category in categories:
        print(f"\n  Category: {category}")

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

        heads = create_heads(device)

        start_time = time.time()
        train_heads(backbone, heads, train_loader, device, epochs=epochs)

        ensemble_config = {"weights": weights, "normalize_scores": True}
        ensemble = EnsembleAnomalyDetector(ensemble_config)
        for name, head in heads.items():
            ensemble.add_head(name, head)
        ensemble._fitted = True

        result = evaluate_ensemble(backbone, ensemble, test_loader, device)
        result["category"] = category
        result["eval_time_seconds"] = time.time() - start_time

        all_results.append(result)
        print(f"    Ensemble AUROC: {result['ensemble_auroc']:.4f}")
        print(f"    Individual: {result['individual_aurocs']}")

        del heads, ensemble
        torch.cuda.empty_cache()

    avg_ensemble = np.mean([r["ensemble_auroc"] for r in all_results])
    avg_individual = {
        name: np.mean([r["individual_aurocs"][name] for r in all_results])
        for name in all_results[0]["individual_aurocs"].keys()
    }

    summary = {
        "experiment": "ensemble_patchcore_fastflow_rectflow",
        "backbone": "dinov3_vitl16",
        "weights": weights,
        "image_size": image_size,
        "epochs": epochs,
        "avg_ensemble_auroc": float(avg_ensemble),
        "avg_individual_aurocs": avg_individual,
        "per_category_results": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Average Ensemble AUROC: {avg_ensemble:.4f}")
    print(f"  Average Individual: {avg_individual}")

    del backbone
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Ensemble experiments")
    parser.add_argument("--data_root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output_dir", type=str, default="results/ensemble")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--patchcore_weight", type=float, default=0.5)
    parser.add_argument("--fastflow_weight", type=float, default=0.3)
    parser.add_argument("--rectflow_weight", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    categories = args.categories or MVTEC_AD_CATEGORIES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = {
        "patchcore": args.patchcore_weight,
        "fastflow": args.fastflow_weight,
        "rectflow": args.rectflow_weight,
    }

    print(f"Device: {device}")
    print(f"Categories: {len(categories)}")

    summary = run_ensemble_experiment(
        data_root=args.data_root,
        output_dir=output_dir,
        image_size=args.image_size,
        categories=categories,
        device=device,
        epochs=args.epochs,
        weights=weights,
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
