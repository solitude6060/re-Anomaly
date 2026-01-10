import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def build_test_dataloader(cfg: DictConfig) -> DataLoader:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = Path(cfg.dataset.root_path) / "test"
    dataset = ImageFolder(str(data_path), transform=transform)

    return DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.get("num_workers", 4),
        pin_memory=True,
    )


def compute_metrics(
    scores: list[float],
    labels: list[int],
    threshold: float | None = None,
) -> dict[str, float]:
    import numpy as np

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    auroc = roc_auc_score(labels_arr, scores_arr)

    if threshold is None:
        threshold = np.percentile(scores_arr[labels_arr == 0], 95)

    predictions = (scores_arr >= threshold).astype(int)

    tp = ((predictions == 1) & (labels_arr == 1)).sum()
    fp = ((predictions == 1) & (labels_arr == 0)).sum()
    fn = ((predictions == 0) & (labels_arr == 1)).sum()
    tn = ((predictions == 0) & (labels_arr == 0)).sum()

    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    overkill_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "auroc": auroc,
        "miss_rate": miss_rate,
        "overkill_rate": overkill_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Evaluation Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    backbone = build_backbone(cfg)
    backbone.load_pretrained(cfg.backbone.get("checkpoint"))
    backbone.freeze()
    backbone = backbone.to(device)
    backbone.eval()

    head = build_head(cfg)

    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path:
        log.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        head.load_state_dict(ckpt["head"])

    head = head.to(device)
    head.eval()

    test_loader = build_test_dataloader(cfg)

    all_scores = []
    all_labels = []
    inference_times = []

    log.info("Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            features = backbone(images)
            scores = head.get_anomaly_score(features)
            end.record()

            torch.cuda.synchronize()
            inference_times.append(start.elapsed_time(end))

            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = compute_metrics(all_scores, all_labels)

    avg_inference_time = (
        sum(inference_times) / len(inference_times) if inference_times else 0
    )

    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"  AUROC:         {metrics['auroc']:.4f}")
    log.info(f"  Miss Rate:     {metrics['miss_rate']:.4f} (target: 0%)")
    log.info(f"  Overkill Rate: {metrics['overkill_rate']:.4f} (target: <10%)")
    log.info(f"  Precision:     {metrics['precision']:.4f}")
    log.info(f"  Recall:        {metrics['recall']:.4f}")
    log.info(f"  F1 Score:      {metrics['f1']:.4f}")
    log.info(f"  Threshold:     {metrics['threshold']:.4f}")
    log.info(f"  Avg Inference: {avg_inference_time:.2f}ms")
    log.info("=" * 50)

    results_path = (
        Path(cfg.output.root_dir) / cfg.output.experiment_name / "eval_results.yaml"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        **metrics,
        "avg_inference_time_ms": avg_inference_time,
        "num_samples": len(all_scores),
    }

    with open(results_path, "w") as f:
        OmegaConf.save(OmegaConf.create(results), f)

    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
