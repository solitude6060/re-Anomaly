#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_AD="data/mvtec_ad"
DATA_LOCO="data/mvtec_loco"
DATA_LOCO_SEG="data/mvtec_loco_composition_maps"
SALAD_ROOT="/tmp/SALAD"
SALAD_TEACHER_WEIGHTS="$SALAD_ROOT/models/teacher_medium.pth"
OUTPUT_ROOT="results/full_experiments"
IMAGE_SIZE=224
EPOCHS=100
BATCH_SIZE=16

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "Missing $label at: $path" >&2
    exit 1
  fi
}

preflight_ad() {
  require_path "$DATA_AD" "MVTec AD dataset"
}

preflight_loco() {
  require_path "$DATA_LOCO" "MVTec LOCO dataset"
  require_path "$DATA_LOCO_SEG" "MVTec LOCO composition maps"
}

preflight_salad() {
  require_path "$SALAD_ROOT" "SALAD repo"
  require_path "$SALAD_TEACHER_WEIGHTS" "SALAD teacher weights"
}

preflight_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv command (install uv or adjust script)." >&2
    exit 1
  fi
}

AD_BACKBONES=(
  dinov2_vitb14
  dinov2_vitl14
  dinov3_vitl16
  clip_vitl14
  siglip_so400m_384
  swin_base
  swin_large
  swinv2_base
  convnext_tiny
  convnext_base
)

AD_HEADS=(
  patchcore
  dinomaly
  fastflow
  rectflow
  simplenet
  mambaad
  msflow
  afrclip
)

LOCO_BACKBONES=(
  dinov2_vitb14
  dinov2_vitl14
  dinov3_vitl16
  pixio_vitl16
)

FEWSHOT_KS=(1 5 10 20 50 100 200)

run_mvtec_ad() {
  preflight_uv
  preflight_ad
  echo "[MVTec AD] Running full experiment matrix"
  PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
    --data_root "$DATA_AD" \
    --output_dir "$OUTPUT_ROOT/mvtec_ad" \
    --image_size "$IMAGE_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --backbones "${AD_BACKBONES[@]}" \
    --heads "${AD_HEADS[@]}"
}

run_mvtec_loco_plan_a() {
  preflight_uv
  preflight_loco
  echo "[MVTec LOCO] Running Plan A (PatchCore) for all backbones"
  for backbone in "${LOCO_BACKBONES[@]}"; do
    echo "  Backbone: $backbone"
    PYTHONPATH=. uv run python scripts/run_plan_a_loco.py \
      --data_root "$DATA_LOCO" \
      --output_dir "$OUTPUT_ROOT/mvtec_loco_plan_a/$backbone" \
      --image_size "$IMAGE_SIZE" \
      --backbone "$backbone"
  done
}

run_mvtec_loco_salad() {
  preflight_uv
  preflight_loco
  preflight_salad
  echo "[MVTec LOCO] Running SALAD (Plan B)"
  PYTHONPATH=. uv run python scripts/run_salad.py --all
}

run_fewshot() {
  preflight_uv
  preflight_ad
  echo "[Few-shot] Running PatchCore few-shot k sweep"
  mkdir -p "$OUTPUT_ROOT/fewshot"
  for k in "${FEWSHOT_KS[@]}"; do
    echo "  k=$k"
    K="$k" PYTHONPATH=. uv run python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.mvtec import MVTecADDataset, MVTEC_AD_CATEGORIES, FewShotSampler
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.heads.patchcore import PatchCoreHead

k = int(os.environ["K"])
output_dir = Path("results/full_experiments/fewshot")
output_dir.mkdir(parents=True, exist_ok=True)

backbone = DINOv3Backbone(
    {
        "variant": "dinov3_vitl16",
        "patch_size": 16,
        "output_layers": [8, 11, 17, 23],
        "pretrained": True,
        "freeze_backbone": True,
        "image_size": 224,
        "interpolate_pos_encoding": True,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device).eval()

results = []
for category in MVTEC_AD_CATEGORIES:
    train_ds = MVTecADDataset("data/mvtec_ad", category=category, split="train", image_size=224)
    test_ds = MVTecADDataset("data/mvtec_ad", category=category, split="test", image_size=224)

    sampler = FewShotSampler(train_ds, k=k, seed=42)
    train_subset = sampler.sample()

    head = PatchCoreHead(
        {
            "k_nearest": 9,
            "coreset_sampling_ratio": 0.1,
            "coreset_method": "greedy",
            "feature_aggregation": "concat",
            "memory_bank": {"max_size": 100000, "normalize": True, "use_faiss": True},
            "anomaly_score": {"normalize": True},
        }
    )

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
    features_batches = []
    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(device)
            features_batches.append([f.cpu() for f in backbone(images)])

    merged = [torch.cat([f[i] for f in features_batches], dim=0) for i in range(len(features_batches[0]))]
    head.fit(merged)

    scores, labels = [], []
    with torch.no_grad():
        for sample in test_ds:
            image = sample["image"].unsqueeze(0).to(device)
            output = head(backbone(image))
            scores.append(output["anomaly_score"].cpu().item())
            labels.append(sample["label"])

    auroc = roc_auc_score(labels, scores)
    results.append({"category": category, "auroc": float(auroc)})

avg_auroc = float(np.mean([r["auroc"] for r in results]))

payload = {"k": k, "avg_auroc": avg_auroc, "results": results}
with open(output_dir / f"k{k}_results.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"Average AUROC (k={k}): {avg_auroc:.4f}")
PY
  done
}

run_all() {
  run_mvtec_ad
  run_mvtec_loco_plan_a
  run_mvtec_loco_salad
  run_fewshot
}

case "${1:-}" in
  ad) run_mvtec_ad ;;
  loco-plan-a) run_mvtec_loco_plan_a ;;
  loco-salad) run_mvtec_loco_salad ;;
  fewshot) run_fewshot ;;
  all) run_all ;;
  *)
    echo "Usage: $0 {ad|loco-plan-a|loco-salad|fewshot|all}"
    exit 1
    ;;
 esac
