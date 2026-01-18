#!/bin/bash
# Pending Experiments for re-Anomaly
# Run these when GPU is available (after Dinomaly exp-1 completes)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# EXP-2: Test PatchCore with larger image sizes (448 and 518)
# ============================================================================
# Current baseline: 224px = 96.51% AUROC
# Expected improvement: DINOv3 was trained at 518px, higher res should help

run_exp2_448() {
    echo "Running EXP-2a: PatchCore with image_size=448"
    PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
        --data_root data/mvtec_ad \
        --output_dir results/patchcore_448 \
        --image_size 448 \
        --epochs 1 \
        --batch_size 8 \
        --backbones dinov3_vitl16 \
        --heads patchcore
}

run_exp2_518() {
    echo "Running EXP-2b: PatchCore with image_size=518"
    PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
        --data_root data/mvtec_ad \
        --output_dir results/patchcore_518 \
        --image_size 518 \
        --epochs 1 \
        --batch_size 8 \
        --backbones dinov3_vitl16 \
        --heads patchcore
}

# ============================================================================
# EXP-3: Test PixIO backbone
# ============================================================================
# PixIO uses 8 class tokens with MAE-style training
# Should provide complementary features to DINOv3

run_exp3() {
    echo "Running EXP-3: PixIO backbone evaluation"
    PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
        --data_root data/mvtec_ad \
        --output_dir results/pixio_eval \
        --image_size 224 \
        --epochs 1 \
        --batch_size 16 \
        --backbones pixio_vitl16 \
        --heads patchcore
}

# ============================================================================
# EXP-4: Few-shot experiments
# ============================================================================
# Test performance with limited training samples: k=1,5,10,20,50,100,200

run_exp4() {
    echo "Running EXP-4: Few-shot experiments"
    for k in 1 5 10 20 50 100 200; do
        echo "  k=$k shots..."
        PYTHONPATH=. uv run python -c "
from src.data.mvtec import MVTecADDataset, FewShotSampler
from src.models.backbones.dinov3 import DINOv3Backbone
from src.models.heads.patchcore import PatchCoreHead
from torch.utils.data import DataLoader
import torch
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np

device = torch.device('cuda')
k = $k
results = []

backbone = DINOv3Backbone({
    'variant': 'dinov3_vitl16',
    'patch_size': 16,
    'output_layers': [8, 11, 17, 23],
})
backbone = backbone.to(device).eval()

categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
              'transistor', 'wood', 'zipper']

for cat in categories:
    train_ds = MVTecADDataset('data/mvtec_ad', category=cat, split='train', image_size=224)
    test_ds = MVTecADDataset('data/mvtec_ad', category=cat, split='test', image_size=224)
    
    # Few-shot sampling
    sampler = FewShotSampler(train_ds, k=k, seed=42)
    train_subset = sampler.sample()
    
    head = PatchCoreHead({
        'k_nearest': 9,
        'coreset_sampling_ratio': 0.1,
        'coreset_method': 'greedy',
        'feature_aggregation': 'concat',
        'memory_bank': {'max_size': 100000, 'normalize': True, 'use_faiss': True},
        'anomaly_score': {'normalize': True},
    })
    
    # Fit on few-shot samples
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
    all_features = []
    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].to(device)
            features = backbone(images)
            all_features.append([f.cpu() for f in features])
    
    merged = [torch.cat([f[i] for f in all_features], dim=0) for i in range(len(all_features[0]))]
    head.fit(merged)
    
    # Evaluate
    scores, labels = [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            image = sample['image'].unsqueeze(0).to(device)
            features = backbone(image)
            output = head(features)
            scores.append(output['anomaly_score'].cpu().item())
            labels.append(sample['label'])
    
    auroc = roc_auc_score(labels, scores)
    results.append({'category': cat, 'k': k, 'auroc': float(auroc)})
    print(f'  {cat}: {auroc:.4f}')

avg_auroc = np.mean([r['auroc'] for r in results])
print(f'  Average AUROC (k={k}): {avg_auroc:.4f}')

Path('results/fewshot').mkdir(parents=True, exist_ok=True)
with open(f'results/fewshot/k{k}_results.json', 'w') as f:
    json.dump({'k': k, 'avg_auroc': avg_auroc, 'results': results}, f, indent=2)
"
    done
}

# ============================================================================
# EXP-5: SALAD on MVTec LOCO
# ============================================================================
# SALAD for logical anomaly detection

run_exp5() {
    echo "Running EXP-5: SALAD on MVTec LOCO"
    PYTHONPATH=. uv run python scripts/run_salad.py \
        --data_root data/mvtec_loco \
        --output_dir results/salad_loco \
        --epochs 100 \
        --batch_size 8
}

# ============================================================================
# Run all pending experiments
# ============================================================================
run_all() {
    echo "Running all pending experiments..."
    run_exp2_448
    run_exp2_518
    run_exp3
    run_exp4
    run_exp5
}

# Parse command line
case "${1:-}" in
    exp2a) run_exp2_448 ;;
    exp2b) run_exp2_518 ;;
    exp2) run_exp2_448 && run_exp2_518 ;;
    exp3) run_exp3 ;;
    exp4) run_exp4 ;;
    exp5) run_exp5 ;;
    all) run_all ;;
    *)
        echo "Usage: $0 {exp2a|exp2b|exp2|exp3|exp4|exp5|all}"
        echo ""
        echo "Experiments:"
        echo "  exp2a - PatchCore image_size=448"
        echo "  exp2b - PatchCore image_size=518"
        echo "  exp2  - Both image size experiments"
        echo "  exp3  - PixIO backbone evaluation"
        echo "  exp4  - Few-shot experiments (k=1,5,10,20,50,100,200)"
        echo "  exp5  - SALAD on MVTec LOCO"
        echo "  all   - Run all pending experiments"
        ;;
esac
