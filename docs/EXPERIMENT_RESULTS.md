# MVTec Anomaly Detection Experiment Results

## Executive Summary

This report summarizes experiments on MVTec AD and MVTec LOCO datasets using various backbone and head combinations. The goal is to achieve state-of-the-art anomaly detection using only OK/NG image-level labels (no pixel-level segmentation masks).

### Key Results

| Dataset | Best Configuration | Avg AUROC | Status |
|---------|-------------------|-----------|--------|
| **MVTec AD** | DINOv3 + PatchCore | **96.51%** | ✅ Complete |
| **MVTec LOCO** | DINOv3 + PatchCore | 73.53% | ✅ Complete |

---

## MVTec AD Results

### Summary Table

| Backbone | Head | Avg AUROC | Avg Prec@100R | Best Category | Worst Category |
|----------|------|-----------|---------------|---------------|----------------|
| dinov3_vitl16 | patchcore | **96.51%** | 90.13% | bottle (100%) | screw (83.8%) |
| dinov3_vitl16 | fastflow | 🔄 Running | - | - | - |
| dinov3_vitl16 | simplenet | ⏳ Pending | - | - | - |
| dinov3_vitl16 | rectflow | ⏳ Pending | - | - | - |

### Per-Category Results (DINOv3 + PatchCore)

| Category | AUROC | Precision@100R | Train Samples | Test Samples |
|----------|-------|----------------|---------------|--------------|
| 🥇 bottle | 100.00% | 100.00% | 209 | 83 |
| 🥇 hazelnut | 100.00% | 100.00% | 391 | 110 |
| 🥇 tile | 100.00% | 100.00% | 230 | 117 |
| leather | 99.97% | 98.92% | 245 | 124 |
| zipper | 99.08% | 97.54% | 240 | 151 |
| carpet | 99.00% | 90.82% | 280 | 117 |
| metal_nut | 98.83% | 90.29% | 220 | 115 |
| pill | 98.28% | 94.00% | 267 | 167 |
| grid | 97.24% | 95.00% | 264 | 78 |
| capsule | 94.81% | 90.83% | 219 | 132 |
| transistor | 95.29% | 80.00% | 213 | 100 |
| toothbrush | 94.72% | 85.71% | 60 | 42 |
| wood | 93.51% | 88.24% | 247 | 79 |
| cable | 93.05% | 61.74% | 224 | 150 |
| ⚠️ screw | 83.77% | 77.78% | 320 | 160 |
| **Average** | **96.51%** | **90.13%** | - | - |

### Category Analysis

**Perfect Detection (100% AUROC)**:
- bottle, hazelnut, tile - Clear texture/shape anomalies, good contrast

**Near-Perfect (>98% AUROC)**:
- leather, zipper, carpet, metal_nut, pill - High-quality features, distinct defect patterns

**Challenging Categories (<95% AUROC)**:
- **screw (83.77%)**: Very subtle thread defects, low contrast differences
- **cable (93.05%)**: Multiple cable types, high intra-class variation
- **wood (93.51%)**: Natural texture variation, hard to distinguish from defects

---

## MVTec LOCO Results

MVTec LOCO is a more challenging dataset with **logical anomalies** (wrong arrangements, missing parts) in addition to structural defects.

### Summary Table

| Backbone | Head | Avg AUROC | Logical AUROC | Structural AUROC |
|----------|------|-----------|---------------|------------------|
| dinov3_vitl16 | patchcore | 73.53% | 69.57% | 79.17% |

### Per-Category Results

| Category | Image AUROC | Logical AUROC | Structural AUROC | Prec@100R |
|----------|-------------|---------------|------------------|-----------|
| juice_bottle | 86.82% | 83.51% | 91.83% | 72.17% |
| breakfast_box | 82.38% | 79.65% | 84.90% | 62.91% |
| splicing_connectors | 74.82% | 76.17% | 73.10% | 62.46% |
| screw_bag | 63.16% | 53.33% | 79.57% | 64.22% |
| pushpins | 60.47% | 55.16% | 66.44% | 56.21% |
| **Average** | **73.53%** | **69.57%** | **79.17%** | **63.59%** |

### Observations

1. **Structural > Logical**: PatchCore performs better on structural anomalies (79.17%) than logical anomalies (69.57%)
2. **Challenging Categories**: pushpins and screw_bag have high intra-class variation
3. **Opportunity**: Logical anomaly detection requires specialized approaches (e.g., SALAD head)

---

## Unified Model Results (Preliminary)

The unified model combines:
- Multi-scale feature aggregation
- Dual-head architecture (FastFlow + Discriminator)
- Memory bank for k-NN scoring
- SDG (Synthetic Defect Generation) augmentation

### Configuration

```python
{
    "backbone": "dinov3_vitl16",
    "projection_dim": 256,
    "flow_blocks": 8,
    "flow_weight": 0.5,
    "disc_weight": 0.3,
    "memory_weight": 0.2,
    "use_sdg": True
}
```

### Quick Test Results (5 epochs)

| Category | AUROC | Status |
|----------|-------|--------|
| bottle | 76.98% | Test only |

**Note**: Full training with 100 epochs pending GPU availability.

---

## Architecture Components

### Backbones Tested

| Backbone | Params | Features | Status |
|----------|--------|----------|--------|
| DINOv3-ViT-L/16 | 307M | 1024-dim, 14×14 patches | ✅ Primary |
| DINOv2-ViT-L/14 | 307M | 1024-dim, 16×16 patches | ✅ Available |
| Swin-L | 197M | Hierarchical features | ⏳ Pending tests |

### Detection Heads

| Head | Type | Trainable | Best Use Case |
|------|------|-----------|---------------|
| PatchCore | Memory Bank | No | Standard textures, fast deployment |
| FastFlow | Normalizing Flow | Yes | Complex distributions |
| SimpleNet | Discriminative | Yes | Lightweight inference |
| RectFlow | Rectified Flow | Yes | Multi-scale anomalies |

### SDG Defect Types

The Synthetic Defect Generation module supports:
- `cutout` - Random rectangular regions
- `noise` - Gaussian/salt-pepper noise
- `blur` - Local blur patches
- `color_shift` - HSV color perturbations
- `texture` - Texture overlays
- `scratch` - Line-based scratches
- `stain` - Blob-like stains

---

## Experimental Setup

### Hardware
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Training: 100 epochs per configuration
- Batch size: 32

### Metrics
- **AUROC**: Area Under ROC Curve (image-level)
- **Precision@100%Recall**: Precision when all anomalies are detected (operational metric)

### Data Splits
- MVTec AD: 15 categories, 3,629 training images, 1,725 test images
- MVTec LOCO: 5 categories, 1,778 training images, 1,568 test images

---

## Conclusions

1. **DINOv3 + PatchCore** achieves **96.51% AUROC** on MVTec AD, competitive with published SOTA
2. **Perfect detection** on bottle, hazelnut, tile categories
3. **screw** category (83.77%) requires targeted optimization (small defects, low contrast)
4. **MVTec LOCO logical anomalies** need specialized approaches beyond feature matching
5. **Unified model** shows promise but requires full training to evaluate

## Next Steps

1. Complete DINOv3 + FastFlow/SimpleNet/RectFlow experiments
2. Run Swin backbone experiments for comparison
3. Train unified model for full 100 epochs
4. Investigate screw category optimization
5. Implement SALAD head for logical anomaly detection

---

*Report generated: 2026-01-13*
*Experiment framework: re-Anomaly (re-fastflow)*
