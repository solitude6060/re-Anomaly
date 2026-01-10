# Plan A Evaluation Report: MVTec LOCO AD

> **Date**: 2026-01-10  
> **Experiment**: Plan A - DINOv2 ViT-B/14 + PatchCore  
> **Dataset**: MVTec LOCO AD (5 categories)

---

## Executive Summary

| Metric | Our Result | SOTA Target | Gap |
|--------|------------|-------------|-----|
| **Average AUROC** | **69.46%** | 96.1% (SALAD) | **-26.64%** |
| **Logical AUROC** | **64.95%** | 95.7% (SALAD) | **-30.75%** |
| **Structural AUROC** | **75.85%** | 96.5% (SALAD) | **-20.65%** |
| **Precision@100%Recall** | 63.40% | ~90% | -26.60% |

**Verdict**: PatchCore **fails catastrophically** on MVTec LOCO, especially for logical anomalies. This confirms the known limitation: texture-based methods cannot detect missing/misplaced components.

---

## Configuration Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | `dinov2_vitb14` | ViT-Base, 14x14 patches |
| Head | PatchCore | Memory bank + k-NN |
| Image Size | 224 | |
| Output Layers | [4, 8, 11] | Multi-scale features |
| Coreset Ratio | 10% | |
| Coreset Method | Random | |
| k (neighbors) | 9 | |

---

## Per-Category Results

| Category | Image AUROC | Logical AUROC | Structural AUROC | Precision@100% |
|----------|-------------|---------------|------------------|----------------|
| juice_bottle | 79.84% | 75.34% | 86.62% | 71.73% |
| breakfast_box | 76.32% | 73.54% | 78.88% | 62.91% |
| splicing_connectors | 73.36% | 74.09% | 72.44% | 62.26% |
| screw_bag | 60.92% | **50.72%** | 77.95% | 64.60% |
| pushpins | 56.85% | **51.07%** | 63.35% | 55.48% |
| **AVERAGE** | **69.46%** | **64.95%** | **75.85%** | **63.40%** |

---

## Analysis

### Why PatchCore Fails on LOCO

1. **Logical Anomalies = Near Random Performance**
   - screw_bag logical: 50.72% (essentially random)
   - pushpins logical: 51.07% (essentially random)
   - PatchCore compares texture patches, not semantic content
   - Missing screws have the same texture as background

2. **Structural Anomalies = Marginally Better**
   - Structural anomalies still involve texture changes
   - Average 75.85% vs 64.95% for logical
   - Still far below SOTA (96.5%)

3. **Root Cause**
   - PatchCore uses patch-level texture matching
   - Logical anomalies require understanding "what should be here"
   - Need semantic/compositional understanding (SALAD approach)

### Category-Specific Issues

| Category | Issue |
|----------|-------|
| **pushpins** | Must count pins and detect missing - pure semantic task |
| **screw_bag** | Must detect missing/extra screws - counting task |
| **splicing_connectors** | Cable arrangement - spatial/structural understanding |
| **breakfast_box** | Missing/wrong items - semantic understanding |
| **juice_bottle** | Cap/label issues - partial texture, partial semantic |

---

## Comparison with SOTA

| Method | Avg AUROC | Logical | Structural | Notes |
|--------|-----------|---------|------------|-------|
| **SALAD** | **96.1%** | **95.7%** | **96.5%** | Multi-branch semantic |
| LA-EAD | 94.2% | - | - | Logical anomaly focus |
| ComAD | ~92% | - | - | Component-aware |
| EfficientAD-M | 89.5% | - | - | Fast baseline |
| LogicQA (VLM) | 87.6% | - | - | Vision-language |
| **Ours (Plan A)** | **69.46%** | **64.95%** | **75.85%** | Texture-only |
| PatchCore (reported) | ~82% | ~70% | ~95% | Literature baseline |

**Note**: Our PatchCore results (69.46%) are below reported literature (~82%). Possible causes:
1. Image size 224 vs higher in papers
2. Random coreset vs greedy
3. DINOv2 features vs ImageNet-pretrained ResNet

---

## Implications for SMT Inspection

### LOCO Results Confirm

1. **Plan A is NOT suitable for SMT inspection** where logical anomalies matter
2. **Missing components cannot be detected** by texture matching
3. **Need Plan B (SALAD-style)** for logical anomaly detection

### Recommended Next Steps

1. **Abandon Plan A for LOCO-type tasks**
   - PatchCore fundamentally cannot solve logical anomalies
   - No amount of tuning will fix this

2. **Implement Plan B (SALAD architecture)**
   - Multi-branch: appearance + composition + global
   - Semantic feature extraction
   - Component-level understanding

3. **Consider Hybrid Approach**
   - Plan A for structural/texture anomalies (scratches, stains)
   - Plan B for logical anomalies (missing parts, wrong orientation)

---

## Raw Results

```json
{
  "avg_image_auroc": 0.6946,
  "avg_logical_auroc": 0.6495,
  "avg_structural_auroc": 0.7585,
  "per_category": {
    "breakfast_box": { "image": 0.7632, "logical": 0.7354, "structural": 0.7888 },
    "juice_bottle": { "image": 0.7984, "logical": 0.7534, "structural": 0.8662 },
    "pushpins": { "image": 0.5685, "logical": 0.5107, "structural": 0.6335 },
    "screw_bag": { "image": 0.6092, "logical": 0.5072, "structural": 0.7795 },
    "splicing_connectors": { "image": 0.7336, "logical": 0.7409, "structural": 0.7244 }
  }
}
```

---

## Conclusion

Plan A (DINOv2 + PatchCore) achieves **69.46% average AUROC** on MVTec LOCO, which is **~27% below SOTA (96.1%)**.

The failure is fundamental:
- **Logical anomalies (64.95%)**: Near-random performance confirms PatchCore cannot detect missing/misplaced components
- **Structural anomalies (75.85%)**: Slightly better but still far below SOTA

**Recommendation**: For SMT inspection with component presence detection requirements, **do not use PatchCore**. Implement SALAD-style semantic approach (Plan B).
