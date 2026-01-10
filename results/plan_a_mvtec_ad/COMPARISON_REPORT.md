# Plan A Evaluation Report: DINOv2 + PatchCore

> **Date**: 2026-01-10  
> **Experiment**: Plan A - DINOv2 ViT-B/14 + PatchCore  
> **Dataset**: MVTec AD (15 categories)

---

## Executive Summary

| Metric | Our Result | SOTA Target | Gap |
|--------|------------|-------------|-----|
| **Image AUROC** | **95.67%** | 99.6% (PatchCore/SimpleNet) | **-3.93%** |
| **Precision@100%Recall** | 88.54% | ~95% | -6.46% |

**Verdict**: Current configuration underperforms SOTA by ~4%. Primary bottleneck is small image size (224 vs 518 native DINOv2 resolution).

---

## Configuration Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | `dinov2_vitb14` | ViT-Base, 14x14 patches |
| Head | PatchCore | Memory bank + k-NN |
| Image Size | **224** | Native DINOv2 is 518 |
| Output Layers | [4, 8, 11] | Multi-scale features |
| Coreset Ratio | 10% | Subsampling for memory |
| Coreset Method | Random | Faster but less optimal |
| k (neighbors) | 9 | For anomaly scoring |

---

## Per-Category Results

### Tier 1: Perfect (100% AUROC)

| Category | Image AUROC | Precision@100%Recall | Status |
|----------|-------------|----------------------|--------|
| carpet | 100.00% | 100.00% | SOTA |
| grid | 100.00% | 100.00% | SOTA |
| leather | 100.00% | 100.00% | SOTA |
| tile | 100.00% | 100.00% | SOTA |

### Tier 2: Excellent (>99% AUROC)

| Category | Image AUROC | Precision@100%Recall | Gap to SOTA |
|----------|-------------|----------------------|-------------|
| bottle | 99.92% | 98.44% | ~0% |
| hazelnut | 99.86% | 98.59% | ~0% |
| metal_nut | 99.80% | 98.94% | ~0% |
| zipper | 99.76% | 96.75% | ~0% |

### Tier 3: Good (>95% AUROC)

| Category | Image AUROC | Precision@100%Recall | Gap to SOTA |
|----------|-------------|----------------------|-------------|
| wood | 97.72% | 89.55% | ~2% |
| toothbrush | 95.83% | 90.91% | ~4% |

### Tier 4: Needs Improvement (<95% AUROC)

| Category | Image AUROC | Precision@100%Recall | Gap to SOTA | Issue |
|----------|-------------|----------------------|-------------|-------|
| pill | 91.68% | 88.13% | **~8%** | Small defects |
| cable | 91.04% | 64.79% | **~9%** | Complex texture |
| transistor | 90.75% | 42.55% | **~9%** | Orientation variance |
| capsule | 89.15% | 83.21% | **~10%** | Subtle defects |
| **screw** | **79.48%** | **76.28%** | **~20%** | **Critical gap** |

---

## Gap Analysis

### Why We're Below SOTA

1. **Image Size (PRIMARY)**
   - Current: 224x224
   - DINOv2 native: 518x518
   - Impact: Loses fine-grained details critical for small defects
   - Affected: screw, capsule, pill (small anomalies)

2. **Coreset Sampling**
   - Current: Random 10%
   - Optimal: Greedy coreset selection
   - Impact: Random may miss representative patches
   - Estimated gain: +0.5-1% AUROC

3. **Model Size**
   - Current: ViT-Base (86M params)
   - Alternative: ViT-Large (304M params)
   - Impact: Larger model = richer features
   - Estimated gain: +0.5-1% AUROC

4. **Category-Specific Issues**
   - **screw**: Threaded texture confuses patch matching; requires rotation-invariant features
   - **transistor**: High variance in normal appearance; low anomaly-to-normal ratio in test set
   - **cable**: Complex, varied textures; many defect subtypes

---

## Precision@100%Recall Analysis

For **zero miss rate** manufacturing inspection, Precision@100%Recall is critical:

| Category | Precision@100%Recall | False Positive Rate | Overkill Severity |
|----------|----------------------|---------------------|-------------------|
| carpet | 100.00% | 0% | None |
| grid | 100.00% | 0% | None |
| leather | 100.00% | 0% | None |
| tile | 100.00% | 0% | None |
| metal_nut | 98.94% | 1.06% | Low |
| hazelnut | 98.59% | 1.41% | Low |
| bottle | 98.44% | 1.56% | Low |
| zipper | 96.75% | 3.25% | Low |
| toothbrush | 90.91% | 9.09% | Medium |
| wood | 89.55% | 10.45% | Medium |
| pill | 88.13% | 11.87% | Medium |
| capsule | 83.21% | 16.79% | High |
| screw | 76.28% | 23.72% | High |
| cable | 64.79% | 35.21% | **Very High** |
| transistor | 42.55% | 57.45% | **Unacceptable** |

**Critical Issue**: transistor and cable have unacceptable false positive rates for production use.

---

## Improvement Roadmap

### Quick Wins (Low Effort, High Impact)

| Change | Expected Gain | Effort | Priority |
|--------|---------------|--------|----------|
| Increase image size to 518 | +2-3% AUROC | Low | **P0** |
| Switch to greedy coreset | +0.5-1% AUROC | Low | P1 |
| Increase coreset ratio to 25% | +0.3-0.5% AUROC | Low | P1 |

### Medium Effort

| Change | Expected Gain | Effort | Priority |
|--------|---------------|--------|----------|
| Use DINOv2-Large backbone | +0.5-1% AUROC | Medium | P1 |
| Add data augmentation (rotation) | +1-2% for screw | Medium | P1 |
| Tune output layers | +0.2-0.5% AUROC | Medium | P2 |

### High Effort (Future)

| Change | Expected Gain | Effort | Priority |
|--------|---------------|--------|----------|
| Multi-scale PatchCore | +1-2% AUROC | High | P2 |
| Category-specific thresholds | Better precision | Medium | P2 |
| Ensemble multiple backbones | +0.5-1% AUROC | High | P3 |

---

## Recommended Next Steps

### Immediate Actions

1. **Re-run with image_size=518**
   ```python
   # In run_plan_a.py, change:
   image_size = 518  # Was 224
   ```
   Expected result: ~98-99% AUROC

2. **Switch to greedy coreset**
   ```python
   coreset_method = "greedy"  # Was "random"
   ```
   Note: Slower training but better quality

3. **Increase coreset ratio**
   ```python
   coreset_sampling_ratio = 0.25  # Was 0.1
   ```

### After Quick Wins

If still below 99.5% AUROC:
- Try DINOv2-Large (`dinov2_vitl14`)
- Add test-time augmentation for difficult categories
- Consider category-specific models for screw/transistor

---

## Comparison with SOTA Methods

| Method | Backbone | Image AUROC | Our Gap |
|--------|----------|-------------|---------|
| MSFlow | ResNet-18 | 99.7% | -4.03% |
| PatchCore | WR-50 | 99.6% | -3.93% |
| SimpleNet | WR-50 | 99.6% | -3.93% |
| FastFlow | WR-50 | 99.4% | -3.73% |
| EfficientAD-M | PDN | 99.1% | -3.43% |
| **Ours (Plan A)** | DINOv2-B | **95.67%** | - |
| PaDiM | WR-50 | 97.9% | -2.23% |

**Note**: Most SOTA methods use larger image sizes (typically 256-512) and WideResNet-50 which is optimized for this task.

---

## Conclusion

Plan A (DINOv2 + PatchCore) achieves **95.67% Image AUROC**, which is a reasonable baseline but **~4% below SOTA**. The gap is primarily due to:

1. **Small image size** (224 vs 518) - accounts for ~2-3% gap
2. **Random coreset sampling** - accounts for ~0.5-1% gap
3. **Category-specific weaknesses** (screw, transistor, cable)

**Recommended Path to SOTA**:
1. Increase image size to 518 (priority)
2. Switch to greedy coreset
3. If needed, upgrade to DINOv2-Large

With these changes, we expect to achieve **99%+ AUROC**, competitive with SOTA.

---

## Appendix: Raw Results

```json
{
  "avg_image_auroc": 0.9567,
  "avg_precision_at_100_recall": 0.8854,
  "per_category": {
    "bottle": 0.9992,
    "cable": 0.9104,
    "capsule": 0.8915,
    "carpet": 1.0000,
    "grid": 1.0000,
    "hazelnut": 0.9986,
    "leather": 1.0000,
    "metal_nut": 0.9980,
    "pill": 0.9168,
    "screw": 0.7948,
    "tile": 1.0000,
    "toothbrush": 0.9583,
    "transistor": 0.9075,
    "wood": 0.9772,
    "zipper": 0.9976
  }
}
```
