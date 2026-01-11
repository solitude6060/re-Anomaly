# Experiment Results

> Last Updated: 2026-01-11

## Overview

This document consolidates all experimental results for the re-Anomaly project.

| Experiment | Dataset | Avg AUROC | SOTA | Gap | Status |
|------------|---------|-----------|------|-----|--------|
| Plan A | MVTec AD | 95.67% | 99.6% | -3.93% | ✅ Complete |
| Plan A | MVTec LOCO | 69.46% | 96.1% | -26.64% | ✅ Complete |

---

## Plan A: DINOv2 + PatchCore

### Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | DINOv2 ViT-B/14 (`facebook/dinov2-base`) |
| Head | PatchCore (Memory Bank + k-NN) |
| Image Size | 224 |
| Output Layers | [4, 8, 11] |
| Coreset Ratio | 10% |
| Coreset Method | Random |
| k (neighbors) | 9 |
| Hardware | RTX 4090 (24GB) |

---

### MVTec AD Results

**Summary**: 95.67% Image AUROC (vs 99.6% SOTA)

| Category | Image AUROC | Precision@100%Recall | Time (s) |
|----------|-------------|----------------------|----------|
| bottle | 99.92% | 98.44% | 18.5 |
| cable | 91.04% | 64.79% | 34.8 |
| capsule | 89.15% | 83.21% | 30.8 |
| carpet | **100.00%** | 100.00% | 34.8 |
| grid | **100.00%** | 100.00% | 19.8 |
| hazelnut | 99.86% | 98.59% | 44.2 |
| leather | **100.00%** | 100.00% | 31.1 |
| metal_nut | 99.80% | 98.94% | 23.1 |
| pill | 91.68% | 88.13% | 39.1 |
| screw | 79.48% | 76.28% | 42.3 |
| tile | **100.00%** | 100.00% | 26.7 |
| toothbrush | 95.83% | 90.91% | 5.0 |
| transistor | 90.75% | 42.55% | 24.8 |
| wood | 97.72% | 89.55% | 23.9 |
| zipper | 99.76% | 96.75% | 31.0 |
| **AVERAGE** | **95.67%** | **88.54%** | - |

**Analysis**:
- 4 categories achieve 100%: carpet, grid, leather, tile
- Weak categories: screw (79.48%), capsule (89.15%), transistor (90.75%)
- Gap to SOTA mainly due to small image size (224 vs 518)

---

### MVTec LOCO Results

**Summary**: 69.46% Image AUROC (vs 96.1% SOTA)

| Category | Image AUROC | Logical AUROC | Structural AUROC | Precision@100% |
|----------|-------------|---------------|------------------|----------------|
| breakfast_box | 76.32% | 73.54% | 78.88% | 62.91% |
| juice_bottle | 79.84% | 75.34% | 86.62% | 71.73% |
| pushpins | 56.85% | 51.07% | 63.35% | 55.48% |
| screw_bag | 60.92% | 50.72% | 77.95% | 64.60% |
| splicing_connectors | 73.36% | 74.09% | 72.44% | 62.26% |
| **AVERAGE** | **69.46%** | **64.95%** | **75.85%** | **63.40%** |

**Analysis**:
- Logical anomalies: 64.95% (near-random, fundamental limitation)
- Structural anomalies: 75.85% (better but still far below SOTA)
- PatchCore **cannot detect missing/misplaced components**
- pushpins and screw_bag logical AUROC ~50% = random guessing

---

## SOTA Benchmarks

### MVTec AD

| Method | Backbone | Image AUROC | Year |
|--------|----------|-------------|------|
| MSFlow | ResNet-18 | 99.7% | 2023 |
| PatchCore | WideResNet-50 | 99.6% | 2021 |
| SimpleNet | WideResNet-50 | 99.6% | 2023 |
| FastFlow | WideResNet-50 | 99.4% | 2022 |
| EfficientAD-M | PDN | 99.1% | 2023 |
| **Ours (Plan A)** | DINOv2-B | **95.67%** | - |

### MVTec LOCO

| Method | Avg AUROC | Logical | Structural | Year |
|--------|-----------|---------|------------|------|
| SALAD | **96.1%** | 95.7% | 96.5% | 2025 |
| LA-EAD | 94.2% | - | - | 2024 |
| ComAD | ~92% | - | - | 2023 |
| EfficientAD-M | 89.5% | - | - | 2023 |
| **Ours (Plan A)** | **69.46%** | 64.95% | 75.85% | - |
| PatchCore (reported) | ~82% | ~70% | ~95% | 2021 |

---

## Improvement Roadmap

### MVTec AD (Plan A Optimization)

| Change | Expected Gain | Priority |
|--------|---------------|----------|
| Increase image size to 518 | +2-3% AUROC | **P0** |
| Switch to greedy coreset | +0.5-1% AUROC | P1 |
| Use DINOv2-Large | +0.5-1% AUROC | P1 |
| Increase coreset ratio to 25% | +0.3-0.5% AUROC | P2 |

### MVTec LOCO (Plan B Required)

| Action | Notes |
|--------|-------|
| Abandon Plan A for LOCO | PatchCore fundamentally cannot solve logical anomalies |
| Implement Plan B (SALAD) | Multi-branch: appearance + composition + global |
| Hybrid approach | Plan A for texture, Plan B for logical |

---

## Conclusions

1. **Plan A (DINOv2 + PatchCore)** is suitable for **texture-based anomalies** (MVTec AD) with optimization potential to reach 99%+

2. **Plan A fails on logical anomalies** (MVTec LOCO) - fundamental limitation of patch-matching approach

3. **For SMT inspection**:
   - Use Plan A for scratches, stains, surface defects
   - Need Plan B (SALAD-style) for missing/misplaced components
   - Consider hybrid architecture for production

---

## File References

| File | Description |
|------|-------------|
| `results/plan_a_mvtec_ad/results.json` | Raw MVTec AD results |
| `results/plan_a_mvtec_ad/COMPARISON_REPORT.md` | Detailed MVTec AD analysis |
| `results/plan_a_mvtec_loco/results.json` | Raw MVTec LOCO results |
| `results/plan_a_mvtec_loco/COMPARISON_REPORT.md` | Detailed MVTec LOCO analysis |
| `reference/SOTA_BENCHMARKS.md` | SOTA reference data |
| `scripts/run_plan_a.py` | MVTec AD evaluation script |
| `scripts/run_plan_a_loco.py` | MVTec LOCO evaluation script |
