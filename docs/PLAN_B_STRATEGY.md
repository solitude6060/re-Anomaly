# Plan B: SALAD Implementation Strategy

> Created: 2026-01-11
> Target: MVTec LOCO AD - Logical Anomaly Detection
> Goal: 96%+ AUROC (matching SOTA)

---

## Executive Summary

Plan A (PatchCore) achieved only 69.46% on MVTec LOCO due to fundamental inability to detect logical anomalies (missing/misplaced components). Plan B implements SALAD (Semantics-Aware Logical Anomaly Detection) which achieves 96.1% AUROC on MVTec LOCO.

---

## SALAD Architecture Overview

SALAD uses **three branches** to detect both structural and logical anomalies:

```
Input Image
    │
    ├──► [Appearance Branch] ──► Structural anomaly score
    │         └── Standard feature-based detection (like PatchCore/SimpleNet)
    │
    ├──► [Composition Branch] ──► Logical anomaly score (KEY INNOVATION)
    │         └── Models distribution of object composition maps
    │         └── Uses SAM-HQ + DINO for automatic composition extraction
    │
    └──► [Global Branch] ──► Context anomaly score
              └── Global image-level features
              └── Captures overall scene consistency
    │
    └──► [Fusion] ──► Final anomaly score
```

---

## Key Components

### 1. Composition Map Extraction (Critical)

The composition map is the key innovation. It's extracted automatically using:

1. **SAM-HQ**: Segment Anything Model (high quality) for component segmentation
2. **DINO**: Self-supervised features for semantic embedding of each component
3. **Result**: A map where each pixel has a semantic class representing what component it belongs to

```python
# Pseudo-code for composition map extraction
def extract_composition_map(image):
    # 1. Segment image into components using SAM-HQ
    masks = sam_hq.generate_masks(image)
    
    # 2. Extract DINO features for each segment
    composition_map = torch.zeros(H, W, embed_dim)
    for mask in masks:
        segment = image * mask
        features = dino.extract_features(segment)
        composition_map[mask] = features
    
    return composition_map
```

### 2. Three Branches

| Branch | Purpose | Method |
|--------|---------|--------|
| **Appearance** | Structural anomalies | Feature-based (SimpleNet-style) |
| **Composition** | Logical anomalies | Composition map distribution modeling |
| **Global** | Context anomalies | Global feature aggregation |

### 3. Training

- **Discriminative approach**: Train to distinguish normal from synthetic anomalies
- **Loss**: Binary cross-entropy + anomaly localization loss
- **Data augmentation**: Synthetic logical anomalies (remove/swap components)

---

## Implementation Plan

### Phase 1: Dependencies & Infrastructure (Day 1)

- [ ] Install SAM-HQ and DINO dependencies
- [ ] Create composition map extraction pipeline
- [ ] Verify extraction works on MVTec LOCO samples

### Phase 2: SALAD Architecture (Day 2-3)

- [ ] Implement Composition Branch
  - [ ] Composition map encoder
  - [ ] Distribution modeling (normalizing flow or discriminator)
- [ ] Implement Global Branch
  - [ ] Global feature pooling
  - [ ] Anomaly scoring
- [ ] Refactor existing Appearance Branch (use SimpleNet)
- [ ] Implement fusion module

### Phase 3: Training Pipeline (Day 4)

- [ ] Implement synthetic anomaly generation for training
- [ ] Create training loop with multi-branch losses
- [ ] Add validation monitoring

### Phase 4: Evaluation (Day 5)

- [ ] Run evaluation on MVTec LOCO
- [ ] Compare logical vs structural performance
- [ ] Analyze per-category results

---

## Technical Requirements

### Dependencies to Add

```toml
# pyproject.toml additions
segment-anything-hq = "^0.3"  # SAM-HQ
# DINO already available via transformers
```

### GPU Memory Estimate

| Component | Memory |
|-----------|--------|
| SAM-HQ (ViT-H) | ~4GB |
| DINOv2-B | ~2GB |
| SALAD model | ~2GB |
| Batch processing | ~8GB |
| **Total** | ~16GB (fits RTX 4090) |

---

## Expected Results

| Metric | Plan A | Plan B Target | SOTA |
|--------|--------|---------------|------|
| Avg AUROC | 69.46% | **95%+** | 96.1% |
| Logical AUROC | 64.95% | **94%+** | 95.7% |
| Structural AUROC | 75.85% | **95%+** | 96.5% |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| SAM-HQ quality issues | Use higher quality mode, post-process masks |
| Composition map inconsistency | Normalize features, use robust statistics |
| Training instability | Gradual branch warmup, careful LR tuning |
| GPU OOM | Batch size reduction, gradient checkpointing |

---

## Alternative: Quick Win First

Before full SALAD implementation, consider **Plan A optimization** as a quick win:

| Change | Expected Gain | Effort |
|--------|---------------|--------|
| Image size 224→518 | +2-3% on MVTec AD | 1 hour |
| Greedy coreset | +0.5-1% | 2 hours |

This could get MVTec AD to ~98-99% while working on Plan B for LOCO.

---

## References

1. **SALAD Paper**: [arXiv:2509.02101](https://arxiv.org/abs/2509.02101)
2. **Official Code**: [github.com/MaticFuc/SALAD](https://github.com/MaticFuc/SALAD)
3. **SAM-HQ**: [github.com/SysCV/sam-hq](https://github.com/SysCV/sam-hq)
4. **MVTec LOCO**: [mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco)

---

## Decision Point

**Option A**: Implement full SALAD from scratch (5 days, high effort, high reward)

**Option B**: Port official SALAD repo and adapt to our framework (3 days, medium effort)

**Option C**: Quick win Plan A optimization first, then SALAD (1 day + 3 days)

**Recommendation**: Option C - Get quick wins on MVTec AD first, then focus on SALAD for LOCO.
