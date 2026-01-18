# Task Plan: re-Anomaly Project Revival - SOTA Survey & Experiment Plan

## Goal
Survey latest SOTA papers (2024-2025), update experiment plans, implement new backbones/heads, and create a universal dataset interface - targeting 0% critical miss, 50ppm escape rate, <5% overkill for industrial deployment with limited samples (1-200).

## Current Project Status (2026-01-17 snapshot)

### Completed Experiments
| Configuration | Dataset | Image Size | Avg Image AUROC | Notes | Status |
|--------------|---------|------------|------------------|-------|--------|
| DINOv3-L + PatchCore | MVTec AD | 224 | **96.51%** | Baseline | ✅ Complete |
| DINOv3-L + FastFlow | MVTec AD | 224 | 96.14% | Baseline | ✅ Complete |
| Swin-Base + PatchCore | MVTec AD | 224 | 87.73% | Baseline | ✅ Complete |
| DINOv3-L + PatchCore | MVTec LOCO | 224 | 73.53% | PatchCore fails on logical anomalies | ✅ Complete |
| DINOv3-L + Dinomaly | MVTec AD | 224 | **97.34%** | Best so far in-repo; paper reports 99.6% | ✅ Complete |
| DINOv3-L + PatchCore | MVTec AD | 448 | 96.85% | Higher-res; weak categories still exist | ✅ Complete |

### In-Progress Experiments
- None (Dinomaly full + PatchCore 448 finished)

### Key Gaps
1. **MVTec AD**: 97.34% (DINOv3+Dinomaly @224) vs 99.6% SOTA (-2.3%) - need higher image_size (518) + better training/decoder tuning
2. **MVTec LOCO**: 73.53% (PatchCore @224) vs 96.1% SOTA (SALAD) (-22.6%) - must run Plan B SALAD
3. **Few-shot**: No systematic k-shot matrix run yet (k=1,5,10,20,50,100,200)
4. **Zero-shot**: AFR-CLIP head exists but not wired into experiment runner; need CLIP backbone integration into scripts + evaluation run
5. **Survey methods without public code**: PA-CLIP, SuperAD, AnoPLe (cannot implement faithfully without re-derivation)

## Phases

### Phase 1: SOTA Survey & Documentation ✅
- [x] Read project status and existing experiment results
- [x] Survey AFR-CLIP paper
- [x] Survey PA-CLIP paper
- [x] Survey Dinomaly (CVPR 2025) paper
- [x] Survey SuperAD (VAND 3.0) paper
- [x] Survey MambaAD / AnomalyMoE papers
- [x] Survey few-shot anomaly detection methods
- [x] Document SOTA findings in notes.md
- [x] Update reference/SOTA_BENCHMARKS.md

### Phase 2: Architecture Analysis & New Backbones
- [x] Analyze which backbones are already implemented (DINOv2, DINOv3, Swin, PixIO, CLIP/SigLIP)
- [x] Implement/verify SigLIP backbone (in `src/models/backbones/clip.py`, via OpenCLIP)
- [ ] Verify DINOv2-giant backbone config exists and run a smoke eval
- [ ] Test PixIO backbone (already in project; license constraints apply)
- [ ] Compare backbone performance for few-shot scenarios (k-shot matrix)

### Phase 3: New Method Implementation
- [x] Fix Dinomaly LSP error (_has_hook issue)
- [x] Run Dinomaly full evaluation (completed; results saved to `results/dinomaly_full_v3/`)
- [x] Implement AFR-CLIP for zero-shot detection (head exists in `src/models/heads/afrclip.py`)
- [x] Implement MambaAD decoder (head exists in `src/models/heads/mambaad.py`)
- [ ] Wire CLIP-based heads into `scripts/run_experiment_matrix.py` and run zero-shot eval
- [ ] Implement few-shot experiment matrix runner (or fix existing exp4 script, which references `src.data.mvtec` classes that may not exist anymore)

### Phase 4: Universal Dataset Interface ✅
- [x] Design universal dataset API supporting:
  - MVTec AD, MVTec LOCO, VisA
  - Custom industrial datasets
- [x] Implement DatasetFactory with TDD
- [x] Add FewShotSampler utilities
- [x] Write tests (15 tests passing)

### Phase 5: Experiment Plan & Strategy for Limited Samples
- [x] Create sample-count strategy document (1-200 samples) - in notes.md
- [ ] Design experiment matrix for few-shot evaluation
- [ ] Run experiments with image_size=448/518
- [ ] Test PixIO backbone

## Implementation Completed Today (2026-01-17)

### Universal Dataset Interface
New files:
- `src/data/mvtec.py` - Extended with:
  - `AnomalyDataset` protocol
  - `BaseAnomalyDataset` base class
  - `VisADataset` for VisA benchmark
  - `CustomDataset` for industrial use
  - `DatasetFactory` for unified creation
  - `FewShotSampler` for k-shot learning
- `tests/test_datasets.py` - 15 tests all passing

Usage:
```python
from src.data import DatasetFactory, FewShotSampler

# Create dataset
dataset = DatasetFactory.create("mvtec_ad", root="data/mvtec_ad", category="bottle", split="train")

# Few-shot sampling
sampler = FewShotSampler(dataset, k=10, seed=42)
subset = sampler.get_subset()  # 10-shot subset

# Multi-category loading
datasets = DatasetFactory.create_multi("mvtec_ad", root="data/mvtec_ad", categories=["bottle", "cable"], split="train")
```

## Decisions Made
- [2026-01-17] Using planning-with-files for structured progress tracking
- [2026-01-17] Prioritizing SOTA survey before implementation
- [2026-01-17] Universal Dataset Interface completed with TDD

## Next Steps (Priority Order)
1. Wait for Dinomaly full evaluation to complete
2. Run PatchCore with image_size=448/518
3. Test PixIO backbone
4. Implement AFR-CLIP for zero-shot baseline

## Errors Encountered
- Fixed: dinomaly.py LSP error - replaced `de._has_hook = True` with proper set-based tracking

---

# Task Plan: Project File Inventory

## Goal
Provide a comprehensive file list with brief purpose summaries for the repository.

## Phases
- [ ] Phase 1: Plan and setup
- [ ] Phase 2: Research/gather information
- [ ] Phase 3: Execute/build
- [ ] Phase 4: Review and deliver

## Key Questions
1. What files and directories exist at the project root and within key subdirectories?
2. What is the apparent purpose of each major file or folder based on naming and content?

## Decisions Made
- Use repository traversal with categorized summaries for top-level and key subdirectories.

## Errors Encountered
- None

## Status
**Currently in Phase 2** - Gathering repository structure and content
