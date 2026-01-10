# re-Anomaly Project Status

> Last Updated: 2026-01-10

## Overview

**Project**: re-Anomaly - Next-generation SMT Anomaly Detection System  
**Repository**: https://github.com/solitude6060/re-Anomaly  
**Branch**: main  
**Python**: 3.10+  
**Package Manager**: uv

## Current Phase

✅ **Phase 1: Project Initialization** (Complete)

## Experimental Plans

| Plan | Backbone | Detection Head | Status | Notes |
|------|----------|----------------|--------|-------|
| **A: PatchCore** | DINOv2/v3 | PatchCore (Memory Bank + kNN) | 🟡 Ready | Stable baseline |
| **B: HLSIS** | DINOv3 + DAPT | SimpleNet + SALAD dual-stream | 🟡 Ready | Logic anomalies focus |
| **C: PixIO** | PixIO-H | Linear/Lightweight Head | 🟡 Ready | Micro defects, few-shot |
| **D: MSFlow** | PixIO/DINOv3 | MSFlow + HGAD | 🟡 Ready | Multi-scale, unified |

## Task Progress

### Initialization (Phase 1)

| Task | Status | Notes |
|------|--------|-------|
| Git repository setup | ✅ Done | Connected to GitHub |
| uv environment setup | ✅ Done | Python 3.10 |
| Directory structure | ✅ Done | All directories created |
| .gitignore | ✅ Done | Comprehensive Python/ML patterns |
| pyproject.toml | ✅ Done | All dependencies defined |
| STATUS.md | ✅ Done | This file |
| README.md (EN) | ✅ Done | |
| README_zh-TW.md | ✅ Done | |

### Documentation (Phase 1)

| Document | English | 繁體中文 |
|----------|---------|----------|
| PROJECT_SPEC.md | ✅ Done | ✅ Done |
| PRD.md | ✅ Done | ✅ Done |
| DEVELOPMENT_PLAN.md | ✅ Done | ✅ Done |
| FEATURE_LIST.md | ✅ Done | N/A |
| BACKBONE_RESEARCH.md | ✅ Done | N/A |
| FLOW_MODEL_RESEARCH.md | ✅ Done | N/A |
| CONFIG_GUIDE.md | ✅ Done | N/A |

### Configuration (Phase 1)

| Config | Status | Files |
|--------|--------|-------|
| Main config.yaml | ✅ Done | configs/config.yaml |
| Backbone configs | ✅ Done | dinov2.yaml, dinov3.yaml, pixio.yaml |
| Head configs | ✅ Done | patchcore.yaml, msflow.yaml, simplenet.yaml, salad.yaml, linear.yaml, fastflow.yaml |
| Augmentation configs | ✅ Done | train.yaml, eval.yaml |
| Dataset configs | ✅ Done | mvtec.yaml, smt.yaml |
| Experiment configs | ✅ Done | plan_a.yaml, plan_b.yaml, plan_c.yaml, plan_d.yaml |
| Pretrain configs | ✅ Done | mae.yaml, dino_dapt.yaml |
| Deploy configs | ✅ Done | triton.yaml |
| Training/Evaluation | ✅ Done | default.yaml |

### Source Code (Phase 1)

| Module | Status | Key Files |
|--------|--------|-----------|
| Backbone loaders | ✅ Done | dinov2.py, dinov3.py, pixio.py |
| Detection heads | ✅ Done | patchcore.py, msflow.py, simplenet.py, salad.py, linear.py, fastflow.py |
| Pretraining | ✅ Done | mae_pretrain.py, dino_dapt.py |
| Data pipeline | 🟡 Pending | dataset.py, transforms.py |
| Training | 🟡 Pending | trainer.py |
| Evaluation | 🟡 Pending | metrics.py, evaluator.py |
| Export | 🟡 Pending | onnx_export.py |
| Deploy | 🟡 Pending | triton_client.py |

### Scripts (Phase 1)

| Script | Status | Purpose |
|--------|--------|---------|
| train.py | ✅ Done | Main training entry (Hydra) |
| evaluate.py | ✅ Done | Evaluation entry |
| pretrain.py | ✅ Done | DAPT/MAE pretraining |
| export.py | ✅ Done | ONNX export |

## Key Research Findings

### Backbones

| Model | Source | Key Innovation | Status |
|-------|--------|----------------|--------|
| **PixIO** | Meta, Dec 2025 | Enhanced MAE, 8 class tokens, deeper decoder, 20B images | Available via transformers |
| **DINOv3** | Meta, Aug 2025 | Gram Anchoring, 7B params, 17B images | Available via transformers |
| **DINOv2** | Meta, 2023 | Self-supervised ViT baseline | Available via transformers |

### Flow Models

| Model | Key Innovation | Use Case |
|-------|----------------|----------|
| **MSFlow** | Multi-scale parallel normalizing flows | Multi-resolution anomaly detection |
| **HGAD** | Hierarchical Gaussian Mixture | Unified multi-class detection |
| **FastFlow** | 2D normalizing flows on feature maps | Fast inference |

## Requirements Summary

| Requirement | Target | Priority |
|-------------|--------|----------|
| Miss Rate | 0% | P0 |
| Stability | High | P1 |
| Over-kill Rate | Low | P2 |
| Inference Speed | <100ms | P3 |
| Backbone Load | ≤500ms | P3 |
| Downstream Load | ≤100ms | P3 |

## Hardware Targets

- **Training**: RTX 4090 (24GB VRAM)
- **Inference**: RTX 4090, ONNX Runtime, Triton Server
- **Data**: 3M unlabeled images for pretraining, 20-30 SMT part types

## Quick Start

```bash
# Install dependencies
uv sync

# Run training with Plan A
python scripts/train.py experiment=plan_a dataset=smt

# Run evaluation
python scripts/evaluate.py checkpoint=outputs/final.pth

# Run MAE pretraining
python scripts/pretrain.py

# Export to ONNX
python scripts/export.py --config configs/config.yaml --checkpoint outputs/final.pth
```

## Legend

- ✅ Done
- 🔵 In Progress
- 🟡 Pending/Ready
- ❌ Blocked
- ⏸️ On Hold

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-10 | Initial project setup: git, uv, directory structure, pyproject.toml |
| 2026-01-10 | Documentation complete: all English and Chinese docs |
| 2026-01-10 | Hydra configs complete: all YAML configuration files |
| 2026-01-10 | Core models complete: backbones, heads, pretraining |
| 2026-01-10 | Entry scripts complete: train.py, evaluate.py, pretrain.py, export.py |
