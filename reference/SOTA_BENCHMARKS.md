# State-of-the-Art Benchmarks for Anomaly Detection

> Last Updated: 2026-01-17

## MVTec AD Dataset

The MVTec Anomaly Detection (MVTec AD) dataset is the most widely used benchmark for industrial anomaly detection. It contains 15 categories with 5,354 images.

### Image-Level AUROC (Detection)

| Method | Backbone | Image AUROC | Pixel AUROC | Year | Notes |
|--------|----------|-------------|-------------|------|-------|
| **Dinomaly** | DINOv2/v3 | **99.6%** | 97.8% | 2025 | CVPR 2025, Multi-class SOTA |
| **PatchCore** | WideResNet-50 | 99.6% | 98.1% | 2021 | Memory bank + kNN |
| **SimpleNet** | WideResNet-50 | 99.6% | - | 2023 | Simple feature adaptation |
| **MambaAD** | Mamba | 99.5% | 97.9% | 2024 | NeurIPS 2024, O(N) efficiency |
| **EfficientAD-M** | PDN | 99.1% | 96.8% | 2023 | Fast, lightweight |
| **EfficientAD-S** | PDN | 98.8% | 96.5% | 2023 | Smaller variant |
| **FastFlow** | WideResNet-50 | 99.4% | 98.0% | 2022 | 2D Normalizing flows |
| **DRAEM** | - | 98.0% | 97.3% | 2021 | Synthetic anomaly augmentation |
| **PaDiM** | WideResNet-50 | 97.9% | 97.5% | 2021 | Gaussian modeling |
| **ReverseDistillation** | WideResNet-50 | 98.5% | 97.8% | 2022 | Knowledge distillation |
| **MSFlow** | ResNet-18 | 99.7% | 98.2% | 2023 | Multi-scale normalizing flows |
| **CFLOW-AD** | WideResNet-50 | 98.3% | 98.6% | 2022 | Conditional normalizing flows |

### Zero-Shot Methods (No Target Training)

| Method | Backbone | Image AUROC | Year | Notes |
|--------|----------|-------------|------|-------|
| **AFR-CLIP** | CLIP ViT-L | ~95% | 2025 | Stateless-to-Stateful rectification |
| **PA-CLIP** | CLIP ViT-L | ~94% | 2025 | Pseudo-anomaly awareness |
| **AA-CLIP** | CLIP ViT-L | ~93% | 2025 | Anomaly-aware text anchors |
| **SuperAD** | DINOv2 | ~93% | 2025 | Training-free, VAND 3.0 Challenge |
| **WinCLIP** | CLIP ViT-L | ~91% | 2023 | Window-based CLIP |

### Our Results

| Method | Backbone | Image AUROC | Status |
|--------|----------|-------------|--------|
| **Ours: PatchCore** | DINOv3-L | **96.51%** | ✅ Complete |
| **Ours: FastFlow** | DINOv3-L | 96.14% | ✅ Complete |
| **Ours: Dinomaly** | DINOv3-L | 🔄 Running (partial results below) | 🔄 In Progress |
| **Ours: PatchCore** | Swin-B | 87.73% | ✅ Complete |

#### Dinomaly Per-Category Results (Running)

| Category | Image AUROC | Training Time |
|----------|-------------|---------------|
| bottle | **100.0%** | 447.5s |
| cable | 95.69% | 512.6s |
| capsule | 96.13% | 496.0s |
| carpet | **99.88%** | 601.2s |
| grid | 🔄 Running | - |
| ... | ... | ... |

### Key Observations

1. **Saturation**: Top methods achieve ~99.6% image AUROC - benchmark is becoming saturated
2. **Dinomaly CVPR 2025**: First multi-class model matching single-class SOTA
3. **PatchCore baseline**: Memory bank approach remains competitive and simple
4. **Speed vs Accuracy**: EfficientAD trades ~0.5% accuracy for 10x faster inference
5. **Multi-scale**: MSFlow achieves highest AUROC with multi-scale approach
6. **Zero-shot gap**: Best zero-shot (~95%) still 4-5% behind supervised methods

---

## MVTec LOCO AD Dataset

The MVTec Logical Constraints Anomaly Detection (MVTec LOCO AD) dataset focuses on logical anomalies (missing/misplaced components) in addition to structural anomalies. It contains 5 categories with 3,644 images.

### Image-Level AUROC by Anomaly Type

| Method | Logical AUROC | Structural AUROC | Average AUROC | Year | Notes |
|--------|---------------|------------------|---------------|------|-------|
| **SALAD** | 95.7% | 96.5% | **96.1%** | 2025 | Semantics-aware, SOTA |
| **LA-EAD** | - | - | 94.2% | 2024 | Logical anomaly focus |
| **ComAD** | - | - | ~92% | 2023 | Component-aware |
| **EfficientAD-M** | - | - | 89.5% | 2023 | Baseline |
| **LogicQA** | - | - | 87.6% | 2025 | VLM-based |
| **GCAD** | - | - | ~85% | 2023 | Global context |
| **PatchCore** | ~70% | ~95% | ~82% | 2021 | Poor on logical |

### SALAD Branch-wise Performance

| Branch | Condition Det. | Logical Det. | Struct. Det. |
|--------|----------------|--------------|--------------|
| Only Appearance | 87.5% | 94.1% | 90.8% |
| Only Composition | 88.1% | 82.8% | 85.4% |
| Only Global | 90.8% | 87.3% | 89.1% |
| **SALAD (Full)** | **96.5%** | **95.7%** | **96.1%** |

### Key Observations

1. **Logical vs Structural gap**: Traditional methods (PatchCore) excel at structural but fail on logical anomalies
2. **SALAD is SOTA**: Achieves 96.1% average AUROC with semantics-aware approach
3. **Multi-branch needed**: No single branch solves all anomaly types
4. **VLM approaches emerging**: LogicQA uses vision-language models but underperforms specialized methods

---

## Comparison: Our Target vs SOTA

### MVTec AD Targets

| Metric | SOTA | Our Target | Notes |
|--------|------|------------|-------|
| Image AUROC | 99.6% (PatchCore/SimpleNet) | ≥99.5% | Must match SOTA |
| Pixel AUROC | 98.2% (MSFlow) | ≥97.5% | For localization |
| Recall@100% | ~95% | **100%** | Zero miss rate priority |
| Inference Time | ~5ms (EfficientAD) | <100ms | Acceptable |

### MVTec LOCO AD Targets

| Metric | SOTA | Our Target | Notes |
|--------|------|------------|-------|
| Average AUROC | 96.1% (SALAD) | ≥95% | Match SALAD |
| Logical AUROC | 95.7% (SALAD) | ≥95% | Critical for missing parts |
| Structural AUROC | 96.5% (SALAD) | ≥96% | Must not regress |

---

## Relevant Papers

### Must-Read (2024-2025)

1. **Dinomaly** (CVPR 2025): "The Less Is More Philosophy in Multi-Class UAD" - [GitHub](https://github.com/guojiajeremy/Dinomaly)
2. **AFR-CLIP** (2025): "Enhancing Zero-Shot IAD with Stateless-to-Stateful Rectification" - arXiv:2503.12910
3. **PA-CLIP** (2025): "Zero-Shot AD through Pseudo-Anomaly Awareness" - arXiv:2503.01292
4. **MambaAD** (NeurIPS 2024): "State Space Models for Multi-class UAD" - arXiv:2404.06564
5. **SuperAD** (CVPR 2025 VAND): "Training-free AD for MVTec AD 2" - arXiv:2505.19750
6. **UniVAD** (2024): "Training-free Unified Model for Few-shot VAD" - arXiv:2412.03342
7. **AnomalyMoE** (2025): "Language-free Generalist Model for Unified VAD" - arXiv:2508.06203

### Classic References

1. **PatchCore** (2021): "Towards Total Recall in Industrial Anomaly Detection" - arXiv:2106.08265
2. **SimpleNet** (2023): "A Simple Network for Image Anomaly Detection and Localization" - CVPR 2023
3. **EfficientAD** (2023): "Accurate Visual Anomaly Detection at Millisecond-Level Latencies" - arXiv:2303.14535
4. **SALAD** (2025): "Semantics-Aware Logical Anomaly Detection" - ICCV 2025
5. **MSFlow** (2023): "Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection" - arXiv:2308.02609

### Additional References

- **FastFlow** (2022): "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows"
- **DRAEM** (2021): "DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection"
- **ComAD** (2023): "Component-aware anomaly detection framework for adjustable and logical industrial visual inspection"
- **MVTec AD 2** (2025): "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection" - arXiv:2503.21622

---

## Backbone Comparison for Anomaly Detection

| Backbone | Params | MVTec AD AUROC | Notes |
|----------|--------|----------------|-------|
| ResNet-18 | 11M | ~97% | Fast, lower accuracy |
| WideResNet-50 | 68M | ~99.5% | Standard choice |
| DINOv2-S | 22M | ~99% | Self-supervised, generalizes well |
| DINOv2-B | 86M | ~99.5% | Better features, slower |
| DINOv2-L | 304M | ~99.5% | Best DINOv2 |
| DINOv2-G | 1.1B | ~99.6% | Giant model, highest quality |
| DINOv3-L | 303M | **96.51%** (ours) | Gram anchoring, needs 518px |
| PixIO-L | 304M | TBD | MAE-based, 8 class tokens |
| CLIP ViT-L/14 | 428M | ~95% (zero-shot) | For zero-shot methods |
| SigLIP-L | 428M | TBD | Better than CLIP for vision |

---

## Few-Shot Anomaly Detection Methods

| Method | k-shot | MVTec AD AUROC | Year | Notes |
|--------|--------|----------------|------|-------|
| **UniVAD** | 0-10 | ~93% | 2024 | Training-free unified model |
| **AnoPLe** | 1-16 | ~92% | 2024 | Bi-directional prompt learning |
| **FADE** | 0-10 | ~91% | 2024 | Large VLM-based |
| **IADGPT** | 1-8 | ~90% | 2025 | LVLM + In-context learning |
| **AnomalyDINO** | 4-16 | ~94% | 2025 | WACV 2025, DINOv2-based |
| **FIND** | 1-16 | ~92% | 2025 | Multimodal (2D+3D) |

### Sample Count Strategy Recommendations

| Samples | Best Method | Expected AUROC | Deployment Time |
|---------|-------------|----------------|-----------------|
| 0 | AFR-CLIP/SuperAD | 85-92% | Instant |
| 1-10 | UniVAD/AnoPLe | 88-94% | Minutes |
| 10-50 | DINOv3+PatchCore | 92-96% | Minutes |
| 50-200 | DINOv3+Dinomaly | 95-99% | Hours |
| 200+ | Full training | 99%+ | Hours-Days |

---

## Notes

- MVTec AD is becoming saturated; focus on MVTec AD 2 and LOCO for differentiation
- Zero miss rate requires threshold tuning at inference time
- SALAD approach (multi-branch semantics) is key for logical anomalies
- Our Plan A (DINOv3 + PatchCore) achieved **96.51%** on MVTec AD
- Our Plan B (SALAD-style) targets MVTec LOCO performance

---

## Priority Actions for re-Anomaly Project

### ✅ Completed
1. **Universal dataset interface** - MVTec AD/LOCO/VisA/custom with FewShotSampler ✅
2. **Implement AFR-CLIP** - Zero-shot head implemented (`src/models/heads/afrclip.py`) ✅
3. **CLIP/SigLIP backbone** - Full implementation with text encoding (`src/models/backbones/clip.py`) ✅
4. **DINOv2-Giant verification** - Already in DINOv2Backbone variants ✅
5. **Fix Dinomaly LSP error** - Hook tracking fixed ✅
6. **MambaAD decoder** - SSM-based alternative to Transformer (`src/models/heads/mambaad.py`) ✅

### 🔄 In Progress
7. **Run Dinomaly on full MVTec AD** - Experiment running (bottle: 100%, cable: 95.69%, capsule: 96.13%, carpet: 99.88%)

### Immediate (High Priority)
8. **Test image size 448/518** - Current 224px limits performance (waiting for GPU)
9. **Test PixIO backbone** - Already implemented, needs evaluation (waiting for GPU)

### Short-term (Medium Priority)
10. **Few-shot experiments** - k=1,5,10,20,50,100,200
11. **Complete SALAD training** - MVTec LOCO logical anomaly detection
