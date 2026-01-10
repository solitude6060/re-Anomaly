# State-of-the-Art Benchmarks for Anomaly Detection

> Last Updated: 2026-01-10

## MVTec AD Dataset

The MVTec Anomaly Detection (MVTec AD) dataset is the most widely used benchmark for industrial anomaly detection. It contains 15 categories with 5,354 images.

### Image-Level AUROC (Detection)

| Method | Backbone | Image AUROC | Pixel AUROC | Year | Notes |
|--------|----------|-------------|-------------|------|-------|
| **PatchCore** | WideResNet-50 | 99.6% | 98.1% | 2021 | Memory bank + kNN |
| **SimpleNet** | WideResNet-50 | 99.6% | - | 2023 | Simple feature adaptation |
| **EfficientAD-M** | PDN | 99.1% | 96.8% | 2023 | Fast, lightweight |
| **EfficientAD-S** | PDN | 98.8% | 96.5% | 2023 | Smaller variant |
| **FastFlow** | WideResNet-50 | 99.4% | 98.0% | 2022 | 2D Normalizing flows |
| **DRAEM** | - | 98.0% | 97.3% | 2021 | Synthetic anomaly augmentation |
| **PaDiM** | WideResNet-50 | 97.9% | 97.5% | 2021 | Gaussian modeling |
| **ReverseDistillation** | WideResNet-50 | 98.5% | 97.8% | 2022 | Knowledge distillation |
| **MSFlow** | ResNet-18 | 99.7% | 98.2% | 2023 | Multi-scale normalizing flows |
| **CFLOW-AD** | WideResNet-50 | 98.3% | 98.6% | 2022 | Conditional normalizing flows |

### Key Observations

1. **Saturation**: Top methods achieve ~99.6% image AUROC - benchmark is becoming saturated
2. **PatchCore baseline**: Memory bank approach remains competitive and simple
3. **Speed vs Accuracy**: EfficientAD trades ~0.5% accuracy for 10x faster inference
4. **Multi-scale**: MSFlow achieves highest AUROC with multi-scale approach

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

### Must-Read

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
| DINOv3 | 7B (max) | TBD | Latest, needs evaluation |
| PixIO | - | TBD | Enhanced MAE, needs evaluation |

---

## Notes

- MVTec AD is becoming saturated; focus on MVTec AD 2 and LOCO for differentiation
- Zero miss rate requires threshold tuning at inference time
- SALAD approach (multi-branch semantics) is key for logical anomalies
- Our Plan A (DINOv2 + PatchCore) should match ~99.5% on MVTec AD
- Our Plan B (SALAD-style) targets MVTec LOCO performance
