# Backbone Research: Vision Foundation Models for SMT Anomaly Detection

## Executive Summary
This document outlines the research and selection criteria for vision backbone models in the re-Anomaly project. Selecting the right backbone is critical for capturing the subtle geometric and textural features required for high-precision SMT defect detection. We evaluate three primary generations of self-supervised models: DINOv2, DINOv3, and PixIO.

## DINOv2 (Meta, 2023)

### Architecture
DINOv2 is a self-supervised Vision Transformer (ViT) that utilizes a discriminative self-distillation approach. It produces high-quality features that are robust across various downstream tasks without requiring fine-tuning.

*   **Variants**: ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14
*   **Training**: Trained on the LVD-142M dataset (142 million curated images) using self-distillation with no labels.
*   **Key Features**:
    *   Strong dense features suitable for depth estimation and semantic segmentation.
    *   Robustness to distribution shifts.
    *   Multi-scale feature consistency.
*   **HuggingFace**: `facebook/dinov2-*` (e.g., `facebook/dinov2-base`, `facebook/dinov2-large`)
*   **Pros/Cons for Anomaly Detection**:
    *   **Pros**: Highly stable, well-supported, excellent baseline performance.
    *   **Cons**: May lack the extreme detail resolution required for sub-pixel micro defects compared to newer MAE-based architectures.

## DINOv3 / iBOTv2 (Meta, Aug 2025)

### Architecture
DINOv3 (incorporating iBOTv2 concepts) scales the DINO architecture to larger parameters and datasets, introducing refinements to the self-supervised objective to improve feature stability.

*   **Variants**: Scaled up to a 7B parameter ViT.
*   **Key Innovation**: **Gram Anchoring** — A technique that prevents feature collapse in dense prediction tasks by regularizing the feature distribution, ensuring that spatial tokens remain distinct and informative.
*   **Training**: Trained on 17B images using a self-supervised objective.
*   **Benefits**:
    *   Most stable dense features in the ViT family.
    *   Excellent performance on pixel-level tasks.
    *   Strong zero-shot capabilities.
*   **HuggingFace Availability**: Available via transformers as `facebook/dinov3-*`.
*   **Why it Matters for SMT**: The stability provided by Gram Anchoring is crucial for detecting small defects where feature noise could lead to false positives or missed detections.

## PixIO (Meta, Dec 2025)

### Architecture
PixIO is an Enhanced Masked Autoencoder (MAE) designed specifically for dense prediction and fine-grained reconstruction tasks. Unlike standard ViTs that focus on global classification, PixIO is optimized for local patch fidelity.

*   **Variants**: PixIO-B (Base), PixIO-L (Large), PixIO-H (Huge)
*   **Key Innovations**:
    *   **8 Class Tokens**: Instead of a single [CLS] token, PixIO uses 8 tokens to capture diverse global perspectives.
    *   **Deeper Decoder**: Features an 8-layer decoder (compared to 1-2 layers in standard MAE) to ensure high-fidelity reconstruction.
    *   **75% Masking Ratio**: High masking during training forces the model to learn deep structural dependencies.
    *   **20B Training Images**: Massive scale training on diverse visual data.
*   **Performance**: Outperforms DINOv3 on dense prediction tasks while maintaining a more efficient parameter count (variants range from 300M to 1.2B).
*   **Why it Matters for SMT**: Superior reconstruction understanding makes it ideal for identifying micro defects. By learning what "normal" looks like at a granular level, PixIO can detect anomalies that are invisible to standard discriminative models.

## Comparison Table

| Model | Params | Training Data | Dense Feature Quality | Inference Speed | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DINOv2** | 21M - 1.1B | 142M images | High | Very Fast | General Features, Baseline |
| **DINOv3** | 7B | 17B images | Very High | Moderate | Pixel-level tasks, Stability |
| **PixIO** | 300M - 1.2B | 20B images | Ultra High | Fast | Micro defects, Reconstruction |

## Recommendations

*   **Plan A/B (Baseline Stability)**: Use **DINOv2** or **DINOv3**. DINOv2 provides a fast and reliable baseline, while DINOv3 offers the highest feature stability for standard industrial parts.
*   **Plan C (Micro Defect Focus)**: Use **PixIO-H**. The enhanced reconstruction capabilities of the PixIO architecture are specifically suited for detecting the smallest deviations in high-resolution SMT imagery.
*   **Plan D (Advanced Multi-Scale)**: Either **PixIO** or **DINOv3** combined with an **MSFlow** head. This setup leverages the multi-scale features of these foundation models to detect anomalies across different resolutions simultaneously.

## References

1.  Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193.
2.  Meta AI. (2025). "DINOv3: Scaling Self-Supervised Vision Transformers with Gram Anchoring."
3.  Meta AI. (2025). "PixIO: Enhanced Masked Autoencoders for Dense Visual Understanding."
4.  Damm et al. (2025). "AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2." WACV 2025.
