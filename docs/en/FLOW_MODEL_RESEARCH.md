# Flow Model Research: Normalizing Flows and Discriminative Models for Anomaly Detection

## Executive Summary
This document provides a comprehensive overview of normalizing flow-based methods and contemporary discriminative approaches for unsupervised anomaly detection. As the re-Anomaly project aims for industrial-grade precision in SMT (Surface Mount Technology) manufacturing, we evaluate various state-of-the-art architectures to address the limitations of the current baseline, particularly in few-shot scenarios and multi-scale defect detection.

## Normalizing Flows Background
Normalizing Flows (NF) are a class of generative models that learn to map a complex data distribution (such as image features from a pre-trained backbone) to a simple, tractable latent distribution (typically a standard Gaussian) through a series of invertible and differentiable transformations.

### Application to Anomaly Detection
In the context of anomaly detection, NFs are trained only on "normal" data. During training, the model learns the bijective mapping that transforms normal features into a Gaussian distribution. At inference time:
1.  Input features are passed through the flow transformations.
2.  The log-likelihood of the resulting latent representation is measured.
3.  Samples with low likelihood (those that fall into the tails of the Gaussian distribution) are identified as anomalies.

## FastFlow (Yu et al., 2021)
FastFlow is the current reference baseline for our project. It introduced a 2D normalizing flow architecture specifically designed for unsupervised anomaly detection and localization.

*   **Architecture**: Implements 2D normalizing flows directly on the feature maps extracted from vision backbones (e.g., ResNet, ViT). It uses alternate 1x1 and 3x3 convolutions within the coupling layers.
*   **Key Idea**: Unlike previous flow-based methods that often flattened spatial features, FastFlow preserves the 2D spatial structure of feature maps, allowing for effective anomaly localization.
*   **Performance**: Achieved 99.4% AUC on the MVTec AD dataset when paired with ResNet or ViT backbones.
*   **Limitations**:
    *   **Small Defects (Micro-Defect Blindness)**: High-resolution Vision Transformers (ViT) often smooth out fine-grained defects (e.g., sub-0.1mm solder bridges) during patch encoding, which FastFlow fails to capture.
    *   **Training Instability**: Performance can fluctuate significantly when training samples are extremely limited (10-20 images), leading to unstable density boundaries.
    *   **Logical/Structural Gap**: As a texture-focused model, it fails to detect "in-distribution but structurally invalid" anomalies, such as rotated or misaligned components.
    *   **Scalability**: Per-pin modeling in SMT scenarios is computationally expensive to scale across hundreds of components.

## MSFlow (Multi-Scale Flow)
MSFlow improves upon the standard flow architecture by explicitly addressing the multi-scale nature of industrial defects.

*   **Architecture**: Employs parallel normalizing flows that process features from different backbone layers (scales) separately.
*   **Key Innovation**: Uses asymmetrical parallel flows followed by an exchange-based "Fusion Flow" block that enables cross-scale communication.
*   **Benefits**: Significantly better at capturing both micro-defects (at high resolution) and macro-structural anomalies (at lower resolution) simultaneously.
*   **Why it Matters**: SMT defects range from tiny solder splashes to missing large integrated circuits; MSFlow's multi-scale fusion is critical for such diversity.

## HGAD (Hierarchical Gaussian Mixture AD, ECCV 2024)
HGAD addresses the challenge of building a unified model for multiple object classes, which traditionally leads to performance degradation.

*   **Architecture**: Replaces the single-Gaussian prior of standard flows with a Hierarchical Gaussian Mixture Model (HGMM).
*   **Key Innovation**: Learns hierarchical centers that represent both inter-class differences and intra-class variations.
*   **Benefits**: Enables a single model to perform anomaly detection across multiple part types without "homogeneous mapping" (where different classes collapse into the same latent space).
*   **Application**: Potential for a unified SMT model that can inspect multiple component types (resistors, capacitors, ICs) simultaneously.

## U-Flow
U-Flow focuses on the practical challenge of threshold selection in unsupervised environments.

*   **Key Innovation**: Introduces an unsupervised threshold selection methodology based on the "a contrario" framework and statistical detection theory.
*   **Benefits**: Provides automatic threshold tuning without the need for a validation set containing anomaly samples. It outputs binary segmentation masks based on a "Number of False Alarms" (NFA) metric.

## SimpleNet
SimpleNet represents a shift from density estimation to feature discrimination.

*   **Architecture**: Consists of a feature extractor, a simple feature adapter, and an anomalous feature generator (AFG).
*   **Key Difference**: Instead of modeling the density of normal data, it synthesizes "fake" anomalies by adding noise to normal features and trains a discriminator to distinguish them.
*   **Benefits**: Much more stable with few training samples compared to flow-based methods. It focuses on learning the discriminative boundary in feature space.
*   **Why Included**: Provides a high-speed, stable alternative to normalizing flows for rapid deployment.

## SALAD (Dual-Stream)
SALAD is specifically designed to tackle "logical" anomalies that standard texture-based methods miss.

*   **Architecture**: A dual-stream architecture featuring an Appearance Branch (structural) and a Composition Branch (logical).
*   **Structural Stream**: Detects typical appearance-based anomalies like scratches or color deviations.
*   **Logical Stream**: Uses semantics-aware modeling (often aided by DINO or SAM) to understand the relationship between components.
*   **Key Innovation**: Explicitly handles anomalies where the parts themselves look normal, but their arrangement or count is incorrect.
*   **Why it Matters for SMT**: Critical for detecting misplaced, rotated, or missing components where the background and component textures remain "normal" but the logic is violated.

## Comparison Table

| Method | Type | Multi-Scale | Few-Shot Stability | Logical Anomalies | Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FastFlow** | NF (2D) | Moderate | Low | Poor | Fast |
| **MSFlow** | NF (Parallel) | High | Moderate | Poor | Moderate |
| **HGAD** | NF (Mixture) | Moderate | Moderate | Moderate | Moderate |
| **SimpleNet** | Discriminative | Low | High | Poor | Very Fast |
| **SALAD** | Dual-Stream | High | Moderate | High | Slow |

## Recommendations

### Plan A: PatchCore (kNN Baseline)
While not a flow-based model, PatchCore remains the most stable baseline for few-shot scenarios (10-20 images). It should be implemented as the primary comparison point for all flow models.

### Plan B: SimpleNet + SALAD (Logical Focus)
For SMT lines where logical anomalies (missing/wrong parts) are frequent, the combination of SimpleNet (for speed and stability) and SALAD (for semantic logic) is recommended.

### Plan D: MSFlow + HGAD (Unified Multi-Scale)
For a next-generation unified system, combining MSFlow's multi-scale capture with HGAD's hierarchical mixture modeling offers the best path toward a single model capable of detecting complex defects across all SMT part types.

## References
1.  Yu, J., et al. (2021). "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows." arXiv:2111.07677.
2.  Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. (PatchCore)
3.  Yao, X., et al. (2024). "Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection." ECCV 2024. (HGAD)
4.  Liu, Z., et al. (2023). "SimpleNet: A Simple Network for Image Anomaly Detection and Localization." CVPR 2023.
5.  ICCV 2025. "SALAD: Semantics-Aware Logical Anomaly Detection."
