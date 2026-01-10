# Development Plan for re-Anomaly Project

This document outlines the 10-week development strategy for the re-Anomaly project. The plan is designed to transition from foundation building to advanced model experimentation and production-ready deployment.

## Development Phases Overview

The development process is divided into five distinct phases, each lasting two weeks. The timeline focuses on establishing a robust modular framework, implementing state-of-the-art anomaly detection heads, performing large-scale pretraining, and optimizing for real-world deployment on NVIDIA RTX 4090 hardware.

## Phase 1: Foundation (Weeks 1-2)

The primary goal of this phase is to establish the core infrastructure and data handling capabilities.

- Project setup (completed)
- Backbone loader implementations:
    - DINOv2 integration for high-quality feature extraction
    - DINOv3 (iBOTv2) support for latest self-supervised representations
    - PixIO-style vision transformer integration
- Basic data pipeline:
    - Efficient image loading and transformation
    - Support for standard anomaly detection datasets (e.g., MVTec AD, VisA)
- Configuration system:
    - Implementation of Hydra for flexible experiment management
    - Standardized configuration schemas for models, data, and training

## Phase 2: Core Components (Weeks 3-4)

This phase focuses on the implementation of various anomaly detection architectures and the supporting training/evaluation infrastructure.

- Detection head implementations:
    - PatchCore: Implementation of memory bank and k-Nearest Neighbors (kNN) search
    - MSFlow: Multi-scale normalizing flows for density estimation
    - SimpleNet: Feature-based anomaly detection with simplified architecture
    - SALAD: Dual-stream architecture for enhanced anomaly localization
    - Linear head: Baseline implementation for linear probing
    - FastFlow: Reference baseline for normalizing flow-based detection
- Training pipeline:
    - Unified trainer class with support for multiple backbones and heads
    - Integration of logging tools (e.g., TensorBoard or WandB)
- Evaluation metrics:
    - Area Under the Receiver Operating Characteristic (AUROC)
    - Recall and Precision at various thresholds
    - Pixel-level vs. Image-level metric calculation

## Phase 3: Pretraining (Weeks 5-6)

Leveraging the 3M image dataset, this phase focuses on domain-specific and general-purpose pretraining to enhance feature representations.

- MAE pretraining:
    - PixIO-style Masked Autoencoder pretraining using the 3M image dataset
    - Configuration of masking ratios and reconstruction loss
- DINO Domain-Adaptive Pre-Training (DAPT):
    - Fine-tuning DINO models with domain-specific industrial data
    - Optimization of contrastive and clustering losses
- Pretrained weight integration:
    - Standardization of weight loading mechanisms across the framework
    - Verification of representation quality improvements

## Phase 4: Experiments (Weeks 7-8)

A systematic exploration of different model combinations and ablation studies to identify the optimal configuration.

- Plan A: PatchCore baseline
    - Benchmarking PatchCore performance across all target datasets
- Plan B: HLSIS dual-stream
    - Implementation and testing of High-Level and Semantic Information Synergy
- Plan C: PixIO + Linear
    - Evaluation of PixIO representations with minimal linear heads
- Plan D: MSFlow + HGAD
    - Combining multi-scale flows with Hierarchical Grouping Anomaly Detection
- Ablation studies:
    - Analysis of backbone influence on detection accuracy
    - Impact of pretraining data volume on final performance
    - Comparison of different head architectures

## Phase 5: Optimization & Deployment (Weeks 9-10)

Final preparation for production use, focusing on inference speed and deployment infrastructure.

- ONNX export:
    - Conversion of PyTorch models to ONNX format for cross-platform support
    - Verification of numerical consistency after export
- Triton server integration:
    - Development of model repositories for NVIDIA Triton Inference Server
    - Configuration of dynamic batching and model instances
- Performance optimization:
    - TensorRT acceleration for GPU-specific optimizations
    - Quantization (FP16/INT8) to reduce latency and memory footprint
- Production testing:
    - End-to-end latency testing on target hardware (RTX 4090)
    - Stability testing under continuous load

## Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 2 | Foundation Ready | Hydra-based framework with DINOv2 support |
| 4 | Core Heads Implemented | Library of detection heads (PatchCore, MSFlow, etc.) |
| 6 | Pretraining Complete | Domain-adapted weights for 3M image dataset |
| 8 | Best Model Selection | Comprehensive experiment report and optimal weights |
| 10 | Production Deployment | Optimized TensorRT models on Triton server |

## Risk Assessment

- Data Quality: Noise or lack of diversity in the 3M dataset may affect pretraining effectiveness. Mitigate by implementing rigorous data cleaning.
- Hardware Constraints: Large-scale pretraining may exceed RTX 4090 memory. Mitigate by using gradient accumulation and mixed-precision training.
- Architectural Complexity: Integrating multiple diverse heads may lead to code maintenance issues. Mitigate by strictly adhering to modular design patterns.
- Timeline Compression: Phase 4 experiments might uncover the need for further architectural changes. Mitigate by prioritizing the most promising plans early.
