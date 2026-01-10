# Project Specification: re-Anomaly

## Executive Summary
re-Anomaly is a next-generation anomaly detection system designed for Surface Mount Technology (SMT) production lines. Building upon the limitations of the previous FastFlow + DINOv3 baseline, this project introduces a multi-backbone and multi-head architecture to achieve industrial-grade accuracy and performance. The system focuses on detecting both micro-defects and logical/structural anomalies with extremely low miss rates and high inference speeds.

## Problem Statement
The current baseline approach using FastFlow with DINOv3 (per-pin models) faces several critical challenges:
- Data scarcity: Only 10-20 OK images are typically available per part, with no NG (No-Good) samples for training.
- Small defect detection: Micro-defects on small electronic components are frequently missed.
- Logical and structural anomalies: The system fails to detect anomalies that are context-dependent or structural in nature (e.g., misaligned components that look correct in isolation).
- Training instability: With very few samples, model training often fails to converge or generalizes poorly.
- Current approach limitations: Per-pin modeling is computationally expensive to scale and lacks global context.

## Technical Goals
- Miss rate: 0% (zero false negatives)
- Overkill rate: Less than 10%
- Pass-through rate: 80% or higher
- Inference latency: Less than 100ms per Region of Interest (ROI)
- Backbone load time: Less than 500ms
- Downstream task load time: Less than 100ms

## Proposed Solution Overview
The project employs a multi-strategy experimental approach to address different types of anomalies:

### Experimental Plans
- Plan A: PatchCore - Uses DINOv2/v3 backbones with a PatchCore head (Memory Bank + kNN) to establish a stable baseline for traditional anomaly detection.
- Plan B: HLSIS - Combines DINOv3 with Domain-Adaptive Pretraining (DAPT) and a dual-stream SimpleNet + SALAD head to specifically target logical anomalies.
- Plan C: PixIO - Utilizes the PixIO-H backbone with a linear/lightweight head to improve micro-defect detection in few-shot scenarios.
- Plan D: MSFlow - A multi-scale approach using PixIO or DINOv3 backbones with MSFlow + HGAD heads for unified multi-class detection.

### Multi-Backbone Strategy
Leveraging state-of-the-art vision models:
- PixIO: Enhanced MAE with deeper decoders, pretrained on 20B images.
- DINOv3: Gram Anchoring based self-supervised model.
- DINOv2: Established self-supervised ViT baseline.

### Domain-Adaptive Pretraining (DAPT)
To improve feature representation for industrial parts, the system will undergo pretraining on 3 million unlabeled industrial images.

## System Architecture
```text
[ Input Image ]
      |
[ Preprocessing & ROI Extraction ]
      |
[ Feature Extraction (Multi-Backbone) ]
|--- DINOv2 / DINOv3 (ViT-based)
|--- PixIO-H (Enhanced MAE)
      |
[ Detection Heads (Multi-Head) ]
|--- PatchCore (Memory Bank)
|--- MSFlow (Normalizing Flows)
|--- SimpleNet / SALAD (Dual-stream)
      |
[ Post-processing & Anomaly Scoring ]
      |
[ Result Visualization & API Response ]
```

## Hardware Requirements
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Inference Runtime: ONNX Runtime for optimized execution.
- Deployment: Triton Inference Server for scalable serving.

## Data Requirements
- Pretraining: 3,000,000 unlabeled industrial images for domain adaptation.
- Fine-tuning/Evaluation: 20-30 different SMT part types, each with 10-30 OK samples for baseline setup.

## Success Criteria and Metrics
The project will be evaluated based on the following metrics:
- Anomaly Detection Performance: Area Under ROC (AUROC) and Area Under Precision-Recall (AUPRO).
- Industrial Metrics: Miss Rate (P0 priority), Overkill Rate, and Pass-through Rate.
- Efficiency: End-to-end inference latency and model load times.
- Stability: Convergence rate and variance of results across different part types with limited samples.
