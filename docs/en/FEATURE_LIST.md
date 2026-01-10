# Feature List

This document outlines the features and components of the re-Anomaly project, tracking their current implementation status and priority.

## Backbone Models

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| DINOv2 | Self-supervised ViT support (ViT-S, ViT-B, ViT-L, ViT-g) | P0 | Done |
| DINOv3 | Support for DINOv3 with Gram Anchoring and stable dense features | P0 | In Progress |
| PixIO | Enhanced MAE with 8 class tokens and deeper decoder | P1 | In Progress |

## Detection Heads

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| PatchCore | Memory bank and kNN-based detection with coreset sampling | P0 | Done |
| MSFlow | Multi-scale parallel normalizing flows for complex anomalies | P1 | Planned |
| SimpleNet | Feature discrimination based anomaly detection | P1 | Done |
| SALAD | Dual-stream architecture for logic and structural anomalies | P1 | In Progress |
| Linear Head | Lightweight head specifically optimized for PixIO | P2 | Planned |
| FastFlow | 2D normalizing flows baseline implementation | P2 | Done |

## Pretraining

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| MAE Pretraining | PixIO-style pretraining with 75% masking ratio | P1 | In Progress |
| DINO DAPT | Domain-adaptive pretraining specifically for SMT datasets | P1 | Planned |

## Data Pipeline

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| Multi-resolution | Support for various input resolutions and aspect ratios | P0 | Done |
| Data Augmentation | Rotation, flip, color jitter, and compression simulation | P0 | Done |
| MVTec AD Support | Integration with standard MVTec AD dataset format | P0 | Done |
| Custom SMT Format | Support for proprietary SMT dataset structures | P0 | Done |

## Evaluation

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| Image AUROC | Standard image-level Area Under the ROC Curve metric | P0 | Done |
| Pixel AUROC/PRO | Pixel-level localization metrics (AUROC and PRO) | P0 | Done |
| Threshold Metrics | Recall and Precision calculation at specific thresholds | P1 | Done |
| Visualization | Generation of anomaly heatmaps and result overlays | P0 | Done |

## Deployment

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| ONNX Export | Support for exporting trained models to ONNX format | P1 | In Progress |
| Triton Client | Client implementation for NVIDIA Triton Inference Server | P1 | Planned |
| Batch Inference | Optimized batch processing for high-throughput inference | P2 | In Progress |
