# re-Anomaly

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](#)

## Overview
re-Anomaly is a next-generation SMT (Surface Mount Technology) industrial anomaly detection system. It features a flexible architecture with multi-backbone support, designed for high-precision defect detection in manufacturing environments. The project focuses on achieving a zero miss rate target while maintaining low overkill rates, leveraging state-of-the-art vision models and detection heads.

## Key Features
- **Multi-backbone Support**: Integrated with powerful vision backbones including PixIO, DINOv3, and DINOv2.
- **Multiple Detection Heads**: Implements various state-of-the-art anomaly detection architectures:
  - **PatchCore**: Memory bank and kNN-based detection.
  - **MSFlow**: Multi-scale parallel normalizing flows.
  - **SimpleNet**: Lightweight and efficient anomaly detection.
  - **SALAD**: Specialized in detecting logic and structural anomalies.
  - **FastFlow**: 2D normalizing flows for rapid inference.
- **Few-shot Learning**: Optimized performance with minimal training samples per class.
- **Zero Miss Rate Target**: Engineering focus on eliminating undetected critical defects.

## Experimental Plans
The project follows a structured experimental approach to evaluate different combinations of backbones and detection heads:

| Plan | Components | Description / Target |
|------|------------|----------------------|
| **Plan A** | DINOv2/v3 + PatchCore | Stable baseline for standard anomaly detection tasks. |
| **Plan B** | DINOv3 + DAPT + SimpleNet + SALAD | Dual-stream architecture focused on complex logic anomalies. |
| **Plan C** | PixIO-H + Linear Head | Optimized for micro defects and few-shot learning scenarios. |
| **Plan D** | PixIO/DINOv3 + MSFlow + HGAD | Multi-scale unified detection for diverse and complex defect types. |

## Requirements
- **Python**: 3.10 or higher
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM) recommended for training and high-speed inference.
- **Performance Targets**:
  - Miss Rate: 0%
  - Overkill Rate: < 10%
  - Inference Time: < 100ms

## Installation
The project uses [uv](https://github.com/astral-sh/uv) for efficient Python package management.

```bash
# Install dependencies and setup environment
uv sync

# Install with all extra dependencies
uv sync --all-extras
```

## Quick Start
Example commands for training and evaluation using the Hydra-based configuration system:

```bash
# Run training for Plan A
python main.py experiment=plan_a dataset=smt_data

# Run evaluation with a specific checkpoint
python main.py mode=eval experiment=plan_a checkpoint=outputs/model.ckpt

# Override configuration parameters
python main.py backbone=dinov3 head=patchcore training.batch_size=16
```

## Project Structure
The repository is organized as follows:

```text
re-Anomaly/
├── configs/          # Hydra configuration files
│   ├── augmentation/ # Data augmentation strategies
│   ├── backbone/     # Backbone model configurations
│   ├── dataset/      # Dataset-specific settings
│   ├── experiment/   # Predefined experimental recipes
│   ├── head/         # Detection head configurations
│   └── training/     # Optimizer and trainer settings
├── src/              # Core source code
│   ├── data/         # Data loading and pipelines
│   ├── deploy/       # Deployment modules (e.g., Triton)
│   ├── models/       # Backbones and detection heads
│   ├── training/     # Training loops and logic
│   └── evaluation/   # Metrics and evaluation scripts
├── scripts/          # Utility and helper scripts
├── docs/             # Technical documentation and guides
└── main.py           # Project entry point
```

## Configuration
This project utilizes **Hydra** for comprehensive configuration management. All aspects of the model, data, and training process can be customized via YAML files in the `configs/` directory or overridden through command-line arguments.

Example of command-line override:
```bash
python main.py backbone.type=pixio head.type=msflow training.lr=0.0001
```

## Internationalization
- [繁體中文 (Traditional Chinese)](README_zh-TW.md)

## License
This project is licensed under the MIT License.
