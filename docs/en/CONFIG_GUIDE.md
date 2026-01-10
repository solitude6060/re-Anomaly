# Hydra Configuration Guide

This guide explains the configuration system used in the **re-Anomaly** project. We utilize [Hydra](https://hydra.cc/) to manage complex, hierarchical configurations, enabling flexible experimentation and easy deployment.

## Overview

Hydra is a framework for elegantly configuring complex applications. In this project, Hydra provides:

- **Composable Configurations**: Build complex configs from smaller, reusable modules.
- **CLI Overrides**: Easily modify any parameter from the command line without changing YAML files.
- **Experiment Tracking**: Automatically manage output directories and record configuration states for every run.
- **Dynamic Composition**: Swap backbones, detection heads, and datasets with simple configuration changes.

## Directory Structure

All configuration files are located in the `configs/` directory. The structure is organized by functional components:

```text
configs/
├── config.yaml          # Main entry point
├── backbone/            # Vision backbone architectures
│   ├── dinov2.yaml
│   ├── dinov3.yaml
│   └── pixio.yaml
├── head/                # Anomaly detection head architectures
│   ├── patchcore.yaml
│   ├── msflow.yaml
│   ├── simplenet.yaml
│   ├── salad.yaml
│   ├── linear.yaml
│   └── fastflow.yaml
├── augmentation/        # Data augmentation strategies
│   ├── train.yaml
│   └── eval.yaml
├── dataset/             # Dataset-specific settings
│   ├── mvtec.yaml
│   └── smt.yaml
├── experiment/          # Predefined experimental recipes (overrides)
│   ├── plan_a.yaml
│   ├── plan_b.yaml
│   ├── plan_c.yaml
│   └── plan_d.yaml
├── training/            # Optimizer, scheduler, and trainer settings
│   └── default.yaml
├── evaluation/          # Metrics and evaluation parameters
│   └── default.yaml
├── pretrain/            # Pretraining task configurations
│   ├── mae.yaml
│   └── dino_dapt.yaml
└── deploy/              # Deployment and inference settings
    └── triton.yaml
```

## Configuration Hierarchy

Hydra uses a `defaults` list to compose the final configuration. The `config.yaml` file acts as the primary schema, defining which default sub-configs to load.

### Composition Logic
1. **Base Groups**: Hydra loads the default file from each group (backbone, head, dataset, etc.).
2. **Experiment Overrides**: If an `experiment` is specified, it is loaded last, overriding any parameters defined in the base groups.
3. **CLI Overrides**: Command-line arguments have the highest priority and override both base and experiment configurations.

## Common Configuration Options

### Backbone Configurations
Located in `configs/backbone/`. These define the feature extraction architecture.
- `name`: The model architecture identifier (e.g., `vit_base_patch14`).
- `pretrained`: Path or identifier for pretrained weights.
- `freeze_backbone`: Boolean to determine if backbone weights should be updated.
- `output_layers`: Indices or names of layers from which to extract features.
- `image_size`: Input resolution required by the backbone.

### Head Configurations
Located in `configs/head/`. These define the anomaly detection logic.
- `type`: The head architecture (e.g., `patchcore`, `msflow`).
- `hidden_dims`: Dimensions of intermediate projection layers.
- `num_flows`: Number of flow steps (for MSFlow/FastFlow).
- `k_nearest`: Number of neighbors for kNN-based methods (for PatchCore).

### Training Configurations
Located in `configs/training/`. These control the optimization process.
- `batch_size`: Number of samples per training step.
- `epochs`: Total number of training passes.
- `learning_rate`: Initial learning rate.
- `optimizer`: Configuration for the optimizer (type, weight decay, etc.).
- `scheduler`: Configuration for the learning rate scheduler.

### Dataset Configurations
Located in `configs/dataset/`. These define data loading behavior.
- `root_path`: Path to the dataset directory.
- `image_size`: Target resolution for resized images.
- `train_split`: Name or percentage of the training data split.
- `categories`: List of object categories to include.

## CLI Override Examples

Hydra allows for powerful command-line manipulation.

### Basic Run
Execute training with default parameters defined in `config.yaml`:
```bash
python scripts/train.py
```

### Override Backbone
Switch to a different backbone configuration:
```bash
python scripts/train.py backbone=pixio
```

### Multiple Overrides
Combine multiple overrides and modify specific parameters:
```bash
python scripts/train.py backbone=dinov3 head=patchcore training.batch_size=32
```

### Run Specific Experiment
Use a predefined experimental recipe:
```bash
python scripts/train.py experiment=plan_a
```

### Multirun (Sweep)
Perform a parameter sweep across multiple configurations:
```bash
python scripts/train.py -m backbone=dinov2,dinov3,pixio head=patchcore,msflow
```

## Creating Custom Experiments

Experiment configurations (in `configs/experiment/`) are the best way to maintain reproducible settings for specific runs.

### Example Experiment: `plan_a.yaml`
```yaml
# @package _global_

defaults:
  - override /backbone: dinov3
  - override /head: patchcore

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001

dataset:
  name: smt_data
  image_size: 224
```

## Environment Variables

The following environment variables can be useful when working with the configuration system:

- `HYDRA_FULL_ERROR=1`: Set this to see the full stack trace for configuration-related errors.
- `HYDRA_OUTPUT_DIR`: Can be used to manually specify the output directory, though Hydra handles this automatically by default.

## Best Practices

- **Keep Base Configs Minimal**: Base sub-configs (like `backbone/dinov2.yaml`) should only contain parameters essential to that module.
- **Use Experiment Configs for Reproducibility**: Instead of long CLI commands, save your setup as an experiment file in `configs/experiment/`.
- **Version Control All Configs**: Ensure all YAML files are tracked in Git to maintain a history of your experimental setups.
- **Avoid Hardcoding Paths**: Use relative paths or environment variables within configs to ensure portability across different machines.
