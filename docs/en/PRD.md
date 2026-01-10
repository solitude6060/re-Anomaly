# Product Requirements Document (PRD) - re-Anomaly

## Overview and Background
re-Anomaly is a next-generation anomaly detection system specifically designed for Surface Mount Technology (SMT) production lines. In the high-precision environment of electronics manufacturing, even minor defects can lead to significant failures. Current rule-based inspection systems often suffer from high overkill rates or miss subtle defects. re-Anomaly leverages state-of-the-art vision models and few-shot learning techniques to provide a highly accurate, real-time inspection solution that can identify both structural and logical anomalies with minimal training data.

## Target Users
* Manufacturing QA teams
* ML engineers

## Functional Requirements
* **FR1: Few-shot learning**: The system must support training with 10-200 OK samples and 0 NG samples to accommodate new parts quickly.
* **FR2: Multi-scale defect detection**: Capability to detect defects ranging from micro-level scratches to macro-level missing components.
* **FR3: Anomaly localization with heatmaps**: Provide pixel-level anomaly scores visualized as heatmaps to help QA teams identify defect locations.
* **FR4: Structural/logical anomaly detection**: Beyond simple surface defects, the system must detect logical anomalies such as incorrect component orientation or missing pins.
* **FR5: ONNX export for deployment**: Support exporting trained models to ONNX format for efficient cross-platform deployment.
* **FR6: Triton server integration**: Integration with NVIDIA Triton Inference Server for scalable and high-performance model serving.
* **FR7: Configurable thresholds**: Allow users to adjust anomaly detection thresholds to balance miss rates and overkill rates based on specific production needs.

## Non-Functional Requirements
* **NFR1: 0% miss rate**: Hard requirement to ensure zero false negatives (no defects are missed).
* **NFR2: <10% overkill rate**: The false positive rate (overkill) must be kept under 10% to maintain production efficiency.
* **NFR3: Inference latency <100ms**: Total inference time per image (200x200 ROI) must be less than 100ms to keep up with real-time production speeds.
* **NFR4: Model size constraints for edge deployment**: Optimized model architectures to fit within the memory constraints of edge computing devices used on production lines.
* **NFR5: Robustness to image compression artifacts**: The system must maintain performance even when processing images with standard compression artifacts common in industrial camera outputs.

## User Stories
### Manufacturing QA Team
* **As a** QA operator, **I want to** see a heatmap overlay on the inspection image **so that** I can quickly verify where the defect is located.
    * **Acceptance Criteria**: Heatmap is generated for every detected anomaly with high localization accuracy.
* **As a** production manager, **I want to** ensure that no defective boards pass the station **so that** we avoid costly field failures.
    * **Acceptance Criteria**: 0% miss rate achieved on a validated test set of NG images.

### ML Engineer
* **As an** ML engineer, **I want to** export the model to ONNX **so that** I can deploy it on various hardware accelerators.
    * **Acceptance Criteria**: Model successfully exports to ONNX and produces consistent results with the PyTorch implementation.
* **As an** ML engineer, **I want to** train a model for a new part using only a few dozen OK samples **so that** we can reduce the time-to-market for new products.
    * **Acceptance Criteria**: Model achieves high AUC with only 20 OK training samples.

## Constraints and Assumptions
* Input images are 200x200 pixel ROI crops.
* Lighting conditions on the production line are relatively stable.
* Hardware environment for inference includes GPUs supporting TensorRT/Triton.
* Training is performed on OK samples only (unsupervised/self-supervised approach).

## Out of Scope items
* Hardware camera integration and triggering logic.
* Long-term data storage and historical trend analysis.
* Automatic defect classification (e.g., categorizing "scratch" vs "missing component").

## Dependencies
* PyTorch
* Transformers
* ONNX
* Triton
