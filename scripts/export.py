import argparse
import logging
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

from src.models.backbones import DINOv2Backbone, DINOv3Backbone, PixIOBackbone
from src.models.heads import (
    FastFlowHead,
    LinearHead,
    MSFlowHead,
    PatchCoreHead,
    SALADHead,
    SimpleNetHead,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BACKBONE_REGISTRY = {
    "dinov2": DINOv2Backbone,
    "dinov3": DINOv3Backbone,
    "pixio": PixIOBackbone,
}

HEAD_REGISTRY = {
    "patchcore": PatchCoreHead,
    "msflow": MSFlowHead,
    "fastflow": FastFlowHead,
    "simplenet": SimpleNetHead,
    "salad": SALADHead,
    "linear": LinearHead,
}


class ExportableModel(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output["anomaly_score"]


def export_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> None:
    log.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)

    backbone_type = cfg.backbone.name
    head_type = cfg.head.name

    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head_type}")

    log.info(f"Building backbone: {backbone_type}")
    backbone = BACKBONE_REGISTRY[backbone_type](OmegaConf.to_container(cfg.backbone))
    backbone.load_pretrained(cfg.backbone.get("checkpoint"))
    backbone.freeze()

    log.info(f"Building head: {head_type}")
    head = HEAD_REGISTRY[head_type](OmegaConf.to_container(cfg.head))

    if checkpoint_path:
        log.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        head.load_state_dict(ckpt["head"])

    model = ExportableModel(backbone, head)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    log.info("Tracing model...")
    with torch.no_grad():
        _ = model(dummy_input)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    log.info(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    log.info("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    log.info("Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(str(output_path))
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    log.info(f"ONNX output shape: {ort_outputs[0].shape}")
    log.info(f"Export successful: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"Model size: {file_size_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Use static batch size (default: dynamic)",
    )

    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset,
        dynamic_batch=not args.static_batch,
    )


if __name__ == "__main__":
    main()
