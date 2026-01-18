"""
CLIP and SigLIP Backbone for re-Anomaly.

OpenCLIP provides access to various CLIP and SigLIP models:
- CLIP: Contrastive Language-Image Pre-training (OpenAI, LAION, DataComp)
- SigLIP: Sigmoid Language-Image Pre-training (Google, better for dense features)

These backbones are essential for zero-shot anomaly detection methods like:
- WinCLIP, APRIL-GAN, AnomalyCLIP, AFR-CLIP

Supported variants:
- CLIP: ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14, ViT-g-14, ViT-bigG-14
- SigLIP: ViT-B-16-SigLIP, ViT-L-16-SigLIP-256, ViT-SO400M-14-SigLIP-384

The backbone extracts multi-layer visual features from CLIP's vision encoder.
Text encoding is also available for zero-shot classification.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class CLIPBackbone(BaseBackbone):
    """CLIP/SigLIP backbone using OpenCLIP library."""

    # Variant configurations: (model_name, pretrained, embed_dim, num_layers, patch_size, image_size)
    # Using most common pretrained weights for each architecture
    VARIANTS = {
        # CLIP variants - OpenAI originals
        "clip_vitb32": ("ViT-B-32", "openai", 512, 12, 32, 224),
        "clip_vitb16": ("ViT-B-16", "openai", 512, 12, 16, 224),
        "clip_vitl14": ("ViT-L-14", "openai", 768, 24, 14, 224),
        "clip_vitl14_336": ("ViT-L-14-336", "openai", 768, 24, 14, 336),
        # CLIP variants - LAION trained (larger scale)
        "clip_vith14_laion": ("ViT-H-14", "laion2b_s32b_b79k", 1024, 32, 14, 224),
        "clip_vitg14_laion": ("ViT-g-14", "laion2b_s34b_b88k", 1024, 40, 14, 224),
        "clip_vitbigg14_laion": (
            "ViT-bigG-14",
            "laion2b_s39b_b160k",
            1280,
            48,
            14,
            224,
        ),
        # EVA-CLIP variants (excellent for dense prediction)
        "eva02_vite14": ("EVA02-E-14", "laion2b_s4b_b115k", 1792, 64, 14, 224),
        "eva02_vite14_plus": (
            "EVA02-E-14-plus",
            "laion2b_s9b_b144k",
            1792,
            64,
            14,
            224,
        ),
        # SigLIP variants (Google's improved CLIP, great for dense features)
        "siglip_vitb16": ("ViT-B-16-SigLIP", "webli", 768, 12, 16, 224),
        "siglip_vitl16_256": ("ViT-L-16-SigLIP-256", "webli", 1024, 24, 16, 256),
        "siglip_so400m_384": ("ViT-SO400M-14-SigLIP-384", "webli", 1152, 27, 14, 384),
        # DataComp trained (higher quality data)
        "clip_vitl14_datacomp": ("ViT-L-14", "datacomp_xl_s13b_b90k", 768, 24, 14, 224),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.variant = config.get("variant", "clip_vitl14")
        self.image_size = config.get("image_size")  # Override default if provided
        self.interpolate_pos = config.get("interpolate_pos_encoding", True)

        if self.variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {self.variant}. "
                f"Available: {list(self.VARIANTS.keys())}"
            )

        (
            self.model_name,
            self.pretrained_tag,
            self.embed_dim,
            self.num_layers,
            self.patch_size,
            default_image_size,
        ) = self.VARIANTS[self.variant]

        # Use config image_size if provided, otherwise use model default
        if self.image_size is None:
            self.image_size = default_image_size

        self.model: nn.Module | None = None
        self.tokenizer: Any = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._features: dict[int, torch.Tensor] = {}
        self._preprocess: Any = None

        if config.get("pretrained", True):
            self.load_pretrained(config.get("pretrained_weights"))

        if config.get("freeze_backbone", True):
            self.freeze()

    def load_pretrained(self, path: str | None = None) -> None:
        """Load pretrained CLIP/SigLIP model."""
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for CLIP/SigLIP backbone. "
                "Install with: pip install open_clip_torch"
            )

        if path:
            # Load from custom checkpoint
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=None,
                device=torch.device("cpu"),
            )
            state_dict = torch.load(path, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            # Load from pretrained
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained_tag,
                device=torch.device("cpu"),
            )

        # Store the full model and preprocess transform
        self.model = model
        self._preprocess = preprocess

        # Get tokenizer for text encoding
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers to extract features."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()

        if self.model is None:
            return

        # OpenCLIP visual encoder structure:
        # model.visual.transformer.resblocks[i] for ViT models
        # or model.visual.trunk for some variants
        visual = getattr(self.model, "visual", None)
        if visual is None:
            return

        # Try different layer access patterns
        layers = None

        # Pattern 1: transformer.resblocks (most common for ViT)
        if hasattr(visual, "transformer"):
            transformer = visual.transformer
            if hasattr(transformer, "resblocks"):
                layers = transformer.resblocks

        # Pattern 2: trunk.blocks (some variants like ConvNeXt-CLIP)
        if layers is None and hasattr(visual, "trunk"):
            trunk = visual.trunk
            if hasattr(trunk, "blocks"):
                layers = trunk.blocks
            elif hasattr(trunk, "stages"):
                # For hierarchical models, we need different handling
                layers = None  # Will need special handling

        if layers is None:
            print(
                f"WARNING: Could not find layer structure for {self.model_name}. "
                "Features may not be extracted correctly."
            )
            return

        for layer_idx in self.output_layers:
            if layer_idx < len(layers):

                def hook_fn(
                    module: nn.Module, input: Any, output: Any, idx: int = layer_idx
                ) -> None:
                    if isinstance(output, tuple):
                        output = output[0]
                    self._features[idx] = output

                handle = layers[layer_idx].register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W), normalized to [0, 1] or [-1, 1]

        Returns:
            List of feature tensors at different layers, each of shape (B, D, h, w)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._features.clear()
        batch_size = x.shape[0]

        # Resize if needed
        if self.interpolate_pos and x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Forward through visual encoder only
        visual = self.model.visual
        _ = visual(x)

        features = []
        h = w = self.image_size // self.patch_size

        # CLIP ViT: [batch, 1 + num_patches, embed_dim]
        # First token is CLS, rest are patch tokens
        for layer_idx in sorted(self.output_layers):
            feat = self._features.get(layer_idx)
            if feat is not None:
                if feat.dim() == 3:
                    # Skip CLS token (first token), keep patch tokens
                    feat = feat[:, 1:, :]
                feat = feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                features.append(feat)

        return features

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to CLIP embedding space.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Image embeddings of shape (B, embed_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return self.model.encode_image(x)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """
        Encode text prompts to CLIP embedding space.

        Args:
            text: List of text strings

        Returns:
            Text embeddings of shape (N, embed_dim)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        tokens = self.tokenizer(text)
        tokens = tokens.to(self.device)
        return self.model.encode_text(tokens)

    def get_text_features(
        self, class_names: list[str], templates: list[str] | None = None
    ) -> torch.Tensor:
        """
        Generate text features for zero-shot classification.

        Args:
            class_names: List of class names
            templates: Optional list of prompt templates. Default uses standard CLIP templates.

        Returns:
            Text features of shape (num_classes, embed_dim), normalized
        """
        if templates is None:
            templates = [
                "a photo of a {}.",
                "a photo of the {}.",
                "a good photo of a {}.",
                "a bad photo of a {}.",
                "a photo of a damaged {}.",
                "a photo of a defective {}.",
                "a photo of a {} with defects.",
            ]

        text_features = []
        with torch.no_grad():
            for class_name in class_names:
                texts = [template.format(class_name) for template in templates]
                class_embeddings = self.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                text_features.append(class_embedding)

        return torch.stack(text_features, dim=0)

    def get_anomaly_text_features(
        self, category: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text features for anomaly detection (normal vs anomaly).

        Args:
            category: Object category name

        Returns:
            Tuple of (normal_features, anomaly_features), each of shape (embed_dim,)
        """
        normal_prompts = [
            f"a photo of a normal {category}.",
            f"a good photo of a {category}.",
            f"a flawless {category}.",
            f"a perfect {category}.",
            f"a {category} without any defects.",
        ]

        anomaly_prompts = [
            f"a photo of a damaged {category}.",
            f"a photo of a defective {category}.",
            f"a {category} with defects.",
            f"a {category} with anomalies.",
            f"a broken {category}.",
            f"a flawed {category}.",
        ]

        with torch.no_grad():
            normal_embeddings = self.encode_text(normal_prompts)
            normal_embeddings = normal_embeddings / normal_embeddings.norm(
                dim=-1, keepdim=True
            )
            normal_feature = normal_embeddings.mean(dim=0)
            normal_feature = normal_feature / normal_feature.norm()

            anomaly_embeddings = self.encode_text(anomaly_prompts)
            anomaly_embeddings = anomaly_embeddings / anomaly_embeddings.norm(
                dim=-1, keepdim=True
            )
            anomaly_feature = anomaly_embeddings.mean(dim=0)
            anomaly_feature = anomaly_feature / anomaly_feature.norm()

        return normal_feature, anomaly_feature
