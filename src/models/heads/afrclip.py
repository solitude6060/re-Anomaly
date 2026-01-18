"""
AFR-CLIP: Zero-Shot Anomaly Detection via Anomaly Feature Rectification.

Based on: "AFR-CLIP: Enhancing Zero-Shot Industrial Anomaly Detection with
Stateless-to-Stateful Anomaly Feature Rectification" (Yuan et al., 2025)
arXiv: 2503.12910

Key concepts:
1. Stateless-to-Stateful: Uses a stateless prompt ("a photo of a {object}")
   and rectifies it with visual features to capture anomalous state info.
2. Text-on-Text Scoring: Instead of image-text similarity (dominated by category),
   compares rectified text embedding with normal/abnormal text prototypes.
3. Cross-Modal Feature Rectification (CMFR): Injects visual anomaly cues into text.
4. Multi-Patch Feature Aggregation (MPFA): Aggregates multi-scale spatial context.

This is a ZERO-SHOT method - no training on target dataset required.
Requires CLIPBackbone as the feature extractor.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class CrossModalFeatureRectification(nn.Module):
    """
    Cross-Modal Feature Rectification (CMFR) module.

    Injects visual anomaly information into stateless text embeddings
    to create state-aware text features.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 4

        # Two conv layers with ReLU, then linear projection
        self.conv1 = nn.Conv1d(embed_dim * 2, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, embed_dim * 2, kernel_size=1)
        self.proj = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(
        self, visual_features: torch.Tensor, text_stateless: torch.Tensor
    ) -> torch.Tensor:
        """
        Rectify stateless text embedding with visual features.

        Args:
            visual_features: (B, N, D) patch-level visual features
            text_stateless: (B, D) or (D,) stateless text embedding

        Returns:
            Rectified text embeddings: (B, N, D)
        """
        B, N, D = visual_features.shape

        # Expand text_stateless to match visual features
        if text_stateless.dim() == 1:
            text_stateless = text_stateless.unsqueeze(0).expand(B, -1)  # (B, D)
        text_expanded = text_stateless.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)

        # Concatenate visual and text features
        concat_feat = torch.cat([visual_features, text_expanded], dim=-1)  # (B, N, 2D)

        # Process through conv layers
        # Reshape for 1D conv: (B, 2D, N)
        x = concat_feat.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = x.transpose(1, 2)  # (B, N, 2D)
        x = self.proj(x)  # (B, N, 2D)

        # Split into visual and text weight masks
        Mv, Mt = x.chunk(2, dim=-1)  # Each (B, N, D)

        # Rectify: add visual features weighted by mask to stateless text
        rectified = text_expanded + visual_features * Mv  # (B, N, D)

        return rectified


class MultiPatchFeatureAggregation(nn.Module):
    """
    Multi-Patch Feature Aggregation (MPFA) module.

    Aggregates multi-scale spatial context from neighboring patches
    to improve boundary coherence and small defect detection.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Aggregate features from neighboring patches.

        Args:
            features: (B, N, D) patch features where N = h * w
            h, w: Spatial dimensions of the feature grid

        Returns:
            Aggregated features: (B, N, D)
        """
        B, N, D = features.shape

        # Reshape to spatial grid
        spatial = features.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)

        # Apply adaptive average pooling with padding
        padding = self.kernel_size // 2
        padded = F.pad(spatial, (padding, padding, padding, padding), mode="reflect")

        # Use avg_pool2d with stride 1 to aggregate neighbors
        aggregated = F.avg_pool2d(
            padded, kernel_size=self.kernel_size, stride=1, padding=0
        )  # (B, D, h, w)

        # Reshape back
        return aggregated.permute(0, 2, 3, 1).reshape(B, N, D)


class AFRCLIPHead(BaseHead):
    """
    AFR-CLIP Zero-Shot Anomaly Detection Head.

    Uses CLIP for zero-shot anomaly detection via:
    1. Cross-Modal Feature Rectification (CMFR)
    2. Text-on-Text Anomaly Scoring
    3. Multi-Patch Feature Aggregation (MPFA)

    This head requires a CLIPBackbone and uses its text encoding capabilities.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.embed_dim = config.get("embed_dim", 768)
        self.category = config.get("category", "object")
        self.image_size = config.get("image_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.mpfa_kernel = config.get("mpfa_kernel", 3)
        self.use_mpfa = config.get("use_mpfa", True)
        self.temperature = config.get("temperature", 1.0)

        # Prompt templates
        self.normal_prompts = config.get(
            "normal_prompts",
            [
                "a photo of a normal {}.",
                "a good photo of a {}.",
                "a flawless {}.",
                "a perfect {}.",
                "a {} without any defects.",
            ],
        )
        self.abnormal_prompts = config.get(
            "abnormal_prompts",
            [
                "a photo of a damaged {}.",
                "a photo of a defective {}.",
                "a {} with defects.",
                "a {} with anomalies.",
                "a broken {}.",
                "a flawed {}.",
            ],
        )
        self.stateless_prompt = config.get("stateless_prompt", "a photo of a {}.")

        # Modules
        self.cmfr = CrossModalFeatureRectification(self.embed_dim)
        if self.use_mpfa:
            self.mpfa = MultiPatchFeatureAggregation(self.mpfa_kernel)

        # Linear adapters for domain adaptation
        self.visual_adapter = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_adapter = nn.Linear(self.embed_dim, self.embed_dim)

        # Projection for CLIP visual features (1024-dim) to text dimension (768-dim)
        # This handles the CLIP ViT-L/14 dimension mismatch
        self._visual_to_text_proj: nn.Linear | None = None

        # Cached text embeddings (computed once per category)
        self._normal_embed: torch.Tensor | None = None
        self._abnormal_embed: torch.Tensor | None = None
        self._stateless_embed: torch.Tensor | None = None

        # Reference to backbone (set during training/inference)
        self._backbone: Any = None

    def set_backbone(self, backbone: Any) -> None:
        """Set reference to CLIP backbone for text encoding."""
        self._backbone = backbone

        # Detect visual dimension from backbone and create projection if needed
        # For CLIP/SigLIP, visual and text dimensions may differ:
        # - CLIP ViT-L/14: visual = 1024, text = 768
        # - CLIP ViT-B-16: visual = 512, text = 512
        # - SigLIP SO400M: visual = 1152, text = 1152
        #
        # The backbone.embed_dim is text dimension, not visual.
        # We get the visual dimension from the visual projection layer.
        visual_dim = self.embed_dim  # Default fallback

        if hasattr(backbone, "model") and hasattr(backbone.model, "visual"):
            visual = backbone.model.visual
            proj = getattr(visual, "proj", None)
            if proj is not None:
                # proj.weight shape is (visual_dim, text_dim) for CLIP
                # Use shape[0] as visual dimension
                if hasattr(proj, "weight"):
                    visual_dim = proj.weight.shape[0]
                elif hasattr(proj, "data"):
                    visual_dim = proj.data.shape[0]
                elif hasattr(proj, "shape"):
                    visual_dim = proj.shape[0]

        # Create projection layer if dimensions don't match
        if visual_dim != self.embed_dim:
            self._visual_to_text_proj = nn.Linear(visual_dim, self.embed_dim)
            # Move to same device as other parameters
            if list(self.parameters()):
                self._visual_to_text_proj.to(next(self.parameters()).device)
        else:
            self._visual_to_text_proj = None

    def set_category(self, category: str) -> None:
        """Update category and invalidate cached embeddings."""
        self.category = category
        self._normal_embed = None
        self._abnormal_embed = None
        self._stateless_embed = None

    def _ensure_text_embeddings(self) -> None:
        """Compute and cache text embeddings for current category."""
        if self._backbone is None:
            raise RuntimeError("Backbone not set. Call set_backbone() first.")

        device = next(self.parameters()).device

        if self._normal_embed is None:
            # Compute normal text embeddings
            normal_texts = [p.format(self.category) for p in self.normal_prompts]
            with torch.no_grad():
                normal_embeds = self._backbone.encode_text(normal_texts)
                normal_embeds = normal_embeds / normal_embeds.norm(dim=-1, keepdim=True)
                self._normal_embed = normal_embeds.mean(dim=0)
                self._normal_embed = self._normal_embed / self._normal_embed.norm()
                self._normal_embed = self._normal_embed.to(device)

        if self._abnormal_embed is None:
            # Compute abnormal text embeddings
            abnormal_texts = [p.format(self.category) for p in self.abnormal_prompts]
            with torch.no_grad():
                abnormal_embeds = self._backbone.encode_text(abnormal_texts)
                abnormal_embeds = abnormal_embeds / abnormal_embeds.norm(
                    dim=-1, keepdim=True
                )
                self._abnormal_embed = abnormal_embeds.mean(dim=0)
                self._abnormal_embed = (
                    self._abnormal_embed / self._abnormal_embed.norm()
                )
                self._abnormal_embed = self._abnormal_embed.to(device)

        if self._stateless_embed is None:
            # Compute stateless text embedding
            stateless_text = [self.stateless_prompt.format(self.category)]
            with torch.no_grad():
                stateless_embed = self._backbone.encode_text(stateless_text)
                self._stateless_embed = stateless_embed.squeeze(0)
                self._stateless_embed = (
                    self._stateless_embed / self._stateless_embed.norm()
                )
                self._stateless_embed = self._stateless_embed.to(device)

    def fit(self, features: list[torch.Tensor]) -> None:
        """
        AFR-CLIP is zero-shot, so fit() is a no-op for the core method.
        However, we can optionally fine-tune adapters on auxiliary data.
        """
        pass

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute anomaly scores using AFR-CLIP.

        Args:
            features: List of feature tensors from CLIP backbone.
                     Expected shape per tensor: (B, D, h, w)

        Returns:
            Dictionary with:
                - anomaly_score: (B,) image-level anomaly scores
                - anomaly_map: (B, H, W) pixel-level anomaly map
        """
        self._ensure_text_embeddings()

        # Use last layer features (or aggregate multiple scales)
        if len(features) == 1:
            feat = features[0]  # (B, D, h, w)
        else:
            # Multi-scale: average features (resize if needed)
            target_h, target_w = features[-1].shape[2:]
            resized = []
            for f in features:
                if f.shape[2:] != (target_h, target_w):
                    f = F.interpolate(
                        f,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                resized.append(f)
            feat = torch.stack(resized, dim=0).mean(dim=0)

        B, D, h, w = feat.shape
        N = h * w

        # Reshape to (B, N, D)
        visual_feat = feat.permute(0, 2, 3, 1).reshape(B, N, D)

        # Project visual features to text dimension if needed (CLIP ViT-L/14: 1024 -> 768)
        if self._visual_to_text_proj is not None:
            visual_feat = self._visual_to_text_proj(visual_feat)

        # Apply visual adapter
        visual_feat = self.visual_adapter(visual_feat)

        # Apply MPFA if enabled
        if self.use_mpfa:
            visual_feat = self.mpfa(visual_feat, h, w)

        # Apply text adapter to stateless embedding
        stateless = self.text_adapter(self._stateless_embed)  # (D,)

        # Cross-modal feature rectification
        rectified = self.cmfr(visual_feat, stateless)  # (B, N, D)

        # Normalize all embeddings
        rectified = rectified / rectified.norm(dim=-1, keepdim=True)
        normal_embed = self._normal_embed / self._normal_embed.norm()  # (D,)
        abnormal_embed = self._abnormal_embed / self._abnormal_embed.norm()  # (D,)

        # Text-on-text similarity scoring
        # For each patch, compute similarity with normal and abnormal prototypes
        sim_normal = torch.einsum("bnd,d->bn", rectified, normal_embed)  # (B, N)
        sim_abnormal = torch.einsum("bnd,d->bn", rectified, abnormal_embed)  # (B, N)

        # Apply temperature and softmax to get anomaly probability
        sim_stack = torch.stack([sim_normal, sim_abnormal], dim=-1)  # (B, N, 2)
        probs = F.softmax(sim_stack / self.temperature, dim=-1)
        anomaly_probs = probs[..., 1]  # (B, N) - probability of abnormal

        # Image-level score: use CLS token (first patch) or max/mean
        # Following paper, use mean of all patches for image-level score
        image_score = anomaly_probs.mean(dim=1)  # (B,)

        # Pixel-level anomaly map
        anomaly_map = anomaly_probs.view(B, h, w)  # (B, h, w)

        # Upsample to original resolution
        target_size = self.image_size
        anomaly_map_upsampled = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        return {
            "anomaly_score": image_score,
            "anomaly_map": anomaly_map_upsampled,
            "anomaly_map_raw": anomaly_map,
        }


class AFRCLIPTrainer:
    """
    Trainer for AFR-CLIP with auxiliary dataset.

    While AFR-CLIP is zero-shot capable, training on auxiliary data
    (as described in the paper) improves performance.
    """

    def __init__(
        self,
        head: AFRCLIPHead,
        backbone: Any,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ) -> None:
        self.head = head
        self.backbone = backbone
        self.device = device
        self.lr = learning_rate

        # Set backbone reference
        head.set_backbone(backbone)

        # Optimizer for trainable parameters (adapters and CMFR)
        self.optimizer = torch.optim.Adam(
            [
                {"params": head.cmfr.parameters()},
                {"params": head.visual_adapter.parameters()},
                {"params": head.text_adapter.parameters()},
            ],
            lr=learning_rate,
        )

    def train_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        category: str,
    ) -> float:
        """
        Single training step on auxiliary data.

        Args:
            images: (B, C, H, W) input images
            masks: (B, H, W) ground truth masks
            category: Object category name

        Returns:
            Loss value
        """
        self.head.set_category(category)
        self.head.train()

        # Extract features
        with torch.no_grad():
            features = self.backbone(images)

        # Forward pass
        output = self.head(features)
        pred_map = output["anomaly_map"]

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(pred_map, masks.float())

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
