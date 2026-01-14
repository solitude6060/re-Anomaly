"""
Dinomaly Head for re-Anomaly.

Dinomaly (CVPR 2025) achieves 99.6% AUROC on MVTec AD using:
- LinearAttention2 decoder that "cannot focus" (uses ELU instead of softmax)
- Dropout-based bottleneck for noise injection
- Cosine similarity loss for loose reconstruction
- Multi-scale feature comparison between encoder and decoder

Reference: https://github.com/caoyunkang/Dinomaly
Paper: "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection"
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """MLP module for transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearAttention2(nn.Module):
    """
    Linear Attention that "cannot focus" - key innovation from Dinomaly.

    Uses ELU + 1 activation instead of softmax, which prevents the decoder
    from focusing on specific regions. This forces reconstruction to rely
    on global patterns rather than copying local details.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Key innovation: ELU + 1 instead of softmax
        # This prevents attention from "focusing" on specific positions
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Efficient linear attention computation O(N) instead of O(N^2)
        kv = torch.einsum("...sd,...se->...de", k, v)
        z = 1.0 / torch.einsum("...sd,...d->...s", q, k.sum(dim=-2))
        x = torch.einsum("...de,...sd,...s->...se", kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DecoderBlock(nn.Module):
    """Transformer decoder block with LinearAttention2."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearAttention2(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DinomalyHead(BaseHead):
    """
    Dinomaly anomaly detection head.

    Architecture:
    1. Takes frozen encoder features from DINOv2/v3
    2. Passes through dropout bottleneck (noise injection)
    3. Reconstructs via LinearAttention2 decoder
    4. Computes anomaly score as 1 - cosine_similarity(encoder, decoder)

    Key insights:
    - LinearAttention2 "cannot focus" due to ELU activation
    - Forces decoder to learn global normal patterns
    - Anomalous regions cannot be reconstructed well
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Model dimensions (will be set during fit based on backbone)
        self.embed_dim: int = config.get("embed_dim", 1024)  # DINOv3-L default
        self.num_heads: int = config.get("num_heads", 16)
        self.decoder_depth: int = config.get("decoder_depth", 8)
        self.mlp_ratio: float = config.get("mlp_ratio", 4.0)

        # Dropout bottleneck for noise injection
        self.bottleneck_drop: float = config.get("bottleneck_drop", 0.0)

        # Target encoder layers to use for reconstruction comparison
        self.target_layers: list[int] = config.get(
            "target_layers", [2, 3, 4, 5, 6, 7, 8, 9]
        )

        # Feature fusion config - which layers to compare
        # Default: compare all 8 target layers
        self.fuse_layer_encoder: list[list[int]] = config.get(
            "fuse_layer_encoder", [[0, 1, 2, 3, 4, 5, 6, 7]]
        )
        self.fuse_layer_decoder: list[list[int]] = config.get(
            "fuse_layer_decoder", [[0, 1, 2, 3, 4, 5, 6, 7]]
        )

        # Neighbor masking to prevent local copying
        self.mask_neighbor_size: int = config.get("mask_neighbor_size", 0)

        # Whether to remove CLS token before processing
        self.remove_class_token: bool = config.get("remove_class_token", False)

        # Training config
        self.drop_rate: float = config.get("drop_rate", 0.0)
        self.attn_drop_rate: float = config.get("attn_drop_rate", 0.0)
        self.drop_path_rate: float = config.get("drop_path_rate", 0.0)

        # Number of register tokens (DINOv3 has 4)
        self.num_register_tokens: int = config.get("num_register_tokens", 4)

        # Build decoder
        self.bottleneck: nn.Module | None = None
        self.decoder: nn.ModuleList | None = None
        self._initialized = False

        # Feature normalization for anomaly scoring
        self.normalize_features: bool = config.get("normalize_features", True)

        # Image size for computing spatial dimensions
        self.image_size: int = config.get("image_size", 448)
        self.patch_size: int = config.get("patch_size", 16)

    def _initialize_decoder(self, embed_dim: int) -> None:
        """Initialize decoder after knowing the embed_dim from backbone."""
        if self._initialized and self.embed_dim == embed_dim:
            return

        self.embed_dim = embed_dim

        if self.bottleneck_drop > 0:
            self.bottleneck = nn.Sequential(
                nn.Dropout(self.bottleneck_drop),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(self.bottleneck_drop),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(self.bottleneck_drop),
            )
        else:
            self.bottleneck = nn.Identity()

        # Decoder: LinearAttention2 transformer blocks
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(self.decoder_depth)
            ]
        )

        self._initialized = True

    def fit(self, features: list[torch.Tensor]) -> None:
        """
        Dinomaly is trained via gradient descent, not memory bank fitting.
        This method initializes the decoder based on backbone feature dimensions.
        """
        if len(features) == 0:
            return

        # Get embed_dim from first feature
        # features[0] shape: (B, C, H, W)
        embed_dim = features[0].shape[1]
        self._initialize_decoder(embed_dim)

        # Move to same device
        device = features[0].device
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)
        if isinstance(self.bottleneck, nn.Module):
            self.bottleneck = self.bottleneck.to(device)

    def _fuse_features(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse features by averaging along the layer dimension."""
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _generate_neighbor_mask(
        self, feature_size: int, device: torch.device
    ) -> torch.Tensor:
        """Generate mask to prevent local copying."""
        h = w = feature_size
        hm = wm = self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)

        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0

        mask = mask.view(h * w, h * w)

        if self.remove_class_token:
            return mask

        # Add CLS and register tokens to mask
        total_tokens = h * w + 1 + self.num_register_tokens
        mask_all = torch.ones(total_tokens, total_tokens, device=device)
        mask_all[1 + self.num_register_tokens :, 1 + self.num_register_tokens :] = mask

        return mask_all

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for inference.

        Args:
            features: List of feature maps from backbone, each (B, C, H, W)

        Returns:
            Dictionary with 'anomaly_score' and 'anomaly_map'
        """
        if not self._initialized:
            self.fit(features)

        # Ensure we're on the right device
        device = features[0].device
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)
        if isinstance(self.bottleneck, nn.Module):
            self.bottleneck = self.bottleneck.to(device)

        batch_size = features[0].shape[0]
        h = w = features[0].shape[-1]  # Spatial size

        # Convert features from (B, C, H, W) to (B, H*W, C) for transformer
        # We use the input features as "encoder features"
        en_list = []
        for feat in features:
            # feat: (B, C, H, W)
            b, c, fh, fw = feat.shape
            # Reshape to (B, H*W, C)
            feat_seq = feat.permute(0, 2, 3, 1).reshape(b, fh * fw, c)
            en_list.append(feat_seq)

        # Fuse encoder features
        x = self._fuse_features(en_list)  # (B, H*W, C)

        # Apply bottleneck (dropout)
        assert self.bottleneck is not None
        x = self.bottleneck(x)

        # Generate attention mask if needed
        attn_mask = None
        if self.mask_neighbor_size > 0:
            attn_mask = self._generate_neighbor_mask(h, device)

        # Pass through decoder
        assert self.decoder is not None
        de_list = []
        for blk in self.decoder:
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)

        # Reverse decoder list for multi-scale comparison
        de_list = de_list[::-1]

        # Fuse features for comparison
        # Use the configured layer indices for encoder and decoder
        en_fused = []
        for layer_indices in self.fuse_layer_encoder:
            valid_indices = [i for i in layer_indices if i < len(en_list)]
            if valid_indices:
                fused = self._fuse_features([en_list[i] for i in valid_indices])
                en_fused.append(fused)

        de_fused = []
        for layer_indices in self.fuse_layer_decoder:
            valid_indices = [i for i in layer_indices if i < len(de_list)]
            if valid_indices:
                fused = self._fuse_features([de_list[i] for i in valid_indices])
                de_fused.append(fused)

        # If we don't have proper fused features, fall back to simple comparison
        if not en_fused:
            en_fused = [en_list[0]]
        if not de_fused:
            de_fused = [de_list[0]]

        # Compute anomaly score: 1 - cosine_similarity
        anomaly_maps = []
        for en, de in zip(en_fused, de_fused):
            # Normalize for cosine similarity
            if self.normalize_features:
                en = F.normalize(en, p=2, dim=-1)
                de = F.normalize(de, p=2, dim=-1)

            # Cosine similarity per position: (B, H*W)
            cos_sim = (en * de).sum(dim=-1)
            # Anomaly score = 1 - similarity
            anomaly = 1 - cos_sim
            # Reshape to spatial: (B, H, W)
            anomaly = anomaly.reshape(batch_size, h, w)
            anomaly_maps.append(anomaly)

        # Average anomaly maps from different scales
        anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)  # (B, H, W)

        # Image-level anomaly score: max over spatial dimensions
        anomaly_score = anomaly_map.amax(dim=(1, 2))  # (B,)

        return {"anomaly_score": anomaly_score, "anomaly_map": anomaly_map}

    def compute_loss(
        self, en_features: list[torch.Tensor], de_features: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for training.

        Args:
            en_features: Encoder features, list of (B, C, H, W)
            de_features: Decoder features, list of (B, C, H, W)

        Returns:
            Cosine similarity loss
        """
        loss = torch.tensor(0.0, device=en_features[0].device)

        for en, de in zip(en_features, de_features):
            # Reshape to (B, C, -1) for normalization
            en = en.flatten(2)  # (B, C, H*W)
            de = de.flatten(2)

            # Normalize
            en = F.normalize(en, p=2, dim=1)
            de = F.normalize(de, p=2, dim=1)

            # Cosine similarity: (B, H*W)
            cos_sim = (en * de).sum(dim=1)

            # Loss = 1 - cos_sim
            loss = loss + (1 - cos_sim).mean()

        return loss / len(en_features)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get parameters that should be trained."""
        params = []
        if self.decoder is not None:
            params.extend(self.decoder.parameters())
        if isinstance(self.bottleneck, nn.Module) and not isinstance(
            self.bottleneck, nn.Identity
        ):
            params.extend(self.bottleneck.parameters())
        return params

    def compute_training_loss(self, features: list[torch.Tensor]) -> torch.Tensor:
        if not self._initialized:
            self.fit(features)

        device = features[0].device
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)
        if isinstance(self.bottleneck, nn.Module):
            self.bottleneck = self.bottleneck.to(device)

        batch_size = features[0].shape[0]
        h = w = features[0].shape[-1]

        en_list = []
        for feat in features:
            b, c, fh, fw = feat.shape
            feat_seq = feat.permute(0, 2, 3, 1).reshape(b, fh * fw, c)
            en_list.append(feat_seq)

        x = self._fuse_features(en_list)

        assert self.bottleneck is not None
        x = self.bottleneck(x)

        attn_mask = None
        if self.mask_neighbor_size > 0:
            attn_mask = self._generate_neighbor_mask(h, device)

        assert self.decoder is not None
        de_list = []
        for blk in self.decoder:
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)

        de_list = de_list[::-1]

        en_fused = []
        for layer_indices in self.fuse_layer_encoder:
            valid_indices = [i for i in layer_indices if i < len(en_list)]
            if valid_indices:
                fused = self._fuse_features([en_list[i] for i in valid_indices])
                en_fused.append(fused)

        de_fused = []
        for layer_indices in self.fuse_layer_decoder:
            valid_indices = [i for i in layer_indices if i < len(de_list)]
            if valid_indices:
                fused = self._fuse_features([de_list[i] for i in valid_indices])
                de_fused.append(fused)

        if not en_fused:
            en_fused = [en_list[0]]
        if not de_fused:
            de_fused = [de_list[0]]

        return self._compute_hard_mining_loss(en_fused, de_fused)

    def _compute_hard_mining_loss(
        self,
        en_features: list[torch.Tensor],
        de_features: list[torch.Tensor],
        hard_mining_percent: float = 0.9,
        grad_factor: float = 0.1,
    ) -> torch.Tensor:
        cos_sim_fn = nn.CosineSimilarity(dim=-1)
        loss = torch.tensor(0.0, device=en_features[0].device)

        for en, de in zip(en_features, de_features):
            en_detached = en.detach()

            with torch.no_grad():
                point_dist = 1 - cos_sim_fn(en_detached, de)
                k = int(point_dist.numel() * (1 - hard_mining_percent))
                k = max(1, k)
                thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]
                hard_mask = point_dist >= thresh

            en_flat = en_detached.reshape(en_detached.shape[0], -1)
            de_flat = de.reshape(de.shape[0], -1)
            cos_loss = 1 - cos_sim_fn(en_flat, de_flat)
            loss = loss + cos_loss.mean()

            if self.training and de.requires_grad:

                def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    mask = hard_mask.unsqueeze(-1).expand_as(grad)
                    grad_modified = grad.clone()
                    grad_modified[~mask] = grad_modified[~mask] * grad_factor
                    return grad_modified

                if not hasattr(de, "_has_hook"):
                    de.register_hook(grad_hook)
                    de._has_hook = True

        return loss / len(en_features)
