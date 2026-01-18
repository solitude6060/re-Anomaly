"""
MambaAD Head for re-Anomaly.

MambaAD (NeurIPS 2024) achieves SOTA on multi-class anomaly detection using:
- State Space Models (SSM) for efficient long-range modeling
- Locality-Enhanced State Space (LSS) modules at multi-scales
- Hybrid Scanning (HS) with 5 scan types and 8 directions
- Multi-kernel convolutions for local feature capture

Reference: https://github.com/lewandofskee/MambaAD
Paper: "MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection"
arXiv: 2404.06564
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_

from .base import BaseHead

# Optional mamba-ssm import - fallback to pure PyTorch implementation
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    selective_scan_fn = None

# Optional hilbert/zorder imports for advanced scanning
try:
    from hilbert import decode, encode

    HILBERT_AVAILABLE = True
except ImportError:
    HILBERT_AVAILABLE = False

try:
    from pyzorder import ZOrderIndexer

    ZORDER_AVAILABLE = True
except ImportError:
    ZORDER_AVAILABLE = False


class HScan(nn.Module):
    """
    Hybrid Scanning module for converting 2D feature maps to 1D sequences.

    Supports multiple scan types:
    - sweep: Simple row-major order
    - scan: Serpentine (snake) scanning
    - zorder: Z-order (Morton) curve
    - zigzag: Diagonal zigzag pattern
    - hilbert: Hilbert space-filling curve
    """

    def __init__(
        self,
        size: int = 16,
        dim: int = 2,
        scan_type: str = "scan",
    ) -> None:
        super().__init__()
        size = int(size)
        max_num = size**dim
        indexes = np.arange(max_num)

        if scan_type == "sweep":
            locs_flat = indexes
        elif scan_type == "scan":
            # Serpentine scanning
            indexes_2d = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes_2d[i, :] = indexes_2d[i, :][::-1]
            locs_flat = indexes_2d.reshape(-1)
        elif scan_type == "zorder":
            if not ZORDER_AVAILABLE:
                # Fallback to sweep if pyzorder not available
                locs_flat = indexes
            else:
                zi = ZOrderIndexer((0, size - 1), (0, size - 1))
                locs_flat = []
                for z in indexes:
                    r, c = zi.rc(int(z))
                    locs_flat.append(c * size + r)
                locs_flat = np.array(locs_flat)
        elif scan_type == "zigzag":
            indexes_2d = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes_2d[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes_2d[j, i - j])
            locs_flat = np.array(locs_flat)
        elif scan_type == "hilbert":
            if not HILBERT_AVAILABLE:
                # Fallback to scan if hilbert not available
                indexes_2d = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes_2d[i, :] = indexes_2d[i, :][::-1]
                locs_flat = indexes_2d.reshape(-1)
            else:
                bit = int(math.log2(size))
                locs = decode(indexes, dim, bit)
                locs_flat = self._flat_locs_hilbert(locs, dim, bit)
        else:
            raise ValueError(f"Invalid scan_type: {scan_type}")

        locs_flat = np.array(locs_flat).astype(np.int64)
        locs_flat_inv = np.argsort(locs_flat)

        index_flat = torch.LongTensor(locs_flat).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv).unsqueeze(0).unsqueeze(1)

        self.register_buffer("index_flat", index_flat)
        self.register_buffer("index_flat_inv", index_flat_inv)

    def _flat_locs_hilbert(
        self, locs: np.ndarray, num_dim: int, num_bit: int
    ) -> np.ndarray:
        ret = []
        length = 2**num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (length**j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode 2D features to 1D sequence using scan pattern."""
        return torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(
            2, self.index_flat_inv.expand(img.shape), img
        )

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        """Decode 1D sequence back to 2D features."""
        return torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(
            2, self.index_flat.expand(img.shape), img
        )


class SS2D(nn.Module):
    """
    2D Selective State Space module with multi-directional scanning.

    This is the core SSM module that processes 2D feature maps using
    multiple scan directions for comprehensive spatial modeling.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.num_direction = num_direction

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Multi-directional projections
        x_proj_weights = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False).weight
            for _ in range(self.num_direction)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weights, dim=0))

        dt_projs = [
            self._init_dt_proj(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
            )
            for _ in range(self.num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([p.weight for p in dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([p.bias for p in dt_projs], dim=0)
        )

        # SSM parameters
        self.A_logs = self._init_A_log(
            self.d_state, self.d_inner, copies=self.num_direction
        )
        self.Ds = self._init_D(self.d_inner, copies=self.num_direction)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Scanning module
        self.scans = HScan(size=size, scan_type=scan_type)

    @staticmethod
    def _init_dt_proj(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale

        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Invalid dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def _init_A_log(d_state: int, d_inner: int, copies: int = 1) -> nn.Parameter:
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # type: ignore
        return A_log

    @staticmethod
    def _init_D(d_inner: int, copies: int = 1) -> nn.Parameter:
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True  # type: ignore
        return D

    def _forward_core_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """Core SSM computation using mamba-ssm library."""
        B, C, H, W = x.shape
        L = H * W
        K = self.num_direction

        # Multi-directional scanning
        xs = []
        if K >= 2:
            xs.append(self.scans.encode(x.view(B, -1, L)))
        if K >= 4:
            xs.append(
                self.scans.encode(
                    torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
                )
            )
        if K >= 8:
            xs.append(
                self.scans.encode(
                    torch.rot90(x, k=1, dims=(2, 3)).contiguous().view(B, -1, L)
                )
            )
            xs.append(
                self.scans.encode(
                    torch.transpose(torch.rot90(x, k=1, dims=(2, 3)), dim0=2, dim1=3)
                    .contiguous()
                    .view(B, -1, L)
                )
            )

        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        # Compute SSM parameters
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # Selective scan
        assert selective_scan_fn is not None
        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # Inverse scanning
        inv_y = torch.flip(out_y[:, K // 2 : K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        if K >= 2:
            ys.append(self.scans.decode(out_y[:, 0]))
            ys.append(self.scans.decode(inv_y[:, 0]))
        if K >= 4:
            ys.append(
                torch.transpose(
                    self.scans.decode(out_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3
                )
                .contiguous()
                .view(B, -1, L)
            )
            ys.append(
                torch.transpose(
                    self.scans.decode(inv_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3
                )
                .contiguous()
                .view(B, -1, L)
            )
        if K >= 8:
            ys.append(
                torch.rot90(
                    self.scans.decode(out_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)
                )
                .contiguous()
                .view(B, -1, L)
            )
            ys.append(
                torch.rot90(
                    self.scans.decode(inv_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)
                )
                .contiguous()
                .view(B, -1, L)
            )
            ys.append(
                torch.rot90(
                    torch.transpose(
                        self.scans.decode(out_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3
                    ),
                    k=3,
                    dims=(2, 3),
                )
                .contiguous()
                .view(B, -1, L)
            )
            ys.append(
                torch.rot90(
                    torch.transpose(
                        self.scans.decode(inv_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3
                    ),
                    k=3,
                    dims=(2, 3),
                )
                .contiguous()
                .view(B, -1, L)
            )

        return sum(ys)  # type: ignore

    def _forward_core_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch fallback when mamba-ssm is not available.

        Uses a simplified approach with:
        1. Multi-directional 1D convolutions to capture sequence patterns
        2. Depthwise separable processing for efficiency
        """
        B, C, H, W = x.shape
        L = H * W

        # Flatten to sequence
        x_flat = x.view(B, C, L)

        # Multi-directional processing (simulating SSM scanning)
        outputs = []

        # Direction 1: forward scan (row-major)
        outputs.append(x_flat)

        # Direction 2: backward scan
        outputs.append(torch.flip(x_flat, dims=[-1]))

        # Direction 3: column-major (transpose then flatten)
        x_col = x.permute(0, 1, 3, 2).contiguous().view(B, C, L)
        outputs.append(x_col)

        # Direction 4: reverse column-major
        outputs.append(torch.flip(x_col, dims=[-1]))

        # Apply 1D convolution for temporal modeling
        conv_out = []
        for seq in outputs:
            # Simple causal-like convolution
            kernel_size = 3
            padded = F.pad(seq, (kernel_size - 1, 0))
            smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
            conv_out.append(smoothed)

        # Average all directions
        y = sum(conv_out) / len(conv_out)

        return y

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))

        if MAMBA_SSM_AVAILABLE:
            y = self._forward_core_mamba(x_in)
        else:
            y = self._forward_core_pytorch(x_in)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)

        return out


class HSSBlock(nn.Module):
    """Hybrid State Space Block with residual connection."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            size=size,
            scan_type=scan_type,
            num_direction=num_direction,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class LSSModule(nn.Module):
    """
    Locality-Enhanced State Space Module.

    Combines SSM blocks with multi-kernel convolutions for
    both long-range and local feature modeling.
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        depth: int = 2,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # SSM blocks
        self.ssm_blocks = nn.ModuleList(
            [
                HSSBlock(
                    hidden_dim=hidden_dim,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop_rate,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    num_direction=num_direction,
                    **kwargs,
                )
                for _ in range(depth)
            ]
        )

        # Multi-kernel convolutions for local features
        self.conv1b7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv77 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                padding=3,
                groups=hidden_dim,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )

        self.conv1b5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv55 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                padding=2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )

        # Fusion
        self.final_conv = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM processing
        out_ssm = x
        for blk in self.ssm_blocks:
            out_ssm = blk(out_ssm)

        # Local convolution processing
        x_conv = x.permute(0, 3, 1, 2).contiguous()
        out_77 = self.conv1a7(self.conv77(self.conv1b7(x_conv)))
        out_55 = self.conv1a5(self.conv55(self.conv1b5(x_conv)))

        # Fusion
        out_ssm_conv = out_ssm.permute(0, 3, 1, 2).contiguous()
        output = torch.cat([out_ssm_conv, out_55, out_77], dim=1)
        output = self.final_conv(output).permute(0, 2, 3, 1).contiguous()

        return output + x


class PatchExpand2D(nn.Module):
    """Patch expansion for upsampling in decoder."""

    def __init__(
        self, dim: int, dim_scale: int = 2, norm_layer: type[nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)
        return x


class LSSLayer(nn.Module):
    """Single layer of LSS modules with optional upsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        upsample: type[nn.Module] | None = None,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = "scan",
        num_direction: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dim = dim

        # Build LSS modules based on depth
        num_modules = max(1, depth // 2)
        module_depth = max(2, depth // num_modules)

        self.blocks = nn.ModuleList(
            [
                LSSModule(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=partial(norm_layer, eps=1e-6),
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=module_depth,
                    num_direction=num_direction,
                )
                for i in range(num_modules)
            ]
        )

        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)

        for blk in self.blocks:
            x = blk(x)

        return x


class MambaDecoder(nn.Module):
    """
    Mamba-based U-Net decoder for anomaly detection.

    Multi-scale decoder using LSS modules for reconstruction.
    """

    def __init__(
        self,
        dims_decoder: list[int] = [512, 256, 128, 64],
        depths_decoder: list[int] = [3, 4, 6, 3],
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        scan_type: str = "scan",
        num_direction: int = 4,
    ) -> None:
        super().__init__()

        # Stochastic depth decay
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))
        ][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = LSSLayer(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths_decoder[:i_layer]) : sum(depths_decoder[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                size=8 * (2**i_layer),
                scan_type=scan_type,
                num_direction=num_direction,
            )
            self.layers.append(layer)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = rearrange(x, "b c h w -> b h w c")
        out_features = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Include all features for comparison with encoder
            out_features.insert(0, rearrange(x, "b h w c -> b c h w"))

        return out_features


class MFFModule(nn.Module):
    """
    Multi-scale Feature Fusion with Output Channel Enhancement.

    Fuses features from different scales before passing to decoder.
    """

    def __init__(
        self,
        in_channels: list[int] = [64, 128, 256],
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        # Projection layers for each input scale
        self.proj_layers = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            stride = 2 ** (len(in_channels) - 1 - i)
            if stride > 1:
                proj = nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                proj = nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            self.proj_layers.append(proj)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = []
        target_size = None

        for i, (feat, proj) in enumerate(zip(features, self.proj_layers)):
            p = proj(feat)
            if target_size is None:
                target_size = p.shape[2:]
            else:
                p = F.interpolate(
                    p, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        return self.fusion(fused)


class MambaADHead(BaseHead):
    """
    MambaAD detection head for multi-class unsupervised anomaly detection.

    Uses Mamba (State Space Model) based decoder with:
    - Locality-Enhanced State Space (LSS) modules
    - Hybrid scanning for comprehensive spatial modeling
    - Multi-scale feature comparison for anomaly detection

    Config:
        embed_dim: int = 1024 - Feature dimension from backbone
        decoder_dims: list[int] = [512, 256, 128, 64] - Decoder channel dimensions
        decoder_depths: list[int] = [3, 4, 6, 3] - Depth of each decoder stage
        d_state: int = 16 - State dimension for SSM
        scan_type: str = "scan" - Scanning pattern ("sweep", "scan", "zorder", "zigzag", "hilbert")
        num_direction: int = 4 - Number of scan directions (2, 4, or 8)
        drop_path_rate: float = 0.2 - Stochastic depth rate
        image_size: int = 224 - Input image size
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.embed_dim = config.get("embed_dim", 1024)
        self.decoder_dims = config.get("decoder_dims", [512, 256, 128, 64])
        self.decoder_depths = config.get("decoder_depths", [3, 4, 6, 3])
        self.d_state = config.get("d_state", 16)
        self.scan_type = config.get("scan_type", "scan")
        self.num_direction = config.get("num_direction", 4)
        self.drop_path_rate = config.get("drop_path_rate", 0.2)
        self.image_size = config.get("image_size", 224)

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.decoder_dims[0], kernel_size=1),
            nn.BatchNorm2d(self.decoder_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Mamba decoder
        self.decoder = MambaDecoder(
            dims_decoder=self.decoder_dims,
            depths_decoder=self.decoder_depths,
            d_state=self.d_state,
            drop_path_rate=self.drop_path_rate,
            scan_type=self.scan_type,
            num_direction=self.num_direction,
        )

        # Anomaly map projection
        self.anomaly_proj = nn.Conv2d(self.decoder_dims[-1], 1, kernel_size=1)

        # Store encoder features for comparison
        self._encoder_features: list[torch.Tensor] | None = None

    def fit(self, features: list[torch.Tensor]) -> None:
        """No fitting required - MambaAD is trained end-to-end."""
        pass

    def set_encoder_features(self, features: list[torch.Tensor]) -> None:
        """Store encoder features for anomaly comparison."""
        self._encoder_features = [f.detach() for f in features]

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for anomaly detection.

        Args:
            features: List of multi-scale features from backbone

        Returns:
            Dictionary containing:
                - anomaly_map: (B, H, W) pixel-level anomaly scores
                - anomaly_score: (B,) image-level anomaly scores
                - decoder_features: List of decoder features for training
        """
        # Use last feature map as input
        x = features[-1]  # (B, C, H, W)

        # Project to decoder dimension
        x = self.input_proj(x)

        # Decode
        decoder_features = self.decoder(x)

        # Get anomaly map from final decoder output
        if decoder_features:
            final_feat = decoder_features[0]  # Highest resolution
            anomaly_map = self.anomaly_proj(final_feat).squeeze(1)  # (B, H, W)
        else:
            # Fallback
            anomaly_map = torch.zeros(
                x.shape[0], self.image_size, self.image_size, device=x.device
            )

        # Resize to input size
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Image-level score
        anomaly_score = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1)[0]

        return {
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score,
            "decoder_features": decoder_features,
        }

    def compute_loss(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        hard_mining_percent: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between encoder and decoder features.

        Uses cosine similarity with hard negative mining.
        """
        cos_sim_fn = nn.CosineSimilarity(dim=-1)
        device = encoder_features[0].device
        loss = None

        # Match features by resolution
        for en in encoder_features:
            # Find decoder feature with matching resolution
            en_size = en.shape[2:]
            matched_de = None
            for de in decoder_features:
                if de.shape[2:] == en_size:
                    matched_de = de
                    break

            if matched_de is None:
                continue

            # Resize if channels don't match
            if en.shape[1] != matched_de.shape[1]:
                matched_de = F.conv2d(
                    matched_de,
                    torch.randn(
                        en.shape[1], matched_de.shape[1], 1, 1, device=en.device
                    )
                    / math.sqrt(matched_de.shape[1]),
                )

            # Compute cosine similarity loss with hard mining
            en_flat = en.permute(0, 2, 3, 1).reshape(-1, en.shape[1])
            de_flat = matched_de.permute(0, 2, 3, 1).reshape(-1, matched_de.shape[1])

            with torch.no_grad():
                point_dist = 1 - cos_sim_fn(en_flat.detach(), de_flat.detach())
                k = int(point_dist.numel() * (1 - hard_mining_percent))
                k = max(1, k)
                thresh = torch.topk(point_dist, k=k)[0][-1]
                hard_mask = point_dist >= thresh

            cos_loss = 1 - cos_sim_fn(en_flat.detach(), de_flat)
            cos_loss = (cos_loss * hard_mask.float()).sum() / hard_mask.float().sum()

            if loss is None:
                loss = cos_loss
            else:
                loss = loss + cos_loss

        if loss is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / max(1, len(encoder_features))
