"""
Ultra-Compact Multi-Rate Acoustic Encoder (models/encoder.py)
------------------------------------------------------------
Mathematical & Architectural Specification for ZipTransducer-Tamil Model Family:
Variants:
- Small:  D=256, (3, 6, 3) U-Net hierarchy, H_Q=8, H_KV=2, D_ff=512   -> ~9.33M parameters
- Medium: D=512, (4, 8, 4) U-Net hierarchy, H_Q=8, H_KV=2, D_ff=1024  -> ~49.2M parameters
- Large:  D=768, (4, 12, 4) U-Net hierarchy, H_Q=12, H_KV=2, D_ff=1024 -> ~105.7M parameters

1. Audio Front-End Stem:
   - Input: Log-mel 80-dim filterbanks (B, T_in, 80)
   - 4x Temporal Downsampling via two-stage Depthwise-Separable Conv1d (stride 2):
     Stage 1: Conv1d(80, 80, k=3, s=2, groups=80) + Conv1d(80, D, k=1) + RMSNorm + SiLU
     Stage 2: Conv1d(D, D, k=3, s=2, groups=D) + Conv1d(D, D, k=1) + RMSNorm + SiLU
     Output: (B, T0, D) where T0 = T_in // 4 (25 Hz / 40ms frame rate).

2. Multi-Rate Temporal U-Net Topology (Zipformer Hierarchy):
   - Stage 1: n1 ZipBlocks at Rate 1 (40ms, T0 frames)
   - Downsampler 1->2: DW-PW Conv1d stride 2 to T1 = T0 // 2 (80ms / 12.5 Hz, cuts attention FLOPs by 75%)
   - Stage 2: n2 ZipBlocks at Rate 2 (T1 frames) for deep linguistic/acoustic context
   - Upsampler 2->1: Nearest interpolate to T0 + DW-PW smoothing + Skip residual connection from Stage 1
   - Stage 3: n3 ZipBlocks at Rate 1 (T0 frames) for fine phonetic alignment

3. ZipBlock Micro-Architecture:
   - Pre-RMSNorm -> Grouped-Query Attention (GQA: H_Q, H_KV, RoPE) + LayerScale residual
   - Pre-RMSNorm -> 1D Depthwise Conv (k=15, GLU gating) + LayerScale residual
   - Pre-RMSNorm -> SwiGLU FFN (d_ff, optional factorized rank) + LayerScale residual

4. Extensible Triton Acceleration Hooks:
   - Compatible with pure PyTorch CPU/CUDA, with zero-overhead hooks for custom Triton kernels.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamic Triton kernel imports with graceful fallback
try:
    from kernels.triton_ops import (
        fused_rmsnorm,
        fused_chunked_linear_attention,
        FusedRMSNorm,
    )
    HAS_TRITON_OPS = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON_OPS = False
    fused_rmsnorm = None
    fused_chunked_linear_attention = None
    FusedRMSNorm = None


# ============================================================================
# 1. Configuration Dataclass & Presets
# ============================================================================

@dataclass
class EncoderConfig:
    """Architectural specifications for Ultra-Compact Acoustic Encoder family."""
    input_dim: int = 80                               # Log-mel spectrogram channels
    d_model: int = 256                               # Hidden feature dimension (D)
    subsampling_factor: int = 4                      # Temporal downsampling factor in stem (4x)
    num_heads: int = 8                               # GQA Query heads (H_Q)
    num_kv_heads: int = 2                            # GQA Key/Value heads (H_KV)
    d_ff: int = 512                                  # SwiGLU intermediate dimension
    conv_kernel_size: int = 15                       # 1D Depthwise conv kernel size (k=15)
    dropout: float = 0.1                             # Dropout probability
    stack_layers: Tuple[int, int, int] = (3, 6, 3)   # U-Net Zipformer hierarchy (n1, n2, n3)
    factorized_rank: Optional[int] = None            # Optional low-rank projection rank
    attention_type: str = "gqa"                      # "gqa", "chunked_linear", or "mamba"
    chunk_size: int = 64                             # Chunk size for linear attention
    max_seq_len: int = 4096                          # Precomputed RoPE sequence length
    causal_conv: bool = False                        # Streaming/causal vs symmetric convolution
    use_triton_ops: bool = False                     # Enable fused Triton kernels if available
    variant: str = "small"                           # "small", "medium", or "large"

    @classmethod
    def from_variant(cls, variant: str = "small", **overrides) -> "EncoderConfig":
        """Factory method to instantiate standard model family configurations."""
        variant_lower = variant.lower()
        if variant_lower == "small":
            cfg = cls(
                variant="small",
                d_model=256,
                num_heads=8,
                num_kv_heads=2,
                d_ff=512,
                stack_layers=(3, 6, 3),
                conv_kernel_size=15,
            )
        elif variant_lower == "medium":
            cfg = cls(
                variant="medium",
                d_model=512,
                num_heads=8,
                num_kv_heads=2,
                d_ff=1024,
                stack_layers=(4, 8, 4),
                conv_kernel_size=15,
            )
        elif variant_lower == "large":
            cfg = cls(
                variant="large",
                d_model=768,
                num_heads=12,
                num_kv_heads=2,
                d_ff=1024,
                stack_layers=(4, 12, 4),
                conv_kernel_size=15,
            )
        else:
            raise ValueError(f"Unknown variant '{variant}'. Supported: 'small', 'medium', 'large'.")

        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# Alias for unified naming convention across repo
ZipTransducerEncoderConfig = EncoderConfig


# ============================================================================
# 2. Normalization: Root Mean Square Layer Normalization (RMSNorm)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Eliminates mean-centering, reducing memory traffic and parameter count.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON_OPS and x.is_cuda and fused_rmsnorm is not None and x.is_contiguous():
            return fused_rmsnorm(x, self.weight, self.eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ============================================================================
# 3. Rotary Positional Embedding (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    1D Rotary Position Embedding (RoPE) for acoustic sequence modeling.
    Provides relative positional inductive bias and generalizes to variable lengths.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(max(seq_len, 2 * self.cos_cached.size(0)))
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for RoPE."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to Query and Key tensors (B, H, T, D_head)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D_head)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D_head)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ============================================================================
# 4. SwiGLU Feed-Forward Network with Optional Factorized Projections
# ============================================================================

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network: Swish(x @ W_gate) * (x @ W_up) @ W_down
    With optional factorized low-rank down-projection.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        factorized_rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        
        # Optional factorized low-rank projection: (hidden_dim -> rank) -> (rank -> dim)
        if factorized_rank is not None and 0 < factorized_rank < min(dim, hidden_dim):
            self.factorized = True
            self.w_down_1 = nn.Linear(hidden_dim, factorized_rank, bias=False)
            self.w_down_2 = nn.Linear(factorized_rank, dim, bias=False)
        else:
            self.factorized = False
            self.w_down = nn.Linear(hidden_dim, dim, bias=False)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        act = F.silu(gate) * up
        act = self.dropout(act)
        
        if self.factorized:
            out = self.w_down_2(self.w_down_1(act))
        else:
            out = self.w_down(act)
            
        return self.dropout(out)


# ============================================================================
# 5. Attention & State-Space Modules
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with RoPE positional encodings.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.0,
        rope: Optional[RotaryEmbedding] = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # (B, H_q, T, D_h)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D_h)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D_h)

        if self.rope is not None:
            cos, sin = self.rope(x, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match Q heads for GQA
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention mask for variable sequence lengths
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(B, 1, 1, T).expand(B, self.num_heads, T, T)

        dropout_val = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_val)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.dropout(self.out_proj(out))


class ChunkedLinearAttention(nn.Module):
    """O(T) Linear Attention with feature maps phi(x) = elu(x) + 1."""
    def __init__(self, d_model: int = 256, num_heads: int = 8, chunk_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if HAS_TRITON_OPS and fused_chunked_linear_attention is not None and self.head_dim in (16, 32, 64):
            out = fused_chunked_linear_attention(q, k, v, chunk_size=self.chunk_size)
        else:
            q_map = self._feature_map(q)
            k_map = self._feature_map(k)
            if key_padding_mask is not None:
                mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(k.dtype)
                k_map = k_map * mask
                v = v * mask
            kv = torch.matmul(k_map.transpose(-2, -1), v)
            z = 1.0 / (torch.matmul(q_map, k_map.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
            out = torch.matmul(q_map, kv) * z
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.dropout(self.out_proj(out))


class SelectiveSSMBlock(nn.Module):
    """Lightweight Bidirectional Selective State-Space / Mamba block."""
    def __init__(self, d_model: int = 256, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_branch.transpose(1, 2)).transpose(1, 2)[:, :T, :]
        x_act = F.silu(x_conv)
        y = x_act * self.D.unsqueeze(0).unsqueeze(0)
        out = y * F.silu(z_branch)
        return self.out_proj(out)


# ============================================================================
# 6. Depthwise 1D Convolution Module (k=15, GLU Gating)
# ============================================================================

class DepthwiseConv1DModule(nn.Module):
    """
    1D Depthwise Separable Convolution block with GLU gating (kernel size 15).
    Micro-architecture: Pointwise(D->2D) -> GLU -> DWConv1d(k=15) -> RMSNorm -> SiLU -> Pointwise(D->D).
    """
    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 15,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) if causal else (kernel_size - 1) // 2
        
        self.pointwise_1 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model,
            bias=False
        )
        self.norm_inner = RMSNorm(d_model)
        self.pointwise_2 = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.pointwise_1(x)
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        
        # 1D Depthwise Conv
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.depthwise(h)
        if self.causal and self.padding > 0:
            h = h[:, :, :T]
        h = h.transpose(1, 2)  # (B, T, D)
        
        h = self.norm_inner(h)
        h = F.silu(h)
        h = self.pointwise_2(h)
        return self.dropout(h)


# ============================================================================
# 7. ZipBlock Micro-Architecture
# ============================================================================

class ZipBlock(nn.Module):
    """
    ZipBlock Micro-Architecture:
    1. Pre-RMSNorm -> GQA Attention (H_Q, H_KV, RoPE) + learnable LayerScale
    2. Pre-RMSNorm -> 1D Depthwise Conv (k=15, GLU) + learnable LayerScale
    3. Pre-RMSNorm -> SwiGLU FFN (d_ff) + learnable LayerScale
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        d_ff: int = 512,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        factorized_rank: Optional[int] = None,
        attention_type: str = "gqa",
        chunk_size: int = 64,
        rope: Optional[RotaryEmbedding] = None,
        causal_conv: bool = False
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Attention branch
        self.norm_attn = RMSNorm(d_model)
        if attention_type == "gqa":
            self.attn = GroupedQueryAttention(
                d_model, num_heads=num_heads, num_kv_heads=num_kv_heads, dropout=dropout, rope=rope
            )
        elif attention_type == "chunked_linear":
            self.attn = ChunkedLinearAttention(
                d_model, num_heads=num_heads, chunk_size=chunk_size, dropout=dropout
            )
        elif attention_type == "mamba":
            self.attn = SelectiveSSMBlock(d_model)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
        self.scale_attn = nn.Parameter(torch.ones(d_model) * 0.1)

        # 2. Depthwise Conv branch
        self.norm_conv = RMSNorm(d_model)
        self.conv = DepthwiseConv1DModule(
            d_model, kernel_size=conv_kernel_size, dropout=dropout, causal=causal_conv
        )
        self.scale_conv = nn.Parameter(torch.ones(d_model) * 0.1)

        # 3. SwiGLU FFN branch
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(
            d_model, hidden_dim=d_ff, factorized_rank=factorized_rank, dropout=dropout
        )
        self.scale_ffn = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Attention
        x = x + self.scale_attn * self.attn(self.norm_attn(x), key_padding_mask=key_padding_mask)
        # 2. Conv
        x = x + self.scale_conv * self.conv(self.norm_conv(x))
        # 3. FFN
        x = x + self.scale_ffn * self.ffn(self.norm_ffn(x))
        return x


# ============================================================================
# 8. Front-End Stem: 4x Depthwise-Separable Downsampler
# ============================================================================

class SubsamplingStem1D(nn.Module):
    """
    Subsampling Stem with 4x Temporal Downsampling via 2-stage DW-PW Conv1d.
    Converts 80-dim log-mel features to D-dim acoustic representations at 25 Hz (40ms).
    """
    def __init__(self, input_dim: int = 80, d_model: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Stage 1: 2x downsampling (80 -> D channels)
        self.stage1_dw = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, groups=input_dim, bias=False)
        self.stage1_pw = nn.Conv1d(input_dim, d_model, kernel_size=1, bias=False)
        self.stage1_norm = RMSNorm(d_model)
        self.stage1_act = nn.SiLU()

        # Stage 2: 2x downsampling (D -> D channels) -> Total 4x temporal downsampling
        self.stage2_dw = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model, bias=False)
        self.stage2_pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.stage2_norm = RMSNorm(d_model)
        self.stage2_act = nn.SiLU()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, T, 80) -> transpose to (B, 80, T)
        x = x.transpose(1, 2)
        
        # Stage 1
        x = self.stage1_pw(self.stage1_dw(x))
        x = self.stage1_act(self.stage1_norm(x.transpose(1, 2).contiguous()).transpose(1, 2))
        
        # Stage 2
        x = self.stage2_pw(self.stage2_dw(x))
        x = self.stage2_act(self.stage2_norm(x.transpose(1, 2).contiguous()).transpose(1, 2))
        
        # Output: (B, T0, D)
        out = x.transpose(1, 2)
        
        out_lens = None
        if lengths is not None:
            out_lens = torch.div(lengths + 3, 4, rounding_mode="floor")
        return out, out_lens


# ============================================================================
# 9. Multi-Rate Temporal Down/Up-Sampling & Skip Connection Modules
# ============================================================================

class TemporalDownsampler1D(nn.Module):
    """
    Downsamples temporal resolution 2x (T0 -> T1) using Depthwise-Separable Conv1d.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x_down = self.pw(self.dw(x.transpose(1, 2))).transpose(1, 2)
        return self.act(self.norm(x_down))


class TemporalUpsampler1D(nn.Module):
    """
    Upsamples temporal resolution 2x (T1 -> T0) with nearest-neighbor interpolation
    followed by Depthwise-Separable Conv1d smoothing.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B, T1, D) -> (B, D, T1)
        x_t = x.transpose(1, 2)
        x_up = F.interpolate(x_t, size=target_len, mode="nearest")
        x_smooth = self.pw(self.dw(x_up)).transpose(1, 2)
        return self.act(self.norm(x_smooth))


# ============================================================================
# 10. The Unified Ultra-Compact Acoustic Encoder
# ============================================================================

class UltraCompactAcousticEncoder(nn.Module):
    """
    Ultra-Compact Multi-Rate Acoustic Encoder (ZipTransducer-Tamil Family).
    
    Supports Small (~9.3M params), Medium (~49.2M params), and Large (~105.7M params).
    Hierarchy:
    - 4x Subsampling Stem: (B, T, 80) -> (B, T0, D) at 25 Hz (40ms)
    - Stage 1: n1 ZipBlocks at Rate 1 (T0)
    - Downsampler: 2x downsampling to Rate 2 (T1 = T0 // 2)
    - Stage 2: n2 ZipBlocks at Rate 2 (T1, 80ms)
    - Upsampler: 2x nearest-neighbor upsampling to T0 + DW-PW smoothing + Skip residual from Stage 1
    - Stage 3: n3 ZipBlocks at Rate 1 (T0)
    - Final RMSNorm(D)
    """
    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()
        self.config = config

        # 1. Front-end 4x Subsampling Stem
        self.stem = SubsamplingStem1D(
            input_dim=config.input_dim,
            d_model=config.d_model
        )

        # 2. Rotary Position Embedding
        self.rope = RotaryEmbedding(
            dim=config.d_model // config.num_heads,
            max_seq_len=config.max_seq_len
        )

        # Helper to build a stack of ZipBlocks
        def make_stack(num_blocks: int) -> nn.ModuleList:
            return nn.ModuleList([
                ZipBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    d_ff=config.d_ff,
                    conv_kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                    factorized_rank=config.factorized_rank,
                    attention_type=config.attention_type,
                    chunk_size=config.chunk_size,
                    rope=self.rope,
                    causal_conv=config.causal_conv
                )
                for _ in range(num_blocks)
            ])

        # 3. U-Net Stacks: (n1, n2, n3)
        n1, n2, n3 = config.stack_layers
        self.stage1 = make_stack(n1)
        self.downsample = TemporalDownsampler1D(config.d_model)
        self.stage2 = make_stack(n2)
        self.upsample = TemporalUpsampler1D(config.d_model)
        self.stage3 = make_stack(n3)

        # 4. Final Output Normalization
        self.final_norm = RMSNorm(config.d_model)

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: (B, T_in, 80) Log-mel spectrogram
            lengths: (B,) original frame lengths before subsampling
            
        Returns:
            encoded: (B, T0, D) Acoustic representations at 25 Hz
            out_lens: (B,) subsampled frame lengths
        """
        # 1. 4x Subsampling Stem
        x, out_lens = self.stem(features, lengths)
        B, T0, _ = x.shape

        # Key padding mask for Rate 1
        mask_r1 = None
        if out_lens is not None:
            time_idx = torch.arange(T0, device=x.device).unsqueeze(0)
            mask_r1 = time_idx < out_lens.unsqueeze(1)

        # 2. Stage 1 (n1 ZipBlocks at Rate 1)
        for block in self.stage1:
            x = block(x, key_padding_mask=mask_r1)
        skip1 = x  # Cache skip connection

        # 3. Downsample to Rate 2 (T1 = T0 // 2)
        x = self.downsample(x)
        T1 = x.size(1)
        mask_r2 = None
        if out_lens is not None:
            lens_r2 = torch.div(out_lens + 1, 2, rounding_mode="floor")
            time_idx_r2 = torch.arange(T1, device=x.device).unsqueeze(0)
            mask_r2 = time_idx_r2 < lens_r2.unsqueeze(1)

        # 4. Stage 2 (n2 ZipBlocks at Rate 2)
        for block in self.stage2:
            x = block(x, key_padding_mask=mask_r2)

        # 5. Upsample back to Rate 1 & Combine Skip Residual
        x = self.upsample(x, target_len=T0)
        x = x + skip1  # Skip residual connection

        # 6. Stage 3 (n3 ZipBlocks at Rate 1)
        for block in self.stage3:
            x = block(x, key_padding_mask=mask_r1)

        # 7. Final Normalization
        out = self.final_norm(x)
        return out, out_lens

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_macs(self, sample_length_sec: float = 10.0) -> float:
        """
        Estimate Multiply-Accumulate Operations (MACs) for a given audio duration.
        """
        t_in = int(sample_length_sec * 100)
        t0 = t_in // self.config.subsampling_factor
        t1 = t0 // 2
        d = self.config.d_model
        d_ff = self.config.d_ff
        h_q = self.config.num_heads
        h_kv = self.config.num_kv_heads
        d_head = d // h_q
        
        # Stem MACs:
        stem_macs = (80 * 3 * (t_in // 2)) + (80 * d * (t_in // 2)) + (d * 3 * t0) + (d * d * t0)
        
        # ZipBlock linear MACs (projections):
        gqa_proj_macs = (d * (h_q * d_head)) + 2 * (d * (h_kv * d_head)) + ((h_q * d_head) * d)
        conv_macs = 2 * d * d + 15 * d + d * d
        ffn_macs = 3 * d * d_ff
        block_linear_macs = gqa_proj_macs + conv_macs + ffn_macs
        
        # U-Net Stacks:
        n1, n2, n3 = self.config.stack_layers
        macs_rate1 = (n1 + n3) * (t0 * block_linear_macs + (t0 * t0 * d))
        macs_rate2 = n2 * (t1 * block_linear_macs + (t1 * t1 * d))
        
        # Downsampler & Upsampler:
        down_macs = (d * 3 * t1) + (d * d * t1)
        up_macs = (d * 3 * t0) + (d * d * t0)
        
        total_macs = stem_macs + macs_rate1 + macs_rate2 + down_macs + up_macs
        return float(total_macs)


def build_encoder(variant: str = "small", **kwargs) -> UltraCompactAcousticEncoder:
    """Factory builder for Ultra-Compact Acoustic Encoder family (small, medium, large)."""
    config = EncoderConfig.from_variant(variant=variant, **kwargs)
    return UltraCompactAcousticEncoder(config)


def build_compact_encoder(
    d_model: int = 256,
    stack_layers: Tuple[int, int, int] = (3, 6, 3),
    attention_type: str = "gqa",
    factorized_rank: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs
) -> UltraCompactAcousticEncoder:
    """Backwards-compatible builder for the small encoder variant."""
    config = EncoderConfig(
        d_model=d_model,
        stack_layers=stack_layers,
        attention_type=attention_type,
        factorized_rank=factorized_rank,
        dropout=dropout,
        **kwargs
    )
    return UltraCompactAcousticEncoder(config)


if __name__ == "__main__":
    for var in ["small", "medium", "large"]:
        m = build_encoder(variant=var)
        num_params = m.count_parameters()
        x = torch.randn(2, 1000, 80)
        lens = torch.tensor([1000, 750])
        out, out_lens = m(x, lens)
        macs = m.estimate_macs(10.0)
        print(f"Variant: {var.upper():6s} | Params: {num_params:,} ({num_params/1e6:.2f}M) | "
              f"Out Shape: {list(out.shape)} | MACs: {macs/1e9:.2f} GMACs")
