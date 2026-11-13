"""
Custom High-Performance Triton GPU Kernels & Fused Primitives
--------------------------------------------------------------
Tailored for Ultra-Compact Tamil ASR (ZipTransducer-Tamil / Zipformer).
"""

from kernels.triton_ops import (
    is_triton_available,
    fused_rmsnorm,
    fused_swiglu,
    fused_causal_dwconv1d_swiglu,
    fused_causal_dwconv1d,
    fused_chunked_linear_attention,
    fused_log_softmax,
    fused_ctc_loss,
    fast_ctc_greedy_decode,
    fused_joint_tanh,
    FusedRMSNorm,
    FusedCausalDWConv1d,
    FusedChunkedLinearAttention,
    FusedJointNetwork,
)

__all__ = [
    "is_triton_available",
    "fused_rmsnorm",
    "fused_swiglu",
    "fused_causal_dwconv1d_swiglu",
    "fused_causal_dwconv1d",
    "fused_chunked_linear_attention",
    "fused_log_softmax",
    "fused_ctc_loss",
    "fast_ctc_greedy_decode",
    "fused_joint_tanh",
    "FusedRMSNorm",
    "FusedCausalDWConv1d",
    "FusedChunkedLinearAttention",
    "FusedJointNetwork",
]
