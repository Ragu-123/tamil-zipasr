"""
Unit tests for custom Triton operations and fused PyTorch primitives (tests/test_triton_ops.py).
Verifies mathematical parity, shapes, gradient flow, and CPU/CUDA fallback.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def test_triton_availability_flag():
    avail = is_triton_available()
    assert isinstance(avail, bool)
    print(f"Triton available on current device: {avail}")


def test_fused_rmsnorm_parity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D = 4, 128, 256
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    weight = torch.ones(D, device=device, dtype=torch.float32)

    y_fused = fused_rmsnorm(x, weight, eps=1e-6)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y_ref = (x * torch.rsqrt(var + 1e-6)) * weight

    assert y_fused.shape == (B, T, D)
    max_diff = (y_fused - y_ref).abs().max().item()
    assert max_diff < 1e-5, f"RMSNorm parity error too large: {max_diff}"


def test_fused_rmsnorm_module():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = FusedRMSNorm(dim=256).to(device)
    x = torch.randn(2, 64, 256, device=device, requires_grad=True)
    out = norm(x)
    assert out.shape == (2, 64, 256)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_causal_dwconv1d_swiglu_parity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C = 2, 64, 512
    D_out = C // 2
    K = 15

    x = torch.randn(B, T, C, device=device)
    w = torch.randn(C, K, device=device) * 0.05
    b = torch.randn(C, device=device) * 0.01

    out_fused = fused_causal_dwconv1d_swiglu(x, w, b, kernel_size=K)
    assert out_fused.shape == (B, T, D_out)

    # Reference
    xc = x.transpose(1, 2)
    xp = F.pad(xc, (K - 1, 0))
    cout = F.conv1d(xp, w.unsqueeze(1), b, groups=C).transpose(1, 2)
    ref = cout[:, :, :D_out] * F.silu(cout[:, :, D_out:])

    max_diff = (out_fused - ref).abs().max().item()
    assert max_diff < 1e-4, f"DWConv+SwiGLU parity error: {max_diff}"


def test_fused_swiglu_ffn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D, D_ff = 2, 64, 256, 512
    x = torch.randn(B, T, D, device=device)
    w_gate = torch.randn(D_ff, D, device=device) * 0.05
    w_up = torch.randn(D_ff, D, device=device) * 0.05
    w_down = torch.randn(D, D_ff, device=device) * 0.05

    out_fused = fused_swiglu(x, w_gate, w_up, w_down)
    assert out_fused.shape == (B, T, D)

    # Ref
    g = F.linear(x, w_gate)
    u = F.linear(x, w_up)
    ref = F.linear(F.silu(g) * u, w_down)

    max_diff = (out_fused - ref).abs().max().item()
    assert max_diff < 1e-4, f"SwiGLU FFN parity error: {max_diff}"


def test_chunked_linear_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, T, D = 2, 4, 128, 32
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)

    out = fused_chunked_linear_attention(q, k, v, chunk_size=32, decay_rate=0.99)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out).any()


def test_chunked_linear_attention_module():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = FusedChunkedLinearAttention(d_model=256, num_heads=8, chunk_size=32).to(device)
    x = torch.randn(2, 128, 256, device=device)
    out = attn(x)
    assert out.shape == (2, 128, 256)
    assert not torch.isnan(out).any()


def test_fused_log_softmax_and_ctc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, V = 2, 100, 1024
    logits = torch.randn(B, T, V, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    log_probs = fused_log_softmax(logits)
    assert log_probs.shape == (B, T, V)
    assert log_probs.dtype == torch.float32

    # Verify log-sum-exp equals 0
    lse = torch.logsumexp(log_probs, dim=-1)
    max_lse_err = (lse - 0.0).abs().max().item()
    assert max_lse_err < 1e-4

    # Test CTC loss and backward gradient flow (P0-1 fix)
    logits.requires_grad_(True)
    targets = torch.randint(1, V, (B, 20), device=device, dtype=torch.long)
    input_lens = torch.full((B,), T, device=device, dtype=torch.long)
    target_lens = torch.full((B,), 20, device=device, dtype=torch.long)

    loss = fused_ctc_loss(logits, targets, input_lens, target_lens, blank=0)
    assert loss.ndim == 0
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    assert logits.grad is not None, "P0-1 Failure: fused_log_softmax broke autograd graph for CTC loss!"
    assert not torch.isnan(logits.grad).any(), "NaN detected in CTC loss gradient!"


def test_fast_ctc_greedy_decode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, V = 2, 50, 1024
    logits = torch.randn(B, T, V, device=device)
    input_lens = torch.tensor([35, 20], device=device, dtype=torch.long)
    decoded = fast_ctc_greedy_decode(logits, blank_id=0, input_lengths=input_lens)

    assert len(decoded) == B
    for seq in decoded:
        assert isinstance(seq, torch.Tensor)
        assert seq.ndim == 1


def test_fused_joint_tanh_and_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, U, D = 2, 40, 15, 384
    enc_out = torch.randn(B, T, D, device=device)
    pred_out = torch.randn(B, U, D, device=device)

    out = fused_joint_tanh(enc_out, pred_out)
    assert out.shape == (B, T, U, D)
    ref = torch.tanh(enc_out.unsqueeze(2) + pred_out.unsqueeze(1))
    assert (out - ref).abs().max().item() < 1e-4

    joint_net = FusedJointNetwork(encoder_dim=256, pred_dim=384, joint_dim=384, vocab_size=1024).to(device)
    enc_feats = torch.randn(B, T, 256, device=device)
    pred_feats = torch.randn(B, U, 384, device=device)
    logits = joint_net(enc_feats, pred_feats)
    assert logits.shape == (B, T, U, 1024)


def test_scaling_dimensions_medium_large():
    """Verify that all kernels scale seamlessly to D=512 (Medium) and D=768 (Large)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for D in [512, 768]:
        # 1. RMSNorm
        x = torch.randn(2, 64, D, device=device)
        w = torch.ones(D, device=device)
        norm_out = fused_rmsnorm(x, w)
        assert norm_out.shape == (2, 64, D)
        assert not torch.isnan(norm_out).any()

        # 2. Causal DWConv + SwiGLU
        C = 2 * D
        x_c = torch.randn(2, 64, C, device=device)
        w_c = torch.randn(C, 15, device=device) * 0.05
        conv_out = fused_causal_dwconv1d_swiglu(x_c, w_c, kernel_size=15)
        assert conv_out.shape == (2, 64, D)
        assert not torch.isnan(conv_out).any()

        # 3. SwiGLU FFN
        w_gate = torch.randn(D * 2, D, device=device) * 0.05
        w_up = torch.randn(D * 2, D, device=device) * 0.05
        w_down = torch.randn(D, D * 2, device=device) * 0.05
        ffn_out = fused_swiglu(x, w_gate, w_up, w_down)
        assert ffn_out.shape == (2, 64, D)
        assert not torch.isnan(ffn_out).any()

    # 4. Chunked Linear Attention with D_head=64 (H=8 -> D=512, H=12 -> D=768)
    for H in [8, 12]:
        q = torch.randn(2, H, 64, 64, device=device)
        k = torch.randn(2, H, 64, 64, device=device)
        v = torch.randn(2, H, 64, 64, device=device)
        attn_out = fused_chunked_linear_attention(q, k, v, chunk_size=32, decay_rate=0.99)
        assert attn_out.shape == (2, H, 64, 64)
        assert not torch.isnan(attn_out).any()

    # 5. Joint Network Tanh (D_joint=512, D_joint=640)
    for D_j in [512, 640]:
        e = torch.randn(2, 40, D_j, device=device)
        p = torch.randn(2, 15, D_j, device=device)
        j_out = fused_joint_tanh(e, p)
        assert j_out.shape == (2, 40, 15, D_j)
        assert not torch.isnan(j_out).any()

