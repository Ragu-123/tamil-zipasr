"""
Unit Tests for Ultra-Compact Multi-Rate Acoustic Encoder Family (tamil-zipasr/tests/test_encoder.py)
"""

import sys
import unittest
import torch

# Ensure local modules are accessible
sys.path.insert(0, ".")

from models.encoder import (
    UltraCompactAcousticEncoder,
    EncoderConfig,
    build_encoder,
    build_compact_encoder,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    GroupedQueryAttention,
    ChunkedLinearAttention,
    SelectiveSSMBlock,
    DepthwiseConv1DModule,
    ZipBlock,
    SubsamplingStem1D,
)


class TestUltraCompactEncoder(unittest.TestCase):
    """Test suite for encoder components and full hierarchy across all variants."""

    def setUp(self):
        torch.manual_seed(42)

    def tearDown(self):
        import gc
        gc.collect()

    def test_rmsnorm(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 50, 256)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        var = out.pow(2).mean(dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_rope_embedding(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=512)
        x = torch.randn(2, 4, 100, 32)
        cos, sin = rope(x, 100)
        self.assertEqual(cos.shape, (100, 32))
        self.assertEqual(sin.shape, (100, 32))

    def test_swiglu_feedforward(self):
        ffn = SwiGLUFeedForward(dim=256, hidden_dim=512)
        x = torch.randn(2, 50, 256)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 50, 256))

        ffn_fact = SwiGLUFeedForward(dim=256, hidden_dim=512, factorized_rank=128)
        out_fact = ffn_fact(x)
        self.assertEqual(out_fact.shape, (2, 50, 256))

    def test_gqa_attention(self):
        rope = RotaryEmbedding(dim=32)
        gqa = GroupedQueryAttention(d_model=256, num_heads=8, num_kv_heads=2, rope=rope)
        x = torch.randn(2, 60, 256)
        mask = torch.ones(2, 60, dtype=torch.bool)
        mask[0, 50:] = False
        out = gqa(x, key_padding_mask=mask)
        self.assertEqual(out.shape, (2, 60, 256))

    def test_chunked_linear_attention(self):
        attn = ChunkedLinearAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 60, 256)
        out = attn(x)
        self.assertEqual(out.shape, (2, 60, 256))

    def test_selective_ssm_block(self):
        ssm = SelectiveSSMBlock(d_model=256)
        x = torch.randn(2, 60, 256)
        out = ssm(x)
        self.assertEqual(out.shape, (2, 60, 256))

    def test_depthwise_conv_module(self):
        conv = DepthwiseConv1DModule(d_model=256, kernel_size=15)
        x = torch.randn(2, 60, 256)
        out = conv(x)
        self.assertEqual(out.shape, (2, 60, 256))

    def test_subsampling_stem(self):
        stem = SubsamplingStem1D(input_dim=80, d_model=256)
        x = torch.randn(2, 400, 80)
        lens = torch.tensor([400, 300])
        out, out_lens = stem(x, lens)
        self.assertEqual(out.shape, (2, 100, 256))
        self.assertEqual(out_lens.tolist(), [100, 75])

    def test_encoder_family_variants(self):
        expected = {
            "small": {"dim": 256, "min_params": 8_000_000, "max_params": 12_000_000},
            "medium": {"dim": 512, "min_params": 40_000_000, "max_params": 55_000_000},
            "large": {"dim": 768, "min_params": 95_000_000, "max_params": 125_000_000},
        }
        for variant, exp in expected.items():
            model = build_encoder(variant=variant)
            params = model.count_parameters()
            print(f"\n[Test] Variant {variant.upper()}: {params:,} ({params/1e6:.2f}M) params")
            self.assertGreaterEqual(params, exp["min_params"])
            self.assertLessEqual(params, exp["max_params"])
            
            # Forward pass
            x = torch.randn(1, 400, 80)
            lens = torch.tensor([400])
            out, out_lens = model(x, lens)
            self.assertEqual(out.shape, (1, 100, exp["dim"]))
            self.assertEqual(out_lens.tolist(), [100])
            del model, out, out_lens, x, lens
            import gc
            gc.collect()

    def test_small_forward_backward_gradient_integrity(self):
        model = build_encoder(variant="small")
        x = torch.randn(2, 600, 80, requires_grad=True)
        lens = torch.tensor([600, 480])
        
        out, out_lens = model(x, lens)
        self.assertEqual(out.shape, (2, 150, 256))
        self.assertEqual(out_lens.tolist(), [150, 120])
        
        loss = out.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients")

    def test_padding_mask_zeroing_and_isolation(self):
        """[P1-2] Test that padded positions are strictly zeroed out and do not leak."""
        model = build_encoder(variant="small")
        model.eval()
        x = torch.randn(2, 400, 80)
        lens = torch.tensor([300, 200])
        
        with torch.no_grad():
            out, out_lens = model(x, lens)
            
        t1, t2 = out_lens[0].item(), out_lens[1].item()
        self.assertTrue((out[0, t1:] == 0.0).all().item(), "Padded frames in batch 0 are not zeroed")
        self.assertTrue((out[1, t2:] == 0.0).all().item(), "Padded frames in batch 1 are not zeroed")


if __name__ == "__main__":
    unittest.main()
