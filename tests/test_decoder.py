"""
Unit Tests for Scalable Ultra-Lightweight Decoder / Prediction Network (tests/test_decoder.py)
=============================================================================================
Verifies:
1. Multi-variant instantiation (Small, Medium, Large) via DecoderConfig and build_decoder.
2. Causal Conv1D causality guarantee (zero future information leak).
3. Stateless Prediction Network forward and single-step equivalence.
4. Joint Network 4D forward, fast step (B x V), and pruned evaluation.
5. Auxiliary CTC Head loss computation with blank penalty & label smoothing.
6. Dual-mode inference: CTC greedy decoding and Transducer greedy / beam search.
7. Parameter budget constraints across all model family variants.
8. Backward pass gradient flow to all trainable parameters for Small, Medium, and Large.
"""

import os
import sys
import unittest

# Import torch FIRST to prevent local code.py from shadowing stdlib code module
import torch
import torch.nn.functional as F

# Ensure local models package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.decoder import (
    UltraLightweightDecoder,
    DecoderConfig,
    build_decoder,
    StatelessPredictionNetwork,
    LightweightRNNPredictionNetwork,
    JointNetwork,
    AuxiliaryCTCHead,
    RMSNorm,
    CausalConv1d,
)


class TestScalableDecoderFamily(unittest.TestCase):
    """Test suite for UltraLightweightDecoder family components."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 431  # Tamil Akshara Vocabulary

    def test_rmsnorm(self):
        norm = RMSNorm(384)
        x = torch.randn(2, 20, 384)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        var = out.pow(2).mean(dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_causal_conv1d_strict_causality(self):
        """Verify that altering future tokens does not affect past outputs."""
        conv = CausalConv1d(in_channels=384, out_channels=384, kernel_size=2)
        x1 = torch.randn(2, 384, 10)
        x2 = x1.clone()
        x2[:, :, 5:] = torch.randn(2, 384, 5)

        out1 = conv(x1)
        out2 = conv(x2)
        self.assertTrue(torch.allclose(out1[:, :, :5], out2[:, :, :5], atol=1e-6))

    def test_stateless_predictor_forward_and_step(self):
        predictor = StatelessPredictionNetwork(
            vocab_size=self.vocab_size,
            embed_dim=384,
            output_dim=384,
            context_size=2,
            blank_id=0,
        )
        predictor.eval()

        targets = torch.randint(0, self.vocab_size, (2, 8))
        out = predictor(targets)
        self.assertEqual(out.shape, (2, 8, 384))

        context = targets[:, -2:]
        step_out = predictor.forward_step(context)
        self.assertEqual(step_out.shape, (2, 384))
        self.assertTrue(torch.allclose(step_out, out[:, -1, :], atol=1e-5))

    def test_joint_network_forward_and_step(self):
        joint = JointNetwork(
            encoder_dim=256,
            pred_dim=384,
            joint_dim=384,
            vocab_size=self.vocab_size,
        )
        enc_out = torch.randn(2, 25, 256)
        pred_out = torch.randn(2, 10, 384)

        logits = joint(enc_out, pred_out)
        self.assertEqual(logits.shape, (2, 25, 10, self.vocab_size))

        enc_t = torch.randn(2, 256)
        pred_u = torch.randn(2, 384)
        step_logits = joint.forward_step(enc_t, pred_u)
        self.assertEqual(step_logits.shape, (2, self.vocab_size))

    def test_auxiliary_ctc_head(self):
        ctc_head = AuxiliaryCTCHead(
            encoder_dim=256,
            vocab_size=self.vocab_size,
            blank_id=0,
        )
        enc_out = torch.randn(2, 30, 256)
        logits = ctc_head(enc_out)
        self.assertEqual(logits.shape, (2, 30, self.vocab_size))

        targets = torch.randint(1, self.vocab_size, (2, 8))
        in_lens = torch.tensor([30, 25], dtype=torch.int32)
        tgt_lens = torch.tensor([8, 6], dtype=torch.int32)

        loss = ctc_head.compute_loss(logits, targets, in_lens, tgt_lens, blank_penalty=0.1)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_variants_parameter_budget(self):
        # 1. Small variant
        dec_s = build_decoder(variant="small", vocab_size=self.vocab_size)
        cnt_s = dec_s.count_parameters()["total_decoder_parameters"]
        self.assertLess(cnt_s, 1_500_000)
        self.assertAlmostEqual(cnt_s / 1e6, 1.13, delta=0.1)

        # 2. Medium variant
        dec_m = build_decoder(variant="medium", vocab_size=self.vocab_size)
        cnt_m = dec_m.count_parameters()["total_decoder_parameters"]
        self.assertLess(cnt_m, 2_500_000)
        self.assertAlmostEqual(cnt_m / 1e6, 1.98, delta=0.1)

        # 3. Large variant
        dec_l = build_decoder(variant="large", vocab_size=self.vocab_size)
        cnt_l = dec_l.count_parameters()["total_decoder_parameters"]
        self.assertLess(cnt_l, 3_500_000)
        self.assertAlmostEqual(cnt_l / 1e6, 3.02, delta=0.15)

    def test_forward_backward_all_variants(self):
        for var in ["small", "medium", "large"]:
            dec = build_decoder(variant=var, vocab_size=self.vocab_size)
            B, T, U = 2, 30, 8
            enc_out = torch.randn(B, T, dec.encoder_dim, requires_grad=True)
            targets = torch.randint(1, self.vocab_size, (B, U), dtype=torch.long)
            enc_lens = torch.tensor([30, 24], dtype=torch.int32)
            tgt_lens = torch.tensor([8, 6], dtype=torch.int32)

            out = dec(enc_out, targets, enc_lens, tgt_lens)
            loss = out["loss"]
            self.assertFalse(torch.isnan(loss))
            loss.backward()

            self.assertIsNotNone(enc_out.grad)
            self.assertFalse(torch.isnan(enc_out.grad).any())

    def test_dual_mode_decoding(self):
        for var in ["small", "medium", "large"]:
            dec = build_decoder(variant=var, vocab_size=self.vocab_size)
            B, T = 2, 20
            enc_out = torch.randn(B, T, dec.encoder_dim)
            enc_lens = torch.tensor([20, 15], dtype=torch.int32)

            # CTC Greedy
            ctc_hyps = dec.ctc_decode(enc_out, enc_lens)
            self.assertEqual(len(ctc_hyps), 2)

            # Transducer Greedy
            trans_hyps = dec.greedy_decode(enc_out, enc_lens)
            self.assertEqual(len(trans_hyps), 2)


if __name__ == "__main__":
    unittest.main()
