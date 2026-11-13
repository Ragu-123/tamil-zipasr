"""Unit tests for UltraCompactTamilASR unified end-to-end model."""

import unittest
import torch
from models.asr_model import build_ultra_compact_tamil_asr, UltraCompactTamilASRConfig


class TestUltraCompactTamilASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_ultra_compact_tamil_asr()

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        import gc
        gc.collect()

    def test_parameter_budget(self):
        breakdown = self.model.get_parameter_breakdown()
        # Parameter budget: sub-25M, target ~10M-15M
        self.assertLess(breakdown["total_million"], 25.0)
        self.assertLess(breakdown["total_million"], 15.0)
        self.assertGreater(breakdown["total_million"], 8.0)
        # Encoder should be ~8-12M
        self.assertLess(breakdown["encoder"]["total_million"], 12.0)
        self.assertGreater(breakdown["encoder"]["total_million"], 8.0)
        # Decoder should be ~1-3M
        self.assertLess(breakdown["decoder"]["total_million"], 5.0)

    def test_forward_pass_training(self):
        B, C, T = 2, 80, 200
        dummy_audio = torch.randn(B, T, C)
        dummy_lengths = torch.tensor([200, 160], dtype=torch.long)
        sample_text = ["வணக்கம் தமிழ்நாடு", "தமிழ் மொழி"]
        padded_tokens, token_lengths = self.model.tokenizer.batch_encode(sample_text)

        out = self.model(
            audio_features=dummy_audio,
            audio_lengths=dummy_lengths,
            targets=padded_tokens,
            target_lengths=token_lengths,
        )

        self.assertIn("loss", out)
        self.assertIn("rnnt_loss", out)
        self.assertIn("ctc_loss", out)
        self.assertIn("encoder_outputs", out)
        self.assertIn("ctc_logits", out)
        self.assertIn("joint_logits", out)

        # 4x downsampling check: 200 frames -> 50 frames
        self.assertEqual(out["encoder_outputs"].shape, (B, 50, self.model.config.d_model))
        self.assertEqual(out["ctc_logits"].shape, (B, 50, self.model.config.vocab_size))

        # Backward pass gradient verification
        out["loss"].backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient!")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!")
                self.assertFalse(torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient!")

        # Explicit non-zero gradient flow checks (auditing CTC, predictor, and encoder)
        self.assertGreater(
            self.model.decoder.ctc_head.head.weight.grad.abs().sum().item(),
            0.0,
            "CTC Head weight received zero gradient!",
        )
        self.assertGreater(
            self.model.decoder.predictor.causal_conv.conv.weight.grad.abs().sum().item(),
            0.0,
            "Predictor conv weight received zero gradient!",
        )
        self.assertGreater(
            self.model.encoder.stem.stage1_dw.weight.grad.abs().sum().item(),
            0.0,
            "Encoder stem weight received zero gradient!",
        )

    def test_inference_transcription(self):
        B, C, T = 1, 80, 100
        dummy_audio = torch.randn(B, T, C)
        ctc_text = self.model.transcribe(dummy_audio, method="ctc")
        self.assertEqual(len(ctc_text), 1)
        self.assertIsInstance(ctc_text[0], str)

        rnnt_text = self.model.transcribe(dummy_audio, method="transducer")
        self.assertEqual(len(rnnt_text), 1)
        self.assertIsInstance(rnnt_text[0], str)


if __name__ == "__main__":
    unittest.main()
