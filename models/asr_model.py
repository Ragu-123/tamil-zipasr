"""
ZipTransducer-Tamil Model Family (models/asr_model.py)
======================================================
Unified End-to-End Speech Recognition System for Indic / Tamil.
Supports 3 scaled model family variants:
- Small  (~10.5M params): Ultra-lightweight edge & real-time streaming deployment (RTF < 0.003)
- Medium (~51.2M params): Production high-accuracy & throughput balance
- Large  (~115.3M params): High-fidelity representations for noisy & diverse multi-speaker domains

Architecture:
1. Acoustic Encoder (models.encoder.UltraCompactAcousticEncoder):
   - Progressive temporal downsampling stem (4x downsampling from 80-dim log-mel).
   - Multi-Rate U-Net Zipformer hierarchy with Asymmetric Conformer blocks.
   - Grouped-Query Attention (GQA) with Rotary Positional Embeddings (RoPE).
   - Depthwise separable 1D convolutions with GLU gating (k=15).
   - Pre-RMSNorm and SwiGLU feedforward layers.
2. Prediction Network & Joint Network (models.decoder.UltraLightweightDecoder):
   - Stateless causal predictor over finite context window (K=2..3).
   - Zero recurrent hidden state: eliminates KV-cache bloat and drift.
   - Joint network combining acoustic and linguistic representations via Tanh.
3. Multi-Task CTC & Transducer Head:
   - Auxiliary CTC head attached to encoder output.
   - Joint loss: L_total = L_transducer + 0.25 * L_ctc.
   - Dual-mode inference:
     * Fast CTC Greedy Decoding (RTF < 0.003)
     * High-Accuracy Frame-Synchronous Transducer Greedy Search
4. Tamil-Specific Phonetic Tokenizer (models.tokenizer.TamilAksharaTokenizer):
   - Unicode-aware akshara segmentation preserving compound glyphs (உயிர்மெய்).
   - Subscript digit normalization (₀-₉ -> 0-9).
   - Compact vocabulary (431 tokens), consuming <0.3M parameters in projection heads.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import UltraCompactAcousticEncoder, EncoderConfig
from models.decoder import UltraLightweightDecoder, DecoderConfig
from models.tokenizer import TamilAksharaTokenizer, normalize_tamil_text

# Dynamic Triton Acceleration Hooks
try:
    from kernels.triton_ops import (
        is_triton_available,
        fast_ctc_greedy_decode,
        fused_log_softmax,
        fused_ctc_loss,
        fused_joint_tanh,
    )
    HAS_TRITON_OPS = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON_OPS = False
    is_triton_available = lambda: False
    fast_ctc_greedy_decode = None
    fused_log_softmax = None
    fused_ctc_loss = None
    fused_joint_tanh = None


@dataclass
class ZipTransducerConfig:
    """Unified Configuration for ZipTransducer-Tamil Model Family."""
    variant: str = "small"                  # "small", "medium", or "large"
    vocab_size: Optional[int] = None       # Auto-detected from tokenizer (431)
    
    # Audio Front-End
    input_dim: int = 80                     # Log-mel spectrogram channels
    subsampling_factor: int = 4            # Stem temporal downsampling
    
    # Encoder
    d_model: int = 256                     # Encoder hidden dimension
    num_heads: int = 8                     # GQA query heads
    num_kv_heads: int = 2                  # GQA Key/Value heads
    conv_kernel_size: int = 15             # Depthwise conv kernel size (k=15)
    d_ff: int = 512                        # SwiGLU hidden dim
    encoder_stack_layers: Tuple[int, int, int] = (3, 6, 3)  # U-Net Zipformer hierarchy
    dropout: float = 0.1
    
    # Decoder & Transducer
    embed_dim: int = 384                   # Predictor embedding dimension
    pred_dim: int = 384                    # Predictor feature dimension
    joint_dim: int = 384                   # Joint representation dimension
    context_size: int = 2                  # Stateless context lookback (bigram C=2)
    predictor_mode: str = "stateless"      # "stateless" or "rnn"
    ctc_weight: float = 0.25               # Weight for CTC loss (L_total = L_rnnt + 0.25 * L_ctc)
    blank_penalty: float = 0.1             # Blank regularizer for Tamil agglutinative text
    
    # Special Tokens
    blank_id: int = 0
    pad_id: int = 1
    unk_id: int = 2
    bos_id: int = 3
    eos_id: int = 4
    
    # Triton Acceleration
    use_triton_ops: bool = False

    @classmethod
    def from_variant(
        cls,
        variant: str = "small",
        vocab_size: Optional[int] = None,
        use_triton_ops: bool = False,
        **overrides,
    ) -> "ZipTransducerConfig":
        """Factory method to construct model configurations by variant name."""
        v = variant.lower()
        if v == "small":
            cfg = cls(
                variant="small",
                vocab_size=vocab_size,
                d_model=256,
                num_heads=8,
                num_kv_heads=2,
                d_ff=512,
                encoder_stack_layers=(3, 6, 3),
                embed_dim=384,
                pred_dim=384,
                joint_dim=384,
                use_triton_ops=use_triton_ops,
            )
        elif v == "medium":
            cfg = cls(
                variant="medium",
                vocab_size=vocab_size,
                d_model=512,
                num_heads=8,
                num_kv_heads=2,
                d_ff=1024,
                encoder_stack_layers=(4, 8, 4),
                embed_dim=512,
                pred_dim=512,
                joint_dim=512,
                use_triton_ops=use_triton_ops,
            )
        elif v == "large":
            cfg = cls(
                variant="large",
                vocab_size=vocab_size,
                d_model=768,
                num_heads=12,
                num_kv_heads=2,
                d_ff=1024,
                encoder_stack_layers=(4, 12, 4),
                embed_dim=640,
                pred_dim=640,
                joint_dim=640,
                use_triton_ops=use_triton_ops,
            )
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: 'small', 'medium', 'large'.")

        for k, val in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, val)
        return cfg


# Backward compatibility alias
UltraCompactTamilASRConfig = ZipTransducerConfig


class ZipTransducerTamil(nn.Module):
    """
    Unified ZipTransducer-Tamil Architecture.
    
    Integrates the Acoustic Encoder, Stateless Predictor, Joint Network,
    Auxiliary CTC Head, and Tamil Akshara Tokenizer.
    """
    def __init__(
        self,
        config: Optional[ZipTransducerConfig] = None,
        tokenizer: Optional[TamilAksharaTokenizer] = None,
    ):
        super().__init__()
        self.config = config or ZipTransducerConfig()
        
        # 1. Initialize Tamil Akshara Tokenizer
        self.tokenizer = tokenizer or TamilAksharaTokenizer()
        if self.config.vocab_size is None:
            self.config.vocab_size = self.tokenizer.vocab_size
            
        # 2. Build Acoustic Encoder
        enc_cfg = EncoderConfig(
            input_dim=self.config.input_dim,
            d_model=self.config.d_model,
            subsampling_factor=self.config.subsampling_factor,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            d_ff=self.config.d_ff,
            conv_kernel_size=self.config.conv_kernel_size,
            dropout=self.config.dropout,
            stack_layers=self.config.encoder_stack_layers,
            use_triton_ops=self.config.use_triton_ops,
            variant=self.config.variant,
        )
        self.encoder = UltraCompactAcousticEncoder(enc_cfg)
        
        # 3. Build Ultra-Lightweight Decoder System
        dec_cfg = DecoderConfig(
            vocab_size=self.config.vocab_size,
            encoder_dim=self.config.d_model,
            embed_dim=self.config.embed_dim,
            pred_dim=self.config.pred_dim,
            joint_dim=self.config.joint_dim,
            context_size=self.config.context_size,
            dropout=self.config.dropout,
            blank_id=self.config.blank_id,
            predictor_mode=self.config.predictor_mode,
            ctc_weight=self.config.ctc_weight,
            blank_penalty=self.config.blank_penalty,
            variant=self.config.variant,
        )
        self.decoder = UltraLightweightDecoder(config=dec_cfg)

    # =========================================================================
    # Forward Pass (Training & Validation)
    # =========================================================================

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-End Forward Pass.
        
        Args:
            audio_features: (Batch, Time, Channels=80) or (Batch, Channels=80, Time)
            audio_lengths: (Batch,) length of input audio frames
            targets: Optional (Batch, U) integer target tokens
            target_lengths: Optional (Batch,) valid target lengths
        Returns:
            Dictionary containing:
                "encoder_outputs": (Batch, T_sub, d_model)
                "encoder_lengths": (Batch,)
                "ctc_logits": (Batch, T_sub, vocab_size)
                If targets provided:
                    "loss": Total multi-task loss
                    "rnnt_loss": Transducer loss
                    "ctc_loss": CTC loss
                    "joint_logits": (Batch, T_sub, U+1, vocab_size)
        """
        # Ensure audio_features has shape (Batch, Time, Channels=80) for encoder
        if audio_features.dim() == 3 and audio_features.size(1) == self.config.input_dim and audio_features.size(2) != self.config.input_dim:
            audio_features = audio_features.transpose(1, 2)
            
        # 1. Acoustic Encoder Forward Pass
        enc_out, enc_lens = self.encoder(audio_features, audio_lengths)
        
        # 2. If inference (no targets given), return acoustic representations and CTC logits
        if targets is None:
            ctc_logits = self.decoder.ctc_head(enc_out)
            return {
                "encoder_outputs": enc_out,
                "encoder_lengths": enc_lens,
                "ctc_logits": ctc_logits,
            }
            
        # 3. Training: Full Multi-Task Joint + CTC Loss computation
        if target_lengths is None:
            target_lengths = torch.full(
                (targets.size(0),), targets.size(1), dtype=torch.long, device=targets.device
            )
            
        decoder_outputs = self.decoder(
            encoder_outputs=enc_out,
            targets=targets,
            encoder_lengths=enc_lens,
            target_lengths=target_lengths,
        )
        
        decoder_outputs["encoder_outputs"] = enc_out
        decoder_outputs["encoder_lengths"] = enc_lens
        return decoder_outputs

    # =========================================================================
    # Inference Transcription Methods
    # =========================================================================

    @torch.no_grad()
    def transcribe(
        self,
        audio_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        method: str = "ctc",
        max_symbols_per_step: int = 3,
    ) -> List[str]:
        """
        Transcribes input audio features into normalized Tamil text.
        
        Args:
            audio_features: (Batch, Time, 80) or (Batch, 80, Time)
            audio_lengths: Optional (Batch,) lengths
            method: "ctc" (ultra-fast, RTF < 0.003) or "transducer" (high accuracy)
            max_symbols_per_step: Max non-blank symbol emissions per frame for transducer
        Returns:
            List of decoded Tamil text strings.
        """
        self.eval()
        # Ensure audio_features has shape (Batch, Time, Channels=80) for encoder
        if audio_features.dim() == 3 and audio_features.size(1) == self.config.input_dim and audio_features.size(2) != self.config.input_dim:
            audio_features = audio_features.transpose(1, 2)
            
        enc_out, enc_lens = self.encoder(audio_features, audio_lengths)
        
        if method.lower() == "ctc":
            if self.config.use_triton_ops and HAS_TRITON_OPS and audio_features.is_cuda and fast_ctc_greedy_decode is not None:
                ctc_logits = self.decoder.ctc_head(enc_out)
                gpu_tokens = fast_ctc_greedy_decode(ctc_logits, blank_id=self.config.blank_id)
                token_ids_list = [t.tolist() for t in gpu_tokens]
            else:
                token_ids_list = self.decoder.ctc_decode(enc_out, enc_lens)
        elif method.lower() == "transducer":
            token_ids_list = self.decoder.greedy_decode(
                enc_out, enc_lens, max_symbols_per_step=max_symbols_per_step
            )
        else:
            raise ValueError(f"Unknown transcription method: {method}. Choose 'ctc' or 'transducer'.")
            
        # Decode token sequences to Tamil strings
        decoded_texts = [self.tokenizer.decode(seq) for seq in token_ids_list]
        return decoded_texts

    # =========================================================================
    # Parameter and Memory Profiling Utilities
    # =========================================================================

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Calculates precise parameter counts and percentage breakdown."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        stem_params = sum(p.numel() for p in self.encoder.stem.parameters())
        block_params = enc_params - stem_params
        
        pred_params = sum(p.numel() for p in self.decoder.predictor.parameters())
        joint_params = sum(p.numel() for p in self.decoder.joint.parameters())
        ctc_params = sum(p.numel() for p in self.decoder.ctc_head.parameters())
        dec_total = sum(p.numel() for p in self.decoder.parameters())
        
        total = enc_params + dec_total
        
        return {
            "variant": self.config.variant,
            "encoder": {
                "subsampling_stem": stem_params,
                "multi_rate_blocks": block_params,
                "total": enc_params,
                "total_million": round(enc_params / 1e6, 3),
            },
            "decoder": {
                "stateless_predictor": pred_params,
                "joint_network": joint_params,
                "auxiliary_ctc_head": ctc_params,
                "total": dec_total,
                "total_million": round(dec_total / 1e6, 3),
            },
            "total_parameters": total,
            "total_million": round(total / 1e6, 3),
            "vocab_size": self.config.vocab_size,
        }

    def print_architecture_summary(self):
        """Prints a human-readable summary of model dimensions and parameter budgets."""
        b = self.get_parameter_breakdown()
        print("=" * 70)
        print(f"   ZipTransducer-Tamil Architecture Summary [Variant: {b['variant'].upper()}]")
        print("=" * 70)
        print(f" Vocabulary Size:          {b['vocab_size']} (Tamil Akshara)")
        print(f" Encoder Model Dim:        {self.config.d_model} (GQA {self.config.num_heads}Q / {self.config.num_kv_heads}KV)")
        print(f" Subsampling Stem:         4x Downsampling ({b['encoder']['subsampling_stem']:,} params)")
        print(f" Encoder Multi-Rate Blocks:{b['encoder']['multi_rate_blocks']:,} params")
        print(f" Total Encoder:            {b['encoder']['total_million']}M params")
        print("-" * 70)
        print(f" Stateless Predictor (K={self.config.context_size}):{b['decoder']['stateless_predictor']:,} params (D={self.config.pred_dim})")
        print(f" Joint Network (Tanh):     {b['decoder']['joint_network']:,} params (D={self.config.joint_dim})")
        print(f" Auxiliary CTC Head:       {b['decoder']['auxiliary_ctc_head']:,} params")
        print(f" Total Decoder & Heads:    {b['decoder']['total_million']}M params")
        print("=" * 70)
        print(f" TOTAL MODEL PARAMETERS:   {b['total_million']}M parameters")
        print("=" * 70)


# Backward compatibility alias
UltraCompactTamilASR = ZipTransducerTamil


def build_zip_transducer(
    variant: str = "small",
    vocab_size: Optional[int] = None,
    use_triton_ops: bool = False,
    **kwargs,
) -> ZipTransducerTamil:
    """Factory function for ZipTransducer-Tamil model family."""
    tokenizer = TamilAksharaTokenizer()
    v_size = vocab_size or tokenizer.vocab_size
    config = ZipTransducerConfig.from_variant(
        variant=variant,
        vocab_size=v_size,
        use_triton_ops=use_triton_ops,
        **kwargs,
    )
    model = ZipTransducerTamil(config=config, tokenizer=tokenizer)
    return model


# Backward compatibility factory
build_ultra_compact_tamil_asr = build_zip_transducer


if __name__ == "__main__":
    for var in ["small", "medium", "large"]:
        m = build_zip_transducer(variant=var)
        m.print_architecture_summary()
