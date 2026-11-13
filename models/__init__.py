"""
models package for Ultra-Compact Tamil ASR (ZipTransducer-Tamil).
"""

from .asr_model import (
    ZipTransducerTamil,
    ZipTransducerConfig,
    build_zip_transducer,
    UltraCompactTamilASR,
    UltraCompactTamilASRConfig,
    build_ultra_compact_tamil_asr,
)
from .encoder import (
    UltraCompactAcousticEncoder,
    EncoderConfig,
    build_encoder,
)
from .decoder import (
    UltraLightweightDecoder,
    DecoderConfig,
    build_decoder,
    StatelessPredictionNetwork,
    LightweightRNNPredictionNetwork,
    JointNetwork,
    AuxiliaryCTCHead,
    TransducerLoss,
    JointCTCTransducerLoss,
    RMSNorm,
    CausalConv1d,
)
from .tokenizer import (
    TamilAksharaTokenizer,
    normalize_tamil_text,
)

__all__ = [
    # Top-level unified ASR models
    "ZipTransducerTamil",
    "ZipTransducerConfig",
    "build_zip_transducer",
    "UltraCompactTamilASR",
    "UltraCompactTamilASRConfig",
    "build_ultra_compact_tamil_asr",
    # Encoder
    "UltraCompactAcousticEncoder",
    "EncoderConfig",
    "build_encoder",
    # Decoder & Heads
    "UltraLightweightDecoder",
    "DecoderConfig",
    "build_decoder",
    "StatelessPredictionNetwork",
    "LightweightRNNPredictionNetwork",
    "JointNetwork",
    "AuxiliaryCTCHead",
    "TransducerLoss",
    "JointCTCTransducerLoss",
    "RMSNorm",
    "CausalConv1d",
    # Tokenizer
    "TamilAksharaTokenizer",
    "normalize_tamil_text",
]
