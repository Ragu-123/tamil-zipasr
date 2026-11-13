"""Unit tests for the Streaming Dataset & Audio Processing Pipeline."""

import io
import math
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from data.audio_utils import (
    decode_audio_from_buffer,
    LogMelSpectrogramExtractor,
    apply_spec_augment,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MELS,
)
from data.collator import ASRDataCollator
from data.dataset import IndicVoicesStreamingDataset
from models.tokenizer import TamilAksharaTokenizer, normalize_tamil_text


def test_in_memory_audio_decoding():
    """Verifies audio decoding from raw in-memory bytes without touching disk."""
    sr = 16000
    duration = 1.0  # 1 second
    num_samples = int(sr * duration)
    # Generate 440 Hz sine wave
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio_np = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Encode to WAV in-memory
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)
    raw_bytes = buf.read()

    # Test decoding raw bytes
    waveform, out_sr = decode_audio_from_buffer(raw_bytes, target_sr=sr)
    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (1, num_samples)
    assert out_sr == sr
    assert waveform.dtype == torch.float32

    # Test decoding Hugging Face audio dict
    hf_dict = {"bytes": raw_bytes, "path": "in_memory.wav"}
    waveform_dict, out_sr_dict = decode_audio_from_buffer(hf_dict, target_sr=sr)
    assert waveform_dict.shape == (1, num_samples)
    assert out_sr_dict == sr

    # Test decoding array dict
    arr_dict = {"array": audio_np, "sampling_rate": sr}
    waveform_arr, out_sr_arr = decode_audio_from_buffer(arr_dict, target_sr=sr)
    assert waveform_arr.shape == (1, num_samples)


def test_stereo_to_mono_and_resampling():
    """Verifies stereo to mono conversion and sample rate resampling."""
    src_sr = 32000
    duration = 1.0
    num_samples = int(src_sr * duration)
    # Stereo signal: 2 channels
    stereo_np = np.zeros((num_samples, 2), dtype=np.float32)
    stereo_np[:, 0] = 0.4
    stereo_np[:, 1] = 0.6

    buf = io.BytesIO()
    sf.write(buf, stereo_np, src_sr, format="WAV")
    raw_bytes = buf.getvalue()

    waveform, out_sr = decode_audio_from_buffer(raw_bytes, target_sr=DEFAULT_SAMPLE_RATE)
    assert out_sr == DEFAULT_SAMPLE_RATE
    assert waveform.shape[0] == 1  # Mono
    # Expected samples at 16 kHz for 1.0s is 16,000
    assert abs(waveform.shape[1] - 16000) <= 10


def test_log_mel_spectrogram_extractor():
    """Verifies 80-dim log-mel spectrogram extraction."""
    extractor = LogMelSpectrogramExtractor(
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        win_length=400,
        n_fft=400,
    )
    # 2-second audio (32,000 samples)
    waveform = torch.randn(1, 32000)
    mel = extractor(waveform)

    # Expected shape: (T, 80) where T ≈ 32000 / 160 = 201 frames
    assert mel.dim() == 2
    assert mel.shape[-1] == 80
    assert 195 <= mel.shape[0] <= 205
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


def test_spec_augment():
    """Verifies frequency and time masking in SpecAugment."""
    mel = torch.ones(100, 80)
    augmented = apply_spec_augment(
        mel,
        freq_mask_param=20,
        time_mask_param=20,
        num_freq_masks=2,
        num_time_masks=2,
        mask_value=0.0,
    )

    assert augmented.shape == mel.shape
    # Masked areas must be 0.0, so the total sum must be strictly less than original
    assert augmented.sum().item() < mel.sum().item()
    assert (augmented == 0.0).any()


def test_streaming_dataset_iteration():
    """Verifies streaming dataset yields correct tensors, text, and metadata."""
    dataset = IndicVoicesStreamingDataset(
        split="train",
        mock_streaming=True,
        apply_spec_augment=False,
    )

    # Take first 4 samples
    samples = []
    for idx, sample in enumerate(dataset):
        samples.append(sample)
        if idx >= 3:
            break

    assert len(samples) == 4
    for sample in samples:
        assert "mel" in sample
        assert "tokens" in sample
        assert "text" in sample
        assert "duration" in sample
        assert sample["mel"].dim() == 2
        assert sample["mel"].shape[1] == 80
        assert sample["tokens"].dim() == 1
        assert len(sample["tokens"]) > 0
        assert isinstance(sample["text"], str)


def test_asr_data_collator():
    """Verifies dynamic batch padding and tensor shape creation in collator."""
    collator = ASRDataCollator(pad_id=0, mel_pad_value=0.0)

    batch_input = [
        {"mel": torch.randn(120, 80), "tokens": torch.tensor([5, 12, 18, 22]), "text": "தமிழ்", "duration": 1.2},
        {"mel": torch.randn(80, 80), "tokens": torch.tensor([5, 12]), "text": "வளர்கிறது", "duration": 0.8},
        {"mel": torch.randn(150, 80), "tokens": torch.tensor([5, 12, 18, 22, 35, 41]), "text": "சோதனை", "duration": 1.5},
    ]

    batch = collator(batch_input)

    assert "inputs" in batch
    assert "input_lengths" in batch
    assert "targets" in batch
    assert "target_lengths" in batch
    assert "texts" in batch

    # Check padded inputs shape: (B=3, T_max=150, 80)
    assert batch["inputs"].shape == (3, 150, 80)
    assert torch.equal(batch["input_lengths"], torch.tensor([120, 80, 150]))

    # Check padded targets shape: (B=3, U_max=6)
    assert batch["targets"].shape == (3, 6)
    assert torch.equal(batch["target_lengths"], torch.tensor([4, 2, 6]))

    # Verify padding value 0 on trailing positions
    assert batch["targets"][1, 2].item() == 0
    assert batch["targets"][1, 5].item() == 0


def test_dataloader_streaming_integration():
    """Verifies end-to-end integration between streaming dataset and PyTorch DataLoader."""
    dataset = IndicVoicesStreamingDataset(
        split="valid",
        mock_streaming=True,
        apply_spec_augment=False,
    )
    collator = ASRDataCollator(pad_id=0)

    loader = DataLoader(dataset, batch_size=3, collate_fn=collator)

    batch = next(iter(loader))
    assert batch["inputs"].dim() == 3
    assert batch["inputs"].shape[0] == 3
    assert batch["inputs"].shape[2] == 80
    assert batch["targets"].dim() == 2
    assert batch["targets"].shape[0] == 3
    assert len(batch["texts"]) == 3
