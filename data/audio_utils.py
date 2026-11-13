"""Audio utilities for Ultra-Compact Tamil ASR.

Features:
1. In-memory audio decoding from raw bytes (WAV, FLAC, MP3, OGG) via soundfile / torchaudio
   without writing temporary files to disk (zero disk footprint).
2. Resampling to target 16kHz mono audio.
3. 80-dimensional log-mel filterbank feature extraction matching ZipTransducer acoustic stem.
4. SpecAugment (frequency masking and time masking) for speech augmentation during training.
"""

from __future__ import annotations

import io
import math
import random
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 400
DEFAULT_WIN_LENGTH = 400      # 25 ms at 16 kHz
DEFAULT_HOP_LENGTH = 160      # 10 ms at 16 kHz -> 100 frames/sec
DEFAULT_N_MELS = 80
DEFAULT_F_MIN = 0.0
DEFAULT_F_MAX = 8000.0


def decode_audio_from_buffer(
    audio_data: Union[bytes, io.BytesIO, np.ndarray, torch.Tensor, Dict[str, Any]],
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[torch.Tensor, int]:
    """Decodes audio from in-memory bytes, array, or dictionary without touching disk.

    Args:
        audio_data: Audio source. Supported formats:
            - Raw compressed audio bytes (WAV, FLAC, MP3, etc.)
            - io.BytesIO buffer
            - HuggingFace dataset audio dict: {"bytes": ..., "path": ...} or {"array": ..., "sampling_rate": ...}
            - NumPy array (samples,) or (channels, samples)
            - PyTorch Tensor (samples,) or (channels, samples)
        target_sr: Desired output sample rate in Hz (default: 16,000 Hz).

    Returns:
        waveform: Mono audio tensor of shape (1, num_samples) as float32 in [-1.0, 1.0].
        sample_rate: Sample rate (equal to target_sr).
    """
    # 0. Handle Hugging Face AudioDecoder / objects with get_all_samples()
    if hasattr(audio_data, "get_all_samples"):
        samples_obj = audio_data.get_all_samples()
        waveform = samples_obj.data if hasattr(samples_obj, "data") else samples_obj
        sr = getattr(samples_obj, "sample_rate", target_sr)
        return _process_waveform(waveform, sr, target_sr)

    # 1. Handle Hugging Face Audio dict
    if isinstance(audio_data, dict):
        if "array" in audio_data and audio_data["array"] is not None:
            arr = audio_data["array"]
            sr = audio_data.get("sampling_rate", target_sr)
            waveform = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
            return _process_waveform(waveform, sr, target_sr)
        elif "bytes" in audio_data and audio_data["bytes"] is not None:
            audio_data = audio_data["bytes"]
        elif "path" in audio_data and audio_data["path"]:
            with open(audio_data["path"], "rb") as f:
                audio_data = f.read()
        else:
            raise ValueError(f"Unrecognized audio dictionary keys: {list(audio_data.keys())}")

    # 2. Handle NumPy array or PyTorch Tensor directly
    if isinstance(audio_data, (np.ndarray, torch.Tensor)):
        waveform = torch.from_numpy(audio_data) if isinstance(audio_data, np.ndarray) else audio_data
        return _process_waveform(waveform, target_sr, target_sr)

    # 3. Handle raw bytes or BytesIO buffer
    buf = io.BytesIO(audio_data) if isinstance(audio_data, bytes) else audio_data
    buf.seek(0)

    # Prefer soundfile for in-memory buffer decoding
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(buf, dtype="float32")
            waveform = torch.from_numpy(data)
            return _process_waveform(waveform, sr, target_sr)
        except Exception:
            buf.seek(0)

    # Fallback to torchaudio
    if HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(buf)
            return _process_waveform(waveform, sr, target_sr)
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio in-memory: {e}")

    raise RuntimeError("Neither soundfile nor torchaudio is available to decode audio buffer.")


def _process_waveform(waveform: torch.Tensor, src_sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Normalizes waveform to float32, mono (1, N), and resamples to target_sr."""
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        waveform = waveform.to(torch.float32)

    # Ensure shape is (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)

    # Stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if src_sr != target_sr:
        if HAS_TORCHAUDIO:
            waveform = torchaudio.functional.resample(waveform, src_sr, target_sr)
        else:
            # Linear interpolation fallback
            num_samples = int(waveform.shape[-1] * target_sr / src_sr)
            waveform = F.interpolate(waveform.unsqueeze(0), size=num_samples, mode="linear", align_corners=False).squeeze(0)

    # Peak normalization to prevent clipping
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val

    return waveform, target_sr


class LogMelSpectrogramExtractor(torch.nn.Module):
    """Computes 80-dimensional log-mel filterbank energies matching ZipTransducer acoustic stem."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        win_length: int = DEFAULT_WIN_LENGTH,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        f_min: float = DEFAULT_F_MIN,
        f_max: float = DEFAULT_F_MAX,
        eps: float = 1e-5,
        normalize_instance: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        self.normalize_instance = normalize_instance

        if HAS_TORCHAUDIO:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=2.0,
                norm="slaney",
                mel_scale="slaney",
            )
        else:
            self.mel_transform = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts log-mel filterbank features.

        Args:
            waveform: Tensor of shape (1, num_samples) or (B, num_samples).

        Returns:
            log_mel: Tensor of shape (T, 80) or (B, T, 80) representing 80-dim log-mel features.
        """
        squeeze_batch = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        elif waveform.dim() == 2 and waveform.shape[0] == 1:
            squeeze_batch = True

        if self.mel_transform is not None:
            # mel: (B, n_mels, T)
            mel = self.mel_transform(waveform)
        else:
            # Fallback STFT + triangular mel filterbanks
            mel = self._fallback_mel(waveform)

        # Log compression: log(clamp(mel, min=eps))
        log_mel = torch.log(torch.clamp(mel, min=self.eps))

        # Transpose to (B, T, n_mels)
        log_mel = log_mel.transpose(-2, -1)

        # Instance-level cepstral mean & variance normalization (CMVN)
        if self.normalize_instance:
            mean = log_mel.mean(dim=-2, keepdim=True)
            std = log_mel.std(dim=-2, keepdim=True)
            log_mel = (log_mel - mean) / (std + self.eps)

        if squeeze_batch and log_mel.dim() == 3 and log_mel.shape[0] == 1:
            log_mel = log_mel.squeeze(0)

        return log_mel

    def _fallback_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch fallback for STFT power spectrogram."""
        window = torch.hann_window(DEFAULT_WIN_LENGTH, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=DEFAULT_N_FFT,
            hop_length=DEFAULT_HOP_LENGTH,
            win_length=DEFAULT_WIN_LENGTH,
            window=window,
            return_complex=True,
        )
        power_spec = stft.abs().pow(2.0)  # (B, n_fft // 2 + 1, T)
        # Simplified linear pooling into 80 bins
        bins_per_mel = max(1, power_spec.shape[-2] // self.n_mels)
        mel_approx = []
        for i in range(self.n_mels):
            start = i * bins_per_mel
            end = min(power_spec.shape[-2], (i + 1) * bins_per_mel)
            mel_approx.append(power_spec[:, start:end, :].mean(dim=-2, keepdim=True))
        return torch.cat(mel_approx, dim=-2)


def apply_spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 35,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Applies SpecAugment (frequency and time masking) to 80-dim log-mel features.

    Reference: Park et al., SpecAugment: A Simple Data Augmentation Method for ASR.

    Args:
        mel: Tensor of shape (T, 80) or (B, T, 80).
        freq_mask_param: Maximum frequency channels to mask (F).
        time_mask_param: Maximum time frames to mask (T_mask).
        num_freq_masks: Number of frequency masks to apply (m_F).
        num_time_masks: Number of time masks to apply (m_T).
        mask_value: Fill value for masked regions (default: 0.0).

    Returns:
        augmented_mel: Augmented log-mel tensor with identical shape.
    """
    augmented = mel.clone()
    is_batched = (augmented.dim() == 3)
    if not is_batched:
        augmented = augmented.unsqueeze(0)

    B, T, F_dim = augmented.shape

    for b in range(B):
        # 1. Frequency Masking
        for _ in range(num_freq_masks):
            f_len = random.randint(0, min(freq_mask_param, F_dim))
            if f_len > 0:
                f0 = random.randint(0, F_dim - f_len)
                augmented[b, :, f0 : f0 + f_len] = mask_value

        # 2. Time Masking
        max_time_allowed = min(time_mask_param, max(1, int(0.20 * T)))
        for _ in range(num_time_masks):
            t_len = random.randint(0, max_time_allowed)
            if t_len > 0 and T > t_len:
                t0 = random.randint(0, T - t_len)
                augmented[b, t0 : t0 + t_len, :] = mask_value

    if not is_batched:
        augmented = augmented.squeeze(0)

    return augmented
