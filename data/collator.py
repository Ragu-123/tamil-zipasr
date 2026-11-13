"""Data Collator with Dynamic Padding for Ultra-Compact Tamil ASR.

Batches variable-length log-mel acoustic features and Tamil target token sequences,
producing properly padded tensor tensors and lengths for multi-task RNN-T + CTC training.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
from torch.nn.utils.rnn import pad_sequence


class ASRDataCollator:
    """Collates variable-length speech samples into padded mini-batches.

    Expected input item structure:
    {
        "mel": torch.Tensor of shape (T_i, 80),
        "tokens": torch.Tensor or List[int] of shape (U_i,),
        "text": Optional[str],
        "duration": Optional[float],
        "speaker_id": Optional[str]
    }
    """

    def __init__(
        self,
        pad_id: int = 0,
        mel_pad_value: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """Initializes the collator.

        Args:
            pad_id: Token ID used for target sequence padding (default: 0 / <blank>).
            mel_pad_value: Floating-point value used for mel-spectrogram padding (default: 0.0).
            device: Optional target torch.device to automatically transfer tensors to.
        """
        self.pad_id = pad_id
        self.mel_pad_value = mel_pad_value
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates and pads a list of sample dictionaries.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Dictionary containing:
                - "inputs": FloatTensor of shape (B, T_max, 80)
                - "input_lengths": LongTensor of shape (B,) containing each acoustic length T_i
                - "targets": LongTensor of shape (B, U_max) containing padded token IDs
                - "target_lengths": LongTensor of shape (B,) containing each target length U_i
                - "texts": List[str] of original text transcripts
                - "durations": FloatTensor of shape (B,)
        """
        # Filter out any None or empty items
        valid_items = [
            item for item in batch
            if item is not None and "mel" in item and "tokens" in item and len(item["tokens"]) > 0
        ]

        if not valid_items:
            raise ValueError("All items in batch are empty or invalid.")

        mels = []
        tokens = []
        texts = []
        durations = []
        speaker_ids = []

        for item in valid_items:
            mel = item["mel"]
            if not isinstance(mel, torch.Tensor):
                mel = torch.tensor(mel, dtype=torch.float32)
            # Ensure shape is (T, 80)
            if mel.dim() == 3 and mel.shape[0] == 1:
                mel = mel.squeeze(0)
            mels.append(mel)

            tok = item["tokens"]
            if not isinstance(tok, torch.Tensor):
                tok = torch.tensor(tok, dtype=torch.long)
            else:
                tok = tok.to(torch.long)
            tokens.append(tok)

            texts.append(item.get("text", ""))
            durations.append(float(item.get("duration", mel.shape[0] * 0.01)))
            speaker_ids.append(item.get("speaker_id", ""))

        # 1. Acoustic lengths and target lengths
        input_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
        target_lengths = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)

        # 2. Dynamic padding
        # inputs: (B, T_max, 80)
        padded_inputs = pad_sequence(mels, batch_first=True, padding_value=self.mel_pad_value)
        # targets: (B, U_max)
        padded_targets = pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)

        durations_tensor = torch.tensor(durations, dtype=torch.float32)

        output = {
            "inputs": padded_inputs,
            "input_lengths": input_lengths,
            "targets": padded_targets,
            "target_lengths": target_lengths,
            "texts": texts,
            "durations": durations_tensor,
            "speaker_ids": speaker_ids,
        }

        if self.device is not None:
            output["inputs"] = output["inputs"].to(self.device)
            output["input_lengths"] = output["input_lengths"].to(self.device)
            output["targets"] = output["targets"].to(self.device)
            output["target_lengths"] = output["target_lengths"].to(self.device)

        return output
