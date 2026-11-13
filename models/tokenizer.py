"""Specialized Tamil Unicode-aware Akshara & Byte-Pair Tokenizer for Indic ASR.

Key Design Principles:
1. Grapheme Cluster / Akshara Segmentation:
   In Indic languages (specifically Tamil), speech acoustics correspond directly to
   atomic syllables (Aksharas: Consonant + Dependent Vowel or Consonant + Virama).
   Splitting these into separate Unicode codepoints (e.g. consonant and vowel marker)
   causes acoustic-symbol misalignment, sequence length bloat, and fragile CTC alignments.
2. Compact Parameter Budget:
   Traditional ASR tokenizers use 16,000 to 32,000 subwords, which would require
   8M to 16M parameters just for the embedding layer and CTC projection head,
   blowing our entire <15M parameter target budget.
   Our Akshara-based vocabulary contains ~380-512 tokens covering 100% of Tamil phonology,
   numerals, and code-mixed Latin tokens, consuming only ~0.2M parameters in the projection head!
3. Preprocessing & Normalization:
   - Subscript digits (₀-₉) and superscript digits (⁰-⁹) normalized to standard 0-9.
   - Traditional Tamil numerals (௦-௯) normalized to standard 0-9.
   - Split composite vowel signs (e.g. ெ + ா -> ொ, ே + ா -> ோ, ெ + ௗ -> ௌ) canonicalized into single NFC code units.
   - Unicode NFC canonical composition.
"""

from __future__ import annotations

import os
import re
import json
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
import torch


# Normalization translation tables
SUBSCRIPT_TO_STANDARD = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_TO_STANDARD = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
TAMIL_NUMERALS_TO_STANDARD = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

# Canonicalization of two-part Tamil vowel signs
COMPOSITE_VOWEL_REPLACEMENTS = [
    ("\u0BC6\u0BBE", "\u0BCA"),  # ெ + ா -> ொ
    ("\u0BC7\u0BBE", "\u0BCB"),  # ே + ா -> ோ
    ("\u0BC6\u0BD7", "\u0BCC"),  # ெ + ௗ -> ௌ
]

# Tamil Unicode Ranges & Akshara regex pattern
# Independent Vowels: \u0B85-\u0B94 (அ to ஔ)
# Ayutham: \u0B83 (ஃ)
# Base Consonants: \u0B95-\u0BB9 (க to ஹ)
# Dependent Vowel Signs: \u0BBE-\u0BD7 (ா, ி, ீ, ு, ூ, ெ, ே, ை, ொ, ோ, ௌ, ், ௗ)
# Virama / Pulli: \u0BCD (்)
# Special Ligature Sri: \u0BB8\u0BCD\u0BB0\u0BC0 (ஸ்ரீ)
# Special Conjunct Ksha: \u0B95\u0BCD\u0BB7 (க்ஷ)
AKSHARA_PATTERN = re.compile(
    r"\u0BB8\u0BCD\u0BB0\u0BC0"              # Grantha ligature: ஸ்ரீ
    r"|\u0B95\u0BCD\u0BB7\u0BCD"            # க்ஷ் (virama)
    r"|\u0B95\u0BCD\u0BB7[\u0BBE-\u0BD7]"  # க்ஷ + dependent vowel signs
    r"|\u0B95\u0BCD\u0BB7"                  # க்ஷ base
    r"|[\u0B95-\u0BB9]\u0BCD"               # Consonant + Virama (மெய்: க், ச், ட், த், etc.)
    r"|[\u0B95-\u0BB9][\u0BBE-\u0BD7]"      # Consonant + Dependent Vowel (உயிர்மெய்: கா, மி, டு, etc.)
    r"|[\u0B95-\u0BB9]"                     # Standalone Consonant with inherent 'a' (க, ச, etc.)
    r"|[\u0B85-\u0B94\u0B83]"                # Independent Vowels (அ-ஔ) and Ayutham (ஃ)
    r"|[0-9]"                               # Normalized digits
    r"|[a-zA-Z]"                            # Latin characters for code-switching / acronyms
    r"|\s"                                  # Whitespace (preserved for word boundary restoration)
    r"|."                                   # Punctuation & miscellaneous symbols
)


def normalize_tamil_text(text: str) -> str:
    """Normalizes Tamil text, canonicalizing composite glyphs and standardizing digits."""
    if not text:
        return ""
    # 1. Normalize subscript, superscript, and Tamil numerals to standard 0-9
    text = text.translate(SUBSCRIPT_TO_STANDARD)
    text = text.translate(SUPERSCRIPT_TO_STANDARD)
    text = text.translate(TAMIL_NUMERALS_TO_STANDARD)

    # 2. Canonicalize split composite vowel signs
    for comp, canonical in COMPOSITE_VOWEL_REPLACEMENTS:
        text = text.replace(comp, canonical)

    # 3. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # 4. Standardize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class TamilAksharaTokenizer:
    """Deterministic, zero-dependency Tamil Akshara / Syllable Tokenizer.
    
    Provides 100% phonetic coverage of the Tamil language without vocabulary bloat.
    Aligned for CTC and Transducer speech models:
      Token 0: <blank> (CTC blank)
      Token 1: <pad>   (Padding)
      Token 2: <unk>   (Out of vocabulary)
      Token 3: <bos>   (Beginning of sentence)
      Token 4: <eos>   (End of sentence)
    """

    SPECIAL_TOKENS = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>"]
    BLANK_ID = 0
    PAD_ID = 1
    UNK_ID = 2
    BOS_ID = 3
    EOS_ID = 4

    def __init__(self, vocab_file: Optional[str] = None):
        self.special_tokens = list(self.SPECIAL_TOKENS)
        self.blank_id = self.BLANK_ID
        self.pad_id = self.PAD_ID
        self.unk_id = self.UNK_ID
        self.bos_id = self.BOS_ID
        self.eos_id = self.EOS_ID

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocabulary()

    def _build_default_vocabulary(self):
        """Constructs the canonical Tamil akshara vocabulary covering all phonetic combinations."""
        tokens = list(self.SPECIAL_TOKENS)

        # 1. Standalone vowels (Uyir: 12) + Ayutham (1)
        uyir = [
            "\u0B85", "\u0B86", "\u0B87", "\u0B88", "\u0B89", "\u0B8A",
            "\u0B8E", "\u0B8F", "\u0B90", "\u0B92", "\u0B93", "\u0B94",
            "\u0B83"
        ]
        tokens.extend(uyir)

        # 2. Consonants: 18 traditional + 5 Grantha
        consonants = [
            "\u0B95", "\u0B99", "\u0B9A", "\u0B9E", "\u0B9F", "\u0BA3",
            "\u0BA4", "\u0BA8", "\u0BAA", "\u0BAE", "\u0BAF", "\u0BB0",
            "\u0BB2", "\u0BB5", "\u0BB4", "\u0BB3", "\u0BB1", "\u0BA9",
            # Grantha consonants
            "\u0B9C", "\u0BB7", "\u0BB8", "\u0BB9", "\u0B95\u0BCD\u0BB7"  # ஜ, ஷ, ஸ, ஹ, க்ஷ
        ]

        # 3. Pure consonants with Virama (Mei)
        for c in consonants:
            tokens.append(f"{c}\u0BCD")

        # 4. Consonant with inherent vowel 'a'
        for c in consonants:
            tokens.append(c)

        # 5. Consonants with dependent vowel signs (Uyirmei)
        vowel_signs = [
            "\u0BBE", "\u0BBF", "\u0BC0", "\u0BC1", "\u0BC2", "\u0BC6",
            "\u0BC7", "\u0BC8", "\u0BCA", "\u0BCB", "\u0BCC", "\u0BD7"
        ]
        for c in consonants:
            for v in vowel_signs:
                tokens.append(f"{c}{v}")

        # 6. Special Grantha Sri
        tokens.append("\u0BB8\u0BCD\u0BB0\u0BC0")  # ஸ்ரீ

        # 7. Common punctuation, symbols, whitespace
        symbols = [
            " ", ".", ",", "!", "?", ":", ";", "'", '"', "-", "_",
            "(", ")", "[", "]", "{", "}", "/", "\\", "%", "@", "#",
            "&", "*", "+", "=", "<", ">"
        ]
        tokens.extend(symbols)

        # 8. Digits 0-9
        for d in "0123456789":
            tokens.append(d)

        # 9. ASCII Latin characters (lowercase and uppercase for acronyms / English loanwords)
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            tokens.append(ch)

        # Deduplicate while preserving insertion order
        seen = set()
        deduped = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(deduped)}
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(deduped)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, text: str) -> List[str]:
        """Segments normalized Tamil text into a list of atomic akshara tokens."""
        norm_text = normalize_tamil_text(text)
        if not norm_text:
            return []
        return AKSHARA_PATTERN.findall(norm_text)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes text to a sequence of token IDs."""
        tokens = self.tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        result = []
        special_ids = {self.blank_id, self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.id_to_token.get(tid, "")
            result.append(tok)

        text = "".join(result)
        return normalize_tamil_text(text)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
        return_tensors: bool = True,
    ) -> Tuple[Union[torch.Tensor, List[List[int]]], Union[torch.Tensor, List[int]]]:
        """Encodes a batch of texts into padded token tensors and length tensors."""
        encoded_list = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        lengths = [len(seq) for seq in encoded_list]

        if not padding:
            if return_tensors:
                return [torch.tensor(s, dtype=torch.long) for s in encoded_list], torch.tensor(lengths, dtype=torch.long)
            return encoded_list, lengths

        max_len = max_length if max_length is not None else max(lengths, default=0)
        padded = []
        for seq in encoded_list:
            if len(seq) < max_len:
                pad_seq = seq + [self.pad_id] * (max_len - len(seq))
            else:
                pad_seq = seq[:max_len]
            padded.append(pad_seq)

        if return_tensors:
            return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
        return padded, lengths

    def batch_decode(
        self,
        token_tensor: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decodes a 2D batch of token IDs."""
        if isinstance(token_tensor, torch.Tensor):
            token_tensor = token_tensor.tolist()
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_tensor]

    def save_vocab(self, file_path: str):
        """Saves tokenizer vocabulary mapping to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        payload = {
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "token_to_id": self.token_to_id,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path: str):
        """Loads tokenizer vocabulary mapping from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.id_to_token = {int(k): v for k, v in payload["id_to_token"].items()}
        self.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        self.special_tokens = payload.get("special_tokens", self.SPECIAL_TOKENS)


# Module-level factory function
def get_tamil_tokenizer(vocab_path: Optional[str] = None) -> TamilAksharaTokenizer:
    return TamilAksharaTokenizer(vocab_file=vocab_path)


if __name__ == "__main__":
    tokenizer = TamilAksharaTokenizer()
    print(f"Constructed TamilAksharaTokenizer with vocab size: {tokenizer.vocab_size}")

    sample_sentences = [
        "வணக்கம் தமிழ்நாடு! தமிழ் மொழி மிகத் தொன்மையானது.",
        "ஸ்ரீ ராமர் கோவில் திருவிழா 2026-ல் நடைபெறுகிறது.",
        "சீரமைக்கப்பட்ட குறிகள்: ₀₁₂₃₄₅₆₇₈₉ மற்றும் ௦௧௨௩௪.",
        "Artificial Intelligence மற்றும் ASR தொழில்நுட்பம்.",
    ]

    for sent in sample_sentences:
        tokens = tokenizer.tokenize(sent)
        ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(ids)
        norm_orig = normalize_tamil_text(sent)
        print(f"\nOriginal: {sent}")
        print(f"Normalized: {norm_orig}")
        print(f"Token count: {len(tokens)} (tokens: {tokens[:8]}...)")
        print(f"Decoded: {decoded}")
        assert decoded == norm_orig, f"Mismatch: '{decoded}' != '{norm_orig}'"

    print("\n✓ All Tamil tokenizer assertions passed losslessly!")
