"""Unit tests for Tamil Unicode-aware Akshara & Byte-Pair Tokenizer."""

import unittest
from models.tokenizer import TamilAksharaTokenizer, normalize_tamil_text


class TestTamilTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TamilAksharaTokenizer()

    def test_vocab_size(self):
        # Akshara vocab should be compact (<600 tokens)
        self.assertLess(self.tokenizer.vocab_size, 600)
        self.assertGreater(self.tokenizer.vocab_size, 350)
        self.assertEqual(self.tokenizer.blank_id, 0)
        self.assertEqual(self.tokenizer.pad_id, 1)

    def test_digit_normalization(self):
        # Subscript digits ₀-₉ -> 0-9
        subscript_text = "எண்: ₀₁₂₃₄₅₆₇₈₉"
        norm = normalize_tamil_text(subscript_text)
        self.assertEqual(norm, "எண்: 0123456789")

        # Tamil numerals ௦-௯ -> 0-9
        tamil_num = "விலை: ௧௨௩௪௫"
        norm_num = normalize_tamil_text(tamil_num)
        self.assertEqual(norm_num, "விலை: 12345")

    def test_lossless_roundtrip(self):
        sentences = [
            "வணக்கம் தமிழ்நாடு! தமிழ் மொழி மிகத் தொன்மையானது.",
            "ஸ்ரீ ராமர் கோவில் திருவிழா 2026-ல் நடைபெறுகிறது.",
            "குறிகள்: ₀₁₂₃ மற்றும் ௦௧௨௩.",
            "Artificial Intelligence பேச்சு அறிதல் முறை.",
        ]
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            ids = self.tokenizer.encode(sent)
            decoded = self.tokenizer.decode(ids)
            expected = normalize_tamil_text(sent)
            self.assertEqual(decoded, expected)

    def test_batch_encoding(self):
        batch = [
            "வணக்கம் தமிழ்நாடு",
            "தமிழ் மொழி மிகத் தொன்மையானது மற்றும் இனிமையானது",
        ]
        padded, lengths = self.tokenizer.batch_encode(batch)
        self.assertEqual(padded.dim(), 2)
        self.assertEqual(padded.size(0), 2)
        self.assertEqual(lengths.tolist(), [len(self.tokenizer.encode(t)) for t in batch])

        decoded_batch = self.tokenizer.batch_decode(padded)
        for dec, orig in zip(decoded_batch, batch):
            self.assertEqual(dec, normalize_tamil_text(orig))


if __name__ == "__main__":
    unittest.main()
