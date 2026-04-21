"""Tests for the BPE tokenizer adapter."""

import pytest
from src.bpe_tokenizer import BPETokenizer, HAS_TRANSFORMERS


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestBPETokenizer:
    """Exercises the wrapper against the default cached HF tokenizer.

    Skipped on minimal installs so the suite stays green without
    transformers. The default ``SmolLM2-135M`` tokenizer is already
    downloaded by ``scripts/bench_real_data.py``; if it isn't present
    locally these tests will trigger a small download (~1 MB) on first
    run, same as any HF AutoTokenizer call.
    """

    @pytest.fixture(scope="class")
    def tok(self):
        return BPETokenizer()

    def test_name_roundtrips(self, tok):
        assert tok.name == "HuggingFaceTB/SmolLM2-135M"

    def test_vocab_size_is_positive(self, tok):
        assert tok.vocab_size > 0
        # SmolLM2 ships a 49k-ish vocab; being strict about the exact
        # number couples the test to an HF release. Range check is enough.
        assert 10_000 < tok.vocab_size < 200_000

    def test_encode_returns_int_list(self, tok):
        ids = tok.encode("machine learning is a subset of")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) >= 5  # at least one id per space-separated word

    def test_encode_decode_roundtrips_lossily(self, tok):
        """BPE decode may normalise whitespace/case on some tokenizers,
        but the content should match after a case-and-strip pass."""
        original = "Machine learning is great."
        restored = tok.decode(tok.encode(original))
        assert restored.strip().lower() == original.strip().lower()

    def test_empty_text_encodes_to_empty(self, tok):
        assert tok.encode("") == []

    def test_unknown_name_raises(self):
        """Unknown repo names should fail fast, not silently fall back."""
        with pytest.raises(Exception):
            BPETokenizer(name="this/repo-does-not-exist-12345")
