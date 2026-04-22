"""Tests for the SentencePiece tokenizer adapter.

Guarded on `HAS_SENTENCEPIECE` — the dep is optional, just like torch
and transformers. On a minimal install these tests skip.
"""

from __future__ import annotations

import pytest

from src.sp_tokenizer import HAS_SENTENCEPIECE
from src.data_loader import load_sample_data

pytestmark = pytest.mark.skipif(
    not HAS_SENTENCEPIECE, reason="sentencepiece not installed"
)


class TestTrainAndUse:
    def test_train_unigram_and_round_trip(self, tmp_path):
        from src.sp_tokenizer import SPTokenizer

        prefix = str(tmp_path / "spm")
        tok = SPTokenizer.train_from_corpus(
            load_sample_data(), prefix,
            vocab_size=200, model_type="unigram",
        )

        text = "machine learning is a subset of artificial intelligence"
        ids = tok.encode(text)
        back = tok.decode(ids)

        assert tok.vocab_size == 200
        assert tok.name.startswith("sp:")
        assert tok.unk_id == 0  # SentencePiece convention
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        # Decode should reproduce the input (SentencePiece handles
        # whitespace deterministically).
        assert back.strip() == text

    def test_train_bpe_variant(self, tmp_path):
        """Confirm the other model type also works."""
        from src.sp_tokenizer import SPTokenizer

        prefix = str(tmp_path / "spm_bpe")
        tok = SPTokenizer.train_from_corpus(
            load_sample_data(), prefix,
            vocab_size=200, model_type="bpe",
        )
        ids = tok.encode("deep learning")
        assert len(ids) > 0

    def test_reload_from_disk(self, tmp_path):
        """A fresh SPTokenizer(model_path=...) should produce identical
        encodings to the one that trained the model."""
        from src.sp_tokenizer import SPTokenizer

        prefix = str(tmp_path / "spm")
        fresh = SPTokenizer.train_from_corpus(
            load_sample_data(), prefix, vocab_size=200,
        )
        reloaded = SPTokenizer(
            model_path=f"{prefix}.model", name="sp:reloaded",
        )
        text = "machine learning"
        assert fresh.encode(text) == reloaded.encode(text)
        assert reloaded.name == "sp:reloaded"

    def test_missing_model_file_raises(self, tmp_path):
        from src.sp_tokenizer import SPTokenizer

        with pytest.raises(FileNotFoundError):
            SPTokenizer(model_path=str(tmp_path / "does-not-exist.model"))


class TestFitsIntoModelContract:
    """SPTokenizer should be usable as ``tokenizer=`` for the neural
    models — same surface as BPETokenizer."""

    def test_drives_lstm_fit(self, tmp_path):
        from src.neural_model import HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        from src.sp_tokenizer import SPTokenizer
        from src.neural_model import LSTMModel

        prefix = str(tmp_path / "spm")
        tok = SPTokenizer.train_from_corpus(
            load_sample_data(), prefix, vocab_size=200,
        )
        model = LSTMModel(embed_dim=16, hidden_dim=32, num_layers=1)
        model.fit(
            load_sample_data(),
            epochs=1, seq_len=8, batch_size=8, lr=1e-3,
            tokenizer=tok,
        )
        preds = model.predict_next(["machine", "learning"], top_k=3)
        assert len(preds) == 3
