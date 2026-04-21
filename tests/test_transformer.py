"""Tests for the decoder-only transformer language model."""

import pytest
from src.transformer_model import TransformerModel, HAS_TORCH


class TestTransformerInit:
    def test_init_default(self):
        model = TransformerModel(vocab_size=100, d_model=32, n_heads=2, n_layers=2, ff_dim=64)
        assert model is not None

    def test_init_custom(self):
        model = TransformerModel(vocab_size=200, d_model=64, n_heads=4, n_layers=3, ff_dim=128)
        assert model is not None


class TestTransformerPredict:
    def test_predict_without_fit_returns_unk(self):
        model = TransformerModel()
        preds = model.predict_next(["anything"], top_k=3)
        assert preds[0][0] == "<unk>"
        assert preds[0][1] == 0.0

    def test_predict_empty_context_returns_unk(self):
        model = TransformerModel()
        preds = model.predict_next([], top_k=3)
        assert preds[0][0] == "<unk>"


@pytest.mark.skipif(not HAS_TORCH, reason="Transformer requires PyTorch")
class TestTransformerPersistence:
    """Round-trip save/load + perplexity for the transformer. Skipped without torch."""

    @staticmethod
    def _tiny_corpus():
        return ("the cat sat on the mat . the dog sat on the log . "
                "a cat and a dog are friends .").split()

    def test_fit_and_predict(self):
        tokens = self._tiny_corpus()
        model = TransformerModel(
            d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16,
        )
        model.fit(tokens, epochs=1, seq_len=4, batch_size=2, lr=1e-3)
        preds = model.predict_next(["the", "cat"], top_k=3)
        assert len(preds) > 0
        assert all(prob >= 0.0 for _, prob in preds)

    def test_perplexity_is_finite_after_fit(self):
        from src.evaluation import compute_perplexity

        tokens = self._tiny_corpus()
        model = TransformerModel(
            d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16,
        )
        model.fit(tokens, epochs=1, seq_len=4, batch_size=2, lr=1e-3)

        ppl_direct = model.perplexity(tokens)
        ppl_dispatched = compute_perplexity(model, tokens)
        assert ppl_direct < float("inf")
        assert ppl_direct > 0.0
        assert ppl_direct == ppl_dispatched

    def test_perplexity_returns_inf_before_fit(self):
        model = TransformerModel()
        assert model.perplexity(["anything"]) == float("inf")

    def test_save_then_load_round_trip(self, tmp_path):
        tokens = self._tiny_corpus()
        original = TransformerModel(
            d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16,
        )
        original.fit(tokens, epochs=1, seq_len=4, batch_size=2, lr=1e-3)

        path = tmp_path / "xformer_tiny"
        original.save(str(path))
        assert (tmp_path / "xformer_tiny.safetensors").exists()
        assert (tmp_path / "xformer_tiny.json").exists()

        reloaded = TransformerModel.load(str(path))
        assert reloaded.vocab_size == original.vocab_size
        assert reloaded._word_to_id == original._word_to_id

        ctx = ["the", "cat"]
        top_orig = original.predict_next(ctx, top_k=3)
        top_reload = reloaded.predict_next(ctx, top_k=3)
        assert [w for w, _ in top_orig] == [w for w, _ in top_reload]

    def test_save_untrained_raises(self, tmp_path):
        model = TransformerModel(vocab_size=10, d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16)
        with pytest.raises(RuntimeError, match="untrained"):
            model.save(str(tmp_path / "nope"))

    def test_load_rejects_wrong_model_type(self, tmp_path):
        import json
        path = tmp_path / "wrong"
        (tmp_path / "wrong.json").write_text(
            json.dumps({"model_type": "lstm"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="model_type"):
            TransformerModel.load(str(path))

    def test_vocab_cap_collapses_rare_tokens_to_unk(self):
        tokens = (
            ["a"] * 10 + ["b"] * 8 + ["c"] * 6 + ["d"] * 4
            + ["rare1", "rare2", "rare3"]
        )
        model = TransformerModel(
            d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16, vocab_cap=4,
        )
        model.fit(tokens, epochs=1, seq_len=3, batch_size=2, lr=1e-3)
        assert model.vocab_size == 4
        assert "<unk>" in model._word_to_id
        for common in ("a", "b", "c"):
            assert common in model._word_to_id
        for rare in ("d", "rare1", "rare2", "rare3"):
            assert rare not in model._word_to_id
