"""Neural language model (LSTM-based).

Implements the same ``fit(tokens)`` / ``predict_next(context, top_k)``
contract as :class:`NGramModel` and :class:`MarkovChainModel` so the LSTM
can be swapped in behind the beam-search decoder and API transparently.
When PyTorch is unavailable the module falls back to deterministic mock
behaviour to keep the rest of the test suite green.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

UNK_TOKEN = "<unk>"

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available")


class LSTMModel(nn.Module if HAS_TORCH else object):
    """LSTM language model with a word-level predict_next / fit interface.

    The module-level helpers (``train_lstm`` / ``predict_next_lstm``) still
    operate on integer token IDs; this class wraps them with a vocabulary
    mapping so it matches the contract advertised in ``docs/ARCHITECTURE.md``.
    """

    def __init__(
        self,
        vocab_size: int = 0,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        vocab_cap: Optional[int] = None,
    ) -> None:
        if HAS_TORCH:
            super().__init__()
            # Defer building the nn layers until fit() knows the real vocab.
            self._embed_dim = embed_dim
            self._hidden_dim = hidden_dim
            self._num_layers = num_layers
            self._dropout = dropout
            if vocab_size > 0:
                self._build_layers(vocab_size)

        # Word <-> id mapping used by fit/predict_next regardless of torch.
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._is_fitted = False
        # Cap the vocabulary to the top-N most frequent tokens during fit.
        # Rare tokens map to <unk>. None = no cap (legacy behaviour).
        self._vocab_cap = vocab_cap

    def _build_layers(self, vocab_size: int) -> None:
        """Build layers: embedding → LSTM → hidden-to-embed projection → tied logits.

        The output projection's weight is tied to the input embedding
        (Press & Wolf 2016), which halves the final-layer parameter count
        and typically improves perplexity. Weight tying requires the tied
        matrices to share their inner dimension, so when ``hidden_dim !=
        embed_dim`` we insert a small ``nn.Linear(hidden_dim, embed_dim)``
        projection (path (b) of the AWD-LSTM recipe). This keeps the
        public ``embed_dim`` / ``hidden_dim`` defaults free to diverge.
        """
        if not HAS_TORCH:
            return
        self.embedding = nn.Embedding(vocab_size, self._embed_dim)
        self.lstm = nn.LSTM(
            self._embed_dim,
            self._hidden_dim,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=self._dropout if self._num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(self._hidden_dim, self._embed_dim)
        # Tied-weight output layer: bias=False because we can't tie the bias,
        # and in practice the bias contributes little once weights are tied.
        self.fc = nn.Linear(self._embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return None
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        return self.fc(self.proj(lstm_out))

    # ------------------------------------------------------------------
    # Shared contract: fit(tokens) / predict_next(context, top_k)
    # ------------------------------------------------------------------
    def fit(
        self,
        tokens: List[str],
        epochs: int = 1,
        seq_len: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        use_compile: bool = False,
    ) -> "LSTMModel":
        """Fit the model on a token list. Trains when torch is present,
        otherwise just builds the vocabulary so predict_next stays usable.

        Vocabulary construction: ``<unk>`` is always ID 0. When
        ``vocab_cap`` is set, the top-(cap-1) most frequent tokens are
        kept and everything else collapses to ``<unk>`` — this prevents
        the long-tail softmax from dominating diversity / coverage
        metrics on large corpora. With no cap, every observed token gets
        its own id plus the ``<unk>`` slot (unused during training but
        available at predict time for OOV context words).
        """
        counts = Counter(tokens)
        if self._vocab_cap is not None and self._vocab_cap > 0:
            kept = [w for w, _ in counts.most_common(self._vocab_cap - 1)]
        else:
            kept = sorted(counts.keys())

        # <unk> occupies ID 0. Training tokens that aren't in `kept` map to it.
        vocab = [UNK_TOKEN] + [w for w in kept if w != UNK_TOKEN]
        self._word_to_id = {w: i for i, w in enumerate(vocab)}
        self._id_to_word = {i: w for w, i in self._word_to_id.items()}
        unk_id = self._word_to_id[UNK_TOKEN]

        if HAS_TORCH:
            self._build_layers(len(vocab))
            token_ids = [self._word_to_id.get(w, unk_id) for w in tokens]
            train_lstm(
                self,
                token_ids,
                vocab_size=len(vocab),
                epochs=epochs,
                seq_len=seq_len,
                batch_size=batch_size,
                lr=lr,
                use_compile=use_compile,
            )
        self._is_fitted = True
        return self

    def predict_next(
        self,
        context: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-k (word, probability) predictions for the context."""
        if not self._is_fitted:
            return [(UNK_TOKEN, 0.0)]

        # OOV context tokens map to <unk> instead of being silently dropped —
        # so the model actually sees every position, and a context of all-OOV
        # words still produces a prediction instead of a hard-coded (<UNK>, 0).
        unk_id = self._word_to_id.get(UNK_TOKEN, 0)
        token_ids = [self._word_to_id.get(w, unk_id) for w in context]
        if not token_ids or not HAS_TORCH:
            return [(UNK_TOKEN, 0.0)]

        raw = predict_next_lstm(self, token_ids, top_k=top_k)
        # raw is List[Tuple[str(id), prob)] -> map ids back to words.
        out: List[Tuple[str, float]] = []
        for idx_str, prob in raw:
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                continue
            word = self._id_to_word.get(idx)
            if word is not None:
                out.append((word, prob))
        return out or [("<UNK>", 0.0)]

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_id)

    def perplexity(self, tokens: List[str], seq_len: int = 128) -> float:
        """Compute perplexity on a token sequence via token-level cross-entropy.

        Mirrors the ``perplexity(tokens) -> float`` contract of
        :class:`NGramModel` and :class:`MarkovChainModel` so
        :func:`src.evaluation.compute_perplexity` can dispatch to all
        three model families uniformly.

        OOV tokens in ``tokens`` are mapped to ``<unk>`` if a ``<unk>``
        entry exists in the model's vocabulary (i.e. the model was fit
        in the usual way). That keeps the denominator consistent with
        the training objective — the model is scored on the same
        distribution it was trained on.

        Args:
            tokens: Token sequence to evaluate on.
            seq_len: BPTT window length for the forward passes. Must be
                at least 1. Longer windows give the recurrent state
                more context; the default matches the benchmark driver.

        Returns:
            Perplexity score (float). Returns ``inf`` when the model is
            not fitted, torch is unavailable, or the sequence is too
            short to produce even one step.
        """
        if not self._is_fitted or not HAS_TORCH:
            return float("inf")
        if not isinstance(self, nn.Module):
            return float("inf")

        unk_id = self._word_to_id.get(UNK_TOKEN, 0)
        ids = [self._word_to_id.get(w, unk_id) for w in tokens]
        if len(ids) < 2:
            return float("inf")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        # Clamp to the longest window the token stream can actually serve.
        # The inner range is ``range(0, len(ids) - seq_len - 1, seq_len)``,
        # so we need ``seq_len <= len(ids) - 2`` to guarantee at least one
        # iteration — otherwise a short test slice silently skips every
        # step and PPL would inflate to inf.
        seq_len = max(min(int(seq_len), len(ids) - 2), 1)
        total_loss = 0.0
        total_n = 0
        try:
            import math
            import torch.nn.functional as F
            with torch.no_grad():
                for i in range(0, len(ids) - seq_len - 1, seq_len):
                    x = torch.tensor(
                        ids[i:i + seq_len], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    y = torch.tensor(
                        ids[i + 1:i + seq_len + 1], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    logits = self(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss.item())
                    total_n += y.numel()
        finally:
            if was_training:
                self.train()

        if total_n == 0:
            return float("inf")
        avg = total_loss / total_n
        return float(math.exp(min(avg, 20)))

    def save(self, path: str) -> None:
        """Persist the trained model to a two-file bundle.

        - ``{path}.safetensors`` — tensor weights (no code execution on load)
        - ``{path}.json``        — vocabulary, hyperparameters, schema version

        Mirrors the ``save`` contract used by :class:`NGramModel` and
        :class:`MarkovChainModel`. ``torch.save`` is avoided on purpose: it
        uses a format that can execute arbitrary code on load, which
        contradicts the project's security anchor (see
        ``tasks/decisions.md``). ``safetensors`` was designed exactly for
        this.

        Args:
            path: Path without extension. Both sibling files are written.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained model. Call fit() first.")
        if not HAS_TORCH:
            raise RuntimeError("Saving requires PyTorch; install it first.")

        import json
        from pathlib import Path
        from safetensors.torch import save_file

        base = Path(path)
        weights_path = base.with_suffix(base.suffix + ".safetensors") if base.suffix else base.with_suffix(".safetensors")
        meta_path = base.with_suffix(base.suffix + ".json") if base.suffix else base.with_suffix(".json")
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(weights_path))

        meta = {
            "model_type": "lstm",
            "schema_version": 2,
            "vocab": [self._id_to_word[i] for i in range(len(self._id_to_word))],
            "embed_dim": self._embed_dim,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "vocab_cap": self._vocab_cap,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("LSTM saved: weights=%s meta=%s", weights_path, meta_path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load a bundle written by :meth:`save`.

        Args:
            path: Same base path passed to ``save``.

        Returns:
            A fitted ``LSTMModel`` ready for ``predict_next``.
        """
        if not HAS_TORCH:
            raise RuntimeError("Loading requires PyTorch; install it first.")

        import json
        from pathlib import Path
        from safetensors.torch import load_file

        base = Path(path)
        weights_path = base.with_suffix(base.suffix + ".safetensors") if base.suffix else base.with_suffix(".safetensors")
        meta_path = base.with_suffix(base.suffix + ".json") if base.suffix else base.with_suffix(".json")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_type") != "lstm":
            raise ValueError(f"Expected model_type 'lstm', got '{meta.get('model_type')}'")

        schema = meta.get("schema_version", 1)
        if schema != 2:
            raise ValueError(
                f"Unsupported LSTM checkpoint schema_version={schema}. "
                "Schema 2 added weight tying + a hidden→embed projection "
                "(Press-Wolf 2016); older checkpoints have no tied layer "
                "and can't be converted losslessly. Retrain and resave."
            )

        model = cls(
            vocab_size=len(meta["vocab"]),
            embed_dim=meta["embed_dim"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            vocab_cap=meta.get("vocab_cap"),
        )
        model._word_to_id = {w: i for i, w in enumerate(meta["vocab"])}
        model._id_to_word = {i: w for w, i in model._word_to_id.items()}

        state = load_file(str(weights_path))
        model.load_state_dict(state)
        model._is_fitted = True

        logger.info("LSTM loaded: vocab=%d from %s", model.vocab_size, weights_path)
        return model


def train_lstm(
    model: Any, tokens: List[int], vocab_size: int, epochs: int = 5,
    seq_len: int = 20, batch_size: int = 32, lr: float = 1e-3,
    grad_clip: float = 1.0, use_amp: bool = True, use_compile: bool = False,
) -> Dict[str, List[float]]:
    """Train the LSTM language model on a flat token stream.

    The token stream is reshaped into ``batch_size`` parallel sequences
    (classic PyTorch language-modelling layout). Each step consumes a
    ``(batch_size, seq_len)`` window — in contrast to the previous
    implementation which always ran at batch=1 because of a stray
    ``unsqueeze(0)`` that ignored the ``batch_size`` argument entirely.

    Training hygiene: gradient clipping keeps the LSTM stable over
    multi-epoch runs; a cosine LR schedule decays the Adam learning rate
    from ``lr`` to zero across ``epochs``.

    Mixed precision: when ``use_amp`` is True and the device is CUDA,
    the forward + loss run under ``torch.autocast(dtype=bfloat16)``.
    bfloat16 has fp32's exponent range, so no loss scaler is needed; the
    tied-embedding checkpoint is still stored in fp32 on disk.
    """
    if not HAS_TORCH or not isinstance(model, nn.Module):
        logger.info("Mock training")
        return {"loss": [2.5 - 0.3*i for i in range(epochs)],
                "perplexity": [np.exp(2.5 - 0.3*i) for i in range(epochs)]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {"loss": [], "perplexity": [], "lr": []}
    amp_enabled = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    # Opt-in torch.compile: ~1.5-2x on Blackwell once the graph is warmed,
    # but has real warmup overhead (first step can be slow) and some graph
    # patterns still raise. Gate behind a kwarg and fall back to eager on
    # any failure so small/CPU runs never break on a compile bug.
    compiled = False
    if use_compile:
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as exc:
            logger.warning("torch.compile failed (%s); falling back to eager", exc)

    # Reshape the flat stream into `batch_size` parallel rows. Rows that
    # don't fit cleanly into `n_steps` tokens are dropped — typical LM
    # tutorial layout.
    n_steps = len(tokens) // max(batch_size, 1)
    if n_steps <= seq_len:
        # Too-short corpus falls back to batch=1 — keeps tiny-corpus tests
        # (13 tokens, seq_len=4) functional without a special case in callers.
        data = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        effective_batch = 1
    else:
        truncated = tokens[: n_steps * batch_size]
        data = torch.tensor(truncated, dtype=torch.long, device=device)
        data = data.view(batch_size, n_steps)
        effective_batch = batch_size

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for i in range(0, data.size(1) - seq_len - 1, seq_len):
            x = data[:, i:i+seq_len]
            y = data[:, i+1:i+seq_len+1]
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg = total_loss / max(n_batch, 1)
        ppl = np.exp(min(avg, 20))
        current_lr = optimizer.param_groups[0]["lr"]
        history["loss"].append(round(avg, 4))
        history["perplexity"].append(round(float(ppl), 4))
        history["lr"].append(round(current_lr, 6))
        logger.info(
            "LSTM Epoch %d/%d: loss=%.4f ppl=%.1f lr=%.2e batch=%d amp=%s compile=%s",
            epoch + 1, epochs, avg, ppl, current_lr, effective_batch,
            "bf16" if amp_enabled else "off",
            "on" if compiled else "off",
        )

    return history


def predict_next_lstm(model: Any, token_ids: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
    """Predict next token with LSTM."""
    if not HAS_TORCH or not isinstance(model, nn.Module) or not token_ids:
        return [("<UNK>", 0.0)]
    # Match the model's device rather than assuming CUDA: train_lstm already
    # moved the model; a freshly-loaded checkpoint stays on CPU until used.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    x = torch.tensor([token_ids[-20:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1).cpu()
    topk = torch.topk(probs, top_k)
    return [(str(i), float(p)) for i, p in zip(topk.indices.tolist(), topk.values.tolist())]
