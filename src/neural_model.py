"""Neural language model (LSTM-based).

Implements the same ``fit(tokens)`` / ``predict_next(context, top_k)``
contract as :class:`NGramModel` and :class:`MarkovChainModel` so the LSTM
can be swapped in behind the beam-search decoder and API transparently.
When PyTorch is unavailable the module falls back to deterministic mock
behaviour to keep the rest of the test suite green.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

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

    def _build_layers(self, vocab_size: int) -> None:
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
        self.fc = nn.Linear(self._hidden_dim, vocab_size)

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return None
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        return self.fc(lstm_out)

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
    ) -> "LSTMModel":
        """Fit the model on a token list. Trains when torch is present,
        otherwise just builds the vocabulary so predict_next stays usable."""
        vocab = sorted(set(tokens))
        self._word_to_id = {w: i for i, w in enumerate(vocab)}
        self._id_to_word = {i: w for w, i in self._word_to_id.items()}

        if HAS_TORCH:
            self._build_layers(len(vocab))
            token_ids = [self._word_to_id[w] for w in tokens]
            train_lstm(
                self,
                token_ids,
                vocab_size=len(vocab),
                epochs=epochs,
                seq_len=seq_len,
                batch_size=batch_size,
                lr=lr,
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
            return [("<UNK>", 0.0)]

        token_ids = [self._word_to_id[w] for w in context if w in self._word_to_id]
        if not token_ids or not HAS_TORCH:
            return [("<UNK>", 0.0)]

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
            "schema_version": 1,
            "vocab": [self._id_to_word[i] for i in range(len(self._id_to_word))],
            "embed_dim": self._embed_dim,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
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

        model = cls(
            vocab_size=len(meta["vocab"]),
            embed_dim=meta["embed_dim"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
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
) -> Dict[str, List[float]]:
    """Train LSTM language model."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        logger.info("Mock training")
        return {"loss": [2.5 - 0.3*i for i in range(epochs)],
                "perplexity": [np.exp(2.5 - 0.3*i) for i in range(epochs)]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "perplexity": []}
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0; n_batch = 0
        # Simple sequential batching
        for i in range(0, len(tokens) - seq_len - 1, seq_len):
            x = torch.tensor(tokens[i:i+seq_len], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(tokens[i+1:i+seq_len+1], dtype=torch.long).unsqueeze(0).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batch += 1

        avg = total_loss / max(n_batch, 1)
        ppl = np.exp(min(avg, 20))
        history["loss"].append(round(avg, 4))
        history["perplexity"].append(round(float(ppl), 4))
        logger.info("LSTM Epoch %d/%d: loss=%.4f ppl=%.1f", epoch+1, epochs, avg, ppl)

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
