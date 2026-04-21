"""Decoder-only transformer language model.

Same ``fit(tokens)`` / ``predict_next(context, top_k)`` / ``perplexity(tokens)``
contract as the other three model families.
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

UNK_TOKEN = "<unk>"
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available")


class TransformerModel(nn.Module if HAS_TORCH else object):
    """Decoder-only transformer LM sharing the project's LM contract.

    Architecture: embedding + learned absolute positional embedding → N
    pre-norm decoder blocks (causal multi-head self-attention + MLP) →
    tied LM head. Weight tying mirrors the Press-Wolf 2016 trick used by
    LSTMModel at schema v2.

    Vocabulary semantics are identical to LSTMModel: ``<unk>`` at id 0,
    optional ``vocab_cap`` caps the trained vocab to the top-(N-1) most
    frequent tokens, OOV at predict time routes through ``<unk>``.

    Persistence: safetensors + JSON metadata sidecar, schema_version=1
    (independent from LSTM's schema version — different state-dict
    layouts version independently). ``torch.save`` is avoided for the
    same security reason it's avoided elsewhere.
    """

    def __init__(
        self,
        vocab_size: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.2,
        vocab_cap: Optional[int] = None,
    ) -> None:
        if HAS_TORCH:
            super().__init__()
            self._d_model = d_model
            self._n_heads = n_heads
            self._n_layers = n_layers
            self._ff_dim = ff_dim
            self._max_seq_len = max_seq_len
            self._dropout = dropout
            if vocab_size > 0:
                self._build_layers(vocab_size)

        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._is_fitted = False
        self._vocab_cap = vocab_cap

    def _build_layers(self, vocab_size: int) -> None:
        if not HAS_TORCH:
            return
        self.embedding = nn.Embedding(vocab_size, self._d_model)
        self.pos_embedding = nn.Embedding(self._max_seq_len, self._d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=self._n_heads,
            dim_feedforward=self._ff_dim,
            dropout=self._dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=self._n_layers)
        self.ln_final = nn.LayerNorm(self._d_model)
        self.lm_head = nn.Linear(self._d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return None
        b, t = x.shape
        if t > self._max_seq_len:
            x = x[:, -self._max_seq_len:]
            t = self._max_seq_len
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        emb = self.embedding(x) + self.pos_embedding(pos)
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1,
        )
        hidden = self.blocks(emb, mask=mask, is_causal=True)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_id)

    # ------------------------------------------------------------------
    # Shared contract: fit / predict_next / perplexity
    # ------------------------------------------------------------------
    def fit(
        self,
        tokens: List[str],
        epochs: int = 1,
        seq_len: int = 64,
        batch_size: int = 32,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ) -> "TransformerModel":
        """Fit on a token list. When torch is absent, builds vocab only so
        predict_next still returns a sensible ``<unk>`` sentinel and the
        rest of the test suite stays green on a minimal install."""
        counts = Counter(tokens)
        if self._vocab_cap is not None and self._vocab_cap > 0:
            kept = [w for w, _ in counts.most_common(self._vocab_cap - 1)]
        else:
            kept = sorted(counts.keys())
        vocab = [UNK_TOKEN] + [w for w in kept if w != UNK_TOKEN]
        self._word_to_id = {w: i for i, w in enumerate(vocab)}
        self._id_to_word = {i: w for w, i in self._word_to_id.items()}
        unk_id = self._word_to_id[UNK_TOKEN]

        if HAS_TORCH:
            self._build_layers(len(vocab))
            ids = [self._word_to_id.get(w, unk_id) for w in tokens]
            _train_transformer(
                self,
                ids,
                vocab_size=len(vocab),
                epochs=epochs,
                seq_len=seq_len,
                batch_size=batch_size,
                lr=lr,
                grad_clip=grad_clip,
                use_amp=use_amp,
            )
        self._is_fitted = True
        return self

    def predict_next(
        self,
        context: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        if not self._is_fitted or not HAS_TORCH:
            return [(UNK_TOKEN, 0.0)]

        unk_id = self._word_to_id.get(UNK_TOKEN, 0)
        ids = [self._word_to_id.get(w, unk_id) for w in context][-self._max_seq_len:]
        if not ids:
            return [(UNK_TOKEN, 0.0)]

        device = next(self.parameters()).device
        self.train(False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(x)
        probs = torch.softmax(logits[0, -1], dim=-1).cpu()
        topk = torch.topk(probs, min(top_k, probs.numel()))
        return [
            (self._id_to_word.get(int(i), UNK_TOKEN), float(p))
            for i, p in zip(topk.indices.tolist(), topk.values.tolist())
        ]

    def perplexity(self, tokens: List[str], seq_len: int = 128) -> float:
        """Token-level cross-entropy perplexity. OOV through ``<unk>``,
        seq_len clamps to both the test slice length and max_seq_len so
        short slices don't silently inflate to inf."""
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
        self.train(False)
        seq_len = max(min(int(seq_len), len(ids) - 2, self._max_seq_len), 1)
        total_loss = 0.0
        total_n = 0
        try:
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
                self.train(True)

        if total_n == 0:
            return float("inf")
        avg = total_loss / total_n
        return float(math.exp(min(avg, 20)))

    # ------------------------------------------------------------------
    # Persistence: safetensors + JSON metadata
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
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
            "model_type": "transformer",
            "schema_version": 1,
            "vocab": [self._id_to_word[i] for i in range(len(self._id_to_word))],
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "ff_dim": self._ff_dim,
            "max_seq_len": self._max_seq_len,
            "dropout": self._dropout,
            "vocab_cap": self._vocab_cap,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Transformer saved: weights=%s meta=%s", weights_path, meta_path)

    @classmethod
    def load(cls, path: str) -> "TransformerModel":
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
        if meta.get("model_type") != "transformer":
            raise ValueError(f"Expected model_type 'transformer', got '{meta.get('model_type')}'")
        if meta.get("schema_version", 1) != 1:
            raise ValueError(
                f"Unsupported transformer schema_version={meta.get('schema_version')}"
            )

        model = cls(
            vocab_size=len(meta["vocab"]),
            d_model=meta["d_model"],
            n_heads=meta["n_heads"],
            n_layers=meta["n_layers"],
            ff_dim=meta["ff_dim"],
            max_seq_len=meta["max_seq_len"],
            dropout=meta["dropout"],
            vocab_cap=meta.get("vocab_cap"),
        )
        model._word_to_id = {w: i for i, w in enumerate(meta["vocab"])}
        model._id_to_word = {i: w for w, i in model._word_to_id.items()}

        state = load_file(str(weights_path))
        model.load_state_dict(state)
        model._is_fitted = True

        logger.info("Transformer loaded: vocab=%d from %s", model.vocab_size, weights_path)
        return model


def _train_transformer(
    model: Any, ids: List[int], vocab_size: int, epochs: int,
    seq_len: int, batch_size: int, lr: float, grad_clip: float, use_amp: bool,
) -> Dict[str, List[float]]:
    """Same batched-parallel-rows layout as train_lstm, minus the
    stateful-BPTT option (transformers attend directly; no hidden state
    to carry). AdamW + cosine LR + optional bf16 autocast on CUDA."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        logger.info("Mock training")
        return {"loss": [2.5 - 0.3*i for i in range(epochs)],
                "perplexity": [np.exp(2.5 - 0.3*i) for i in range(epochs)]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {"loss": [], "perplexity": [], "lr": []}
    amp_enabled = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())

    n_steps = len(ids) // max(batch_size, 1)
    if n_steps <= seq_len:
        data = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        effective_batch = 1
    else:
        truncated = ids[: n_steps * batch_size]
        data = torch.tensor(truncated, dtype=torch.long, device=device).view(batch_size, n_steps)
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
            "Transformer Epoch %d/%d: loss=%.4f ppl=%.1f lr=%.2e batch=%d amp=%s",
            epoch + 1, epochs, avg, ppl, current_lr, effective_batch,
            "bf16" if amp_enabled else "off",
        )

    return history
