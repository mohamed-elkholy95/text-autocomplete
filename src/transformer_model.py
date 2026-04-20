"""
Transformer Language Model (HuggingFace causal LM)
====================================================

Wraps a pretrained HuggingFace causal language model (default
``HuggingFaceTB/SmolLM2-135M``, Apache-2.0, 135M params) in the same
``fit(tokens)`` / ``predict_next(context, top_k)`` contract exposed by
:class:`NGramModel`, :class:`MarkovChainModel`, and :class:`LSTMModel`.
That lets the beam-search decoder, the API, and the Streamlit UI treat
the transformer interchangeably with the classical models.

EDUCATIONAL CONTEXT
-------------------
A transformer language model differs from the n-gram / Markov / LSTM
models in this project in three ways that matter for the contract:

1. **No re-training.** The model ships with pretrained weights — so
   ``fit(tokens)`` does NOT learn from the passed tokens. It records the
   training vocabulary for coverage metrics, warms up CUDA kernels, and
   flips ``_is_fitted`` to ``True``. Actual prediction uses the
   pretrained weights directly.

2. **Subword (BPE) tokenization.** SmolLM2 / GPT-2 / Pythia all split
   text into *subword pieces* (e.g. ``"intelligence"`` → ``"intel",
   "ligence"``). A single top-k prediction may therefore be a word piece,
   not a whole word. This class surfaces the raw subword pieces (with
   the leading-space marker stripped) and documents the distinction
   rather than hiding it behind an extra beam-expansion step — teaching
   transparency is the goal.

3. **Perplexity is measured at the subword level.** The
   :meth:`perplexity` method uses the HuggingFace-recommended
   *sliding-window stride* estimator (Wu et al. style), which is the
   methodology behind the published WikiText-103 numbers. That number
   is directly comparable to the distilgpt2 (21.1) / GPT-2 (16.3)
   baselines on the same test split.

RUNTIME
-------
- GPU (CUDA present):  FP16 on the first visible device. SmolLM2-135M
  needs ~300 MB VRAM in FP16; inference latency on an RTX 5080 is a
  few milliseconds per token.
- CPU fallback:  FP32 on CPU if ``torch.cuda.is_available()`` returns
  False. Slower but still correct.
- transformers missing: ``HAS_TRANSFORMERS = False``; all methods return
  deterministic mock outputs so the rest of the test suite stays green.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    HAS_TRANSFORMERS = True
except Exception as e:  # pragma: no cover - import guard
    HAS_TRANSFORMERS = False
    logger.info("transformers/torch not available for TransformerModel: %s", e)


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


class TransformerModel:
    """HuggingFace causal LM behind the shared predict_next contract.

    Attributes:
        model_id: HuggingFace repository ID.
        device: "cuda" or "cpu" (auto-selected when device=None).
        dtype: torch dtype used for model weights (fp16 on GPU, fp32 on CPU).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        self.model_id = model_id
        self._is_fitted = False
        self._vocab: set = set()

        if not HAS_TRANSFORMERS:
            # Mock branch for installs without transformers.
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
            self.dtype = None
            return

        # Auto-select device + dtype. FP16 on GPU for memory/latency; FP32
        # on CPU because fp16 matmul on CPU is slow + lossy.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype

        logger.info("Loading %s on %s (%s)...", model_id, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype
        ).to(device).eval()

        # Some tokenizers (GPT-2 family) have no pad token; set one so
        # batched calls / padded sequences behave predictably.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Shared contract
    # ------------------------------------------------------------------
    def fit(self, tokens: List[str]) -> "TransformerModel":
        """Calibrate the model for this corpus.

        Unlike NGramModel / MarkovChainModel / LSTMModel, this does NOT
        train weights — the pretrained checkpoint is already fit. What
        it does:

        1. Records the training vocabulary so vocabulary-coverage metrics
           still make sense when the transformer is evaluated alongside
           the other models.
        2. Runs a warmup forward pass so the first real request is not
           artificially slow because of lazy CUDA kernel compilation.
        """
        self._vocab = set(tokens)

        if HAS_TRANSFORMERS and self.model is not None:
            with torch.inference_mode():
                warm = self.tokenizer("warmup", return_tensors="pt").input_ids.to(self.device)
                self.model(warm)

        self._is_fitted = True
        return self

    def predict_next(
        self,
        context: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the top-k most likely **subword pieces** that follow the context.

        Because SmolLM2 (and every BPE-based causal LM) operates on
        subword pieces, the returned string values may occasionally be
        word fragments (e.g. ``"ligence"`` rather than ``"intelligence"``)
        rather than whole words. This is a real property of the model;
        callers that want strict word-level suggestions should post-
        process via beam expansion. See the module docstring for why we
        expose this directly.

        Args:
            context: List of word-level tokens (same format as the other
                models in this project). Joined with spaces before being
                fed to the model's own BPE tokenizer.
            top_k: Number of suggestions to return.

        Returns:
            List of (subword, probability) tuples sorted descending.
        """
        if not self._is_fitted or not HAS_TRANSFORMERS or self.model is None:
            return [("<UNK>", 0.0)]

        prompt = " ".join(context) if context else ""
        # An empty prompt still yields a valid distribution over the model's
        # token vocabulary; HF tokenizers tolerate the empty string.
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)

        # Edge case: some tokenizers return zero-length input for empty
        # strings. Seed with the BOS/EOS token so the model has something
        # to condition on.
        if input_ids.shape[-1] == 0:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if bos is None:
                return [("<UNK>", 0.0)]
            input_ids = torch.tensor([[bos]], device=self.device)

        with torch.inference_mode():
            logits = self.model(input_ids).logits[0, -1]          # next-token logits
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        out: List[Tuple[str, float]] = []
        for idx, p in zip(top.indices.tolist(), top.values.tolist()):
            # decode single token; .strip() drops the BPE leading-space marker.
            piece = self.tokenizer.decode([idx], skip_special_tokens=True).strip()
            if piece:
                out.append((piece, float(p)))
        return out or [("<UNK>", 0.0)]

    def perplexity(
        self,
        text,
        max_length: int = 1024,
        stride: int = 512,
    ) -> float:
        """Sliding-window perplexity on a string OR a list of word tokens.

        Uses the HuggingFace-recommended stride approach: advance by
        ``stride`` tokens per window but only score the new ``stride``
        tokens, so each position is predicted with the maximum possible
        context up to ``max_length``. This is the methodology behind the
        published WikiText-103 numbers for distilgpt2 (21.1) and GPT-2
        (16.3) — so the number returned here is directly comparable.

        Args:
            text: Either a raw string (preferred — passed to the model's
                BPE tokenizer as-is) or a list of word-level tokens
                (joined with spaces before tokenization).
            max_length: Max context length per window.
            stride: Number of new tokens scored per window.

        Returns:
            Token-level perplexity (float). Returns +inf if not fitted.
        """
        if not self._is_fitted or not HAS_TRANSFORMERS or self.model is None:
            return float("inf")

        if isinstance(text, list):
            text = " ".join(text)
        if not text:
            return float("inf")

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            return float("inf")

        # Keep windows inside what the model was trained on.
        ctx_len = min(max_length, getattr(self.model.config, "max_position_embeddings", max_length))
        stride = min(stride, ctx_len)

        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + ctx_len, seq_len)
            target_len = end - prev_end          # new tokens to score this window
            window = input_ids[:, begin:end]
            # Mask positions we've already scored so they don't contribute twice.
            target = window.clone()
            target[:, : -target_len] = -100

            with torch.inference_mode():
                out = self.model(window, labels=target)
            # out.loss is already mean-NLL over unmasked tokens; multiply by
            # the number of unmasked tokens to get the summed NLL.
            n_tokens = max(target_len, 1)
            nlls.append(out.loss.float() * n_tokens)

            prev_end = end
            if end == seq_len:
                break

        total_nll = torch.stack(nlls).sum()
        total_tokens = min(seq_len, prev_end)
        avg_nll = total_nll / max(total_tokens, 1)
        return float(torch.exp(avg_nll).item())

    @property
    def vocab_size(self) -> int:
        """Return the model tokenizer's vocabulary size (not the training word set)."""
        if not HAS_TRANSFORMERS or self.tokenizer is None:
            return 0
        return self.tokenizer.vocab_size
