"""SentencePiece tokenizer adapter — educational alternative to BPE.

Two subword tokenizers, two implementations worth comparing:

- ``BPETokenizer`` (``src/bpe_tokenizer.py``) wraps a HuggingFace
  byte-level BPE model (e.g. SmolLM2's ~49 k vocab). Byte-level means
  the tokenizer never emits `<unk>` — every byte has a guaranteed id.

- ``SPTokenizer`` (this file) wraps a **Google SentencePiece** model,
  trained directly on the project's corpus rather than pulled from HF.
  SentencePiece is what T5, ALBERT, and LLaMA use; it supports two
  algorithms (BPE and Unigram LM) and handles whitespace as a literal
  token (the ``▁`` marker), which is a different design from
  byte-level BPE.

Both satisfy the same ``encode`` / ``decode`` / ``vocab_size`` /
``unk_id`` / ``name`` surface that :class:`LSTMModel` and
:class:`TransformerModel` expect via their ``tokenizer=`` kwarg — so
passing either into ``fit()`` Just Works.

SentencePiece is an **optional** dependency. Install with:

    pip install sentencepiece

On a minimal install, importing this module is still safe — only
instantiation raises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import sentencepiece as spm  # type: ignore
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    logger.info("sentencepiece not available")


class SPTokenizer:
    """Adapter around a SentencePiece processor.

    Two construction modes:

    1. **Load a pre-trained model file.** Pass ``model_path=...``
       pointing at a ``.model`` produced by ``spm.SentencePieceTrainer``.

    2. **Train on a corpus.** Call :meth:`train_from_corpus` to fit a
       fresh model from a text file (or in-memory string) and save it
       alongside an HF-style repo name for persistence. Useful for
       teaching — students see a tokenizer being trained on their own
       data rather than downloaded.

    The resulting processor exposes the minimal surface the project's
    neural models need; it's a drop-in for :class:`BPETokenizer` in
    every ``LSTMModel.fit(tokenizer=...)`` / ``TransformerModel.fit``
    call.
    """

    def __init__(self, model_path: str, name: Optional[str] = None) -> None:
        if not HAS_SENTENCEPIECE:
            raise ImportError(
                "sentencepiece is not installed. Install with "
                "`pip install sentencepiece`."
            )
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"SentencePiece model not found at {model_path!r}. "
                "Train one first via SPTokenizer.train_from_corpus()."
            )
        self._sp = spm.SentencePieceProcessor(model_file=model_path)
        self._path = model_path
        self._name = name or f"sp:{Path(model_path).stem}"

    # ------------------------------------------------------------------
    # Contract shared with BPETokenizer — keep these method signatures
    # identical so a caller can swap implementations without code changes.
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self._sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    @property
    def unk_id(self) -> Optional[int]:
        # SentencePiece always reserves id 0 for <unk> by convention.
        uid = self._sp.unk_id()
        return int(uid) if uid >= 0 else None

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Training helper — teaches the "fit a tokenizer on a corpus" story
    # ------------------------------------------------------------------
    @classmethod
    def train_from_corpus(
        cls,
        corpus: str | Path,
        out_prefix: str,
        vocab_size: int = 4000,
        model_type: str = "unigram",
        character_coverage: float = 1.0,
    ) -> "SPTokenizer":
        """Train a SentencePiece model from a text file or raw string.

        Writes ``<out_prefix>.model`` and ``<out_prefix>.vocab`` to disk
        (SentencePiece's native format), then returns an ``SPTokenizer``
        bound to the new model.

        ``model_type`` is either ``"unigram"`` (the LLaMA/T5 default; a
        probabilistic LM over subword pieces) or ``"bpe"`` (classic
        greedy merges, same family as HuggingFace BPE). ``vocab_size``
        is hard-capped by the input corpus; 4k is a reasonable default
        for the sample corpus or a small WikiText slice.
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError(
                "sentencepiece is not installed. Install with "
                "`pip install sentencepiece`."
            )

        # Training can take a raw text string or a file path. Decide
        # which by checking for newline / length (path-looking strings
        # are short and never contain newlines); otherwise dump to a
        # temp file so SentencePieceTrainer can read a file input.
        corpus_path: Path
        if isinstance(corpus, Path) or (
            isinstance(corpus, str) and len(corpus) < 4096 and "\n" not in corpus
            and Path(corpus).exists()
        ):
            corpus_path = Path(str(corpus))
        else:
            import re
            import tempfile
            text = str(corpus)
            # SentencePieceTrainer expects newline-separated sentences.
            # If the input is one continuous paragraph, split on sentence
            # punctuation so the trainer gets distinct lines to count.
            if "\n" not in text:
                text = re.sub(r"(?<=[.!?])\s+", "\n", text)
            tmp = tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".txt", encoding="utf-8",
            )
            tmp.write(text)
            tmp.flush()
            tmp.close()
            corpus_path = Path(tmp.name)

        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=out_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            # Silence progress bars in tests.
            train_extremely_large_corpus=False,
        )
        return cls(model_path=f"{out_prefix}.model", name=f"sp:{Path(out_prefix).stem}")
