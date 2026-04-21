"""BPE subword tokenizer wrapper.

Thin adapter around a HuggingFace ``AutoTokenizer`` that exposes the
minimal surface the project's models need:

    ``encode(text) -> List[int]``      — raw text → subword ids
    ``decode(ids)  -> str``            — subword ids → raw text
    ``vocab_size``                     — total vocab size
    ``unk_id``                         — id used for unknown bytes, or
                                          ``None`` for byte-level
                                          tokenizers that never emit
                                          ``<unk>``
    ``name``                           — HF repo name, used for
                                          reproducible persistence

Why this exists: the project ships three model families
(:class:`NGramModel`, :class:`MarkovChainModel`, :class:`LSTMModel`,
:class:`TransformerModel`) that currently take ``tokens: List[str]`` and
maintain their own word-level vocabulary. Word-level hits the
documented sparse-data problem on every held-out slice:
``docs/ARCHITECTURE.md §5`` calls this out explicitly. BPE subword
tokenization is the fix.

This PR keeps the model contracts unchanged and only adds the
infrastructure. A follow-up PR can teach the model classes to accept an
external tokenizer; until then, users who want to train an LSTM or a
transformer on subword tokens can drive the pipeline by hand — see
``scripts/bpe_train_lstm.py`` for a worked example.

Transformers is an **optional** dependency (same contract as PyTorch is
with :mod:`src.neural_model`). When it's not installed, importing this
module succeeds but instantiation raises a clear error so the rest of
the test suite stays green on a minimal install.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.info("transformers not available")


DEFAULT_BPE_NAME = "HuggingFaceTB/SmolLM2-135M"
"""Default HF repo: SmolLM2's tokenizer is already cached by
``scripts/bench_real_data.py`` and has a reasonable vocab size (~49 k)
for a teaching project, so the first BPE run doesn't force an extra
download."""


class BPETokenizer:
    """Adapter around a HuggingFace tokenizer.

    Usage:

        >>> tok = BPETokenizer()          # defaults to SmolLM2's tokenizer
        >>> ids = tok.encode("Hello, world!")
        >>> text = tok.decode(ids)
        >>> tok.vocab_size
        49152

    The constructor is a thin pass-through to ``AutoTokenizer.from_pretrained``
    so any HF-registered tokenizer name works.
    """

    def __init__(self, name: str = DEFAULT_BPE_NAME) -> None:
        if not HAS_TRANSFORMERS:
            raise RuntimeError(
                "BPETokenizer requires the 'transformers' package. "
                "Install it with 'pip install transformers' or pick the "
                "word-level tokenizer in src.data_loader instead."
            )
        self.name = name
        self._tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to a list of subword ids.

        ``add_special_tokens=False`` by default so the output is a flat
        stream suitable for language-model training — callers that want
        BOS/EOS can set it to True.
        """
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of subword ids back to text."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size)

    @property
    def unk_id(self) -> Optional[int]:
        """The id used for unknown tokens, if any. Byte-level BPE
        tokenizers (like SmolLM2's) never emit ``<unk>`` and return
        ``None`` here — the byte fallback covers everything."""
        return self._tok.unk_token_id
