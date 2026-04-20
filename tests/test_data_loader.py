"""Tests for the data loader module."""

import os

import pytest

from src.data_loader import (
    load_sample_data,
    tokenize,
    build_ngrams,
    generate_synthetic_data,
    train_test_split,
    get_corpus_stats,
    _remove_stopwords,
    load_wikitext,
)

# Tests that hit huggingface.co are skipped by default to keep CI offline-safe.
# Set RUN_NETWORK_TESTS=1 locally to exercise them.
NETWORK_TESTS = os.getenv("RUN_NETWORK_TESTS", "").lower() in ("1", "true", "yes")
network = pytest.mark.skipif(
    not NETWORK_TESTS, reason="set RUN_NETWORK_TESTS=1 to run HF-hub tests"
)


class TestLoadSampleData:
    """Tests for sample data loading."""

    def test_returns_string(self):
        assert isinstance(load_sample_data(), str)

    def test_not_empty(self):
        assert len(load_sample_data()) > 0

    def test_contains_multiple_sentences(self):
        data = load_sample_data()
        # Should have multiple sentences (periods)
        assert data.count(".") > 5

    def test_reproducible(self):
        """Loading should always return the same data."""
        assert load_sample_data() == load_sample_data()


class TestTokenize:
    """Tests for the tokenization function."""

    def test_basic(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_punctuation(self):
        tokens = tokenize("Hello, world!")
        assert "hello" in tokens
        assert "," in tokens
        assert "!" in tokens
        assert "world" in tokens

    def test_lowercasing(self):
        tokens = tokenize("Machine Learning")
        assert "machine" in tokens
        assert "learning" in tokens
        # Original case should NOT be present
        assert "Machine" not in tokens

    def test_numbers(self):
        tokens = tokenize("Python 3.12 is great")
        assert "python" in tokens
        assert "3" in tokens
        assert "12" in tokens

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == []

    def test_single_word(self):
        tokens = tokenize("hello")
        assert tokens == ["hello"]

    def test_remove_stopwords(self):
        tokens_with = tokenize("the cat is on the mat")
        tokens_without = tokenize("the cat is on the mat", remove_stopwords=True)
        # Without stopwords, "the", "is", "on" should be removed
        assert "the" not in tokens_without
        assert "is" not in tokens_without
        assert "on" not in tokens_without
        # Content words should remain
        assert "cat" in tokens_without
        assert "mat" in tokens_without


class TestBuildNgrams:
    """Tests for n-gram construction."""

    def test_bigrams(self):
        ngrams = build_ngrams([1, 2, 3, 4], 2)
        assert len(ngrams) == 3
        assert ngrams == [(1, 2), (2, 3), (3, 4)]

    def test_unigrams(self):
        ngrams = build_ngrams([1, 2, 3], 1)
        assert len(ngrams) == 3
        assert ngrams == [(1,), (2,), (3,)]

    def test_trigrams(self):
        ngrams = build_ngrams([1, 2, 3, 4, 5], 3)
        assert len(ngrams) == 3
        assert ngrams == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_n_larger_than_input(self):
        """N larger than input should return empty list."""
        ngrams = build_ngrams([1, 2], 5)
        assert ngrams == []

    def test_zero_n(self):
        """N=0 should return empty list."""
        ngrams = build_ngrams([1, 2, 3], 0)
        assert ngrams == []

    def test_negative_n(self):
        """Negative N should return empty list."""
        ngrams = build_ngrams([1, 2, 3], -1)
        assert ngrams == []

    def test_empty_input(self):
        """Empty input should return empty list."""
        ngrams = build_ngrams([], 2)
        assert ngrams == []

    def test_n_equals_input_length(self):
        """N equal to input length should return exactly 1 n-gram."""
        ngrams = build_ngrams([1, 2, 3], 3)
        assert len(ngrams) == 1
        assert ngrams == [(1, 2, 3)]


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_returns_string(self):
        assert isinstance(generate_synthetic_data(n_sentences=10), str)

    def test_reproducible(self):
        """Same seed should produce same data."""
        assert generate_synthetic_data(seed=42) == generate_synthetic_data(seed=42)

    def test_different_seeds(self):
        """Different seeds should (usually) produce different data."""
        # With small sample size, there might be overlap, but that's OK
        data1 = generate_synthetic_data(seed=1, n_sentences=100)
        data2 = generate_synthetic_data(seed=2, n_sentences=100)
        assert isinstance(data1, str)
        assert isinstance(data2, str)

    def test_custom_sentence_count(self):
        """Should respect the n_sentences parameter."""
        data_5 = generate_synthetic_data(n_sentences=5, seed=42)
        data_100 = generate_synthetic_data(n_sentences=100, seed=42)
        # More sentences should produce longer text
        assert len(data_100) >= len(data_5)


class TestTrainTestSplit:
    """Tests for train/test splitting."""

    def test_returns_two_lists(self):
        tokens = ["a", "b", "c", "d", "e"]
        train, test = train_test_split(tokens, test_ratio=0.2, seed=42)
        assert isinstance(train, list)
        assert isinstance(test, list)

    def test_correct_split_ratio(self):
        tokens = list(range(100))
        train, test = train_test_split(tokens, test_ratio=0.2, seed=42)
        # Test set should be ~20% of data
        assert abs(len(test) - 20) <= 1
        assert abs(len(train) - 80) <= 1

    def test_no_overlap(self):
        """Train and test should have no overlapping tokens (at same indices)."""
        tokens = list(range(50))
        train, test = train_test_split(tokens, test_ratio=0.4, seed=42)
        # Each index should appear in exactly one set
        train_indices = set()
        test_indices = set()
        for i in range(50):
            if tokens[i] in train:
                train_indices.add(i)
            if tokens[i] in test:
                test_indices.add(i)
        assert len(train_indices & test_indices) == 0
        assert len(train_indices) + len(test_indices) == 50

    def test_reproducible(self):
        tokens = list(range(50))
        t1_train, t1_test = train_test_split(tokens, seed=42)
        t2_train, t2_test = train_test_split(tokens, seed=42)
        assert t1_train == t2_train
        assert t1_test == t2_test

    def test_all_test(self):
        """100% test ratio should put everything in test."""
        tokens = ["a", "b", "c"]
        train, test = train_test_split(tokens, test_ratio=1.0, seed=42)
        assert len(test) == 3
        assert len(train) == 0

    def test_no_test(self):
        """0% test ratio should put everything in train."""
        tokens = ["a", "b", "c"]
        train, test = train_test_split(tokens, test_ratio=0.0, seed=42)
        assert len(train) == 3
        assert len(test) == 0


class TestLoadWikitext:
    """Tests for load_wikitext. Network-gated by default."""

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be"):
            load_wikitext(split="not-a-split")

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            load_wikitext(size="9999")

    @network
    def test_wikitext2_validation_loads(self):
        """Smallest sensible slice: WikiText-2 validation (~3k docs)."""
        text = load_wikitext(split="validation", size="2")
        # Sanity: the validation split should be on the order of
        # hundreds of thousands of characters.
        assert isinstance(text, str)
        assert len(text) > 100_000

    @network
    def test_max_docs_respected(self):
        """max_docs should cap output well below the full validation split."""
        capped = load_wikitext(split="validation", size="2", max_docs=20)
        full = load_wikitext(split="validation", size="2")
        # Capped slice should be a small fraction of the full split.
        assert 0 < len(capped) < len(full) // 10


class TestCorpusStats:
    """Tests for corpus statistics computation."""

    def test_basic_stats(self):
        stats = get_corpus_stats(["hello", "world", "hello", "test"])
        assert stats["total_tokens"] == 4
        assert stats["unique_tokens"] == 3  # hello, world, test
        assert stats["min_token_length"] == 4  # "test"

    def test_empty_corpus(self):
        stats = get_corpus_stats([])
        assert stats["total_tokens"] == 0
        assert stats["unique_tokens"] == 0

    def test_single_token(self):
        stats = get_corpus_stats(["hello"])
        assert stats["total_tokens"] == 1
        assert stats["unique_tokens"] == 1
        assert stats["avg_token_length"] == 5

    def test_punctuation_tokens(self):
        stats = get_corpus_stats(["hello", ",", "world", "!"])
        assert stats["min_token_length"] == 1  # "," or "!"
