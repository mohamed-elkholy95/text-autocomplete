"""Tests for data_loader."""
import pytest
from src.data_loader import load_sample_data, tokenize, build_ngrams, generate_synthetic_data


class TestLoadSampleData:
    def test_returns_string(self):
        assert isinstance(load_sample_data(), str)

    def test_not_empty(self):
        assert len(load_sample_data()) > 0


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_punctuation(self):
        tokens = tokenize("Hello, world!")
        assert "hello" in tokens
        assert "," in tokens


class TestBuildNgrams:
    def test_bigrams(self):
        ngrams = build_ngrams([1, 2, 3, 4], 2)
        assert len(ngrams) == 3
        assert ngrams[0] == (1, 2)

    def test_unigrams(self):
        ngrams = build_ngrams([1, 2, 3], 1)
        assert len(ngrams) == 3


class TestGenerateSyntheticData:
    def test_returns_string(self):
        assert isinstance(generate_synthetic_data(n_sentences=10), str)
    def test_reproducible(self):
        assert generate_synthetic_data(seed=42) == generate_synthetic_data(seed=42)
