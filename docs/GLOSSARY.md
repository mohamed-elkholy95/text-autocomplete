# NLP & Language Modeling Glossary

A reference guide for the concepts and terminology used throughout this project.
Each term includes a plain-English definition and where it appears in the code.

---

## Core Concepts

### Token
A unit of text after splitting. Usually a word or punctuation mark.
- `"Machine learning is great."` → `["machine", "learning", "is", "great", "."]`
- **Code:** `src/data_loader.py → tokenize()`

### Vocabulary (Vocab)
The set of all unique tokens in the training data. Vocabulary size `V`
determines model complexity — more unique words = more parameters.
- **Code:** `NGramModel._vocab`, `MarkovChainModel._word_to_idx`

### Corpus
A collection of text used for training. Our corpus includes sentences about
ML, software engineering, data science, and technology.
- **Code:** `src/data_loader.py → SAMPLE_TEXTS`

### Language Model (LM)
A model that assigns probabilities to sequences of words. The core question:
"Given some words, what word is most likely to come next?"
- **Code:** All models in `src/` are language models.

---

## Statistical Concepts

### N-gram
A contiguous sequence of `n` items from text.
- **Unigram** (n=1): `"cat"` — individual word frequency
- **Bigram** (n=2): `"the cat"` — word pair frequency
- **Trigram** (n=3): `"the cat sat"` — three-word pattern
- **Code:** `src/ngram_model.py`, `src/data_loader.py → build_ngrams()`

### Markov Property
The assumption that the future depends only on the present state, not the
entire history. For language: P(next_word | all_previous) ≈ P(next_word | last_word).
- **Code:** `src/markov_model.py` — first-order Markov chain

### Smoothing
Techniques to handle unseen n-grams (words or combinations not in training data).
Without smoothing, unseen events get probability 0, breaking the model.

| Technique | How It Works | Trade-off |
|-----------|-------------|-----------|
| Laplace (Add-1) | Add 1 to every count | Simple but over-smooths |
| Backoff | Fall back to shorter n-gram | Ignores lower-order when higher exists |
| Interpolation | Blend all n-gram orders | Better but needs tuning |

- **Code:** `NGramModel.predict_next()` (backoff), `predict_next_interpolated()` (interpolation),
  `MarkovChainModel._get_transition_probs()` (Laplace)

### Transition Matrix
A matrix where entry T[i][j] = P(word_j follows word_i). Each row sums to 1.0.
Think of it as a probability map: "from this word, where can I go?"
- **Code:** `MarkovChainModel._transitions`

---

## Evaluation Metrics

### Perplexity (PPL)
The primary metric for language models. Measures how "surprised" the model
is by test data. Lower is better.
- **PPL = 1:** Perfect prediction (impossible in practice)
- **PPL = 10:** Model narrows each prediction to ~10 equally likely words
- **PPL = V:** No better than random guessing
- **Formula:** PPL = exp(-1/N × Σ log P(w_i | context))
- **Code:** `src/evaluation.py → compute_perplexity()`

### Top-k Accuracy
The fraction of test cases where the correct next word appears in the
model's top-k predictions. Higher is better.
- **Top-1:** Strict — only the #1 prediction counts
- **Top-5:** Lenient — correct word anywhere in top 5
- **Code:** `src/evaluation.py → autocomplete_accuracy()`

### Prediction Diversity
Measures how varied the model's top predictions are across different
inputs. A model that always predicts "the" has zero diversity.
- **Code:** `src/evaluation.py → prediction_diversity()`

### Vocabulary Coverage
What fraction of the vocabulary appears in the model's predictions.
High coverage + high accuracy = ideal model.
- **Code:** `src/evaluation.py → vocabulary_coverage()`

---

## Decoding Strategies

### Greedy Decoding
Always pick the single most probable next token. Fast but often produces
repetitive or suboptimal sequences. Equivalent to beam search with width=1.

### Beam Search
Maintain multiple candidate sequences in parallel, expanding and pruning
at each step. Finds better sequences than greedy at the cost of more computation.
- **Beam width:** Number of parallel candidates (higher = more thorough)
- **Length penalty:** Prevents bias toward shorter sequences
- **Code:** `src/beam_search.py → BeamSearchDecoder`

### Temperature Sampling
Reshapes the probability distribution before sampling:
- **T < 1.0:** Sharper distribution → more deterministic (focused)
- **T = 1.0:** Original probabilities → natural diversity
- **T > 1.0:** Flatter distribution → more random (creative)
- **Code:** `MarkovChainModel.generate_text()` — temperature parameter

---

## Neural Network Concepts

### Embedding
A learned mapping from discrete tokens to dense vectors. Words with similar
meanings end up with similar vectors: vec("king") - vec("man") ≈ vec("queen") - vec("woman").
- **Code:** `LSTMModel.embedding`

### LSTM (Long Short-Term Memory)
A recurrent neural network variant that can learn long-range dependencies.
Uses gates (forget, input, output) to control information flow through time.
- **Code:** `src/neural_model.py → LSTMModel`

### Dropout
Randomly zeroes neurons during training to prevent overfitting. Forces the
network to be robust — no single neuron can become a "single point of failure."
- **Code:** `LSTMModel.__init__()` — dropout=0.2

---

## API & Engineering Concepts

### Token Bucket (Rate Limiting)
An algorithm that controls request flow. Each client has a "bucket" of tokens;
each request consumes one. Tokens refill at a fixed rate. Empty bucket = rejected.
Used by AWS, Stripe, and most cloud APIs.
- **Code:** `src/api/main.py → _check_rate_limit()`

### Model Caching
Store trained models in memory so they persist across API requests. Without
caching, every request would retrain the model (expensive and slow).
- **Code:** `src/api/main.py → _model_cache`

### CORS (Cross-Origin Resource Sharing)
A browser security mechanism that controls which domains can call your API.
Without CORS headers, a web app on `example.com` can't call an API on
`api.example.com`.
- **Code:** `src/api/main.py → CORSMiddleware`
