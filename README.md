<div align="center">

# ✍️ Text Autocomplete

**N-gram and neural language models** for text completion with perplexity evaluation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-24%20passed-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)

</div>

## Overview

A **text autocomplete system** combining classic n-gram language models with a neural approach. Features configurable n-gram order, backoff smoothing, perplexity evaluation, and a REST API for real-time completion suggestions.

## Features

- 📊 **N-gram Models** — Bigram and trigram models with configurable order
- 🔄 **Backoff Smoothing** — Kneser-Ney inspired smoothing for unseen n-grams
- 🧠 **Neural Language Model** — Simple neural network for next-token prediction
- 📏 **Perplexity Evaluation** — Standard language model evaluation metric
- 🔗 **REST API** — Real-time autocomplete endpoint
- 📝 **Synthetic Data** — Template-based corpus generation

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/text-autocomplete.git
cd text-autocomplete
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
