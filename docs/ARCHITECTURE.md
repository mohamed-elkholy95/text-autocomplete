# Text Autocomplete — Architecture Guide

## Overview

This project implements text autocomplete using two complementary approaches:
1. **N-gram Language Model**: Statistical approach using Markov assumption
2. **Neural Language Model**: Deep learning approach using LSTM

## Design Decisions

- Dual-model approach for comparison and educational purposes
- Vocab management with unknown token handling
- Backoff smoothing for unseen n-gram combinations
