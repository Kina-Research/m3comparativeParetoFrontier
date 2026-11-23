# 🔬 Empirical Pareto-Frontier Analysis of Architecture Paradigms and Joint-Embedding Predictive Architectures (JEPA) for Low-Resource Semitic Languages

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This repository contains the complete experimental suite, datasets, benchmarks, evaluation pipelines, and Pareto-frontier analysis code for evaluating **Transformer Architectural Paradigms** and **Joint-Embedding Predictive Architectures (JEPA)** on low-resource Semitic languages, specifically focusing on **Amharic**.

### Key Research Questions & Evaluation Axes
1. **Multi-Objective Pareto Optimality**: Where does the empirical Pareto frontier lie across parameter efficiency, computational FLOPs, inference throughput, latency, and predictive performance?
2. **Generative Autoregressive vs. Non-Generative Self-Supervised JEPA**: How do representation-learning architectures with non-autoregressive latent prediction (Text-JEPA) compare in computational complexity and semantic stability against autoregressive generative models?
3. **Semitic Morphological Robustness**: How do differing tokenization strategies (WordPiece, SentencePiece/Unigram, Byte-level BPE) and architecture inductive biases interact with Amharic non-concatenative root-and-pattern templatic morphology?

---

## 🏗️ Evaluated Architectural Paradigms

| Paradigm | Exemplar Architecture | Checkpoint / Framework | Size | Tokenizer | Tokenizer Family |
|---|---|---|---|---|---|
| **Encoder-only** | mBERT-Amharic | [`Davlan/bert-base-multilingual-cased-finetuned-amharic`](https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic) | ~177M | WordPiece | WordPiece |
| **Encoder-Decoder** | mT5-small | [`google/mt5-small`](https://huggingface.co/google/mt5-small) | ~300M | SentencePiece | Unigram |
| **Decoder-only** | BLOOMZ-560M | [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m) | ~559M | Byte-level BPE | BPE |
| **Self-Supervised JEPA** | Text-JEPA | Latent Predictive Energy Architecture | ~148M | SentencePiece | Unigram |

---
## ⚡ Quickstart & Reproducibility

### 1. Installation
Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/Kina-Research/m3comparativeParetoFrontier.git
cd m3comparativeParetoFrontier

# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Datasets
Verify the cryptographic SHA-256 integrity of the included processed datasets:

```bash
python scripts/prepare_data.py
```

### 3. Run Pareto Frontier Analysis & Figure Generation
Generate all Pareto frontier curves, surface visualizations, and Scopus/WoS-ready formatted tables:

```bash
python scripts/pareto_frontier_analysis.py
```

### 4. Execute Unit Tests
Verify model implementations, FLOPs counters, and morphological analyzers:

```bash
pytest tests/
```

---

## 📊 Summary of Key Findings

1. **Parameter Efficiency**: LoRA fine-tuning restricts trainable parameters to $<0.55\%$ of total weights across all architectures. mT5-small achieves 0.39% trainable parameter footprint while BLOOMZ-560M restricts trainable parameters to 0.28%.
2. **Computational Scaling (FLOPs)**: Non-autoregressive latent prediction (Text-JEPA) exhibits $O(N)$ computational complexity during inference, whereas standard autoregressive decoder-only models incur quadratic growth $O(N^2)$ across extended context windows.
3. **Inference Throughput**: mT5-small provides superior throughput for short-context sequence-to-sequence translation and summarization, while Text-JEPA offers 2.8× higher inference throughput for latent representations.
4. **Morphological Subword Fertility**: The 250k SentencePiece vocabulary of mT5 minimizes Amharic subword fertility ($1.82$ tokens/word) compared to Byte-level BPE ($3.45$ tokens/word), leading to preserved root-and-pattern templatic integrity in downstream tasks.

---
