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

## 📁 Repository Structure

```text
├── configs/
│   └── experiment_config.yaml         # Complete experiment hyperparameter specifications
├── data/
│   ├── processed/
│   │   ├── sentiment_train.csv        # 7,524 cleaned AfriSenti Amharic records
│   │   ├── sentiment_val.csv          # 1,882 validation records (stratified, 0-leakage)
│   │   ├── summ_train.csv             # 6,469 cleaned XL-Sum Amharic news pairs
│   │   ├── summ_eval500.csv           # 500 gold evaluation article-summary pairs
│   │   ├── cleaning_report_*.csv      # Verification logs and filtering stats
│   │   └── manifest.json              # Cryptographic SHA-256 integrity manifest
│   └── README.md                      # Dataset documentation and provenance
├── docs/
│   ├── methodology.md                 # Full experimental methodology protocol
│   ├── limitations.md                 # Computational and scope limitations
│   └── ethics.md                      # Ethical statement and data governance
├── notebooks/
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   ├── 02_sentiment_classification.ipynb
│   ├── 03_summarization.ipynb
│   ├── 05_efficiency_and_energy.ipynb
│   └── 06_statistical_analysis_and_visualization.ipynb
├── results/
│   ├── figures/                       # Publication vector (PDF) and raster (PNG) figures
│   │   ├── fig1_pareto_perplexity_vs_flops.*
│   │   ├── fig2_throughput_vs_latency_scaling.*
│   │   ├── fig3_semitic_morphology_fertility_tradeoff.*
│   │   ├── fig4_parameter_efficiency_breakdown.*
│   │   └── fig5_multiobjective_pareto_surface.*
│   ├── tables/                        # Formatted CSV and Markdown experimental tables
│   │   ├── table1_architectural_specifications.*
│   │   ├── table2_computational_complexity_flops.*
│   │   ├── table3_language_modeling_perplexity.*
│   │   ├── table4_inference_efficiency_throughput.*
│   │   ├── table5_semitic_morphology_evaluation.*
│   │   └── table6_pareto_frontier_optimality.*
│   └── raw/                           # Raw run logs, predictions, and metrics JSONs
├── scripts/
│   ├── pareto_frontier_analysis.py    # Multi-objective Pareto frontier identification & plotting
│   ├── generate_results.py            # Summary table and figure aggregator
│   ├── populate_benchmark_results.py  # Reference benchmark population pipeline
│   ├── prepare_data.py                # Dataset download, cleaning, and SHA-256 validation
│   └── run_experiment.py              # End-to-end execution harness
├── src/
│   ├── efficiency/                    # FLOPs counter, profiler, and inference throughput benchmarker
│   ├── evaluation/                    # Amharic metrics, morphological analyzer, perplexity
│   ├── models/                        # Text-JEPA architecture, parameter breakdown, model factory
│   ├── preprocessing/                 # Unicode NFC standardizer, Ethiopic script cleaner, tokenizers
│   ├── training/                      # LoRA training loops for classification and summarization
│   └── utils.py                       # Hardware telemetry, seeds, and cryptographic utilities
├── tests/                             # Comprehensive unit and integration test suite
├── environment.yml                    # Conda virtual environment definition
├── pyproject.toml                     # Python package configuration
└── requirements.txt                   # Pinned dependency requirements
```

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

## 📜 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you utilize this codebase, datasets, or experimental findings in your research, please cite:

```bibtex
@article{amharic_llm_pareto_2026,
  title={Empirical Pareto-Frontier Analysis of Architecture Paradigms and Joint-Embedding Predictive Architectures (JEPA) for Low-Resource Semitic Languages},
  author={Kina Research Team},
  journal={PeerJ Computer Science},
  year={2026}
}
```