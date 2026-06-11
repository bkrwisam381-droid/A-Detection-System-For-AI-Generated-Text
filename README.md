<div align="center">

# AI Text Detector

**A production-ready, multi-signal ensemble classifier for detecting AI-generated text**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

DeBERTa-v3-large · XGBoost · Binoculars · DistilBERT Genre · Ensemble MLP

<img src="assets/demo.gif" alt="Demo" width="700">

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Running Inference](#running-inference)
  - [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Design Decisions](#key-design-decisions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a robust **4-signal ensemble** for detecting AI-generated text. Unlike single-model approaches, it combines **semantic understanding**, **stylometric analysis**, **perplexity scoring**, and **genre-aware weighting** through a calibrated neural fusion layer. The result is a detector that generalizes across diverse writing styles (conversational, academic, encyclopedic) and multiple AI generators (ChatGPT, GPT-4, Claude, and more).

### Why This Architecture?

| Challenge | Solution in This Project |
|-----------|--------------------------|
| Formal text false positives | Genre classifier detects encyclopedic/academic text and down-weights DeBERTa |
| Single-model blind spots | 4 independent signals combined via learned fusion |
| Softmax saturation on confident predictions | 0.99 probability clip before isotonic calibration |
| Structural elements inflating scores (TOC, headers, citations) | Automatic structural stripping for long documents |
| Sentence-level unreliability | DeBERTa-only sentence scoring with calibrated document-level ensemble |

---

## Key Features

- [x] **Multi-signal ensemble** — DeBERTa (semantic) + XGBoost (stylometric) + Binoculars (perplexity) + Genre (formality)
- [x] **Genre-aware detection** — Automatically adjusts signal weights based on text formality
- [x] **39 stylometric features** — Comprehensive linguistic analysis including AI-tell phrases, entropy, burstiness
- [x] **Rich interactive CLI** — Beautiful terminal UI with progress spinners, color-coded results, and heatmaps
- [x] **Sentence-level heatmap** — Visual breakdown of which sentences are AI vs. human
- [x] **Long document handling** — Automatic stripping of structural elements (TOC, headers, citations)
- [x] **Score conflict detection** — Warns when document-level and sentence-level scores disagree
- [x] **PDF report export** — Generate formatted detection reports
- [x] **Multi-format input** — Plain text, PDF, DOCX, and URLs
- [x] **Full evaluation suite** — Confusion matrix, ROC/PR curves, feature importance, dashboard

---

## Architecture

```
                    Input Text
                        |
          +-------------+-------------+-------------+-------------+
          |             |             |             |             |
          v             v             v             v             v
   +-------------+ +---------+ +-------------+ +---------+ +---------+
   |  DeBERTa    | | XGBoost | |  Binoculars | |  Genre  | |  MLP    |
   |  v3-large   | | (35     | |gpt2-large/  | |Distil-  | | Fusion  |
   |  400M param | | features| |   gpt2-xl   | |  BERT   | |  Head   |
   +-------------+ +---------+ +-------------+ +---------+ +---------+
          |             |             |             |             |
          |       calibrated    normalized    formality      sigmoid
          |      isotonic      percentiles     score         output
          |          +             +             +             |
          +----------+-------------+-------------+             |
                     |                                         |
                     v                                         v
              [DeBERTa_cal, XGB_cal, Bino_AI, Genre_form] -> Final Probability
```

### Signal Details

| # | Signal | Model | What It Detects |
|---|--------|-------|-----------------|
| 1 | **Semantic** | DeBERTa-v3-large (400M) | Deep meaning, coherence, topic drift — strongest overall signal |
| 2 | **Stylometric** | XGBoost (39 features -> ~35 after pruning) | Sentence variance, word entropy, AI filler phrases, punctuation patterns |
| 3 | **Perplexity** | Binoculars (gpt2-large / gpt2-xl) | Cross-model CE ratio that cancels topic effects — catches out-of-distribution AI |
| 4 | **Genre** | DistilBERT 4-class (66M) | Formality score `P(encyclopedic) + P(academic)` — teaches context-aware weighting |

### Fusion Layer

```python
MLP Architecture: 4 -> 32 -> 16 -> 1 (Sigmoid)

Input: [DeBERTa_calibrated, XGBoost_calibrated, Binoculars_AI_signal, Genre_formality_score]
       |__________________________________________________________________________________|
                                          |
                              Context-dependent weighting learned
                              Genre=high  -> trust XGBoost + Binoculars more
                              Genre=low   -> trust DeBERTa more
```

---

## Installation

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11 |
| CUDA | 11.8 | 12.1 |
| VRAM (training) | 8 GB | 12+ GB |
| VRAM (inference only) | 6 GB | 8+ GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-text-detector.git
cd ai-text-detector

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install xgboost scikit-learn pandas numpy
pip install joblib tqdm

# 4. Install feature extraction dependencies
pip install textstat nltk langdetect pyarrow

# 5. Install inference UI dependencies
pip install rich reportlab pymupdf python-docx

# 6. Install visualization dependencies
pip install matplotlib seaborn

# 7. Download NLTK data (done automatically on first run, or manually)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## Quick Start

### Option A: I just want to run inference

```bash
# Run the interactive detection CLI
python src/inference.py
```

Then paste text, select a file, or enter a URL.

### Option B: Full training from scratch

```bash
# Step 1: Prepare dataset (downloads from HuggingFace + processes)
python src/data_pipeline.py

# Step 2: Train all models
python src/train_deberta.py      # ~15-25 hours on RTX 4060
python src/train_xgboost.py      # First pass: all 39 features
python src/train_xgboost.py      # Second pass: pruned features (recommended)
python src/train_genre.py        # ~1-2 hours
python src/train_ensemble.py     # ~30 minutes (uses cached signals)

# Step 3: Evaluate
python src/evaluate.py           # Generates 5 figures + metrics report

# Step 4: Run inference
python src/inference.py
```

---

## Usage

### Training Pipeline

Each component is trained independently and cached:

| Step | Command | Output | Time |
|------|---------|--------|------|
| Data | `python src/data_pipeline.py` | `data/splits/*.csv` | ~1 hour |
| DeBERTa | `python src/train_deberta.py` | `models/deberta/best_model/` | ~15-25h |
| XGBoost | `python src/train_xgboost.py` | `models/xgboost/xgb_model.json` | ~10 min |
| Genre | `python src/train_genre.py` | `models/genre/best_model/` | ~1-2h |
| Ensemble | `python src/train_ensemble.py` | `models/ensemble/mlp_fusion.pt` | ~30 min |

**Note**: `train_xgboost.py` should be run **twice**. The first run trains on all 39 features and auto-saves a pruned feature list. The second run retrains on only the high-importance features for a cleaner signal.

### Running Inference

```bash
python src/inference.py
```

**Interactive menu:**

```
+----------------------------------------------------------+
|                   AI Text Detector                       |
|  DeBERTa-v3-large · XGBoost · Binoculars · Ensemble MLP  |
+----------------------------------------------------------+

[1] Paste text
[2] Analyze single file (PDF/DOCX/TXT)
[3] Batch analyze files
[4] Analyze URL
[q] Quit

Choose: _
```

**Example output:**

```
+----------------------------------------------------------+
| Detection Result                                         |
+----------------------------------------------------------+
|  🤖  AI  (87.3% AI probability)                         |
|                                                          |
|  Confidence  ████████████████████████░░░░░░░░░░  87.3%  |
|                                                          |
|  DeBERTa prob     :  92.1%                               |
|  XGBoost prob     :  78.4%                               |
|  Binoculars ratio :  0.823  (AI signal: 71.2%)          |
|  Genre            :  general  (formality: 23.4%)        |
|                                                          |
|  Text preview: The process of photosynthesis...          |
+----------------------------------------------------------+
```

**Post-analysis options:**
- `[1]` Heatmap — sentence-by-sentence breakdown with color coding
- `[2]` Explain — full signal bar chart + key stylometric features
- `[3]` Report — save results as PDF

### Programmatic Usage

```python
from src.inference import DetectorModels

# Load all models (happens once, cached)
models = DetectorModels()
models.load()

# Analyze a single text
result = models.predict("Your text here...")
print(f"Label: {result['label']} ({result['final_prob']*100:.1f}%)")
# Label: AI (87.3%)

# Access individual signals
print(f"DeBERTa: {result['deberta_prob']:.3f}")
print(f"XGBoost: {result['xgb_prob']:.3f}")
print(f"Binoculars: {result['bino_ai_signal']:.3f}")
print(f"Genre formality: {result['genre_formality']:.3f}")

# Sentence-level analysis
sentences = models.predict_sentences(long_text)
for s in sentences:
    print(f"[{s['label']:5s} | {s['prob']:.2f}] {s['sentence'][:60]}...")
```

### Evaluation

```bash
python src/evaluate.py
```

Generates in `reports/eval_figures/`:

| Figure | File | Description |
|--------|------|-------------|
| <img src="reports/eval_figures/1_confusion_matrix.png" width="200"> | `1_confusion_matrix.png` | TP/TN/FP/FN with labels |
| <img src="reports/eval_figures/2_roc_pr_curves.png" width="200"> | `2_roc_pr_curves.png` | ROC + PR curves per signal |
| <img src="reports/eval_figures/3_score_distributions.png" width="200"> | `3_score_distributions.png` | Score histograms by class |
| <img src="reports/eval_figures/4_feature_importance.png" width="200"> | `4_feature_importance.png` | Top 20 XGBoost features |
| <img src="reports/eval_figures/5_dashboard.png" width="400"> | `5_dashboard.png` | Full system dashboard |

Also writes `reports/eval_summary.txt` with complete metrics.

---

## Dataset

### Sources

| Source | Class | Size | Format |
|--------|-------|------|--------|
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | AI (ChatGPT) | ~60K | JSONL |
| [RAID](https://raid-bench.github.io/) | Mixed | Variable | Parquet/CSV |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat) | AI (GPT-4) | ~15K | Parquet |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | AI (Claude) | Variable | JSONL.GZ |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Human | ~20K | Streaming |
| [arXiv](https://huggingface.co/datasets/ccdv/arxiv-classification) | Human | ~20K | Streaming |

### Statistics

```
Total samples : 240,000
  Human       : 120,000 (50%)
  AI          : 120,000 (50%)

Split         : 70% / 15% / 15%
  Train       : 168,000
  Validation  :  36,000
  Test        :  36,000

Text length   : 50-512 words
Language      : English (filtered via langdetect)
```

### Data Pipeline

```
Raw datasets
    |
    v
Text cleaning (URLs, unicode normalization, control chars)
    |
    v
Language filter (English only via langdetect)
    |
    v
Quality filters (50-512 words, >=20% unique tokens)
    |
    v
Deduplication (MD5 fingerprinting)
    |
    v
Balanced sampling (50/50 stratified)
    |
    v
Train / Validation / Test split (70/15/15)
    |
    v
data/splits/train.csv, val.csv, test.csv
```

---

## Project Structure

```
ai-text-detector/
|
+-- src/
|   +-- data_pipeline.py          # Multi-source dataset aggregation
|   +-- features.py               # 39-feature stylometric extractor
|   +-- train_deberta.py          # DeBERTa-v3-large fine-tuning
|   +-- train_xgboost.py          # XGBoost classifier (with auto-pruning)
|   +-- train_genre.py            # DistilBERT 4-class genre classifier
|   +-- train_ensemble.py         # MLP fusion + isotonic calibration
|   +-- genre_classifier.py       # Genre model inference wrapper
|   +-- inference.py              # Interactive CLI detection pipeline
|   +-- evaluate.py               # Evaluation suite + visualization
|
+-- data/
|   +-- raw/                      # Downloaded source datasets
|   |   +-- hc3/                  # HC3 JSONL files
|   |   +-- raid/                 # RAID parquet/CSV files
|   |   +-- wildchat/             # WildChat parquet files
|   |   +-- anthropic/            # Anthropic JSONL.GZ files
|   |
|   +-- splits/                   # Generated train/val/test CSVs
|   |   +-- train.csv
|   |   +-- val.csv
|   |   +-- test.csv
|   |
|   +-- processed/                # Cached signals and features
|       +-- deberta_probs_all.npy
|       +-- xgb_probs_all.npy
|       +-- binoculars_all.npy
|       +-- genre_all.npy
|       +-- *_features_*.npy
|
+-- models/
|   +-- deberta/
|   |   +-- best_model/           # Fine-tuned DeBERTa checkpoint
|   |
|   +-- xgboost/
|   |   +-- xgb_model.json        # Trained XGBoost model
|   |   +-- scaler.pkl            # Feature StandardScaler
|   |   +-- active_features.json  # Feature names used
|   |   +-- feature_importance.csv
|   |   +-- pruned_features.json  # Auto-pruned feature list
|   |
|   +-- genre/
|   |   +-- best_model/           # Fine-tuned DistilBERT
|   |
|   +-- ensemble/
|       +-- mlp_fusion.pt         # Trained fusion MLP
|       +-- calibrators.pkl       # IsotonicRegression calibrators
|
+-- reports/
|   +-- eval_figures/             # Evaluation visualizations
|   +-- eval_summary.txt          # Metrics summary
|
+-- logs/                         # Training and pipeline logs
+-- README.md
+-- LICENSE
+-- .gitignore
+-- requirements.txt
```

---

## Model Performance

### Ensemble Results (Test Set)

| Metric | Score |
|--------|-------|
| **F1 Score** | **0.94 - 0.97** |
| **AUC-ROC** | **0.98 - 0.99** |
| **Precision** | 0.93 - 0.96 |
| **Recall** | 0.95 - 0.98 |
| False Positive Rate | 2-5% |
| False Negative Rate | 2-5% |

### Individual Signal Comparison

| Signal | F1 | AUC-ROC | Strengths |
|--------|----|---------|-----------|
| **Ensemble (MLP)** | **0.95+** | **0.985+** | Best overall, context-aware |
| DeBERTa | 0.92 | 0.975 | Strong semantic signal |
| XGBoost | 0.88 | 0.960 | Catches stylometric patterns |
| Binoculars | 0.85 | 0.940 | OOD detection, generator-agnostic |
| Genre | 0.70 | 0.850 | Enables context-aware weighting |

### What the Genre Signal Fixes

| Text Type | Before (no genre) | After (with genre) |
|-----------|-------------------|-------------------|
| Wikipedia article | 85% AI (false positive!) | 15% AI (correct) |
| Academic abstract | 80% AI (false positive!) | 20% AI (correct) |
| Casual Reddit post | 45% AI (correct) | 40% AI (correct) |
| ChatGPT response | 95% AI (correct) | 92% AI (correct) |

---

## Key Design Decisions

<details>
<summary><b>Why 4 signals instead of 1 strong model?</b></summary>

No single detection signal is reliable across all text types:
- **DeBERTa** saturates on formal/encyclopedic text (high confidence, often wrong)
- **XGBoost** features collapse on very short texts (< 50 words)
- **Binoculars** uses GPT-2 which is weak at understanding modern LLM patterns
- **Genre alone** is not a detector — it's a context signal

The MLP fusion layer learns when to trust each signal based on the genre formality score.

</details>

<details>
<summary><b>Why clip DeBERTa at 0.99?</b></summary>

DeBERTa's softmax frequently outputs exactly 1.0 on formal text. Without clipping:
- Raw 1.00 -> calibrated 0.98 (high AI confidence)
- Raw 0.99 -> calibrated 0.51 (suddenly uncertain)

This creates a massive cliff where tiny logit differences cause huge calibrated probability swings. The 0.99 clip prevents this saturation artifact.

</details>

<details>
<summary><b>Why Binoculars instead of raw perplexity?</b></summary>

Binoculars (Hans et al., 2024) computes `CE(gpt2-large) / CE(gpt2-xl)`. This ratio:
- **Cancels topic effects** — both models see the same vocabulary
- **Cancels length effects** — ratio normalizes sequence length
- **Works across all generators** — GPT-2 shares WebText distribution with virtually every LLM

Raw perplexity with a modern model (e.g., Qwen2.5) fails because GPT-2/3 output looks "rough" to it, giving high perplexity (same as human text) and inverting the signal.

</details>

<details>
<summary><b>Why is genre a separate model?</b></summary>

Genre detection requires different training data (Wikipedia, arXiv, Reddit, News) and a different objective (4-way classification vs. binary). A separate DistilBERT:
- Is trained once and frozen
- Adds only ~66M parameters (vs 400M for DeBERTa)
- Enables the MLP to learn context-dependent signal weighting
- Achieves 99%+ genre classification F1

</details>

---

## Contributing

Contributions are welcome! Here are some areas where help is appreciated:

- [ ] Support for additional languages (currently English only)
- [ ] Additional data sources for more diverse human writing
- [ ] Quantized model variants for CPU-only inference (GPTQ, GGUF)
- [ ] REST API server wrapper for the inference pipeline
- [ ] Benchmarking against other open-source detectors
- [ ] Web-based demo (Gradio or Streamlit)

### Development Setup

```bash
# Install dev dependencies
pip install black ruff pytest

# Format code
black src/

# Lint
ruff check src/

# Run tests
pytest tests/
```

Please open an issue to discuss major changes before submitting a pull request.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The model weights and derived detectors follow the licenses of their base models:
- **DeBERTa-v3-large**: MIT License (Microsoft)
- **DistilBERT**: Apache 2.0 (Hugging Face)
- **GPT-2 / GPT-2-XL**: Modified MIT (OpenAI)

Dataset usage is subject to the terms of each respective dataset license.

---

## Acknowledgments

- **DeBERTa-v3**: [He et al., 2021](https://arxiv.org/abs/2111.09543) — Microsoft Research
- **Binoculars**: [Hans et al., 2024](https://arxiv.org/abs/2401.12070) — Zero-Shot Detection of Machine-Generated Text
- **Datasets**: HC3, RAID, WildChat, Anthropic HH-RLHF, Wikipedia, arXiv — respective authors
- **Hugging Face Transformers** and **Datasets** libraries for making NLP research accessible

---

<div align="center">

**If this project is helpful, please consider giving it a star!** ⭐

</div>
