# SimpleTM — Simple Baseline for Multivariate Time Series Forecasting

> **BTP Extension**: This repository extends the original [SimpleTM](https://github.com/vsingh-group/SimpleTM) with **SWT-Only** and **FFT-Only** tokenization variants for ablation study.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Variants](#model-variants)
- [Pipeline Flow](#pipeline-flow)
- [Code Structure & Function Reference](#code-structure--function-reference)
- [Setup & Installation](#setup--installation)
- [Running Experiments](#running-experiments)
- [Dataset](#dataset)

---

## Overview

SimpleTM is a lightweight yet effective baseline for **Multivariate Time Series Forecasting (MTSF)**. It combines:

1. **Channel-independent inverted embedding** — treats each variate as a token
2. **Stationary Wavelet Transform (SWT)** — multi-scale signal decomposition for tokenization
3. **Geometric Attention** — novel attention combining dot product and wedge product scores

This project extends SimpleTM with two ablation variants that change **only the tokenization strategy** while keeping the geometric attention mechanism intact:

| Model | Tokenization | Attention | Description |
|-------|-------------|-----------|-------------|
| `SimpleTM` | SWT | Geometric (Wedge) | Original model |
| `SimpleTM_SWT` | SWT | Geometric (Wedge) | SWT-only baseline (same as original, for clean comparison) |
| `SimpleTM_FFT` | FFT | Geometric (Wedge) | FFT spectral band tokenization |

---

## Architecture

### Original SimpleTM Pipeline

```
Input: (B, L, N) — Batch, Sequence Length, Number of Variates
         │
         ▼
┌─────────────────────────────┐
│  Instance Normalization     │  Subtract mean, divide by std (RevIN-style)
│  (RevIN)                    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Inverted Embedding         │  (B, L, N) → (B, N, d_model)
│  DataEmbedding_inverted     │  Transpose + Linear projection
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  SimpleTM Encoder Layer (×e_layers)                 │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  GeomAttentionLayer                         │    │
│  │                                             │    │
│  │  1. SWT Decomposition  (B,N,d) → (B,N,m+1,d)│   │
│  │     └─ Multi-level wavelet transform        │    │
│  │                                             │    │
│  │  2. Q/K/V Linear Projections                │    │
│  │     └─ + Dropout                            │    │
│  │                                             │    │
│  │  3. GeomAttention (Wedge Product Scoring)   │    │
│  │     └─ scores = (1-α)·dot + α·wedge_norm   │    │
│  │     └─ softmax → weighted values            │    │
│  │                                             │    │
│  │  4. Output Projection + SWT Reconstruction  │    │
│  │     └─ (B,N,m+1,d) → (B,N,d)              │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  5. Residual Connection + LayerNorm                 │
│  6. Feed-Forward Network (Conv1d → Conv1d)          │
│  7. Residual Connection + LayerNorm                 │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Output Projection          │  (B, N, d_model) → (B, H, N)
│  Linear(d_model, pred_len)  │  Permute to forecast shape
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  De-normalization           │  Reverse the RevIN normalization
└────────────┬────────────────┘
             │
             ▼
Output: (B, H, N) — Batch, Horizon, Number of Variates
```

### FFT-Only Variant Difference

In `SimpleTM_FFT`, the SWT decomposition/reconstruction is replaced:

```
SWT Model:  SWT Decomposition → [Attention] → SWT Reconstruction
FFT Model:  FFT Band Decomposition → [Attention] → FFT Band Summation

FFT Decomposition:
    Input signal → torch.fft.rfft → Split spectrum into (m+1) bands
    → torch.fft.irfft per band → (m+1) time-domain band signals

    Band 0: Lowest frequencies  (≈ SWT approximation coefficients)
    Band 1: Low-mid frequencies (≈ SWT level-1 detail coefficients)
    ...
    Band m: Highest frequencies (≈ SWT level-m detail coefficients)

FFT Reconstruction:
    Sum all weighted frequency band signals → Reconstructed output
```

---

## Model Variants

### 1. SimpleTM (Original)

**Files**: `model/SimpleTM.py`, `layers/SWTAttention_Family.py`

Uses Stationary Wavelet Transform for multi-scale decomposition and geometric attention with wedge product scoring.

### 2. SimpleTM_SWT (SWT-Only Baseline)

**Files**: `model/SimpleTM_SWT.py`, `layers/SWTAttention_Family.py`

Functionally identical to the original SimpleTM. Maintained as a separate model file for clean ablation study comparison with consistent naming.

### 3. SimpleTM_FFT (FFT-Only Tokenization)

**Files**: `model/SimpleTM_FFT.py`, `layers/FFTAttention_Family.py`

Replaces SWT tokenization with FFT spectral band decomposition. The frequency spectrum is divided into `m+1` bands with learnable weights. Geometric attention (wedge product) is unchanged.

---

## Pipeline Flow

```
Raw CSV Data
    │
    ▼
┌──────────────────┐    data_provider/data_loader.py
│  Dataset Loading  │    (Dataset_Custom, Dataset_ETT_hour, etc.)
│  + Normalization  │    StandardScaler + train/val/test split
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    data_provider/data_factory.py
│  DataLoader       │    Batching, shuffling, num_workers
│  (PyTorch)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    experiments/exp_long_term_forecasting.py
│  Training Loop    │    AdamW optimizer, early stopping
│  Exp_Long_Term    │    MSE/L1 loss + L1 attention regularization
│  _Forecast        │    Learning rate scheduling
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    model/SimpleTM*.py + layers/*
│  Model Forward    │    Normalization → Embedding → Encoder → Projection
│  Pass             │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    utils/metrics.py
│  Evaluation       │    MSE, MAE, RMSE, MAPE, MSPE
│  Metrics          │
└──────────────────┘
```

---

## Code Structure & Function Reference

```
simpleTMG/
├── run.py                          # Entry point — argument parsing, training/testing loop
├── model/
│   ├── SimpleTM.py                 # Original model
│   ├── SimpleTM_SWT.py            # SWT-only baseline (identical to original)
│   └── SimpleTM_FFT.py            # FFT tokenization variant
├── layers/
│   ├── Embed.py                    # Inverted embedding layer
│   ├── SWTAttention_Family.py     # SWT tokenization + Geometric attention
│   ├── FFTAttention_Family.py     # FFT tokenization + Geometric attention
│   ├── Transformer_Encoder.py    # Encoder + EncoderLayer
│   └── StandardNorm.py            # RevIN normalization
├── experiments/
│   ├── exp_basic.py               # Base experiment class (model registry)
│   └── exp_long_term_forecasting.py  # Training/validation/testing logic
├── data_provider/
│   ├── data_factory.py            # Dataset and DataLoader factory
│   └── data_loader.py            # Dataset classes for all benchmarks
├── utils/
│   ├── metrics.py                 # MSE, MAE, RMSE, MAPE, MSPE
│   ├── tools.py                   # EarlyStopping, LR scheduling, visualization
│   ├── masking.py                 # Masking utilities
│   └── timefeatures.py           # Time feature engineering
├── scripts/
│   └── multivariate_forecasting/  # Shell scripts for each dataset
└── SimpleTM_Kaggle.ipynb          # Kaggle notebook for running experiments
```

### Detailed Function Reference

#### `run.py`
- **Main entry point**: Parses all arguments, creates experiment, runs train/test loop
- Key arguments: `--model` (SimpleTM/SimpleTM_SWT/SimpleTM_FFT), `--wv` (wavelet type), `--m` (decomposition levels), `--alpha` (dot vs wedge weight)

#### `layers/Embed.py`
- **`DataEmbedding_inverted`**: Inverted channel embedding
  - `__init__(c_in, d_model, ...)`: Creates `nn.Linear(seq_len, d_model)` projection
  - `forward(x, x_mark)`: Transposes `(B,L,N)` → `(B,N,L)`, applies linear projection → `(B,N,d_model)`

#### `layers/SWTAttention_Family.py`
- **`WaveletEmbedding`**: SWT decomposition and reconstruction
  - `__init__(d_channel, swt, requires_grad, wv, m, kernel_size)`: Initializes wavelet filters (h0, h1) from PyWavelets
  - `forward(x)`: Calls `swt_decomposition` or `swt_reconstruction`
  - `swt_decomposition(x, h0, h1, depth, kernel_size)`: Multi-level SWT using circular padding + grouped 1D convolution with increasing dilation. Returns `(B,N,m+1,L)` — stacked [approx, detail_m, ..., detail_1]
  - `swt_reconstruction(coeffs, g0, g1, m, kernel_size)`: Inverse SWT using reconstruction filters
  
- **`GeomAttentionLayer`**: Wraps tokenization + attention
  - `__init__(attention, d_model, ...)`: Creates SWT embeddings, Q/K/V projections, output projection with inverse SWT
  - `forward(queries, keys, values, ...)`: SWT decompose → project Q/K/V → geometric attention → output project + SWT reconstruct
  
- **`GeomAttention`**: Novel geometric attention mechanism
  - `__init__(mask_flag, factor, scale, attention_dropout, output_attention, alpha)`: Sets up attention parameters
  - `forward(queries, keys, values, ...)`: Computes attention scores as `(1-α)·dot_product + α·wedge_norm`, where wedge_norm = `√(|q|²|k|² - (q·k)²)` (exterior product magnitude). Returns weighted values and mean absolute attention scores for L1 regularization.

#### `layers/FFTAttention_Family.py` *(New)*
- **`FFTEmbedding`**: FFT-based tokenization
  - `__init__(d_channel, decompose, m)`: Initialize with learnable band weights
  - `fft_decomposition(x)`: `rfft` → split into `m+1` frequency bands → `irfft` per band → stack as `(B,N,m+1,L)`
  - `fft_reconstruction(coeffs)`: Weighted sum of all frequency bands → `(B,N,L)`
  
- **`FFTGeomAttentionLayer`**: FFT tokenization + Geometric attention
  - Same interface as `GeomAttentionLayer` but uses `FFTEmbedding` instead of `WaveletEmbedding`

#### `layers/Transformer_Encoder.py`
- **`EncoderLayer`**: Single transformer encoder layer
  - `forward(x, ...)`: Attention → residual + LayerNorm → FFN (Conv1d) → residual + LayerNorm
  
- **`Encoder`**: Stack of encoder layers
  - `forward(x, ...)`: Sequential pass through all layers, applies final LayerNorm

#### `layers/StandardNorm.py`
- **`Normalize`**: Reversible instance normalization (RevIN)
  - `forward(x, mode='norm'|'denorm')`: Normalize or de-normalize

#### `model/SimpleTM.py` (and `SimpleTM_SWT.py`, `SimpleTM_FFT.py`)
- **`Model`**: Main model class
  - `__init__(configs)`: Builds embedding → encoder → projector
  - `forecast(x_enc, ...)`: Full forward pass: normalize → embed → encode → project → denormalize
  - `forward(x_enc, ...)`: Calls `forecast()`

#### `experiments/exp_basic.py`
- **`Exp_Basic`**: Base experiment class
  - `__init__(args)`: Device setup, model building, parameter counting
  - `model_dict`: Registry mapping model names to modules

#### `experiments/exp_long_term_forecasting.py`
- **`Exp_Long_Term_Forecast`**: Full experiment lifecycle
  - `train(setting)`: Training loop — data loading, AdamW optimizer, MSE loss + L1 attention regularization, early stopping, LR scheduling
  - `vali(vali_data, vali_loader, criterion)`: Validation loop
  - `test(setting)`: Testing loop — computes MSE, MAE, RMSE, MAPE, MSPE, saves visualizations
  - `predict(setting)`: Inference on unseen data

#### `data_provider/data_factory.py`
- **`data_provider(args, flag)`**: Factory function returning dataset + dataloader for given flag (train/val/test/pred)

#### `data_provider/data_loader.py`
- **`Dataset_Custom`**: General CSV dataset with 70/10/20 train/val/test split
- **`Dataset_ETT_hour`** / **`Dataset_ETT_minute`**: ETT benchmark datasets
- **`Dataset_PEMS`**: Traffic flow dataset (`.npz` format)
- **`Dataset_Solar`**: Solar energy dataset
- **`Dataset_Pred`**: Prediction-only dataset for inference

#### `utils/metrics.py`
- **`metric(pred, true)`**: Returns (MAE, MSE, RMSE, MAPE, MSPE)
- Individual functions: `RSE`, `CORR`, `MAE`, `MSE`, `RMSE`, `MAPE`, `MSPE`

#### `utils/tools.py`
- **`EarlyStopping`**: Patience-based early stopping with model checkpointing
- **`adjust_learning_rate`**: Multiple LR schedules (type1 halving, type2 manual, TST OneCycleLR)
- **`visual(true, preds, name)`**: Plot ground truth vs predictions
- **`StandardScaler`**: Manual z-score normalization

---

## Setup & Installation

### Environment

```bash
# Create conda environment
conda create -n simpletm python=3.10
conda activate simpletm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio

# Install dependencies
pip install numpy pandas scikit-learn matplotlib PyWavelets
```

### From `environment.yml`

```bash
conda env create -f environment.yml
conda activate simpletm
```

---

## Running Experiments

### Original SimpleTM (SWT + Geometric Attention)

```bash
python -u run.py \
  --is_training 1 \
  --model SimpleTM \
  --model_id ETTh1 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --wv db1 \
  --m 3 \
  --alpha 1.0 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10
```

### SWT-Only Baseline

```bash
python -u run.py \
  --is_training 1 \
  --model SimpleTM_SWT \
  --model_id ETTh1_SWT \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --wv db1 \
  --m 3 \
  --alpha 1.0 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10
```

### FFT-Only Tokenization

```bash
python -u run.py \
  --is_training 1 \
  --model SimpleTM_FFT \
  --model_id ETTh1_FFT \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --m 3 \
  --alpha 1.0 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10
```

> **Note**: For the FFT model, `--wv` and `--kernel_size` arguments are ignored since there are no wavelet filters. The `--m` argument controls the number of FFT frequency bands (m+1 total).

---

## Dataset

The primary dataset for this BTP project is available at:

📦 **[Google Drive Link](https://drive.google.com/file/d/1hTpUrhe1yEIGa9mCiGxM5rDyzlYKAnyx/view?usp=sharing)**

Place the downloaded data in `./dataset/` directory following the structure expected by `data_provider/data_loader.py`.

### Supported Benchmark Datasets

| Dataset | Type | Features | Data Loader |
|---------|------|----------|-------------|
| ETTh1/ETTh2 | Electricity Transformer | 7 | `Dataset_ETT_hour` |
| ETTm1/ETTm2 | Electricity Transformer | 7 | `Dataset_ETT_minute` |
| ECL | Electricity Consumption | 321 | `Dataset_Custom` |
| Weather | Weather Stations | 21 | `Dataset_Custom` |
| Traffic | Road Occupancy | 862 | `Dataset_Custom` |
| PEMS | Traffic Flow | 170-883 | `Dataset_PEMS` |
| Solar | Solar Energy | 137 | `Dataset_Solar` |

---

## References

- **SimpleTM Paper**: [A Simple Baseline for Multivariate Time Series Forecasting](https://openreview.net/forum?id=SimpleTM)
- **Original Repository**: [https://github.com/vsingh-group/SimpleTM](https://github.com/vsingh-group/SimpleTM)