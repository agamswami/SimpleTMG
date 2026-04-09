# SimpleTMG

SimpleTMG extends the original SimpleTM baseline into a modular forecasting framework with:

- four tokenization branches: `SWT`, `FFT`, `Conv`, and `Hybrid`
- two attention modes: `original` and `dual`
- one shared training pipeline for controlled ablations across datasets

The codebase keeps the original SimpleTM path intact while making tokenization and attention independently switchable.

## Overview

The project studies multivariate time-series forecasting under two independent design axes.

1. Tokenization
- `SimpleTM` or `SimpleTM_SWT`: stationary wavelet tokenization
- `SimpleTM_FFT`: FFT band tokenization
- `SimpleTM_Conv`: multi-scale convolutional tokenization
- `SimpleTM_Hybrid`: gated fusion of `SWT + FFT + Conv`

2. Attention
- `attention_mode=original`: original SimpleTM geometric attention
- `attention_mode=dual`: original branch plus a parallel temporal self-attention branch

That gives the following experimental matrix:

| Tokenization | Original Attention | Dual Attention |
| --- | --- | --- |
| SWT | `SimpleTM_SWT` | `SimpleTM_SWT --attention_mode dual` |
| FFT | `SimpleTM_FFT` | `SimpleTM_FFT --attention_mode dual` |
| Conv | `SimpleTM_Conv` | `SimpleTM_Conv --attention_mode dual` |
| Hybrid | `SimpleTM_Hybrid` | `SimpleTM_Hybrid --attention_mode dual` |

`SimpleTM` is also available as the original SWT-based baseline.

## Architecture

At a high level the forecasting path is still:

```text
Input series (B, L, N)
  -> reversible normalization
  -> inverted embedding
  -> encoder layer(s)
  -> horizon projection
  -> de-normalization
  -> forecast (B, H, N)
```

### Inverted Embedding

The embedding layer in [layers/Embed.py](layers/Embed.py) maps:

```text
(B, L, N) -> (B, N, d_model)
```

Each variate becomes a token, and its temporal history is projected into a latent vector.

### Tokenization Branches

All tokenizer variants preserve the same downstream shape contract:

```text
(B, N, d_model) -> (B, N, m + 1, d_model)
```

This keeps the geometric encoder structure comparable across branches.

#### SWT

Implemented in [layers/SWTAttention_Family.py](layers/SWTAttention_Family.py).

- stationary wavelet transform over the embedded representation
- multi-scale decomposition with `m + 1` coefficients
- strong fit for non-stationary, locally varying structure

#### FFT

Implemented in [layers/FFTAttention_Family.py](layers/FFTAttention_Family.py).

- splits the latent representation into `m + 1` frequency bands
- reconstructs bands in the time domain for attention processing
- useful when periodic or compact spectral structure dominates

#### Conv

Implemented in [layers/ConvAttention_Family.py](layers/ConvAttention_Family.py).

- applies multi-scale depthwise 1D convolutions with configurable odd kernel sizes
- learns local motifs and short-range patterns directly from data
- useful for sharp transitions, local bursts, and short receptive-field structure

#### Hybrid

Implemented in [layers/HybridAttention_Family.py](layers/HybridAttention_Family.py).

- runs `SWT`, `FFT`, and `Conv` tokenizers in parallel
- fuses them with a learned softmax gate
- keeps the encoder fixed while making the tokenizer adaptive

Conceptually:

```text
T_swt  = SWT(x)
T_fft  = FFT(x)
T_conv = Conv(x)
T_fused = softmax_gate(T_swt, T_fft, T_conv)
```

### Original Geometric Attention

The original SimpleTM attention is preserved across all branches. It mixes:

- dot-product similarity
- wedge-product magnitude

Conceptually:

```text
score = (1 - alpha) * dot(q, k) + alpha * wedge_norm(q, k)
```

Implemented through:

- [layers/SWTAttention_Family.py](layers/SWTAttention_Family.py)
- [layers/FFTAttention_Family.py](layers/FFTAttention_Family.py)
- [layers/ConvAttention_Family.py](layers/ConvAttention_Family.py)
- [layers/HybridAttention_Family.py](layers/HybridAttention_Family.py)

### Dual Attention

Implemented in [layers/ParallelAttention_Family.py](layers/ParallelAttention_Family.py).

The dual mode adds a parallel temporal attention branch beside the original branch:

```text
embedded tokens
  -> original branch (tokenizer + geometric attention)
  -> temporal branch (time-axis self-attention)
  -> learned fusion gate
  -> fused encoder output
```

The temporal branch attends across the latent time axis after transposing the inverted embedding. The fusion gate is a small MLP that outputs one scalar gate per sample and mixes:

```text
out = g * original_branch + (1 - g) * temporal_branch
```

This keeps the original mechanism available while testing whether explicit temporal reasoning adds value.

## Model Files

Model entrypoints:

- [model/SimpleTM.py](model/SimpleTM.py)
- [model/SimpleTM_SWT.py](model/SimpleTM_SWT.py)
- [model/SimpleTM_FFT.py](model/SimpleTM_FFT.py)
- [model/SimpleTM_Conv.py](model/SimpleTM_Conv.py)
- [model/SimpleTM_Hybrid.py](model/SimpleTM_Hybrid.py)

Experiment and training flow:

- [experiments/exp_basic.py](experiments/exp_basic.py)
- [experiments/exp_long_term_forecasting.py](experiments/exp_long_term_forecasting.py)
- [run.py](run.py)

Data pipeline:

- [data_provider/data_factory.py](data_provider/data_factory.py)
- [data_provider/data_loader.py](data_provider/data_loader.py)

## Running Experiments

Core switches:

- `--model`: `SimpleTM`, `SimpleTM_SWT`, `SimpleTM_FFT`, `SimpleTM_Conv`, `SimpleTM_Hybrid`
- `--attention_mode`: `original` or `dual`
- `--conv_kernel_sizes`: comma-separated odd kernels for the Conv and Hybrid branches

Examples:

```bash
python run.py --is_training 1 --model SimpleTM_SWT --attention_mode original ...
python run.py --is_training 1 --model SimpleTM_FFT --attention_mode dual ...
python run.py --is_training 1 --model SimpleTM_Conv --attention_mode original --conv_kernel_sizes 3,5,7,11 ...
python run.py --is_training 1 --model SimpleTM_Hybrid --attention_mode dual --conv_kernel_sizes 3,5,7,11 ...
```

## Metrics

The test pipeline reports:

- `MAE`
- `MSE`
- `RMSE`
- `MAPE`
- `MSPE`
- `RSE`
- `CORR`
- `SMAPE`
- `WAPE`
- `R2`

Metric code is in [utils/metrics.py](utils/metrics.py).

## Kaggle Notebooks

Main experiment notebooks in this repo:

- [results_complete/all-data-set-fixed-graph.ipynb](results_complete/all-data-set-fixed-graph.ipynb)
  original SWT and FFT comparison
- [results_complete/all-data-set-dual-attention.ipynb](results_complete/all-data-set-dual-attention.ipynb)
  SWT and FFT with `original` and `dual` attention
- [simpletmg-alldataset-conv-hybrid.ipynb](simpletmg-alldataset-conv-hybrid.ipynb)
  all eight variants: SWT, FFT, Conv, and Hybrid with both attention modes

EDA notebooks and notes:

- [eda-analysis.ipynb](eda-analysis.ipynb)
- [EDA-model-justification-all-datasets-with-smart.ipynb](EDA-model-justification-all-datasets-with-smart.ipynb)
- [readmenew.md](readmenew.md)

## Datasets

The repo works with the benchmark datasets already wired through the loader layer, including:

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `electricity`
- `weather`
- `traffic`
- `Solar`
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`
- `smartbuilding`

Dataset-specific loading is handled in [data_provider/data_loader.py](data_provider/data_loader.py).

## Current Status

Implemented:

- SWT tokenization
- FFT tokenization
- Conv tokenization
- Hybrid gated tokenization
- original geometric attention
- dual attention with a temporal branch
- all-variant Kaggle notebook support

Useful technical note:

- `SimpleTM` and `SimpleTM_SWT` belong to the same SWT family
- `SimpleTM_Hybrid` uses a softmax branch-fusion gate
- `attention_mode=dual` uses a learned gate between original and temporal branches

## References

- Base paper: [A Simple Baseline for Multivariate Time Series Forecasting](https://openreview.net/forum?id=gieyCN1b4d)
- Original repo: [https://github.com/vsingh-group/SimpleTM](https://github.com/vsingh-group/SimpleTM)
