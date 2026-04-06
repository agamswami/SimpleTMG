# SimpleTMG: Architecture Guide with Dual Attention

This document describes the current architecture of the project as it exists in this repository, including:

- the original SimpleTM baseline
- the SWT and FFT tokenization variants
- the new switchable dual-attention extension
- how the data, model, training loop, and experiment notebooks fit together

This file is intended to be a technical project note for implementation, experiments, and report writing.

---

## 1. Project Summary

This repository extends the original **SimpleTM** architecture for multivariate time series forecasting.

The project now has **two independent axes of variation**:

1. **Tokenization choice**
   - `SimpleTM` / `SimpleTM_SWT`: SWT tokenization
   - `SimpleTM_FFT`: FFT tokenization

2. **Attention choice**
   - `attention_mode=original`: original SimpleTM geometric attention only
   - `attention_mode=dual`: original attention branch plus a new parallel temporal attention branch

So the project is no longer just a tokenization ablation. It is now a **tokenization x attention** study.

---

## 2. Main Idea of the Whole Architecture

At a high level, the model still follows the same forecasting pipeline:

```text
Input time series
  -> instance normalization
  -> inverted embedding
  -> encoder layer(s)
  -> linear horizon projection
  -> de-normalization
  -> forecast
```

The main design principle is:

- keep the original SimpleTM path intact
- allow tokenization to be switched independently
- allow attention to be switched independently
- make the new temporal branch parallel, not destructive

That means the original base model remains available for fair comparison.

---

## 3. Full Pipeline

### 3.1 Input and Shapes

The input tensor uses the standard shape:

```text
x: (B, L, N)
```

where:

- `B` = batch size
- `L` = input sequence length (`seq_len`)
- `N` = number of variates / channels

The model predicts:

```text
y_hat: (B, H, N)
```

where `H = pred_len`.

### 3.2 Normalization

Inside the model, each sample is normalized before encoding:

```text
x_norm = (x - mean) / std
```

The predicted output is later de-normalized back to the original scale.

### 3.3 Inverted Embedding

The embedding layer is implemented in `layers/Embed.py`.

Instead of treating time points as tokens, SimpleTM uses **inverted embedding**:

```text
(B, L, N) -> (B, N, d_model)
```

Each variate becomes a token, and its temporal history is projected into a latent vector of size `d_model`.

This is one of the key ideas of SimpleTM:

- token identity = channel / variate
- latent content = compressed temporal history

---

## 4. Tokenization Branches

After inverted embedding, the project supports two tokenization strategies.

### 4.1 SWT Tokenization

Used by:

- `SimpleTM`
- `SimpleTM_SWT`

Implemented in:

- `layers/SWTAttention_Family.py`

Core idea:

- each embedded variate token is decomposed into multi-scale coefficients using a stationary wavelet transform
- the transform preserves length and produces `m + 1` sub-representations

Shape:

```text
(B, N, d_model) -> (B, N, m+1, d_model)
```

This branch emphasizes:

- multi-resolution decomposition
- time-localized structure
- separation of trend and detail information

### 4.2 FFT Tokenization

Used by:

- `SimpleTM_FFT`

Implemented in:

- `layers/FFTAttention_Family.py`

Core idea:

- the latent signal is split into `m + 1` frequency bands using `rfft`
- each band is brought back to the time domain with `irfft`
- reconstruction is a weighted sum of the bands

Shape:

```text
(B, N, d_model) -> (B, N, m+1, d_model)
```

This branch emphasizes:

- spectral decomposition
- frequency-band separation
- comparison with wavelet-style tokenization under the same downstream architecture

---

## 5. Original SimpleTM Attention

The original SimpleTM attention is the geometric attention mechanism from the base paper.

Implemented in:

- `layers/SWTAttention_Family.py`
  - `GeomAttention`
  - `GeomAttentionLayer`
- `layers/FFTAttention_Family.py`
  - `FFTGeomAttentionLayer`

### 5.1 What it does

It computes attention scores using a mixture of:

- dot product similarity
- wedge-product magnitude

In code form, the score is:

```text
score = (1 - alpha) * dot(q, k) + alpha * wedge_norm(q, k)
```

This is designed to capture not only similarity, but also geometric complementarity between tokens.

### 5.2 Role in the Project

This remains the **reference attention mechanism**.

When:

```text
attention_mode = original
```

the model behaves like the original branch only.

---

## 6. New Dual Attention Extension

The new addition is a **parallel temporal attention branch**, inspired by the idea in the DAG paper of using temporal and channel-style reasoning in parallel.

Important clarification:

- DAG is designed for exogenous-aware forecasting
- this repository does **not** have DAG's exogenous-variable setting
- so the implementation here is an **adaptation of the architectural idea**, not a direct reproduction of DAG

The core borrowed idea is:

> use a temporal branch in parallel with the existing branch, then fuse the two.

---

## 7. Dual Attention: Conceptual View

The dual-attention encoder has two branches:

```text
                         +-----------------------------+
                         | Original SimpleTM Branch    |
input embedding -------->| tokenization + geom attn    |----+
                         +-----------------------------+    |
                                                            | fusion
                         +-----------------------------+    v
                         | Temporal Attention Branch   |--> gate --> fused output
input embedding -------->| time-axis self-attention    |
                         +-----------------------------+
```

### Branch 1: Original branch

This is unchanged.

For SWT:

```text
embedded x
  -> SWT decomposition
  -> geometric attention
  -> SWT reconstruction
```

For FFT:

```text
embedded x
  -> FFT decomposition
  -> geometric attention
  -> FFT reconstruction
```

### Branch 2: Temporal branch

This is new.

It operates directly on the inverted embedding output:

```text
(B, N, d_model)
```

To attend over the latent time axis, it transposes the tensor to:

```text
(B, d_model, N)
```

Now:

- sequence axis = latent temporal axis
- feature dimension = channel dimension

Then a standard self-attention style computation is applied:

```text
Q = Wq(X)
K = Wk(X)
V = Wv(X)
A = softmax(Q K^T / sqrt(N))
out = A V
```

and finally it is transposed back to:

```text
(B, N, d_model)
```

This branch is implemented in:

- `layers/ParallelAttention_Family.py`
  - `TemporalAxisAttention`

---

## 8. Dual Attention: Fusion Mechanism

The two branch outputs have the same shape:

```text
channel_out:  (B, N, d_model)
temporal_out: (B, N, d_model)
```

So they can be fused directly.

### 8.1 Current Fusion

The repository currently uses a **learned scalar gate per sample**:

```text
pooled = concat(mean(channel_out over channels),
                mean(temporal_out over channels))

g = sigmoid(MLP(pooled))

fused = g * channel_out + (1 - g) * temporal_out
```

This is implemented in:

- `layers/ParallelAttention_Family.py`
  - `_ParallelAttentionBase`

### 8.2 Why this design was chosen

This is a good compromise for the project because it:

- keeps the model close to the original SimpleTM
- adds very few extra parameters
- makes the branch combination explicit and interpretable
- preserves a clean ablation between original and dual attention

### 8.3 What it is not

The current gate is **not**:

- token-wise
- channel-wise
- head-wise

It is a simple sample-level scalar gate.

That means it is lightweight and easy to analyze, but not the most expressive possible fusion.

### 8.4 Possible future upgrades

The next stronger versions would be:

1. feature-wise gate
2. token-wise gate
3. concat + projection fusion
4. residual sum baseline

These would be useful ablations in a future report or paper section.

---

## 9. Mathematical Summary of the Dual Layer

Let:

- `E(X)` be the inverted embedding
- `C(.)` be the original channel/scale branch
- `T(.)` be the new temporal attention branch
- `G(.)` be the learned gate

Then:

```text
Z = E(X)
H_channel  = C(Z)
H_temporal = T(Z)
g = sigmoid(G([pool(H_channel), pool(H_temporal)]))
H_fused = g * H_channel + (1 - g) * H_temporal
```

The fused representation is then passed into the same encoder residual path and feed-forward block as before.

So the encoder layer logic remains:

```text
attention output
  -> residual add
  -> layer norm
  -> feed-forward network
  -> residual add
  -> layer norm
```

---

## 10. Where the New Code Lives

### 10.1 Core Attention Files

- `layers/SWTAttention_Family.py`
  - original SWT tokenization and geometric attention

- `layers/FFTAttention_Family.py`
  - original FFT tokenization and geometric attention

- `layers/ParallelAttention_Family.py`
  - new temporal branch
  - branch fusion gate
  - wrappers for SWT + dual attention
  - wrappers for FFT + dual attention

### 10.2 Model Files

- `model/SimpleTM.py`
  - original/SWT model
  - now supports `attention_mode`

- `model/SimpleTM_SWT.py`
  - explicit SWT baseline
  - now supports `attention_mode`

- `model/SimpleTM_FFT.py`
  - FFT tokenization model
  - now supports `attention_mode`

### 10.3 Training Entry Point

- `run.py`
  - adds the CLI flag:

```text
--attention_mode [original | dual]
```

---

## 11. Available Model Combinations

The project now supports these meaningful combinations:

| Tokenization | Attention | How to run |
|---|---|---|
| SWT | original | `--model SimpleTM_SWT --attention_mode original` |
| SWT | dual | `--model SimpleTM_SWT --attention_mode dual` |
| FFT | original | `--model SimpleTM_FFT --attention_mode original` |
| FFT | dual | `--model SimpleTM_FFT --attention_mode dual` |

Because `SimpleTM` and `SimpleTM_SWT` are functionally the same tokenization family, `SimpleTM` can also be used with:

```text
--model SimpleTM --attention_mode original
--model SimpleTM --attention_mode dual
```

---

## 12. End-to-End Training Flow

The complete training path is:

```text
run.py
  -> experiments/exp_long_term_forecasting.py
  -> data_provider/data_factory.py
  -> data_provider/data_loader.py
  -> model/SimpleTM*.py
  -> layers/*
  -> utils/metrics.py + utils/tools.py
```

### 12.1 Optimization

The training loop currently uses:

- `AdamW`
- task loss from `MSELoss` for most datasets
- extra `L1` regularization term from the returned attention magnitude
- early stopping
- learning-rate scheduling

### 12.2 Output artifacts

Training produces:

- model checkpoints
- result text logs
- PDF prediction plots during testing

---

## 13. Experiment Notebooks

### Existing notebook

- `results_complete/all-data-set-fixed-graph.ipynb`

This is the original multi-dataset notebook for:

- SWT with original attention
- FFT with original attention

### New notebook

- `results_complete/all-data-set-dual-attention.ipynb`

This extends the Kaggle workflow to run:

- `SWT_original`
- `SWT_dual`
- `FFT_original`
- `FFT_dual`

across all configured datasets.

---

## 14. Why the Dual Attention Matters

The original SimpleTM branch is strong at:

- channel-aware latent interactions
- scale-aware processing through SWT/FFT tokenization
- geometric scoring beyond plain dot-product attention

The new temporal branch adds:

- explicit attention over the latent time axis
- a second path that can focus on temporal ordering and cross-position dependencies
- a way to test whether channel/scale attention alone is sufficient

In short:

- original branch = channel/scale-centric latent reasoning
- temporal branch = time-axis latent reasoning
- dual model = fused reasoning from both views

This is the main architectural contribution added in the current version of the project.

---

## 15. Design Tradeoff Notes

### Why gating instead of just summing the two branches?

Because gating gives:

- explicit control of contribution
- better interpretability
- a cleaner ablation
- less architectural disturbance than concatenation

### Why not full DAG-style correlation discovery/injection?

Because this project is not solving the same problem DAG solves.

DAG is designed for:

- endogenous variables
- historical exogenous variables
- future exogenous variables

This repository currently works on the simpler standard forecasting setup used by SimpleTM.

So the correct adaptation was:

- keep SimpleTM's core structure
- borrow the **parallel temporal + channel reasoning idea**
- avoid importing exogenous-specific components that do not match the project data pipeline

---

## 16. Suggested Report Language

If this project is described in a report or presentation, a clean wording is:

> We extend SimpleTM with a switchable dual-attention encoder that preserves the original geometric channel/scale attention branch and adds a DAG-inspired temporal self-attention branch in parallel. The two branches are fused through a learned gating mechanism, enabling a controlled comparison between original and dual-attention settings under both SWT and FFT tokenization.

That description is technically accurate for the current code.

---

## 17. Current Status

What is already implemented:

- SWT tokenization
- FFT tokenization
- original geometric attention
- dual attention with a temporal branch
- CLI switch for attention mode
- Kaggle notebook for all tokenization x attention combinations

What is still open for future work:

- token-wise or feature-wise fusion gates
- additional fusion baselines
- larger experimental study of `original` vs `dual`
- cleaner documentation merge into the main `README.md`

---

## 18. Quick Usage Examples

### SWT + original attention

```bash
python run.py --is_training 1 --model SimpleTM_SWT --attention_mode original ...
```

### SWT + dual attention

```bash
python run.py --is_training 1 --model SimpleTM_SWT --attention_mode dual ...
```

### FFT + original attention

```bash
python run.py --is_training 1 --model SimpleTM_FFT --attention_mode original ...
```

### FFT + dual attention

```bash
python run.py --is_training 1 --model SimpleTM_FFT --attention_mode dual ...
```

---

## 19. Final Takeaway

The repository now supports a clean experimental matrix:

```text
tokenization: SWT or FFT
attention:    original or dual
```

The most important implementation change is the addition of a **parallel temporal attention branch** that can be turned on without removing the original SimpleTM mechanism.

That makes the project suitable for:

- controlled architecture ablations
- BTP reporting
- Kaggle-scale comparative runs
- future extensions on fusion and temporal-channel interaction
