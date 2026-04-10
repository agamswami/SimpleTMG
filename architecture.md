# Conv Architecture in SimpleTMG

This note explains the **Conv branch** used by `SimpleTM_Conv` in this repository.

It is not a separate CNN forecasting model. Instead, it is a **convolutional tokenization variant** of SimpleTM that keeps the original forecasting backbone and replaces the SWT/FFT tokenization stage with a multi-scale convolutional tokenizer.

---

## 1. Main Idea

The Conv model follows the same overall SimpleTM pipeline:

```text
Input series
  -> normalization
  -> inverted embedding
  -> Conv tokenization
  -> geometric attention encoder
  -> output projection
  -> de-normalization
  -> forecast
```

So the main change is only in the **tokenization stage**.

Implemented in:

- [model/SimpleTM_Conv.py](model/SimpleTM_Conv.py)
- [layers/ConvAttention_Family.py](layers/ConvAttention_Family.py)

---

## 2. Why Use Conv Tokenization

The Conv branch is designed to capture:

- local temporal motifs
- short-range variations
- spikes and abrupt local changes
- patterns at multiple receptive fields

Compared with the other branches:

- `SWT` uses fixed wavelet filters
- `FFT` uses frequency-band decomposition
- `Conv` uses **learnable local filters**

So the Conv branch is the most data-adaptive of the single-branch tokenizers.

---

## 3. Full Data Flow

The input to the model is:

```text
x_enc: (B, L, N)
```

where:

- `B` = batch size
- `L` = input sequence length
- `N` = number of variates

### Step 1: Normalization

In [model/SimpleTM_Conv.py](model/SimpleTM_Conv.py), the model first applies instance-style normalization if `use_norm=1`:

```text
x_norm = (x - mean) / std
```

This is the same behavior as the other SimpleTM variants.

### Step 2: Inverted Embedding

The embedding layer [layers/Embed.py](layers/Embed.py) maps:

```text
(B, L, N) -> (B, N, d_model)
```

This means:

- each variate becomes a token
- its whole input history is projected into a latent vector of length `d_model`

This is important: the Conv branch does **not** convolve directly on the raw `(L)` time axis.  
It convolves over the latent axis after inverted embedding.

---

## 4. Conv Tokenization

The tokenizer is implemented by [ConvEmbedding](layers/ConvAttention_Family.py) in [layers/ConvAttention_Family.py](layers/ConvAttention_Family.py).

### 4.1 Kernel Selection

Kernel sizes are chosen by `resolve_conv_kernel_sizes(...)`.

Behavior:

- the branch needs exactly `m + 1` kernels
- if no kernels are provided, it uses defaults:

```text
[3, 5, 7, 11, 15, 21, 31]
```

- if the provided list is shorter than `m + 1`, it keeps extending with larger odd values
- if a kernel size is even, it is converted to the next odd number

So with:

```text
m = 3
```

the branch uses `4` kernels, for example:

```text
[3, 5, 7, 11]
```

### 4.2 Depthwise Circular Convolution

Each scale uses `DepthwiseCircularConv1d`.

Key design:

- `groups = d_channel`
- `in_channels = out_channels = d_channel`
- padding mode is `circular`

This means:

- each latent channel is convolved independently
- there is no channel mixing inside the tokenizer itself
- wrap-around padding is used, similar in spirit to the boundary handling used by the wavelet branch

### 4.3 Shape Transformation

The input to `ConvEmbedding` is:

```text
(B, N, d_model)
```

For each kernel size, one depthwise convolution is applied and produces:

```text
(B, N, d_model)
```

Then all scale outputs are stacked:

```text
(B, N, d_model) -> (B, N, m + 1, d_model)
```

So each variate token now has `m + 1` convolutional sub-representations, one per kernel size.

This shape is intentionally the same as the SWT and FFT branches, so the rest of the encoder can stay unchanged.

---

## 5. Conv + Geometric Attention

The attention wrapper is `ConvGeomAttentionLayer` in [layers/ConvAttention_Family.py](layers/ConvAttention_Family.py).

Its flow is:

```text
queries/keys/values
  -> Conv decomposition
  -> Q/K/V projection
  -> geometric attention
  -> output projection
  -> scale reconstruction
```

### 5.1 Q/K/V Projection

After Conv tokenization, the branch applies linear projections:

- query projection
- key projection
- value projection

Each projection is:

```text
Linear(d_model, d_model) + Dropout
```

This is the same pattern used in the other tokenization variants.

### 5.2 Geometric Attention

The Conv branch still uses the original SimpleTM geometric attention from [layers/SWTAttention_Family.py](layers/SWTAttention_Family.py).

Conceptually, the attention score is:

```text
score = (1 - alpha) * dot(q, k) + alpha * wedge_norm(q, k)
```

So the Conv model changes **tokenization**, not the core attention scoring rule.

This is why the comparison stays fair:

- same embedding idea
- same encoder structure
- same attention mechanism
- different tokenizer

### 5.3 Scale Reconstruction

After attention, the Conv branch must collapse:

```text
(B, N, m + 1, d_model) -> (B, N, d_model)
```

This is done by `ScaleMixerReconstruction`.

Implementation idea:

1. flatten the scale dimension and latent dimension:

```text
(m + 1, d_model) -> ((m + 1) * d_model)
```

2. apply:

```text
Linear((m + 1) * d_model, d_model)
```

So reconstruction is **learned**, not a fixed inverse transform.

That is the main difference from SWT:

- `SWT` has structured wavelet reconstruction
- `Conv` learns how to mix scales back into one latent token

---

## 6. Encoder and Forecast Head

In [model/SimpleTM_Conv.py](model/SimpleTM_Conv.py), the tokenizer-attention block is inserted into the standard encoder:

- `Encoder`
- `EncoderLayer`
- residual connections
- feed-forward network
- layer normalization

After the encoder output `(B, N, d_model)` is produced, the model uses:

```text
Linear(d_model, pred_len)
```

Then it permutes to:

```text
(B, pred_len, N)
```

This is the final forecast.

So the Conv branch changes the internal token processing, but the forecast head stays the same as the rest of SimpleTM.

---

## 7. Dual Attention Version of Conv

If:

```text
attention_mode = dual
```

then [model/SimpleTM_Conv.py](model/SimpleTM_Conv.py) uses:

- `ParallelConvGeomAttentionLayer`

from [layers/ParallelAttention_Family.py](layers/ParallelAttention_Family.py).

In that case the architecture becomes:

```text
embedded tokens
  -> Conv + geometric attention branch
  -> temporal self-attention branch
  -> learned gate
  -> fused output
```

### 7.1 Original Branch

This branch is exactly the Conv path described above.

### 7.2 Temporal Branch

The temporal branch:

- transposes `(B, N, d_model)` to `(B, d_model, N)`
- applies self-attention over the latent time axis
- produces another `(B, N, d_model)` output

### 7.3 Fusion Gate

The two branches are fused with a small MLP gate:

```text
Linear(2 * d_model, d_model)
-> GELU
-> Linear(d_model, 1)
-> sigmoid
```

Then:

```text
out = g * conv_branch + (1 - g) * temporal_branch
```

So the dual version adds temporal reasoning in parallel, while the original Conv branch remains intact.

---

## 8. Tensor Shape Summary

For the original Conv model:

```text
Input                    : (B, L, N)
Embedding output         : (B, N, d_model)
Conv tokenization        : (B, N, m + 1, d_model)
After reconstruction     : (B, N, d_model)
Forecast projection      : (B, pred_len, N)
```

For the dual Conv model:

```text
Embedding output         : (B, N, d_model)
Conv branch output       : (B, N, d_model)
Temporal branch output   : (B, N, d_model)
Fused output             : (B, N, d_model)
Forecast projection      : (B, pred_len, N)
```

---

## 9. Why This Design Is Useful

This architecture is useful because it gives you a third tokenizer family that is different from both wavelets and Fourier bands.

Its strengths are:

- flexible local pattern extraction
- learnable receptive fields
- easy comparability with SWT and FFT because the downstream architecture is unchanged

Its limitations are:

- it does not provide the explicit frequency interpretation of FFT
- it does not provide the structured multi-resolution interpretation of SWT
- reconstruction is learned rather than analytically defined

So the Conv branch is best understood as:

> a learnable multi-scale local tokenizer inside the SimpleTM framework.

---

## 10. Short Comparison with Other Branches

| Branch | Tokenizer Type | Strength |
| --- | --- | --- |
| SWT | fixed wavelet decomposition | non-stationary multi-resolution structure |
| FFT | spectral band decomposition | periodic and frequency-dominant structure |
| Conv | learnable local convolutions | local motifs, spikes, short-range patterns |
| Hybrid | gated combination of all three | adaptive tokenization |

---

## 11. Practical CLI Usage

Original Conv branch:

```bash
python run.py --model SimpleTM_Conv --attention_mode original --conv_kernel_sizes 3,5,7,11 ...
```

Dual Conv branch:

```bash
python run.py --model SimpleTM_Conv --attention_mode dual --conv_kernel_sizes 3,5,7,11 ...
```

If `--conv_kernel_sizes` is omitted, the model uses the default odd kernel list and truncates or extends it to match `m + 1`.

---

## 12. Final Summary

The Conv architecture in this project is:

- not a standalone CNN forecaster
- a **SimpleTM variant with convolutional multi-scale tokenization**
- built to stay comparable with SWT and FFT
- optionally extensible with the dual temporal-attention branch

In one line:

> `SimpleTM_Conv` keeps the SimpleTM backbone, replaces the tokenizer with multi-scale depthwise circular convolutions, and optionally adds a parallel temporal attention branch in dual mode.
