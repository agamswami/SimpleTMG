# Conv and Hybrid Architecture in SimpleTMG

This note explains the **Conv** and **Hybrid** branches used in this repository.

Neither of them is a separate forecasting backbone. Both are **SimpleTM variants**:

- `SimpleTM_Conv` replaces the tokenizer with a multi-scale convolutional tokenizer
- `SimpleTM_Hybrid` runs `SWT`, `FFT`, and `Conv` tokenizers in parallel and fuses them with a learned gate

Both can also be used in `dual` attention mode, where a temporal self-attention branch is added in parallel to the original branch.

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

---

## 13. Hybrid Model Overview

The Hybrid model is implemented in:

- [model/SimpleTM_Hybrid.py](model/SimpleTM_Hybrid.py)
- [layers/HybridAttention_Family.py](layers/HybridAttention_Family.py)

Its goal is different from the Conv-only branch.

Instead of committing to one tokenizer, the Hybrid branch runs:

- `SWT`
- `FFT`
- `Conv`

in parallel on the same embedded token tensor, then fuses them with a learned gate before geometric attention.

So the Hybrid pipeline is:

```text
Input series
  -> normalization
  -> inverted embedding
  -> SWT tokenization
  -> FFT tokenization
  -> Conv tokenization
  -> learned branch fusion
  -> geometric attention
  -> learned scale reconstruction
  -> encoder FFN/residual/norm
  -> output projection
  -> de-normalization
  -> forecast
```

This makes `SimpleTM_Hybrid` the most flexible tokenizer in the project.

---

## 14. Hybrid Tokenization in Detail

The core layer is `HybridGeomAttentionLayer` in [layers/HybridAttention_Family.py](layers/HybridAttention_Family.py).

### 14.1 Shared input shape

Just like the other models, Hybrid starts after inverted embedding:

```text
(B, L, N) -> (B, N, d_model)
```

This embedded tensor is sent to all three tokenizers.

### 14.2 Three parallel tokenizers

For the same input token tensor `x`, the Hybrid layer computes:

```text
T_swt  = SWT(x)
T_fft  = FFT(x)
T_conv = Conv(x)
```

Each of these has the same shape:

```text
(B, N, m + 1, d_model)
```

That shared shape is what makes the fusion possible.

The roles of the three branches are:

- `SWT`: structured multi-resolution decomposition for non-stationary signals
- `FFT`: compact spectral decomposition for periodic structure
- `Conv`: learnable local motif extraction

So Hybrid is trying to combine:

- hand-structured wavelet features
- spectral features
- learned local features

in one token stream.

### 14.3 Fusion gate

The branch fusion module is `BranchFusionGate`.

Its structure is:

```text
Linear(3 * d_model, d_model)
-> GELU
-> Linear(d_model, 3)
-> softmax
```

This is not a single scalar gate for the whole sample.
It is applied on the concatenated branch features at each `(sample, variate, scale)` location.

So for each token location the gate computes 3 weights:

```text
w_swt, w_fft, w_conv
```

with:

```text
w_swt + w_fft + w_conv = 1
```

Then the fused token is:

```text
T_fused = w_swt * T_swt + w_fft * T_fft + w_conv * T_conv
```

This means the model can adapt branch importance depending on the specific token and scale.

### 14.4 Example

Suppose after tokenization, for one sample, one variable, one scale, the three branch vectors are:

```text
T_swt  = [0.2, 0.7, 0.1]
T_fft  = [0.5, 0.2, 0.3]
T_conv = [0.4, 0.6, 0.5]
```

If the learned fusion gate outputs:

```text
w_swt = 0.6
w_fft = 0.1
w_conv = 0.3
```

then the fused token becomes:

```text
T_fused = 0.6*T_swt + 0.1*T_fft + 0.3*T_conv
```

So that token is dominated by the SWT branch, but still uses information from FFT and Conv.

For another token, the weights may be very different.  
That is why Hybrid is more adaptive than selecting one tokenizer globally.

---

## 15. Hybrid Attention and Reconstruction

After fusion, the Hybrid branch behaves like the other tokenizers.

Its flow is:

```text
queries/keys/values
  -> tokenize with SWT, FFT, Conv
  -> fuse branches with softmax gate
  -> Q/K/V projection
  -> geometric attention
  -> learned scale reconstruction
```

### 15.1 Geometric attention is unchanged

The Hybrid model still uses the original geometric attention scoring rule:

```text
score = (1 - alpha) * dot(q, k) + alpha * wedge_norm(q, k)
```

So again, the attention mechanism stays fixed.
The new part is the tokenizer fusion before attention.

### 15.2 Reconstruction

After geometric attention, the Hybrid output is still:

```text
(B, N, m + 1, d_model)
```

It is collapsed back using the same `ScaleMixerReconstruction` used by the Conv branch:

```text
(B, N, m + 1, d_model) -> (B, N, d_model)
```

So Hybrid has:

- learned branch fusion across tokenizers
- learned scale mixing after attention

This gives it two levels of adaptivity.

---

## 16. Dual Attention Version of Hybrid

If:

```text
attention_mode = dual
```

then [model/SimpleTM_Hybrid.py](model/SimpleTM_Hybrid.py) switches to:

- `ParallelHybridGeomAttentionLayer`

from [layers/ParallelAttention_Family.py](layers/ParallelAttention_Family.py).

This means the Hybrid branch itself remains intact, and a temporal branch is added beside it.

So the dual Hybrid architecture is:

```text
embedded tokens
  -> Hybrid branch:
       SWT + FFT + Conv
       -> branch fusion gate
       -> geometric attention
       -> scale reconstruction
  -> temporal self-attention branch
  -> learned branch gate
  -> fused encoder output
```

This is important:

- the `Hybrid` branch already has an internal **3-way tokenization gate**
- the `dual` version adds a second **2-way attention-branch gate**

So there are two different gating levels:

1. tokenization fusion gate inside Hybrid
2. original-vs-temporal fusion gate inside Dual Attention

### 16.1 Temporal branch

The temporal branch is the same one used in the dual Conv/SWT/FFT variants:

- transpose `(B, N, d_model)` to `(B, d_model, N)`
- run self-attention over the latent time axis
- map back to `(B, N, d_model)`

### 16.2 Dual fusion gate

The outer fusion gate in [layers/ParallelAttention_Family.py](layers/ParallelAttention_Family.py) is:

```text
Linear(2 * d_model, d_model)
-> GELU
-> Linear(d_model, 1)
-> sigmoid
```

It produces one scalar gate per sample and fuses:

```text
out = g * hybrid_branch + (1 - g) * temporal_branch
```

So in the dual Hybrid model:

- Hybrid decides how to mix `SWT`, `FFT`, and `Conv`
- Dual attention decides how much to trust the Hybrid path versus the temporal path

---

## 17. Shape Summary for Hybrid

Original Hybrid model:

```text
Input                         : (B, L, N)
Embedding output              : (B, N, d_model)
SWT / FFT / Conv tokens       : (B, N, m + 1, d_model)
After branch fusion           : (B, N, m + 1, d_model)
After scale reconstruction    : (B, N, d_model)
Forecast                      : (B, pred_len, N)
```

Dual Hybrid model:

```text
Embedding output              : (B, N, d_model)
Hybrid branch output          : (B, N, d_model)
Temporal branch output        : (B, N, d_model)
After outer fusion gate       : (B, N, d_model)
Forecast                      : (B, pred_len, N)
```

---

## 18. Why Hybrid Is Different

`SimpleTM_Hybrid` is the most expressive variant in the project because it can:

- use wavelet-style multi-resolution information
- use Fourier-style spectral information
- use learnable convolutional local patterns
- adaptively weight them
- optionally combine all of that with a temporal attention branch

Its strengths are:

- most flexible tokenizer
- strongest ability to adapt across datasets
- can favor different branches for different tokens and scales

Its tradeoffs are:

- more parameters than a single-branch tokenizer
- harder to interpret than pure SWT or pure FFT
- more moving parts because there are two fusion stages in dual mode

In one line:

> `SimpleTM_Hybrid` runs SWT, FFT, and Conv tokenizers in parallel, learns how to fuse them before geometric attention, and in `dual` mode also learns how to fuse that hybrid path with a temporal self-attention branch.
