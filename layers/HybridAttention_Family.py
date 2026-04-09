import torch
import torch.nn as nn

from layers.SWTAttention_Family import WaveletEmbedding
from layers.FFTAttention_Family import FFTEmbedding
from layers.ConvAttention_Family import ConvEmbedding, ScaleMixerReconstruction


class BranchFusionGate(nn.Module):
    """
    Softmax-normalized fusion over SWT, FFT, and Conv tokenizers.

    The gate is computed per sample, variate, and scale, which matches the
    project note's goal of adaptive branch weights instead of one global weight.
    """

    def __init__(self, d_model, num_branches=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_branches * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_branches),
        )

    def forward(self, branches):
        fused_input = torch.cat(branches, dim=-1)
        logits = self.gate(fused_input)
        weights = torch.softmax(logits, dim=-1)
        stacked = torch.stack(branches, dim=-2)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        return fused, weights


class HybridGeomAttentionLayer(nn.Module):
    """
    Three-branch hybrid tokenization followed by the original geometric attention.

    Branches:
    - SWT for multi-scale, non-stationary structure
    - FFT for compact spectral structure
    - Conv for local motifs and sharp short-range changes
    """

    def __init__(
        self,
        attention,
        d_model,
        requires_grad=True,
        wv='db2',
        m=2,
        kernel_size=None,
        d_channel=None,
        geomattn_dropout=0.5,
        conv_kernel_sizes=None,
    ):
        super().__init__()
        self.inner_attention = attention
        self.swt = WaveletEmbedding(
            d_channel=d_channel,
            swt=True,
            requires_grad=requires_grad,
            wv=wv,
            m=m,
            kernel_size=kernel_size,
        )
        self.fft = FFTEmbedding(
            d_channel=d_channel,
            decompose=True,
            m=m,
        )
        self.conv = ConvEmbedding(
            d_channel=d_channel,
            m=m,
            kernel_sizes=conv_kernel_sizes,
        )
        self.fusion_gate = BranchFusionGate(d_model=d_model, num_branches=3)

        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            ScaleMixerReconstruction(d_model=d_model, m=m),
        )

    def _tokenize_and_fuse(self, x):
        swt_tokens = self.swt(x)
        fft_tokens = self.fft(x)
        conv_tokens = self.conv(x)
        fused, weights = self.fusion_gate([swt_tokens, fft_tokens, conv_tokens])
        return fused, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries, _ = self._tokenize_and_fuse(queries)
        keys, _ = self._tokenize_and_fuse(keys)
        values, _ = self._tokenize_and_fuse(values)

        queries = self.query_projection(queries).permute(0, 3, 2, 1)
        keys = self.key_projection(keys).permute(0, 3, 2, 1)
        values = self.value_projection(values).permute(0, 3, 2, 1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0, 3, 2, 1))
        return out, attn
