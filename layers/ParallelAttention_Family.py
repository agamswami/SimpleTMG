import torch
import torch.nn as nn
from math import sqrt

from layers.SWTAttention_Family import GeomAttentionLayer
from layers.FFTAttention_Family import FFTGeomAttentionLayer


class TemporalAxisAttention(nn.Module):
    """Self-attention over the latent time axis after inverted embedding."""

    def __init__(self, d_channel, attention_dropout=0.1):
        super().__init__()
        self.scale = 1.0 / sqrt(d_channel)
        self.query_projection = nn.Linear(d_channel, d_channel)
        self.key_projection = nn.Linear(d_channel, d_channel)
        self.value_projection = nn.Linear(d_channel, d_channel)
        self.out_projection = nn.Linear(d_channel, d_channel)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        # Convert (B, N, D) into time-major tokens (B, D, N).
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        q = self.query_projection(queries)
        k = self.key_projection(keys)
        v = self.value_projection(values)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = self.out_projection(out)

        return out.transpose(1, 2), scores.abs().mean()


class _ParallelAttentionBase(nn.Module):
    def __init__(self, original_attention_layer, d_model, d_channel, branch_dropout=0.1):
        super().__init__()
        self.original_attention_layer = original_attention_layer
        self.temporal_attention = TemporalAxisAttention(
            d_channel=d_channel, attention_dropout=branch_dropout
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        channel_out, channel_attn = self.original_attention_layer(
            queries, keys, values, attn_mask=attn_mask, tau=tau, delta=delta
        )
        temporal_out, temporal_attn = self.temporal_attention(queries, keys, values)

        pooled = torch.cat(
            [channel_out.mean(dim=1), temporal_out.mean(dim=1)], dim=-1
        )
        gate = torch.sigmoid(self.gate(pooled)).unsqueeze(-1)
        out = gate * channel_out + (1.0 - gate) * temporal_out

        return out, 0.5 * (channel_attn + temporal_attn)


class ParallelSWTGeomAttentionLayer(_ParallelAttentionBase):
    """Original SWT geometric attention plus a parallel temporal branch."""

    def __init__(
        self,
        attention,
        d_model,
        requires_grad=True,
        wv="db2",
        m=2,
        kernel_size=None,
        d_channel=None,
        geomattn_dropout=0.5,
    ):
        original_attention_layer = GeomAttentionLayer(
            attention,
            d_model,
            requires_grad=requires_grad,
            wv=wv,
            m=m,
            kernel_size=kernel_size,
            d_channel=d_channel,
            geomattn_dropout=geomattn_dropout,
        )
        super().__init__(
            original_attention_layer=original_attention_layer,
            d_model=d_model,
            d_channel=d_channel,
            branch_dropout=geomattn_dropout,
        )


class ParallelFFTGeomAttentionLayer(_ParallelAttentionBase):
    """Original FFT geometric attention plus a parallel temporal branch."""

    def __init__(self, attention, d_model, m=2, d_channel=None, geomattn_dropout=0.5):
        original_attention_layer = FFTGeomAttentionLayer(
            attention,
            d_model,
            m=m,
            d_channel=d_channel,
            geomattn_dropout=geomattn_dropout,
        )
        super().__init__(
            original_attention_layer=original_attention_layer,
            d_model=d_model,
            d_channel=d_channel,
            branch_dropout=geomattn_dropout,
        )
