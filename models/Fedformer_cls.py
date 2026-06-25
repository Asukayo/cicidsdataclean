import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.layers.Embed import DataEmbedding_wo_pos
from models.layers.AutoCorrelation import AutoCorrelationLayer
from models.layers.FourierCorrelation import FourierBlock
from models.layers.MultiWaveletCorrelation import MultiWaveletTransform


# ============================================================
#  Autoformer-style components (series_decomp, EncoderLayer, etc.)
#  Inlined here to avoid dependency on Autoformer_EncDec.py
# ============================================================

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """Multiple series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, mov = func(x)
            moving_mean.append(mov)
            res.append(sea)
        sea = sum(res) / len(res)
        mov = sum(moving_mean) / len(moving_mean)
        return sea, mov


class my_Layernorm(nn.Module):
    """Special LayerNorm for the seasonal part."""
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class EncoderLayer(nn.Module):
    """Autoformer-style EncoderLayer with series decomposition."""
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# ============================================================
#  FEDformer for Classification
# ============================================================

class Model(nn.Module):
    """
    FEDformer encoder-only, adapted for time series classification.

    Architecture:
        Input [B, seq_len, enc_in]
        → DataEmbedding_wo_pos → [B, seq_len, d_model]
        → Encoder (Fourier / Wavelet attention + decomposition) × e_layers
        → Global Average Pooling → [B, d_model]
        → MLP Classification Head → [B, num_classes]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.version = getattr(configs, 'version', 'Fourier')
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 32)

        # ---- Embedding ----
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            getattr(configs, 'embed', 'timeF'),
            getattr(configs, 'freq', 'h'),
            configs.dropout
        )

        # ---- Frequency-domain attention ----
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model,
                L=getattr(configs, 'L', 1),
                base=getattr(configs, 'base', 'legendre')
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select
            )

        # ---- Encoder ----
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=getattr(configs, 'moving_avg', 25),
                    dropout=configs.dropout,
                    activation=getattr(configs, 'activation', 'gelu')
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ---- Classification Head ----
        self.classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.num_classes)
        )

    def forward(self, x_enc, x_mark_enc=None):
        """
        Args:
            x_enc:      [B, seq_len, enc_in]  — raw time series features
            x_mark_enc: [B, seq_len, time_feat_dim] — temporal features (optional)
                        If None, a zero tensor is used.
        Returns:
            logits: [B, num_classes]
        """
        # handle missing time marks
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(
                x_enc.shape[0], x_enc.shape[1], 4,
                device=x_enc.device, dtype=x_enc.dtype
            )

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)       # [B, L, d_model]

        # encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=None)     # [B, L, d_model]

        # global average pooling over time dimension
        enc_out = enc_out.mean(dim=1)                          # [B, d_model]

        # classification
        logits = self.classifier(enc_out)                      # [B, num_classes]
        return logits