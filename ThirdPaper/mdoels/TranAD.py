# -*- coding: utf-8 -*-
"""
TranAD official-core baseline.

Core:
    src = cat(src, c)
    TransformerEncoder
    TransformerDecoder1 with c=0
    TransformerDecoder2 with c=(x1-src)^2

Input:
    x: [B, W, F]
"""

from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .BaseModel import BaseAnomalyModel
except Exception:
    class BaseAnomalyModel(nn.Module):
        def __init__(self):
            super(BaseAnomalyModel, self).__init__()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(BaseAnomalyModel):
    def __init__(self, input_dim=68, window_size=100, n_heads=None,
                 dim_feedforward=16, dropout=0.1, output_activation="identity"):
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.n_feats = input_dim
        self.n_window = window_size
        self.d_model = 2 * input_dim

        if n_heads is None:
            n_heads = input_dim
        if self.d_model % n_heads != 0:
            raise ValueError("d_model={} must be divisible by n_heads={}".format(self.d_model, n_heads))
        self.n_heads = n_heads

        self.pos_encoder = PositionalEncoding(2 * input_dim, 0.1, window_size)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)

        layers = [nn.Linear(2 * input_dim, input_dim)]
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation != "identity":
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")
        self.fcn = nn.Sequential(*layers)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("TranAD expects [B,W,F], got {}".format(tuple(x.shape)))
        if x.size(1) != self.n_window or x.size(2) != self.n_feats:
            raise ValueError("expected [B,{},{},], got {}".format(
                self.n_window, self.n_feats, tuple(x.shape)
            ))

        src = x.transpose(0, 1).contiguous()  # [W,B,F]
        tgt = src

        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

        return x1.transpose(0, 1).contiguous(), x2.transpose(0, 1).contiguous()

    def compute_loss(self, x, x_mark=None, epoch=None):
        x1, x2 = self.forward(x)
        if epoch is None:
            loss = 0.5 * F.mse_loss(x1, x) + 0.5 * F.mse_loss(x2, x)
        else:
            n = max(int(epoch), 1)
            loss = (1.0 / n) * F.mse_loss(x1, x) + (1.0 - 1.0 / n) * F.mse_loss(x2, x)
        return {"loss": loss, "mse": loss.item()}

    def compute_anomaly_score(self, x, x_mark=None, alpha=0.2):
        with torch.no_grad():
            x1, x2 = self.forward(x)
            s1 = torch.mean((x1 - x) ** 2, dim=(1, 2))
            s2 = torch.mean((x2 - x) ** 2, dim=(1, 2))
            return alpha * s1 + (1.0 - alpha) * s2

    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(4, 100, 68)
    model = TranAD(input_dim=68, window_size=100, n_heads=4)
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
