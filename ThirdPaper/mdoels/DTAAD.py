# -*- coding: utf-8 -*-
"""
DTAAD official-core baseline.

Core:
    Tcn_Local -> TransformerEncoder1 -> decoder1
    callback(src, x1): src + x1 -> Tcn_Global -> TransformerEncoder2
    decoder2

Input:
    x: [B, W, F]
Output:
    x1, x2: [B, 1, F]
Score:
    MSE against the last timestep x[:, -1:, :]
"""

from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

try:
    from .BaseModel import BaseAnomalyModel
except Exception:
    class BaseAnomalyModel(nn.Module):
        def __init__(self):
            super(BaseAnomalyModel, self).__init__()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)


class Tcn_Local(nn.Module):
    def __init__(self, num_outputs, kernel_size=4, dropout=0.2):
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        out_channels = num_outputs
        for _ in range(num_levels):
            layers += [TemporalCnn(
                out_channels, out_channels, kernel_size,
                stride=1, dilation=1, padding=(kernel_size - 1),
                dropout=dropout,
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Tcn_Global(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2((num_inputs - 1) / float(kernel_size - 1) + 1))
        out_channels = num_outputs
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers += [TemporalCnn(
                out_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


class DTAAD(BaseAnomalyModel):
    def __init__(self, input_dim=68, window_size=100, n_heads=None,
                 dropout=0.2, dim_feedforward=16, output_activation="identity"):
        super(DTAAD, self).__init__()
        self.name = "DTAAD"
        self.n_feats = input_dim
        self.n_window = window_size

        if n_heads is None:
            n_heads = input_dim
        if input_dim % n_heads != 0:
            raise ValueError("input_dim={} must be divisible by n_heads={}".format(input_dim, n_heads))
        self.n_heads = n_heads

        self.l_tcn = Tcn_Local(num_outputs=input_dim, kernel_size=4, dropout=dropout)
        self.g_tcn = Tcn_Global(num_inputs=window_size, num_outputs=input_dim, kernel_size=3, dropout=dropout)

        self.pos_encoder = PositionalEncoding(input_dim, 0.1, window_size)
        encoder_layers1 = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
        )
        encoder_layers2 = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
        )
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers1, num_layers=1)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers2, num_layers=1)

        self.fcn = nn.Linear(input_dim, input_dim)

        dec1 = [nn.Linear(window_size, 1)]
        dec2 = [nn.Linear(window_size, 1)]
        if output_activation == "sigmoid":
            dec1.append(nn.Sigmoid())
            dec2.append(nn.Sigmoid())
        elif output_activation != "identity":
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")

        self.decoder1 = nn.Sequential(*dec1)
        self.decoder2 = nn.Sequential(*dec2)

    def callback(self, src, c):
        # src: [B,F,W], c: [B,1,F]
        src2 = src + c.transpose(1, 2)
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("DTAAD expects [B,W,F], got {}".format(tuple(x.shape)))
        if x.size(1) != self.n_window or x.size(2) != self.n_feats:
            raise ValueError("expected [B,{},{},], got {}".format(
                self.n_window, self.n_feats, tuple(x.shape)
            ))

        src = x.transpose(1, 2).contiguous()  # [B,F,W]

        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)

        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0)).permute(0, 2, 1)

        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0)).permute(0, 2, 1)

        return x1, x2

    def compute_loss(self, x, x_mark=None):
        target = x[:, -1:, :]
        x1, x2 = self.forward(x)
        loss = 0.5 * F.mse_loss(x1, target) + 0.5 * F.mse_loss(x2, target)
        return {"loss": loss, "mse": loss.item()}

    def compute_anomaly_score(self, x, x_mark=None):
        with torch.no_grad():
            target = x[:, -1:, :]
            x1, x2 = self.forward(x)
            s1 = torch.mean((x1 - target) ** 2, dim=(1, 2))
            s2 = torch.mean((x2 - target) ** 2, dim=(1, 2))
            return 0.5 * s1 + 0.5 * s2

    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(4, 100, 68)
    model = DTAAD(input_dim=68, window_size=100, n_heads=4)
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
