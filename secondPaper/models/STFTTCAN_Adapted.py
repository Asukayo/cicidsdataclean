# -*- coding: utf-8 -*-
"""
STFT-TCAN adapted / paper-aligned baseline.

No confirmed official code was available here.
This implementation keeps the intended idea:
    time-domain TCN branch + frequency-domain rFFT amplitude branch + attention/gate fusion.

Input:
    x: [B, W, F]
"""

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class STFTTCAN(BaseAnomalyModel):
    def __init__(self, input_dim=68, seq_len=100, d_model=64,
                 tcn_layers=4, dropout=0.1):
        super(STFTTCAN, self).__init__()
        self.name = "STFT-TCAN-Adapted"
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.freq_bins = seq_len // 2 + 1

        self.time_proj = nn.Linear(input_dim, d_model)
        self.tcn = nn.Sequential(*[
            TCNBlock(d_model, kernel_size=3, dilation=2 ** i, dropout=dropout)
            for i in range(tcn_layers)
        ])

        self.freq_encoder = nn.Sequential(
            nn.Linear(self.freq_bins * input_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("STFTTCAN expects [B,W,F], got {}".format(tuple(x.shape)))
        if x.size(1) != self.seq_len or x.size(2) != self.input_dim:
            raise ValueError("expected [B,{},{},], got {}".format(
                self.seq_len, self.input_dim, tuple(x.shape)
            ))

        h_time = self.time_proj(x).transpose(1, 2).contiguous()
        h_time = self.tcn(h_time).transpose(1, 2).contiguous()

        amp = torch.log1p(torch.abs(torch.fft.rfft(x, dim=1)))  # [B,Fq,C]
        h_freq = self.freq_encoder(amp.reshape(x.size(0), -1))
        h_freq = h_freq.unsqueeze(1).expand(-1, x.size(1), -1)

        g = self.gate(torch.cat([h_time, h_freq], dim=-1))
        h = g * h_time + (1.0 - g) * h_freq
        return self.decoder(h)

    def compute_loss(self, x, x_mark=None):
        recon = self.forward(x)
        mse = torch.mean((recon - x) ** 2, dim=(1, 2))
        return {"loss": mse.mean(), "mse": mse.mean().item()}

    def compute_anomaly_score(self, x, x_mark=None):
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((recon - x) ** 2, dim=(1, 2))

    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(8, 100, 68)
    model = STFTTCAN(input_dim=68, seq_len=100)
    print(model(x).shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
