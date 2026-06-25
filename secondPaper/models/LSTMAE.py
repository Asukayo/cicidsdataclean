# -*- coding: utf-8 -*-
"""
Standard LSTM AutoEncoder baseline.

Input:
    x: [B, W, F]
Output:
    recon: [B, W, F]
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


class LSTMAE(BaseAnomalyModel):
    def __init__(self, input_dim=68, hidden_dim=128, latent_dim=64,
                 num_layers=1, dropout=0.0):
        super(LSTMAE, self).__init__()
        self.name = "LSTM-AE"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("LSTMAE expects [B,W,F], got {}".format(tuple(x.shape)))

        _, (h_n, _) = self.encoder(x)
        z = self.to_latent(h_n[-1])
        dec_input = self.from_latent(z).unsqueeze(1).expand(-1, x.size(1), -1)
        dec_out, _ = self.decoder(dec_input)
        return self.out_proj(dec_out)

    def compute_loss(self, x, x_mark=None):
        recon = self.forward(x)
        mse = torch.mean((recon - x) ** 2, dim=(1, 2))
        return {
            "loss": mse.mean(),
            "mse": mse.mean().item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((recon - x) ** 2, dim=(1, 2))

    def score(self, x):
        return self.compute_anomaly_score(x)


# Backward-compatible alias
LSTMAutoEncoder = LSTMAE


if __name__ == "__main__":
    x = torch.randn(8, 100, 68)
    model = LSTMAE(input_dim=68)
    print(model(x).shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
