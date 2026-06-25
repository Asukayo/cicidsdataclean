# -*- coding: utf-8 -*-
"""
USAD official-core baseline.

Reference core:
    Encoder + Decoder1 + Decoder2
    w1 = D1(E(x))
    w2 = D2(E(x))
    w3 = D2(E(w1))
    loss1 = 1/n * MSE(x,w1) + (1-1/n) * MSE(x,w3)
    loss2 = 1/n * MSE(x,w2) - (1-1/n) * MSE(x,w3)

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


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size, hidden_dim=None, use_official_width=False):
        super(Encoder, self).__init__()

        if use_official_width:
            h1 = int(in_size / 2)
            h2 = int(in_size / 4)
        else:
            h1 = int(hidden_dim if hidden_dim is not None else 64)
            h2 = max(int(h1 / 2), latent_size)

        self.linear1 = nn.Linear(in_size, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        z = self.relu(self.linear3(out))
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size, hidden_dim=None, use_official_width=False,
                 output_activation="identity"):
        super(Decoder, self).__init__()

        if use_official_width:
            h2 = int(out_size / 4)
            h1 = int(out_size / 2)
        else:
            h1 = int(hidden_dim if hidden_dim is not None else 64)
            h2 = max(int(h1 / 2), latent_size)

        self.linear1 = nn.Linear(latent_size, h2)
        self.linear2 = nn.Linear(h2, h1)
        self.linear3 = nn.Linear(h1, out_size)
        self.relu = nn.ReLU(True)

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "identity":
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")

    def forward(self, z):
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return self.output_activation(out)


class USAD(BaseAnomalyModel):
    def __init__(self, seq_len=100, input_dim=68, latent_dim=16, hidden_dim=64,
                 output_activation="identity", use_official_width=False):
        super(USAD, self).__init__()
        self.name = "USAD"
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.w_size = seq_len * input_dim
        self.z_size = latent_dim

        self.encoder = Encoder(
            self.w_size, latent_dim,
            hidden_dim=hidden_dim,
            use_official_width=use_official_width,
        )
        self.decoder1 = Decoder(
            latent_dim, self.w_size,
            hidden_dim=hidden_dim,
            use_official_width=use_official_width,
            output_activation=output_activation,
        )
        self.decoder2 = Decoder(
            latent_dim, self.w_size,
            hidden_dim=hidden_dim,
            use_official_width=use_official_width,
            output_activation=output_activation,
        )

    def _flatten(self, x):
        if x.dim() != 3:
            raise ValueError("USAD expects x with shape [B,W,F], got {}".format(tuple(x.shape)))
        if x.size(1) != self.seq_len or x.size(2) != self.input_dim:
            raise ValueError("expected [B,{},{},], got {}".format(
                self.seq_len, self.input_dim, tuple(x.shape)
            ))
        return x.reshape(x.size(0), -1)

    def _unflatten(self, w):
        return w.reshape(w.size(0), self.seq_len, self.input_dim)

    def forward(self, x):
        batch = self._flatten(x)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return self._unflatten(w1), self._unflatten(w2), self._unflatten(w3)

    def training_step(self, x, n):
        batch = self._flatten(x)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        loss1 = 1.0 / n * torch.mean((batch - w1) ** 2) + \
                (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1.0 / n * torch.mean((batch - w2) ** 2) - \
                (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def compute_loss(self, x, x_mark=None, epoch=1):
        loss1, loss2 = self.training_step(x, max(int(epoch), 1))
        return {
            "loss": loss1 + loss2,
            "loss1": loss1.item(),
            "loss2": loss2.item(),
        }

    def validation_step(self, x, n):
        with torch.no_grad():
            loss1, loss2 = self.training_step(x, n)
        return {"val_loss1": loss1, "val_loss2": loss2}

    def compute_anomaly_score(self, x, x_mark=None, alpha=0.5, beta=0.5):
        with torch.no_grad():
            batch = self._flatten(x)
            w1 = self.decoder1(self.encoder(batch))
            w3 = self.decoder2(self.encoder(w1))
            score = alpha * torch.mean((batch - w1) ** 2, dim=1) + \
                    beta * torch.mean((batch - w3) ** 2, dim=1)
        return score

    # Alias for efficiency script compatibility
    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(8, 100, 68)
    model = USAD(seq_len=100, input_dim=68)
    print([y.shape for y in model(x)])
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
