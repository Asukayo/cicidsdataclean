# -*- coding: utf-8 -*-
"""
MemAE adapted official-core baseline.

Official MemAE core:
    MemoryUnit:
        att = softmax(input @ memory^T)
        att = hard_shrink_relu(att)
        att = L1 normalize(att)
        output = att @ memory

This file adapts the official memory addressing module to time-series windows:
    x [B,W,F] -> temporal encoder -> h [B,D,W] -> MemModule -> decoder -> recon [B,W,F]
"""

from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    from .BaseModel import BaseAnomalyModel
except Exception:
    class BaseAnomalyModel(nn.Module):
        def __init__(self):
            super(BaseAnomalyModel, self).__init__()


def hard_shrink_relu(input_tensor, lambd=0.0, epsilon=1e-12):
    output = (F.relu(input_tensor - lambd) * input_tensor) / \
             (torch.abs(input_tensor - lambd) + epsilon)
    return output


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor):
        # input_tensor: [N, C]
        att_weight = F.linear(input_tensor, self.weight)  # [N, M]
        att_weight = F.softmax(att_weight, dim=1)

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_trans)
        return {"output": output, "att": att_weight}


class MemModule(nn.Module):
    """
    Supports 3D feature maps:
        input:  [B, C, T]
        output: [B, C, T]
        att:    [B, M, T]
    """

    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(mem_dim, fea_dim, shrink_thres)

    def forward(self, input_tensor):
        s = input_tensor.data.shape
        if len(s) != 3:
            raise ValueError("MemModule adapted here expects [B,C,T], got {}".format(tuple(s)))

        x = input_tensor.permute(0, 2, 1).contiguous()  # [B,T,C]
        x = x.view(-1, s[1])                            # [B*T,C]

        y_and = self.memory(x)
        y = y_and["output"]
        att = y_and["att"]

        y = y.view(s[0], s[2], s[1]).permute(0, 2, 1).contiguous()
        att = att.view(s[0], s[2], self.mem_dim).permute(0, 2, 1).contiguous()

        return {"output": y, "att": att}


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(True)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x):
        if self.padding > 0:
            return x[:, :, :-self.padding].contiguous()
        return x

    def forward(self, x):
        res = self.residual(x)
        out = self._chomp(self.conv1(x))
        out = self.dropout(self.act(out))
        out = self._chomp(self.conv2(out))
        out = self.dropout(self.act(out))
        return out + res


class MemAE(BaseAnomalyModel):
    def __init__(self, input_dim=68, latent_dim=64, mem_dim=64,
                 shrink_thres=0.0025, hidden_dim=64, dropout=0.1):
        super(MemAE, self).__init__()
        self.name = "MemAE-Adapted"
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ReLU(True),
            TemporalBlock(hidden_dim, latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(latent_dim, latent_dim, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.memory = MemModule(mem_dim=mem_dim, fea_dim=latent_dim, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            TemporalBlock(latent_dim, latent_dim, kernel_size=3, dilation=1, dropout=dropout),
            nn.Conv1d(latent_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim, input_dim, 1),
        )

    @staticmethod
    def entropy_loss(att):
        # att: [B,M,T]
        p = att.permute(0, 2, 1).contiguous().view(-1, att.size(1))
        return -torch.mean(torch.sum(p * torch.log(p + 1e-12), dim=1))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("MemAE expects [B,W,F], got {}".format(tuple(x.shape)))
        h = x.transpose(1, 2).contiguous()  # [B,F,W]
        h = self.encoder(h)
        y = self.memory(h)
        h_mem = y["output"]
        att = y["att"]
        recon = self.decoder(h_mem).transpose(1, 2).contiguous()
        return recon, att

    def compute_loss(self, x, x_mark=None, entropy_weight=2e-4):
        recon, att = self.forward(x)
        mse = torch.mean((recon - x) ** 2, dim=(1, 2))
        ent = self.entropy_loss(att)
        loss = mse.mean() + entropy_weight * ent
        return {
            "loss": loss,
            "mse": mse.mean().item(),
            "entropy": ent.item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        with torch.no_grad():
            recon, _ = self.forward(x)
            return torch.mean((recon - x) ** 2, dim=(1, 2))

    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(8, 100, 68)
    model = MemAE(input_dim=68)
    recon, att = model(x)
    print(recon.shape, att.shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
