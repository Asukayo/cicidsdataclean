# -*- coding: utf-8 -*-
"""
TransDe official-core baseline.

Core kept from official TimeDetector:
    RevIN
    HP filter decomposition
    Multi-scale patch embeddings
    DAC_structure
    series/prior KL loss

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


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class HpFilter(nn.Module):
    """
    Device-safe HP filter.
    Official code uses inverse(I + lambda * D.T @ D).
    Here torch.linalg.solve is used for numerical stability.
    """

    def __init__(self, lamb=6400.0):
        super(HpFilter, self).__init__()
        self.lamb = lamb

    @staticmethod
    def _second_diff_matrix(n, device, dtype):
        d = torch.zeros(n - 2, n, device=device, dtype=dtype)
        idx = torch.arange(n - 2, device=device)
        d[idx, idx] = 1.0
        d[idx, idx + 1] = -2.0
        d[idx, idx + 2] = 1.0
        return d

    def forward(self, x):
        # x: [B,W,F]
        b, t, c = x.shape
        d = self._second_diff_matrix(t, x.device, x.dtype)
        eye = torch.eye(t, device=x.device, dtype=x.dtype)
        a = eye + self.lamb * torch.mm(d.t(), d)

        rhs = x.permute(1, 0, 2).contiguous().view(t, -1)
        if hasattr(torch.linalg, "solve"):
            trend = torch.linalg.solve(a, rhs)
        else:
            trend = torch.solve(rhs, a)[0]
        trend = trend.view(t, b, c).permute(1, 0, 2).contiguous()
        residual = x - trend
        return residual, trend


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DACStructure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True,
                 scale=None, attention_dropout=0.05, output_attention=True):
        super(DACStructure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def representation_learning(self, queries_patch_size, keys_patch_size):
        b, l, h, e = queries_patch_size.shape
        scale_patch_size = self.scale or 1.0 / math.sqrt(e)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size)
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))
        return series_patch_size

    def sampling(self, series_patch_size, patch_index, T=True):
        patch = self.patch_size[patch_index]
        if T:
            series_patch_size = series_patch_size.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
            b_all, h, m, n = series_patch_size.shape
            b = b_all // self.channel
            series_patch_size = series_patch_size.view(b, self.channel, h, m, n).mean(dim=1)
        else:
            repeat = self.window_size // patch
            series_patch_size = series_patch_size.repeat_interleave(repeat, dim=2).repeat_interleave(repeat, dim=3)
            b_all, h, m, n = series_patch_size.shape
            b = b_all // self.channel
            series_patch_size = series_patch_size.view(b, self.channel, h, m, n).mean(dim=1)
        return series_patch_size

    def forward(self, queries_patch_size, queries_patch_num,
                keys_patch_size, keys_patch_num, patch_index, attn_mask=None):
        series_patch_size = self.representation_learning(queries_patch_size, keys_patch_size)
        series_patch_num = self.representation_learning(queries_patch_num, keys_patch_num)

        series_patch_size = self.sampling(series_patch_size, patch_index)
        series_patch_num = self.sampling(series_patch_num, patch_index, False)

        if self.output_attention:
            return series_patch_size, series_patch_num
        return None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size,
                 d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        b, l, _ = x_patch_size.shape
        h = self.n_heads
        queries_patch_size = self.patch_query_projection(x_patch_size).view(b, l, h, -1)
        keys_patch_size = self.patch_key_projection(x_patch_size).view(b, l, h, -1)

        b, l, _ = x_patch_num.shape
        queries_patch_num = self.patch_query_projection(x_patch_num).view(b, l, h, -1)
        keys_patch_num = self.patch_key_projection(x_patch_num).view(b, l, h, -1)

        series, prior = self.inner_attention(
            queries_patch_size,
            queries_patch_num,
            keys_patch_size,
            keys_patch_num,
            patch_index,
            attn_mask,
        )
        return series, prior


class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class TransDe(BaseAnomalyModel):
    def __init__(self, win_size=100, input_dim=68, c_out=None, n_heads=1,
                 d_model=128, e_layers=2, patch_size=(5, 10, 20),
                 dropout=0.0, output_attention=True, lamb=6400.0,
                 temperature=50.0):
        super(TransDe, self).__init__()
        self.name = "TransDe"
        self.output_attention = output_attention
        self.patch_size = list(patch_size)
        self.channel = input_dim
        self.win_size = win_size
        self.hp_lamb = lamb
        self.temperature = temperature

        for p in self.patch_size:
            if win_size % p != 0:
                raise ValueError("win_size={} must be divisible by patch_size={}".format(win_size, p))
        if d_model % n_heads != 0:
            raise ValueError("d_model={} must be divisible by n_heads={}".format(d_model, n_heads))

        self.revin = RevIN(num_features=input_dim)
        self.Decomp1 = HpFilter(lamb=self.hp_lamb)

        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for patchsize in self.patch_size:
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(input_dim, d_model, dropout)

        self.encoder = Encoder([
            AttentionLayer(
                DACStructure(win_size, self.patch_size, input_dim, False,
                             attention_dropout=dropout, output_attention=output_attention),
                d_model,
                self.patch_size,
                input_dim,
                n_heads,
                win_size,
            )
            for _ in range(e_layers)
        ])

    @staticmethod
    def my_kl_loss(p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def one_dual(self, x, x_ori):
        series_patch_mean = []
        prior_patch_mean = []
        b, l, m = x.shape

        for patch_index, patchsize in enumerate(self.patch_size):
            x_bct = x.permute(0, 2, 1).contiguous()

            x_patch_size = x_bct.view(b * m, self.win_size // patchsize, patchsize)
            x_patch_num = x_bct.view(b * m, patchsize, self.win_size // patchsize)

            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)

            series_patch_mean.extend(series)
            prior_patch_mean.extend(prior)

        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        return None

    def cat(self, series_trend, series_residual, prior_trend, prior_residual):
        series, prior = [], []
        for i in range(len(series_trend)):
            series.append(torch.cat((series_trend[i], series_residual[i]), dim=3))
            prior.append(torch.cat((prior_trend[i], prior_residual[i]), dim=3))
        return series, prior

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("TransDe expects [B,W,F], got {}".format(tuple(x.shape)))
        if x.size(1) != self.win_size or x.size(2) != self.channel:
            raise ValueError("expected [B,{},{},], got {}".format(
                self.win_size, self.channel, tuple(x.shape)
            ))

        x = self.revin(x, "norm")
        x_ori = self.embedding_window_size(x)

        res, cyc = self.Decomp1(x)

        series_trend, prior_trend = self.one_dual(cyc, x_ori)
        series_residual, prior_residual = self.one_dual(res, x_ori)

        series, prior = self.cat(series_trend, series_residual, prior_trend, prior_residual)
        return series, prior

    def train_vai_loss(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            p_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12)
            series_loss = series_loss + (
                torch.mean(self.my_kl_loss(series[u], p_norm.detach())) +
                torch.mean(self.my_kl_loss(p_norm.detach(), series[u]))
            )
            prior_loss = prior_loss + (
                torch.mean(self.my_kl_loss(p_norm, series[u].detach())) +
                torch.mean(self.my_kl_loss(series[u].detach(), p_norm))
            )
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return prior_loss - series_loss

    def test_loss(self, series, prior):
        score = 0.0
        for u in range(len(prior)):
            p_norm = prior[u] / (torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12)
            s_loss = self.my_kl_loss(series[u], p_norm.detach()) * self.temperature
            p_loss = self.my_kl_loss(p_norm, series[u].detach()) * self.temperature
            # Positive discrepancy as anomaly score.
            score = score + s_loss + p_loss
        score = score / len(prior)
        return score

    def compute_loss(self, x, x_mark=None):
        series, prior = self.forward(x)
        loss = self.train_vai_loss(series, prior)
        return {
            "loss": loss,
            "assoc_loss": loss.item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        with torch.no_grad():
            series, prior = self.forward(x)
            score_bt = self.test_loss(series, prior)  # [B,W]
            return score_bt.mean(dim=-1)

    def score(self, x):
        return self.compute_anomaly_score(x)


if __name__ == "__main__":
    x = torch.randn(2, 100, 68)
    model = TransDe(win_size=100, input_dim=68, patch_size=(5, 10, 20), d_model=64, n_heads=1)
    series, prior = model(x)
    print(len(series), series[0].shape)
    print(model.compute_loss(x)["loss"].item())
    print(model.compute_anomaly_score(x).shape)
