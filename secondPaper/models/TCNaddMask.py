"""
TCN-based AutoEncoder + Frequency Masked Prediction branch
===========================================================
- Reconstruction branch: unchanged from the original TCNAE
- New branch: masked frequency prediction (route B)
    * Input : (B, F, C) masked amplitude spectrum
    * Output: (B, F, C) predicted amplitude spectrum
    * Loss computed only on masked positions
"""

import torch
import torch.nn as nn
from .BaseModel import BaseAnomalyModel
from ..utils.FrequencyMasking import FrequencyMasking,get_train_freq_weight,get_infer_spike_boost  # adjust import path as needed


# ============================================================
# (Original blocks unchanged — CausalConv1d, TCNBlock, TCNEncoder, TCNDecoder)
# ============================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.ln1 = nn.GroupNorm(1, out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.ln2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.ln2(self.conv2(out)))
        out = self.dropout(out)
        return out + residual


class TCNEncoder(nn.Module):
    def __init__(self, input_dim=38, channels=None, kernel_size=3, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [96, 96, 64, 32, 32]
        dilations = [2 ** i for i in range(len(channels))]
        layers, in_ch = [], input_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

        # 新增：因果Stride-2 下采样，T：100 -> 50
        # kernel = 2 ，stride = 2， 左padding=0，即严格因果且无冗余
        self.downsample = nn.Conv1d(channels[-1], channels[-1]
                                    , kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, F, T]
        z = self.network(x)     # [B, 32, 100]
        z = self.downsample(z)  # [B, 32, 50]
        return z.transpose(1, 2)    # [B, 50, 32]


class TCNDecoder(nn.Module):
    def __init__(self, latent_dim=32, channels=None, output_dim=38, kernel_size=3, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 64, 96, 96]
        dilations = [2 ** i for i in range(len(channels))]

        # 新增:转置卷积上采样,T: 50 -> 100
        self.upsample = nn.ConvTranspose1d(latent_dim, latent_dim,
                                           kernel_size=2, stride=2, padding=0)

        layers, in_ch = [], latent_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(channels[-1], output_dim, 1)

    def forward(self, z):
        z = z.transpose(1, 2) # [B, 32, 50]
        z = self.upsample(z)  # [B, 32, 100]
        out = self.network(z)
        x_hat = self.output_proj(out).transpose(1, 2)
        return x_hat


# ============================================================
# NEW: Frequency-domain predictor
# ============================================================
class FreqPredictor(nn.Module):
    """
    Lightweight predictor operating on the amplitude spectrum.

    Input : (B, F, C)  masked log-amplitude spectrum
    Output: (B, F, C)  predicted log-amplitude spectrum

    Non-causal 1D convs along the frequency axis — frequency has no causal
    ordering, neighboring bins are both informative (unlike time).
    """

    def __init__(self, input_dim=38, hidden_dim=64, num_layers=4,
                 kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=pad),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, amp_masked):
        # amp_masked: (B, F, C) -> (B, C, F) for Conv1d along freq axis
        h = amp_masked.transpose(1, 2)
        h = self.net(h)
        out = self.out_proj(h).transpose(1, 2)   # (B, F, C)
        return out


# ============================================================
# Full model: TCNAE + Frequency branch
# ============================================================
class TCNAEWithFreq(BaseAnomalyModel):
    """
    Two parallel self-supervised branches:
      1. Time-domain reconstruction (original TCNAE, unchanged)
      2. Frequency-domain masked prediction (new)
    """

    def __init__(
        self,
        input_dim=38,
        window_size=100,
        enc_channels=None,
        dec_channels=None,
        kernel_size=3,
        dropout=0.1,

        # ---- frequency branch hyperparams ----
        freq_beta1=0.0,
        freq_beta2=0.5,
        freq_hidden_dim=64,
        freq_num_layers=4,
        freq_kernel_size=5,
        freq_loss_weight=1.0,
        freq_infer_segments=10,
    ):
        super().__init__()
        self.name = 'TCNAE_with_FreqPred'
        if enc_channels is None:
            enc_channels = [64, 64, 64, 32, 32]
        if dec_channels is None:
            dec_channels = [32, 64, 64, 64, 64]

        # ---- branch 1: reconstruction (unchanged) ----
        self.encoder = TCNEncoder(input_dim, enc_channels, kernel_size, dropout)
        self.decoder = TCNDecoder(enc_channels[-1], dec_channels, input_dim,
                                  kernel_size, dropout)

        # ---- branch 2: frequency masked prediction ----
        self.freq_mask = FrequencyMasking(beta1=freq_beta1, beta2=freq_beta2,
                                          log_amp=True, protect_dc=True)
        self.freq_predictor = FreqPredictor(
            input_dim=input_dim,
            hidden_dim=freq_hidden_dim,
            num_layers=freq_num_layers,
            kernel_size=freq_kernel_size,
            dropout=dropout,
        )
        self.freq_loss_weight = freq_loss_weight
        self.freq_infer_segments = freq_infer_segments

        self.input_dim = input_dim
        self.window_size = window_size

    # ---- original reconstruction branch (untouched) ----
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    # ---- new frequency branch helper ----
    def _freq_forward(self, x, mask=None):
        """
        Returns per-sample masked-MSE in the frequency domain.
        """
        out = self.freq_mask(x, mask=mask)                 # dict
        pred = self.freq_predictor(out["amp_masked"])      # (B, F, C)
        inv_mask = (1.0 - out["mask"]).unsqueeze(-1)       # (B, F, 1), 1 at masked
        sq_err = (pred - out["amp_original"]) ** 2 * inv_mask   # (B, F, C)

        # 训练时加上整段的权重
        if self.training:
            freq_weight = get_train_freq_weight(sq_err.shape[1], x.device)
            sq_err = sq_err * freq_weight

        # Normalize: sum over (F, C) divided by (num_masked_freq_bins * C)
        C = x.shape[-1]
        per_sample_err = sq_err.sum(dim=(1, 2)) / (out["num_masked"].float() * C + 1e-8)
        return per_sample_err, out

    # def compute_anomaly_score(self, x, x_mark=None):
    #     self.eval()
    #     B, L, C = x.shape
    #     F = L // 2 + 1
    #     K = self.freq_infer_segments
    #
    #     with torch.no_grad():
    #         x_hat, _ = self.forward(x)
    #         recon_score = ((x - x_hat) ** 2).mean(dim=(1, 2))
    #
    #         freq_indices = torch.arange(1, F, device=x.device)
    #         segments = torch.chunk(freq_indices, K)
    #
    #         total_sq_err = torch.zeros(B, device=x.device)
    #         total_count = torch.zeros(B, device=x.device)
    #
    #         for seg in segments:
    #             if seg.numel() == 0:
    #                 continue
    #             mask = torch.ones(B, F, device=x.device)
    #             mask[:, seg] = 0.0
    #
    #             out = self.freq_mask(x, mask=mask)
    #             pred = self.freq_predictor(out["amp_masked"])
    #             inv_mask = (1.0 - mask).unsqueeze(-1)
    #
    #             raw_err = (pred - out["amp_original"]) ** 2 * inv_mask  # ← 改动
    #             boosted_err = get_infer_spike_boost(raw_err)  # ← 新增
    #             sq_err = boosted_err.sum(dim=(1, 2))  # ← 改动
    #
    #             total_sq_err += sq_err
    #             total_count += inv_mask.sum(dim=(1, 2)) * C
    #
    #         freq_score = total_sq_err / (total_count + 1e-8)
    #
    #     return {
    #         'recon_score': recon_score,
    #         'freq_score': freq_score,
    #     }


    # ---- BaseAnomalyModel interface ----

    def compute_loss(self, x, x_mark=None):
        """
        Total loss = reconstruction MSE + λ * masked frequency MSE
        """
        # branch 1: reconstruction
        x_hat, _ = self.forward(x)
        recon_mse = torch.mean((x - x_hat) ** 2, dim=(1, 2))     # (B,)
        recon_loss = recon_mse.mean()

        # branch 2: frequency masked prediction (single random mask per step)
        freq_err, _ = self._freq_forward(x, mask=None)           # (B,)
        freq_loss = freq_err.mean()

        total = recon_loss + self.freq_loss_weight * freq_loss

        return {
            'loss': total,
            'recon_loss': recon_loss.item(),
            'freq_loss': freq_loss.item(),
            'mse': recon_mse.mean().item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        """
        Deterministic inference returning both scores.

        Reconstruction score: standard MSE.
        Frequency score: rolling deterministic masks, each frequency bin is
        masked exactly once, the resulting masked-MSE is averaged.

        Returns:
            dict with:
                recon_score: (B,)
                freq_score:  (B,)
        """
        self.eval()
        B, L, C = x.shape
        F = L // 2 + 1
        K = self.freq_infer_segments

        with torch.no_grad():
            # ---- branch 1 ----
            x_hat, _ = self.forward(x)
            recon_score = ((x - x_hat) ** 2).mean(dim=(1, 2))    # (B,)

            # ---- branch 2: rolling masks over [1, F) ----
            freq_indices = torch.arange(1, F, device=x.device)   # protect DC
            # torch.chunk preserves order; each bin appears in exactly one segment
            segments = torch.chunk(freq_indices, K)

            # total_sq_err = torch.zeros(B, device=x.device)
            # total_count  = torch.zeros(B, device=x.device)

            # 新增：初始化一张完整的误差图，用于拼接所有 segment 的误差
            total_err_map = torch.zeros(B, F, C, device=x.device)
            total_count = torch.zeros(B, device=x.device)

            for seg in segments:
                if seg.numel() == 0:
                    continue
                mask = torch.ones(B, F, device=x.device)
                mask[:, seg] = 0.0                               # mask only this segment

                out = self.freq_mask(x, mask=mask)
                pred = self.freq_predictor(out["amp_masked"])
                inv_mask = (1.0 - mask).unsqueeze(-1)            # (B, F, 1)
                # sq_err = ((pred - out["amp_original"]) ** 2 * inv_mask).sum(dim=(1, 2))
                # total_sq_err += sq_err

                # 改动：推理时先累积 raw_err，不在循环内做 sum 或 spike boost
                raw_err = ((pred - out["amp_original"]) ** 2) * inv_mask
                total_err_map += raw_err

                total_count  += inv_mask.sum(dim=(1, 2)) * C     # masked_bins * C
            # freq_score = total_sq_err / (total_count + 1e-8)     # (B,)
            # 新增：循环结束后，对完整的误差谱做一次 spike boost
            boosted = get_infer_spike_boost(total_err_map, kernel_size=5, alpha=2.0)
            freq_score = boosted.sum(dim=(1, 2)) / (total_count + 1e-8)  # (B,)

        return {
            'recon_score': recon_score,
            'freq_score':  freq_score,
        }