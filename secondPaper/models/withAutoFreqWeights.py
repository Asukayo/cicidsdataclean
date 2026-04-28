import torch
import torch.nn as nn
from .BaseModel import BaseAnomalyModel
# 注意：这里导入的函数去掉了 get_train_freq_weight
from ..utils.FrequencyMasking import FrequencyMasking, get_infer_spike_boost


# ... [保留原有的 CausalConv1d, TCNBlock, TCNEncoder, TCNDecoder, FreqPredictor 不变] ...
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





class TCNAEWithFreq(BaseAnomalyModel):
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
        # 【新增】：全局频率不确定性参数 log_var (初始值为 0)
        # F_len 是 rFFT 后的长度 (window_size // 2 + 1)
        F_len = window_size // 2 + 1
        self.freq_log_var = nn.Parameter(torch.zeros(1, F_len, 1))

    # ---- original reconstruction branch (untouched) ----
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    # ---- 核心重写：_freq_forward ----
    def _freq_forward(self, x, mask=None):
        out = self.freq_mask(x, mask=mask)
        pred = self.freq_predictor(out["amp_masked"])
        inv_mask = (1.0 - out["mask"]).unsqueeze(-1)

        # 原始的平方误差 (B, F, C)
        raw_sq_err = (pred - out["amp_original"]) ** 2

        if self.training:
            # 【修改】：自适应不确定性 Loss
            precision = torch.exp(-self.freq_log_var)  # (1, F, 1)
            # 加权误差项 + 正则化惩罚项 (只在被掩码的位置计算)
            loss_components = (raw_sq_err * precision + self.freq_log_var) * inv_mask
            C = x.shape[-1]
            per_sample_err = loss_components.sum(dim=(1, 2)) / (out["num_masked"].float() * C + 1e-8)
        else:
            # 推理阶段计算 loss 时通常只看原始误差
            sq_err = raw_sq_err * inv_mask
            C = x.shape[-1]
            per_sample_err = sq_err.sum(dim=(1, 2)) / (out["num_masked"].float() * C + 1e-8)

        return per_sample_err, out

    # ... [compute_loss 保持你原来的不变] ...
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

    # ---- 核心重写：compute_anomaly_score ----
    def compute_anomaly_score(self, x, x_mark=None):
        self.eval()
        B, L, C = x.shape
        F = L // 2 + 1
        K = self.freq_infer_segments

        with torch.no_grad():
            x_hat, _ = self.forward(x)
            recon_score = ((x - x_hat) ** 2).mean(dim=(1, 2))

            freq_indices = torch.arange(1, F, device=x.device)
            segments = torch.chunk(freq_indices, K)

            total_err_map = torch.zeros(B, F, C, device=x.device)
            total_count = torch.zeros(B, device=x.device)

            for seg in segments:
                if seg.numel() == 0:
                    continue
                mask = torch.ones(B, F, device=x.device)
                mask[:, seg] = 0.0

                out = self.freq_mask(x, mask=mask)
                pred = self.freq_predictor(out["amp_masked"])
                inv_mask = (1.0 - mask).unsqueeze(-1)

                raw_err = ((pred - out["amp_original"]) ** 2) * inv_mask
                total_err_map += raw_err
                total_count += inv_mask.sum(dim=(1, 2)) * C

            # 【修改】：利用模型学到的 log_var 进行智能尖峰增强
            # 也可以使用kernel
            boosted = get_infer_spike_boost(total_err_map, self.freq_log_var, alpha=2.0)

            # boosted = get_infer_spike_boost(total_err_map, kernel_size = 5, alpha=2.0)

            freq_score = boosted.sum(dim=(1, 2)) / (total_count + 1e-8)

        return {
            'recon_score': recon_score,
            'freq_score': freq_score,
        }