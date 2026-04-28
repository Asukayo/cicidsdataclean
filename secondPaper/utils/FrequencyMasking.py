import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyMasking(nn.Module):
    """
    Frequency-domain random masking module for masked frequency prediction.

    Input:
        x: (B, L, C) real-valued time-series window

    Output (dict):
        amp_original: (B, F, C) ground-truth amplitude spectrum (optionally log-scaled)
        amp_masked:   (B, F, C) masked amplitude spectrum, used as model input
        mask:         (B, F)    binary mask, 1 = keep, 0 = masked  (shared across channels)
        phase_original: (B, F, C) original phase, kept for optional iFFT reconstruction
        num_masked:   (B,)      number of masked frequency bins per sample

    where F = L // 2 + 1 (rFFT output length).

    Design choices:
        - rFFT is used (real input → half spectrum).
        - Mask is shared across all C channels (window-level anomaly granularity).
        - Mask is independent per sample within a batch.
        - DC component (index 0) is never masked.
        - Per-sample mask ratio is sampled from U(beta1, beta2), following FEI.
        - Log-amplitude transform log(1+A) is applied by default for dynamic-range compression.
    """

    def __init__(self,
                 beta1: float = 0.0,
                 beta2: float = 0.5,
                 log_amp: bool = True,
                 protect_dc: bool = True):
        super().__init__()
        assert 0.0 <= beta1 < beta2 < 1.0, \
            f"Expected 0 <= beta1 < beta2 < 1, got ({beta1}, {beta2})"
        self.beta1 = beta1
        self.beta2 = beta2
        self.log_amp = log_amp
        self.protect_dc = protect_dc

    def _sample_mask(self, batch_size: int, freq_len: int, device) -> torch.Tensor:
        """
        Sample a random binary mask of shape (B, F).
        For each sample, first draw a ratio r ~ U(beta1, beta2),
        then randomly select k = round(r * F_eff) positions to mask.
        """
        F_eff = freq_len - 1 if self.protect_dc else freq_len   # positions eligible for masking
        offset = 1 if self.protect_dc else 0

        # Per-sample mask ratio
        ratios = torch.empty(batch_size, device=device).uniform_(self.beta1, self.beta2)
        ks = (ratios * F_eff).round().long().clamp(min=0, max=F_eff)   # (B,)

        mask = torch.ones(batch_size, freq_len, device=device)
        # Vectorized random selection via argsort on random noise
        noise = torch.rand(batch_size, F_eff, device=device)
        sorted_idx = noise.argsort(dim=-1)      # (B, F_eff), random permutation per row

        # Build a (B, F_eff) boolean matrix indicating which positions to mask
        arange = torch.arange(F_eff, device=device).unsqueeze(0).expand(batch_size, -1)
        mask_flag = arange < ks.unsqueeze(-1)   # True where we should mask
        # mask_flag is aligned with sorted_idx; scatter back
        to_mask = torch.zeros(batch_size, F_eff, device=device, dtype=torch.bool)
        to_mask.scatter_(1, sorted_idx, mask_flag)

        mask[:, offset:] = torch.where(to_mask,
                                       torch.zeros_like(mask[:, offset:]),
                                       torch.ones_like(mask[:, offset:]))
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        :param x: (B, L, C) real-valued window
        :param mask: optional (B, F) predetermined mask. If None, a random mask is sampled.
                     Passing a fixed mask is useful for deterministic inference.
        """
        assert x.dim() == 3, f"Expected (B, L, C), got shape {tuple(x.shape)}"
        B, L, C = x.shape
        F = L // 2 + 1

        # FFT along time axis
        fft_x = torch.fft.rfft(x, dim=1)                     # (B, F, C) complex
        amp = fft_x.abs()                                    # (B, F, C) real, non-negative
        phase = torch.angle(fft_x)                           # (B, F, C)

        # Optional log-amplitude compression
        if self.log_amp:
            amp_original = torch.log1p(amp)
        else:
            amp_original = amp

        # Mask sampling
        if mask is None:
            mask = self._sample_mask(B, F, x.device)         # (B, F), {0,1}
        else:
            assert mask.shape == (B, F), \
                f"Expected mask shape {(B, F)}, got {tuple(mask.shape)}"
            mask = mask.to(x.device).float()

        # Apply mask (broadcast across channel dim)
        amp_masked = amp_original * mask.unsqueeze(-1)       # (B, F, C)

        num_masked = (mask == 0).sum(dim=-1)                 # (B,)

        return {
            "amp_original": amp_original,
            "amp_masked": amp_masked,
            "mask": mask,
            "phase_original": phase,
            "num_masked": num_masked,
        }


def get_train_freq_weight(F_len, device,
                          w_lowfreq=0.2,      # 低频峰值区(漂移最严重)的权重
                          w_midfreq=1.0,      # 中频稳定区的权重(最高)
                          w_highfreq=0.6,     # 高频次稳区的权重(中等)
                          low_end=0.10,       # 低频区结束位置(归一化)
                          mid_start=0.18,     # 中频区开始位置
                          mid_end=0.28,       # 中频区结束位置
                          high_start=0.55):   # 高频区开始位置
    """
    基于频域分布距离分析得到的频段稳定性加权.

    形状(基于 CICIDS2017 的实测漂移曲线):
      - [0, low_end]: 低频,漂移最严重,权重最低
      - [low_end, mid_start]: 过渡区,权重线性上升
      - [mid_start, mid_end]: 中频,最稳定,权重最高
      - [mid_end, high_start]: 过渡区,权重线性下降
      - [high_start, 1.0]: 高频,次稳定,权重中等
    """
    freq_idx = torch.linspace(0, 1, F_len, device=device)
    weight = torch.zeros_like(freq_idx)

    # 低频稳定段
    mask_low = freq_idx <= low_end
    weight[mask_low] = w_lowfreq

    # 低频→中频过渡(线性上升)
    mask_up = (freq_idx > low_end) & (freq_idx < mid_start)
    t = (freq_idx[mask_up] - low_end) / (mid_start - low_end)
    weight[mask_up] = w_lowfreq + t * (w_midfreq - w_lowfreq)

    # 中频稳定段
    mask_mid = (freq_idx >= mid_start) & (freq_idx <= mid_end)
    weight[mask_mid] = w_midfreq

    # 中频→高频过渡(线性下降)
    mask_down = (freq_idx > mid_end) & (freq_idx < high_start)
    t = (freq_idx[mask_down] - mid_end) / (high_start - mid_end)
    weight[mask_down] = w_midfreq + t * (w_highfreq - w_midfreq)

    # 高频稳定段
    mask_high = freq_idx >= high_start
    weight[mask_high] = w_highfreq

    weight[0] = 0.0  # DC 不参与 loss

    return weight.view(1, F_len, 1)


# def get_infer_spike_boost(pred_error, kernel_size=5, alpha=2.0):
#     """
#     推理时使用:在预测误差谱上检测尖峰,额外叠加尖锐异常的惩罚
#     pred_error: (B, F, C) 每个频率位置的预测误差
#     返回: (B, F, C) 增强后的预测误差
#     """
#     x = pred_error.transpose(1, 2)        # (B, C, F)
#     pad = kernel_size // 2
#     local_bg = F.avg_pool1d(x, kernel_size=kernel_size,
#                             stride=1, padding=pad)
#     local_bg = local_bg.transpose(1, 2)   # (B, F, C)
#
#     # 只提取超出局部背景的部分
#     spike = torch.relu(pred_error - local_bg)
#
#     # 加法叠加:原始误差不变,尖峰部分额外加分
#     return pred_error + alpha * spike

# 以前用一维平均池化（avg_pool1d）来估算局部的背景噪声。
# 现在有了极具物理意义的参数 log_var（它就是模型学出来的各个频段的正常噪声方差），直接用它来判定尖峰，
def get_infer_spike_boost(pred_error, log_var, alpha=2.0):
    """
    推理时使用：基于模型学到的同方差不确定性 (log_var) 检测异常尖峰。
    如果预测误差远大于该频段的“正常噪声容忍度”，则予以放大。

    pred_error: (B, F, C) 每个频率位置的原始预测误差
    log_var:    (1, F, 1) 模型学到的对数方差参数
    alpha:      放大系数
    """
    # 1. 计算该频段正常的方差容忍度 exp(s)
    normal_variance = torch.exp(log_var)

    # 2. 提取超出正常容忍度的突变尖峰 (小于容忍度的部分会被 relu 截断为 0)
    spike = torch.relu(pred_error - normal_variance)

    # 3. 原始误差 + 放大的尖峰惩罚
    return pred_error + alpha * spike



if __name__ == "__main__":
    # Sanity check
    torch.manual_seed(0)
    B, L, C = 4, 100, 8
    x = torch.randn(B, L, C)

    module = FrequencyMasking(beta1=0.0, beta2=0.5, log_amp=True, protect_dc=True)
    out = module(x)

    print("Input shape:          ", tuple(x.shape))
    print("amp_original shape:   ", tuple(out["amp_original"].shape))
    print("amp_masked shape:     ", tuple(out["amp_masked"].shape))
    print("mask shape:           ", tuple(out["mask"].shape))
    print("phase_original shape: ", tuple(out["phase_original"].shape))
    print("num_masked per sample:", out["num_masked"].tolist())
    print("DC always kept:       ", bool((out["mask"][:, 0] == 1).all().item()))

    # Check that masked positions are indeed zero in amp_masked
    masked_vals = out["amp_masked"][out["mask"].unsqueeze(-1).expand_as(out["amp_masked"]) == 0]
    print("Masked positions all zero:", bool((masked_vals == 0).all().item()))

    # Check determinism when mask is provided
    fixed_mask = out["mask"]
    out2 = module(x, mask=fixed_mask)
    print("Deterministic mode consistent:",
          bool(torch.allclose(out["amp_masked"], out2["amp_masked"])))