"""
FullModel 张量维度验证脚本
========================
在 PyCharm 中直接运行，无需项目包结构。
逐模块验证每个新增组件的输入输出形状，最后做完整的 train/infer 流程测试。

用法：直接运行本文件即可。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 桩类：替代项目内的 BaseAnomalyModel 和 FrequencyMasking
# ============================================================

class BaseAnomalyModel(nn.Module):
    """桩：仅提供基类接口"""
    pass


class FrequencyMasking(nn.Module):
    """桩：简化版，仅保证维度正确"""

    def __init__(self, beta1=0.0, beta2=0.5, log_amp=True, protect_dc=True):
        super().__init__()
        self.beta1, self.beta2 = beta1, beta2
        self.log_amp = log_amp

    def forward(self, x, mask=None):
        B, L, C = x.shape
        F_len = L // 2 + 1
        fft_x = torch.fft.rfft(x, dim=1)
        amp = fft_x.abs()
        phase = torch.angle(fft_x)
        amp_original = torch.log1p(amp) if self.log_amp else amp

        if mask is None:
            mask = torch.ones(B, F_len, device=x.device)
            k = int(F_len * (self.beta1 + self.beta2) / 2)
            mask[:, 1:1 + k] = 0.0

        amp_masked = amp_original * mask.unsqueeze(-1)
        num_masked = (mask == 0).sum(dim=-1)
        return {
            "amp_original": amp_original,
            "amp_masked": amp_masked,
            "mask": mask,
            "phase_original": phase,
            "num_masked": num_masked,
        }


def get_infer_spike_boost(pred_error, log_var, alpha=2.0):
    """桩：与项目中逻辑一致"""
    normal_variance = torch.exp(log_var)
    spike = torch.relu(pred_error - normal_variance)
    return pred_error + alpha * spike


# ============================================================
# 2. 模型组件（与 FullModel.py 一致，去掉相对 import）
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


class TCNDecoder(nn.Module):
    def __init__(self, latent_dim=32, channels=None, output_dim=38,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 64, 96, 96]
        dilations = [2 ** i for i in range(len(channels))]
        self.upsample = nn.ConvTranspose1d(latent_dim, latent_dim,
                                           kernel_size=2, stride=2, padding=0)
        layers, in_ch = [], latent_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(channels[-1], output_dim, 1)

    def forward(self, z):
        z = z.transpose(1, 2)
        z = self.upsample(z)
        out = self.network(z)
        return self.output_proj(out).transpose(1, 2)


class FreqPredictor(nn.Module):
    def __init__(self, input_dim=38, hidden_dim=64, num_layers=4,
                 kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        layers, in_ch = [], input_dim
        for _ in range(num_layers):
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
        h = amp_masked.transpose(1, 2)
        h = self.net(h)
        return self.out_proj(h).transpose(1, 2)


class LearnableLowPass(nn.Module):
    def __init__(self, num_channels, kernel_size=25):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size,
                              groups=num_channels, padding=0, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.padding, 0))
        return self.conv(h).transpose(1, 2)


class MemoryPrototypeBank(nn.Module):
    def __init__(self, feat_dim, num_prototypes=16, tau=0.1):
        super().__init__()
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim) * 0.02)

    def forward(self, z):
        logits = torch.matmul(z, self.prototypes.t()) / self.tau
        weights = torch.softmax(logits, dim=-1)
        z_hat = torch.matmul(weights, self.prototypes)
        sparse_loss = (weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return z_hat, sparse_loss


class SplitTCNEncoder(nn.Module):
    def __init__(self, input_dim=38, channels=None, kernel_size=3, dropout=0.1,
                 split_after=2, num_prototypes=16, proto_tau=0.1):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 32, 32]
        dilations = [2 ** i for i in range(len(channels))]

        front_layers, in_ch = [], input_dim
        for i in range(split_after):
            front_layers.append(TCNBlock(in_ch, channels[i], kernel_size,
                                         dilations[i], dropout))
            in_ch = channels[i]
        self.front = nn.Sequential(*front_layers)

        self.memory = MemoryPrototypeBank(channels[split_after - 1],
                                          num_prototypes, proto_tau)

        back_layers = []
        for i in range(split_after, len(channels)):
            back_layers.append(TCNBlock(in_ch, channels[i], kernel_size,
                                        dilations[i], dropout))
            in_ch = channels[i]
        self.back = nn.Sequential(*back_layers)
        self.downsample = nn.Conv1d(channels[-1], channels[-1],
                                    kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.front(h)
        h = h.transpose(1, 2)
        h, sparse_loss = self.memory(h)
        h = h.transpose(1, 2)
        h = self.back(h)
        h = self.downsample(h)
        return h.transpose(1, 2), sparse_loss


class FullModel(BaseAnomalyModel):
    def __init__(self, input_dim=38, window_size=100,
                 enc_channels=None, dec_channels=None,
                 kernel_size=3, dropout=0.1,
                 trend_kernel_size=25,
                 num_prototypes=16, proto_tau=0.1,
                 sparse_weight=0.01, split_after=2,
                 freq_beta1=0.0, freq_beta2=0.5,
                 freq_hidden_dim=64, freq_num_layers=4,
                 freq_kernel_size=5, freq_loss_weight=1.0,
                 freq_infer_segments=10):
        super().__init__()
        if enc_channels is None:
            enc_channels = [64, 64, 64, 32, 32]
        if dec_channels is None:
            dec_channels = [32, 64, 64, 64, 64]

        self.sparse_weight = sparse_weight
        self.freq_loss_weight = freq_loss_weight
        self.freq_infer_segments = freq_infer_segments

        self.trend_extractor = LearnableLowPass(input_dim, trend_kernel_size)
        self.encoder = SplitTCNEncoder(input_dim, enc_channels, kernel_size,
                                       dropout, split_after, num_prototypes, proto_tau)
        self.decoder = TCNDecoder(enc_channels[-1], dec_channels, input_dim,
                                  kernel_size, dropout)
        self.freq_mask = FrequencyMasking(freq_beta1, freq_beta2, True, True)
        self.freq_predictor = FreqPredictor(input_dim, freq_hidden_dim,
                                            freq_num_layers, freq_kernel_size, dropout)
        F_len = window_size // 2 + 1
        self.freq_log_var = nn.Parameter(torch.zeros(1, F_len, 1))

    def forward(self, x):
        trend = self.trend_extractor(x)
        stable = x - trend
        z, sparse_loss = self.encoder(stable)
        stable_recon = self.decoder(z)
        return stable_recon, trend, sparse_loss

    def _freq_forward(self, x, mask=None):
        out = self.freq_mask(x, mask=mask)
        pred = self.freq_predictor(out["amp_masked"])
        inv_mask = (1.0 - out["mask"]).unsqueeze(-1)
        raw_sq_err = (pred - out["amp_original"]) ** 2
        if self.training:
            precision = torch.exp(-self.freq_log_var)
            loss_components = (raw_sq_err * precision + self.freq_log_var) * inv_mask
            C = x.shape[-1]
            per_sample_err = loss_components.sum(dim=(1, 2)) / (
                out["num_masked"].float() * C + 1e-8)
        else:
            sq_err = raw_sq_err * inv_mask
            C = x.shape[-1]
            per_sample_err = sq_err.sum(dim=(1, 2)) / (
                out["num_masked"].float() * C + 1e-8)
        return per_sample_err, out

    def compute_loss(self, x, x_mark=None):
        stable_recon, trend, sparse_loss = self.forward(x)
        stable = x - trend
        recon_mse = torch.mean((stable - stable_recon) ** 2, dim=(1, 2))
        recon_loss = recon_mse.mean()
        freq_err, _ = self._freq_forward(x, mask=None)
        freq_loss = freq_err.mean()
        total = recon_loss + self.freq_loss_weight * freq_loss + self.sparse_weight * sparse_loss
        return {
            'loss': total,
            'recon_loss': recon_loss.item(),
            'freq_loss': freq_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'mse': recon_mse.mean().item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        self.eval()
        B, L, C = x.shape
        F_len = L // 2 + 1
        K = self.freq_infer_segments
        with torch.no_grad():
            stable_recon, trend, _ = self.forward(x)
            stable = x - trend
            recon_score = ((stable - stable_recon) ** 2).mean(dim=(1, 2))

            freq_indices = torch.arange(1, F_len, device=x.device)
            segments = torch.chunk(freq_indices, K)
            total_err_map = torch.zeros(B, F_len, C, device=x.device)
            total_count = torch.zeros(B, device=x.device)
            for seg in segments:
                if seg.numel() == 0:
                    continue
                mask = torch.ones(B, F_len, device=x.device)
                mask[:, seg] = 0.0
                out = self.freq_mask(x, mask=mask)
                pred = self.freq_predictor(out["amp_masked"])
                inv_mask = (1.0 - mask).unsqueeze(-1)
                raw_err = ((pred - out["amp_original"]) ** 2) * inv_mask
                total_err_map += raw_err
                total_count += inv_mask.sum(dim=(1, 2)) * C
            boosted = get_infer_spike_boost(total_err_map, self.freq_log_var, alpha=2.0)
            freq_score = boosted.sum(dim=(1, 2)) / (total_count + 1e-8)
        return {'recon_score': recon_score, 'freq_score': freq_score}


# ============================================================
# 3. 测试用例
# ============================================================

def sep(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def test_learnable_lowpass():
    sep("LearnableLowPass")
    B, L, C = 4, 100, 38
    x = torch.randn(B, L, C)
    m = LearnableLowPass(C, 25)
    out = m(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    assert out.shape == x.shape
    print("  PASS")


def test_memory_bank():
    sep("MemoryPrototypeBank")
    B, T, D = 4, 100, 64
    z = torch.randn(B, T, D)
    m = MemoryPrototypeBank(D, 16, 0.1)
    z_hat, loss = m(z)
    print(f"  Input:       {tuple(z.shape)}")
    print(f"  Output:      {tuple(z_hat.shape)}")
    print(f"  Sparse loss: {loss.item():.4f}")
    assert z_hat.shape == z.shape and loss.dim() == 0
    print("  PASS")


def test_split_encoder():
    sep("SplitTCNEncoder")
    B, L, C = 4, 100, 38
    ch = [64, 64, 64, 32, 32]
    m = SplitTCNEncoder(C, ch, 3, 0.1, 2, 16, 0.1)
    z, loss = m(torch.randn(B, L, C))
    expected = (B, L // 2, ch[-1])
    print(f"  Input:  (4, 100, 38)")
    print(f"  Latent: {tuple(z.shape)}  expected {expected}")
    assert z.shape == expected
    print("  PASS")


def test_forward():
    sep("FullModel.forward")
    B, L, C = 4, 100, 38
    x = torch.randn(B, L, C)
    model = FullModel(C, L)
    model.train()
    s_recon, trend, loss = model(x)
    print(f"  Input:        {tuple(x.shape)}")
    print(f"  Trend:        {tuple(trend.shape)}")
    print(f"  Stable recon: {tuple(s_recon.shape)}")
    assert trend.shape == x.shape and s_recon.shape == x.shape
    print("  PASS")


def test_loss():
    sep("FullModel.compute_loss + backward")
    x = torch.randn(4, 100, 38)
    model = FullModel(38, 100)
    model.train()
    result = model.compute_loss(x)
    for k, v in result.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {val:.6f}")
    result['loss'].backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Gradient flow: {'PASS' if grad_ok else 'FAIL'}")
    assert grad_ok


def test_score():
    sep("FullModel.compute_anomaly_score")
    x = torch.randn(4, 100, 38)
    model = FullModel(38, 100, freq_infer_segments=10)
    scores = model.compute_anomaly_score(x)
    print(f"  recon_score: {tuple(scores['recon_score'].shape)}")
    print(f"  freq_score:  {tuple(scores['freq_score'].shape)}")
    assert scores['recon_score'].shape == (4,) and scores['freq_score'].shape == (4,)
    print("  PASS")


def test_params():
    sep("Parameter Count")
    model = FullModel(38, 100)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,}")
    for name, mod in [('trend_extractor', model.trend_extractor),
                      ('encoder+memory', model.encoder),
                      ('decoder', model.decoder),
                      ('freq_predictor', model.freq_predictor)]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"    {name}: {n:,}")
    print(f"    freq_log_var: {model.freq_log_var.numel()}")


if __name__ == '__main__':
    torch.manual_seed(42)
    test_learnable_lowpass()
    test_memory_bank()
    test_split_encoder()
    test_forward()
    test_loss()
    test_score()
    test_params()
    sep("ALL TESTS PASSED")