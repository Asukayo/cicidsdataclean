"""
TCN-based AutoEncoder for Unsupervised Time Series Anomaly Detection
=====================================================================
Designed for: CICIDS2017, Window=100, 38 features
Key design choices:
  - Dilated causal convolutions with exponentially growing dilation
  - BatchNorm after each conv layer (for future Test-Time Adaptive Normalization)
  - Symmetric encoder-decoder structure
  - Bottleneck via channel compression, NOT temporal pooling (preserves time resolution)
"""

import torch
import torch.nn as nn

from .BaseModel import BaseAnomalyModel


# ============================================================
# Building Block: Residual Dilated Causal Conv Block
# ============================================================
class CausalConv1d(nn.Module):
    """Causal convolution: pad only on the left so future info doesn't leak."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove the extra right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Single TCN residual block:
        CausalConv -> BatchNorm -> ReLU -> Dropout ->
        CausalConv -> BatchNorm -> ReLU -> Dropout + Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1x1 conv for residual connection if channel mismatch
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


# ============================================================
# Encoder
# ============================================================
class TCNEncoder(nn.Module):
    """
    Multi-layer TCN encoder with increasing dilation.
    Receptive field = sum over layers of: 2 * (kernel_size - 1) * dilation
    With kernel_size=3, dilation=[1,2,4,8,16]:
        RF = 2*(2)*1 + 2*(2)*2 + 2*(2)*4 + 2*(2)*8 + 2*(2)*16 = 124 > W=100 ✓
    """

    def __init__(self, input_dim=38, channels=None, kernel_size=3, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 32, 32]  # gradually compress channels

        dilations = [2 ** i for i in range(len(channels))]
        layers = []
        in_ch = input_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — standard time series format
        Returns:
            z: (batch, seq_len, latent_dim) — encoded representation
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        z = self.network(x)
        z = z.transpose(1, 2)
        return z


# ============================================================
# Decoder
# ============================================================
class TCNDecoder(nn.Module):
    """
    Mirror of the encoder. Uses non-causal (standard) convolutions
    since the decoder reconstructs the full sequence.
    """

    def __init__(self, latent_dim=32, channels=None, output_dim=38, kernel_size=3, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 64, 64, 64]  # mirror of encoder

        dilations = [2 ** i for i in range(len(channels))]
        layers = []
        in_ch = latent_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(channels[-1], output_dim, 1)

    def forward(self, z):
        """
        Args:
            z: (batch, seq_len, latent_dim)
        Returns:
            x_hat: (batch, seq_len, output_dim)
        """
        z = z.transpose(1, 2)
        out = self.network(z)
        x_hat = self.output_proj(out)
        x_hat = x_hat.transpose(1, 2)
        return x_hat


# ============================================================
# Full AutoEncoder
# ============================================================
class TCNAE(BaseAnomalyModel):
    """
    TCN AutoEncoder for unsupervised anomaly detection.

    Usage:
        model = TCNAE(input_dim=38, window_size=100)
        x = torch.randn(32, 100, 38)  # (batch, seq_len, features)
        loss_dict = model.compute_loss(x)
        loss_dict['loss'].backward()
    """

    def __init__(
        self,
        input_dim=38,
        window_size=100,
        enc_channels=None,
        dec_channels=None,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.name = 'TCNAE'
        if enc_channels is None:
            enc_channels = [64, 64, 64, 32, 32]
        if dec_channels is None:
            dec_channels = [32, 64, 64, 64, 64]

        self.encoder = TCNEncoder(input_dim, enc_channels, kernel_size, dropout)
        self.decoder = TCNDecoder(enc_channels[-1], dec_channels, input_dim, kernel_size, dropout)
        self.input_dim = input_dim
        self.window_size = window_size

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    # ---------- BaseAnomalyModel 接口 ----------

    def compute_loss(self, x, x_mark=None):
        """
        计算训练损失

        Args:
            x:      [B, W, F] 输入窗口
            x_mark: [B, W]    时间掩码（未使用，保持接口一致）

        Returns:
            dict: 包含 'loss' 和 'mse' 键
        """
        x_hat, _ = self.forward(x)
        mse = torch.mean((x - x_hat) ** 2, dim=(1, 2))  # [B]
        loss = mse.mean()
        return {
            'loss': loss,
            'mse': mse.mean().item(),
        }

    def compute_anomaly_score(self, x, x_mark=None):
        """
        计算每个窗口的异常分数（确定性推理，不更新 BN 统计量）

        Args:
            x:      [B, W, F]
            x_mark: [B, W]

        Returns:
            scores: [B] 每个窗口的 MSE 重构误差
        """
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores