import torch
import torch.nn as nn

from .BaseModel import BaseAnomalyModel


class OmniAnomaly(BaseAnomalyModel):
    """
    OmniAnomaly: GRU + VAE 重构式异常检测
    ========================================
    改动说明（相对原始实现）：
      - 移除 Decoder 末层 Sigmoid，适配 StandardScaler 标准化后的数据
      - 实现 BaseAnomalyModel 接口（compute_loss / compute_anomaly_score）
      - 移除跨 batch 传递 hidden state 的逻辑（shuffle=True 时无意义）
    """

    def __init__(self, feats, device, n_hidden=32, n_latent=8, beta=0.01):
        super().__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.beta = beta
        self.n_feats = feats
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        self.gru = nn.GRU(feats, self.n_hidden, num_layers=2, batch_first=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats),  # 无 Sigmoid
        )

    def forward(self, x):
        """
        Args:
            x: [B, W, F]
        Returns:
            x_recon: [B, W*F]
            mu:      [B, W*n_latent]
            logvar:  [B, W*n_latent]
        """
        bs, win = x.shape[0], x.shape[1]

        # GRU: 输入 [W, B, F]
        out, _ = self.gru(x.permute(1, 0, 2))  # [W, B, n_hidden]

        # Encode → (mu, logvar)
        h = self.encoder(out)                    # [W, B, 2*n_latent]
        mu, logvar = torch.split(h, self.n_latent, dim=-1)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Decode
        x_recon = self.decoder(z)                # [W, B, F]

        return (
            x_recon.permute(1, 0, 2).reshape(bs, win * self.n_feats),
            mu.permute(1, 0, 2).reshape(bs, win * self.n_latent),
            logvar.permute(1, 0, 2).reshape(bs, win * self.n_latent),
        )

    # ---------- BaseAnomalyModel 接口 ----------

    def compute_loss(self, x, x_mark=None):
        x_recon, mu, logvar = self.forward(x)
        x_flat = x.reshape(x.shape[0], -1)

        mse = torch.mean((x_recon - x_flat) ** 2, dim=-1)          # [B]
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B]
        kld = torch.clamp(kld, max=1e4)  # 防止单个样本 KLD 爆炸

        loss = torch.mean(mse + self.beta * kld)
        return {
            'loss': loss,
            'mse': mse.mean().item(),
            'kld': kld.mean().item(),
        }

    def encode_decode(self, x):
        """确定性前向：用 mu 代替采样，消除随机性"""
        bs, win = x.shape[0], x.shape[1]
        out, _ = self.gru(x.permute(1, 0, 2))

        h = self.encoder(out)
        mu, logvar = torch.split(h, self.n_latent, dim=-1)

        # 直接用 mu，不采样
        x_recon = self.decoder(mu)

        return x_recon.permute(1, 0, 2).reshape(bs, win * self.n_feats), mu, logvar

    def compute_anomaly_score(self, x, x_mark=None):
        x_recon, _, _ = self.encode_decode(x)
        x_flat = x.reshape(x.shape[0], -1)
        return torch.mean((x_recon - x_flat) ** 2, dim=-1)