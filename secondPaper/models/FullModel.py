"""
FullModel: TCN Autoencoder with Trend-Stable Decomposition,
           Multi-Scale Memory Prototypes, and Frequency Masked Prediction.

Architecture overview:
    x → LearnableLowPass → trend
      → stable = x - trend
      → TCNEncoder_front (Block 1-2)
      → MemoryPrototypeBank (softmax attention over K prototypes)
      → TCNEncoder_back  (Block 3-5 + downsample)
      → TCNDecoder → ŝ (stable reconstruction)
      → x̂ = ŝ + trend
      → anomaly score = MSE(stable, ŝ)   (trend excluded from scoring)

    Frequency branch (unchanged from withAutoFreqWeights):
      x → rFFT → random mask → FreqPredictor → masked prediction loss
      with homoscedastic uncertainty weighting (freq_log_var)
"""

import torch
import torch.nn as nn
from .BaseModel import BaseAnomalyModel
from ..utils.FrequencyMasking import FrequencyMasking, get_infer_spike_boost


# ============================================================
# Reused building blocks (identical to withAutoFreqWeights.py)
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
        z = z.transpose(1, 2)  # (B, D, T')
        z = self.upsample(z)   # (B, D, T)
        out = self.network(z)
        x_hat = self.output_proj(out).transpose(1, 2)  # (B, T, C)
        return x_hat


class FreqPredictor(nn.Module):
    def __init__(self, input_dim=38, hidden_dim=64, num_layers=4,
                 kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        in_ch = input_dim
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
        h = amp_masked.transpose(1, 2)  # (B, C, F)
        h = self.net(h)
        return self.out_proj(h).transpose(1, 2)  # (B, F, C)


# ============================================================
# NEW module 1: Learnable causal low-pass filter
# ============================================================

class LearnableLowPass(nn.Module):
    """
    Per-channel causal depthwise convolution with large kernel.
    Initialized as uniform average (≈ moving average) so the starting
    point is a reasonable low-pass filter; the kernel shape is then
    refined during training.

    Input:  (B, L, C)
    Output: (B, L, C)  — the extracted trend component
    """

    def __init__(self, num_channels, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # causal: pad left only
        self.conv = nn.Conv1d(
            num_channels, num_channels, kernel_size,
            groups=num_channels,  # depthwise: each channel independent
            padding=0, bias=False,
        )
        # Initialize as uniform average: 1/k
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x: (B, L, C)
        h = x.transpose(1, 2)                            # (B, C, L)
        h = nn.functional.pad(h, (self.padding, 0))      # causal left-pad
        trend = self.conv(h).transpose(1, 2)              # (B, L, C)
        return trend


# ============================================================
# NEW module 2: Memory Prototype Bank
# ============================================================

class MemoryPrototypeBank(nn.Module):
    """
    Learnable prototype memory for constraining latent representations
    to the convex hull of normal patterns.

    Input:  (B, T, D)  — intermediate encoder features
    Output: (B, T, D)  — prototype-constrained features
            sparse_loss — entropy regularization scalar

    Each timestep z_t is replaced by a soft combination of K prototypes:
        ẑ_t = Σ_k  softmax(z_t · m_k / τ) · m_k
    """

    def __init__(self, feat_dim, num_prototypes=16, tau=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = tau
        # Prototype vectors: (K, D)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim) * 0.02)

    def forward(self, z):
        """
        z: (B, T, D)
        Returns:
            z_hat:       (B, T, D)
            sparse_loss: scalar (mean negative entropy → encourages peaky attention)
        """
        # Dot-product attention: z @ M^T / τ
        logits = torch.matmul(z, self.prototypes.t()) / self.tau  # (B, T, K)
        weights = torch.softmax(logits, dim=-1)                   # (B, T, K)

        # Weighted combination of prototypes
        z_hat = torch.matmul(weights, self.prototypes)            # (B, T, D)

        # Sparsity loss: minimize negative entropy = Σ w·log(w)
        sparse_loss = (weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

        return z_hat, sparse_loss


# ============================================================
# NEW module 3: Split TCN Encoder with prototype injection
# ============================================================

class SplitTCNEncoder(nn.Module):
    """
    TCN Encoder split into front/back stages with MemoryPrototypeBank
    inserted in between.

    Stage 1 (front):  input_dim → channels[0] → ... → channels[split-1]
    ↓ MemoryPrototypeBank at channels[split-1] dimensionality
    Stage 2 (back):   channels[split-1] → ... → channels[-1]
    ↓ Causal stride-2 downsample: T → T//2
    """

    def __init__(self, input_dim=38, channels=None, kernel_size=3, dropout=0.1,
                 split_after=2, num_prototypes=16, proto_tau=0.1):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 32, 32]
        assert 0 < split_after < len(channels), \
            f"split_after={split_after} must be in (0, {len(channels)})"

        dilations = [2 ** i for i in range(len(channels))]

        # ---- Front stage: Block 0 .. split_after-1 ----
        front_layers, in_ch = [], input_dim
        for i in range(split_after):
            front_layers.append(TCNBlock(in_ch, channels[i], kernel_size,
                                         dilations[i], dropout))
            in_ch = channels[i]
        self.front = nn.Sequential(*front_layers)

        # ---- Memory prototype bank ----
        self.memory = MemoryPrototypeBank(
            feat_dim=channels[split_after - 1],
            num_prototypes=num_prototypes,
            tau=proto_tau,
        )

        # ---- Back stage: Block split_after .. end ----
        back_layers = []
        for i in range(split_after, len(channels)):
            back_layers.append(TCNBlock(in_ch, channels[i], kernel_size,
                                        dilations[i], dropout))
            in_ch = channels[i]
        self.back = nn.Sequential(*back_layers)

        # ---- Causal stride-2 downsample: T → T//2 ----
        self.downsample = nn.Conv1d(channels[-1], channels[-1],
                                    kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        x: (B, L, C)
        Returns:
            z:           (B, L//2, channels[-1])
            sparse_loss: scalar
        """
        h = x.transpose(1, 2)          # (B, C, L)  for Conv1d
        h = self.front(h)              # (B, ch_mid, L)

        h = h.transpose(1, 2)          # (B, L, ch_mid)  for memory bank
        h, sparse_loss = self.memory(h)
        h = h.transpose(1, 2)          # (B, ch_mid, L)  back to Conv1d

        h = self.back(h)               # (B, ch_last, L)
        h = self.downsample(h)         # (B, ch_last, L//2)
        z = h.transpose(1, 2)          # (B, L//2, ch_last)

        return z, sparse_loss


# ============================================================
# Main model
# ============================================================

class FullModel(BaseAnomalyModel):
    """
    Dual-branch anomaly detection model:
      Branch 1: Trend-Stable decomposition + Memory-augmented TCN AE
      Branch 2: Frequency masked prediction with adaptive uncertainty

    Three-layer drift defense:
      Layer 1 (input space):  LearnableLowPass strips trend → stable input
      Layer 2 (latent space): MemoryPrototypeBank constrains z to normal manifold
      Layer 3 (score space):  anomaly score computed on stable component only
    """

    def __init__(
            self,
            input_dim=38,
            window_size=100,
            enc_channels=None,
            dec_channels=None,
            kernel_size=3,
            dropout=0.1,

            # ---- trend decomposition ----
            trend_kernel_size=25,

            # ---- memory prototype bank ----
            num_prototypes=16,
            proto_tau=0.1,
            sparse_weight=0.01,
            split_after=2,

            # ---- frequency branch ----
            freq_beta1=0.0,
            freq_beta2=0.5,
            freq_hidden_dim=64,
            freq_num_layers=4,
            freq_kernel_size=5,
            freq_loss_weight=1.0,
            freq_infer_segments=10,
    ):
        super().__init__()
        self.name = 'FullModel'

        if enc_channels is None:
            enc_channels = [64, 64, 64, 32, 32]
        if dec_channels is None:
            dec_channels = [32, 64, 64, 64, 64]

        self.input_dim = input_dim
        self.window_size = window_size
        self.sparse_weight = sparse_weight
        self.freq_loss_weight = freq_loss_weight
        self.freq_infer_segments = freq_infer_segments

        # ---- Layer 1: Trend-Stable decomposition ----
        self.trend_extractor = LearnableLowPass(input_dim, trend_kernel_size)

        # ---- Layer 2: Split encoder with memory prototypes ----
        self.encoder = SplitTCNEncoder(
            input_dim=input_dim,
            channels=enc_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            split_after=split_after,
            num_prototypes=num_prototypes,
            proto_tau=proto_tau,
        )

        # ---- Decoder ----
        self.decoder = TCNDecoder(
            latent_dim=enc_channels[-1],
            channels=dec_channels,
            output_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # ---- Branch 2: Frequency masked prediction ----
        self.freq_mask = FrequencyMasking(
            beta1=freq_beta1, beta2=freq_beta2,
            log_amp=True, protect_dc=True,
        )
        self.freq_predictor = FreqPredictor(
            input_dim=input_dim,
            hidden_dim=freq_hidden_dim,
            num_layers=freq_num_layers,
            kernel_size=freq_kernel_size,
            dropout=dropout,
        )
        F_len = window_size // 2 + 1
        self.freq_log_var = nn.Parameter(torch.zeros(1, F_len, 1))

    # ----------------------------------------------------------------
    #  Forward: reconstruction branch
    # ----------------------------------------------------------------

    def forward(self, x):
        """
        Returns:
            stable_recon: (B, L, C)  — reconstructed stable component
            trend:        (B, L, C)  — extracted trend (for add-back)
            sparse_loss:  scalar     — prototype sparsity regularization
        """
        trend = self.trend_extractor(x)         # (B, L, C)
        stable = x - trend                      # (B, L, C)

        z, sparse_loss = self.encoder(stable)   # (B, L//2, D), scalar
        stable_recon = self.decoder(z)           # (B, L, C)

        return stable_recon, trend, sparse_loss

    # ----------------------------------------------------------------
    #  Frequency branch (identical to withAutoFreqWeights)
    # ----------------------------------------------------------------

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
                out["num_masked"].float() * C + 1e-8
            )
        else:
            sq_err = raw_sq_err * inv_mask
            C = x.shape[-1]
            per_sample_err = sq_err.sum(dim=(1, 2)) / (
                out["num_masked"].float() * C + 1e-8
            )

        return per_sample_err, out

    # ----------------------------------------------------------------
    #  Training loss
    # ----------------------------------------------------------------

    def compute_loss(self, x, x_mark=None):
        """
        Total = stable_recon_MSE + λ1 * freq_loss + λ2 * sparse_loss

        Note: recon loss is computed on stable component only (trend excluded).
        """
        # Branch 1: reconstruction on stable component
        stable_recon, trend, sparse_loss = self.forward(x)
        stable = x - trend

        recon_mse = torch.mean((stable - stable_recon) ** 2, dim=(1, 2))  # (B,)
        recon_loss = recon_mse.mean()

        # Branch 2: frequency masked prediction (on original x)
        freq_err, _ = self._freq_forward(x, mask=None)
        freq_loss = freq_err.mean()

        # Total
        total = (recon_loss
                 + self.freq_loss_weight * freq_loss
                 + self.sparse_weight * sparse_loss)

        return {
            'loss': total,
            'recon_loss': recon_loss.item(),
            'freq_loss': freq_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'mse': recon_mse.mean().item(),
        }

    # ----------------------------------------------------------------
    #  Inference: anomaly scoring
    # ----------------------------------------------------------------

    def compute_anomaly_score(self, x, x_mark=None):
        self.eval()
        B, L, C = x.shape
        F = L // 2 + 1
        K = self.freq_infer_segments

        with torch.no_grad():
            # ---- Recon score: stable component only ----
            stable_recon, trend, _ = self.forward(x)
            stable = x - trend
            recon_score = ((stable - stable_recon) ** 2).mean(dim=(1, 2))  # (B,)

            # ---- Freq score: deterministic rolling mask ----
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

            boosted = get_infer_spike_boost(
                total_err_map, self.freq_log_var, alpha=2.0
            )
            freq_score = boosted.sum(dim=(1, 2)) / (total_count + 1e-8)

        return {
            'recon_score': recon_score,
            'freq_score': freq_score,
        }
