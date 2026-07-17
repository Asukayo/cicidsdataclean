"""
FreqDAR full model: frequency-aware drift-adaptive reconstruction.

Architecture overview:
    x → LearnableLowPass → trend
      → stable = x - trend
      → TCNEncoder_front (Block 1-2)
      → MemoryPrototypeBank (cosine attention, direction constraint,
                              and input-norm scaling)
      → TCNEncoder_back  (Block 3-5 + downsample)
      → TCNDecoder → ŝ (stable reconstruction)
      → anomaly score = MSE(stable, ŝ)   (trend excluded from scoring)

    Frequency branch:
      x → rFFT → random mask → FreqPredictor → masked prediction loss
      with homoscedastic uncertainty weighting (freq_log_var)

    Final score:
      validation ECDF(reconstruction score)
      + validation ECDF(frequency score)
"""

import torch
import torch.nn as nn
from .BaseModel import BaseAnomalyModel
from ..utils.FrequencyMasking import FrequencyMasking, get_infer_spike_boost
import torch.nn.functional as F

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
    def __init__(self, latent_dim=32, channels=None, output_dim=68,
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
    def __init__(self, input_dim=68, hidden_dim=64, num_layers=4,
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

    def __init__(self, num_channels, kernel_size=25, smooth_weight=0.1):
        super().__init__()
        # smooth_weight is kept for backward compatibility; the actual loss
        # coefficient is controlled by FullModel.smooth_weight.
        self.smooth_weight = smooth_weight
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # causal: pad left only
        # Softmax-normalized kernel implements a non-negative weighted moving average.
        self.raw_weight = nn.Parameter(torch.zeros(num_channels, 1, kernel_size))

    def forward(self, x):
        h = x.transpose(1, 2)  # (B, C, L)
        h = nn.functional.pad(h, (self.padding, 0))  # causal left-pad
        w = torch.softmax(self.raw_weight, dim=-1)  # (C, 1, K), non-negative and sums to 1
        trend = nn.functional.conv1d(h, w, groups=h.shape[1]).transpose(1, 2)
        return trend

    def smoothness_loss(self):
        """Difference smoothness regularization on the normalized low-pass kernel."""
        w = torch.softmax(self.raw_weight, dim=-1)
        return ((w[:, :, 1:] - w[:, :, :-1]) ** 2).mean()


# ============================================================
# NEW module 2: Memory Prototype Bank
# ============================================================

class MemoryPrototypeBank(nn.Module):
    """
    Direction-constrained prototype memory.

    Input:  (B, T, D)  — intermediate encoder features
    Output: (B, T, D)  — prototype-constrained features
            sparse_loss — entropy regularization scalar

    Each timestep z_t is decomposed into direction and magnitude. Attention is
    computed by cosine similarity against normalized prototypes. A convex
    combination of prototype directions is normalized again, and the resulting
    direction is rescaled by the input feature norm.
    """

    def __init__(self, feat_dim, num_prototypes=16, tau=0.1, eps=1e-8):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = tau
        self.eps = eps
        # Prototype vectors: (K, D)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim) * 0.02)

    def forward(self, z):
        # Decompose into direction and magnitude
        z_scale = z.norm(dim=-1, keepdim=True)  # (B, T, 1)
        z_norm = F.normalize(z, dim=-1, eps=self.eps)  # (B, T, D)
        m_norm = F.normalize(self.prototypes, dim=-1, eps=self.eps)  # (K, D)

        # Cosine attention: match direction only
        logits = torch.matmul(z_norm, m_norm.t()) / self.tau  # (B, T, K)
        weights = torch.softmax(logits, dim=-1)  # (B, T, K)

        # Eq. (5): normalize the prototype mixture direction before copying
        # the input norm. Without this second normalization, the output norm
        # also depends on attention dispersion and does not match the paper.
        mixture = torch.matmul(weights, m_norm)  # (B, T, D)
        mixture_direction = mixture / (
            mixture.norm(dim=-1, keepdim=True) + self.eps
        )
        z_hat = z_scale * mixture_direction  # (B, T, D)

        # Entropy sparsity regularization: minimizing it sharpens attention.
        sparse_loss = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

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

    def __init__(self, input_dim=68, channels=None, kernel_size=3, dropout=0.1,
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
            input_dim=68,
            window_size=100,
            enc_channels=None,
            dec_channels=None,
            kernel_size=3,
            dropout=0.1,
            smooth_weight=0.1,

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
            freq_spike_alpha=2.0,
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
        self.freq_spike_alpha = freq_spike_alpha
        self.smooth_weight = smooth_weight

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

        # Validation ECDF references are data-dependent calibration state, not
        # trainable model parameters. They are deliberately excluded from model
        # checkpoints and must be fitted from the validation split after loading.
        self.register_buffer("_val_recon_sorted", torch.empty(0), persistent=False)
        self.register_buffer("_val_freq_sorted", torch.empty(0), persistent=False)

    # ----------------------------------------------------------------
    #  Forward: reconstruction branch
    # ----------------------------------------------------------------

    def forward(self, x):
        """
        Returns:
            stable_recon: (B, L, C)  — reconstructed stable component
            trend:        (B, L, C)  — extracted trend used to form the target
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

        if raw_sq_err.shape[1] != self.freq_log_var.shape[1]:
            raise ValueError(
                "Input window length does not match the configured window_size: "
                f"got {x.shape[1]} time steps ({raw_sq_err.shape[1]} rFFT bins), "
                f"expected {self.window_size} ({self.freq_log_var.shape[1]} bins)."
            )

        # Eq. (1) is the frequency training/validation objective. Inference
        # scoring is implemented separately in compute_anomaly_score(), so this
        # loss must not change merely because the module is in eval mode.
        precision = torch.exp(-self.freq_log_var)
        loss_components = (raw_sq_err * precision + self.freq_log_var) * inv_mask
        C = x.shape[-1]
        per_sample_err = loss_components.sum(dim=(1, 2)) / (
            out["num_masked"].float() * C + 1e-8
        )

        return per_sample_err, out

    # ----------------------------------------------------------------
    #  Training loss
    # ----------------------------------------------------------------

    def compute_loss(self, x, x_mark=None):
        """
        Total = stable_recon_MSE + λ1 * freq_loss + λ2 * sparse_loss + λ3 * smooth_loss

        Note: recon loss is computed on stable component only (trend excluded).
        """
        # Branch 1: reconstruction on stable component
        stable_recon, trend, sparse_loss = self.forward(x)
        stable = x - trend
        # Smoothness is applied to the normalized low-pass kernel, not to trend outputs.
        smooth_loss = self.trend_extractor.smoothness_loss()

        recon_mse = torch.mean((stable - stable_recon) ** 2, dim=(1, 2))  # (B,)
        recon_loss = recon_mse.mean()

        # Branch 2: frequency masked prediction (on original x)
        freq_err, _ = self._freq_forward(x, mask=None)
        freq_loss = freq_err.mean()

        # Total
        total = (recon_loss
                 + self.freq_loss_weight * freq_loss
                 + self.sparse_weight * sparse_loss
                 + self.smooth_weight * smooth_loss)

        return {
            'loss': total,
            'recon_loss': recon_loss.item(),
            'freq_loss': freq_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'mse': recon_mse.mean().item(),
        }

    # ----------------------------------------------------------------
    #  Score fusion
    # ----------------------------------------------------------------

    @staticmethod
    def _as_flat_score(score):
        """Convert an array-like branch score to a one-dimensional float tensor."""
        if not torch.is_tensor(score):
            score = torch.as_tensor(score)
        if not torch.is_floating_point(score):
            score = score.float()
        return score.reshape(-1)

    @classmethod
    def _split_branch_scores(cls, recon_score, freq_score=None):
        """Accept either two score arrays or one branch-score dictionary."""
        if isinstance(recon_score, dict):
            if freq_score is not None:
                raise ValueError(
                    "Pass either a branch-score dictionary or two score arrays, not both."
                )
            freq_score = recon_score["freq_score"]
            recon_score = recon_score["recon_score"]
        if freq_score is None:
            raise ValueError("Both reconstruction and frequency scores are required.")

        recon_score = cls._as_flat_score(recon_score)
        freq_score = cls._as_flat_score(freq_score)
        if recon_score.numel() != freq_score.numel():
            raise ValueError(
                "Reconstruction and frequency score arrays must have the same length."
            )
        return recon_score, freq_score

    @staticmethod
    def _empirical_cdf(score, sorted_reference):
        """Evaluate F_val(score) = count(reference <= score) / len(reference)."""
        if sorted_reference.numel() == 0:
            raise RuntimeError(
                "Validation ECDF is not fitted. Call fit_validation_ecdf() first."
            )
        reference = sorted_reference.to(device=score.device, dtype=score.dtype)
        ranks = torch.searchsorted(reference, score.contiguous(), right=True)
        return ranks.to(dtype=score.dtype) / reference.numel()

    @property
    def validation_ecdf_is_fitted(self):
        return self._val_recon_sorted.numel() > 0 and self._val_freq_sorted.numel() > 0

    def clear_validation_ecdf(self):
        """Remove validation calibration when switching datasets/splits."""
        self._val_recon_sorted = self._val_recon_sorted.new_empty(0)
        self._val_freq_sorted = self._val_freq_sorted.new_empty(0)

    def fit_validation_ecdf(self, recon_score, freq_score=None):
        """Fit the two fixed validation-set ECDF mappings from Eq. (9).

        Args:
            recon_score: validation reconstruction scores, or a dictionary with
                ``recon_score`` and ``freq_score`` entries.
            freq_score: validation frequency scores when ``recon_score`` is not
                a dictionary.

        Returns:
            ``self`` for convenient chaining.
        """
        recon_score, freq_score = self._split_branch_scores(recon_score, freq_score)
        if recon_score.numel() == 0:
            raise ValueError("Cannot fit an ECDF from an empty validation set.")
        if not torch.isfinite(recon_score).all() or not torch.isfinite(freq_score).all():
            raise ValueError("Validation scores must contain only finite values.")

        self._val_recon_sorted = torch.sort(recon_score.detach())[0]
        self._val_freq_sorted = torch.sort(freq_score.detach())[0]
        return self

    def rank_fusion(self, recon_score, freq_score=None):
        """Fuse branch scores using the fixed validation ECDFs from Eq. (9)."""
        recon_score, freq_score = self._split_branch_scores(recon_score, freq_score)
        recon_rank = self._empirical_cdf(recon_score, self._val_recon_sorted)
        freq_rank = self._empirical_cdf(freq_score, self._val_freq_sorted)
        return recon_rank + freq_rank

    def fuse_score_batches(self, score_batches):
        """Fuse collected inference batches using previously fitted ECDFs."""
        score_batches = list(score_batches)
        if not score_batches:
            raise ValueError("score_batches must contain at least one batch.")
        recon_score = torch.cat(
            [self._as_flat_score(batch["recon_score"]) for batch in score_batches],
            dim=0,
        )
        freq_score = torch.cat(
            [self._as_flat_score(batch["freq_score"]) for batch in score_batches],
            dim=0,
        )
        return self.rank_fusion(recon_score, freq_score)

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
                total_err_map, self.freq_log_var, alpha=self.freq_spike_alpha
            )
            freq_score = boosted.sum(dim=(1, 2)) / (total_count + 1e-8)

        # Final fusion is intentionally not performed here: Eq. (9) requires
        # validation-set ECDFs, which are dataset-level statistics rather than
        # mini-batch statistics. Use fit_validation_ecdf() once on validation
        # branch scores, then rank_fusion() for validation/test branch scores.
        return {
            'recon_score': recon_score,
            'freq_score': freq_score,
        }

    def compute_fused_anomaly_score(self, x, x_mark=None):
        """Compute branch scores and fuse them with fitted validation ECDFs."""
        scores = self.compute_anomaly_score(x, x_mark)
        return {
            'anomaly_score': self.rank_fusion(scores),
            **scores,
        }
