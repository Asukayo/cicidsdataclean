"""
FreqDAR — Anomaly Score Distribution comparison (A1 w/o Freq Branch vs FreqDAR).

To use real data, replace the four arrays in the `=== DATA ===` block:
    scores_a1, labels_a1, scores_fd, labels_fd
and set the two validation thresholds thr_a1, thr_fd.
Everything below the DATA block is data-agnostic.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ----------------------------- Style -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "savefig.dpi": 300,
})

NORMAL_COLOR = "#4C72B0"   # blue
ANOM_COLOR   = "#C44E52"   # red
THRESH_COLOR = "#000000"   # black
HIST_ALPHA   = 0.4
N_BINS       = 50

# =============================== DATA ===============================
# --- SIMULATED DATA (replace with your real numpy arrays) ----------
rng = np.random.default_rng(42)

def _mix(rng, comps):
    """comps = list of (mean, std, n). Returns concatenated, clipped to [0,1]."""
    parts = [rng.normal(m, s, n) for (m, s, n) in comps]
    return np.clip(np.concatenate(parts), 0.0, 1.0)

# A1 (w/o Freq Branch): drift inflates a chunk of NORMAL scores -> fat right tail;
# anomalous has a left tail -> heavy overlap; val threshold lands inside overlap.
n_norm, n_anom = 6000, 2000
scores_a1 = np.concatenate([
    _mix(rng, [(0.25, 0.06, int(n_norm * 0.80)),
               (0.46, 0.07, int(n_norm * 0.20))]),   # drift-inflated normals
    _mix(rng, [(0.55, 0.07, int(n_anom * 0.75)),
               (0.42, 0.06, int(n_anom * 0.25))]),   # anomalous w/ left tail
])
labels_a1 = np.concatenate([np.zeros(n_norm, int), np.ones(n_anom, int)])
thr_a1 = 0.47   # migrated validation threshold -> falls inside the overlap

# FreqDAR (full): drift adversarial mechanism keeps NORMAL scores low -> short right
# tail; anomalous cleanly higher -> a gap; threshold lands in the gap.
scores_fd = np.concatenate([
    _mix(rng, [(0.22, 0.05, int(n_norm * 0.95)),
               (0.33, 0.04, int(n_norm * 0.05))]),   # short normal tail
    _mix(rng, [(0.62, 0.07, int(n_anom * 0.90)),
               (0.50, 0.05, int(n_anom * 0.10))]),   # thin left tail
])
labels_fd = np.concatenate([np.zeros(n_norm, int), np.ones(n_anom, int)])
thr_fd = 0.43   # migrated validation threshold -> falls in the gap
# ====================================================================

# Shared x-range over BOTH variants (global min/max with small padding)
_all = np.concatenate([scores_a1, scores_fd])
_gmin, _gmax = float(_all.min()), float(_all.max())
_pad = 0.02 * (_gmax - _gmin)
XLIM = (_gmin - _pad, _gmax + _pad)


def plot_distribution(ax, scores, labels, threshold, variant_name, xlim, bins=N_BINS):
    normal = scores[labels == 0]
    anom   = scores[labels == 1]
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)

    # 1-2. Density-normalized histograms
    ax.hist(normal, bins=bin_edges, density=True, color=NORMAL_COLOR,
            alpha=HIST_ALPHA, edgecolor="none", zorder=1)
    ax.hist(anom, bins=bin_edges, density=True, color=ANOM_COLOR,
            alpha=HIST_ALPHA, edgecolor="none", zorder=1)

    # 3. KDE smooth curves (same color, solid, no fill)
    xs = np.linspace(xlim[0], xlim[1], 400)
    for data, color in [(normal, NORMAL_COLOR), (anom, ANOM_COLOR)]:
        kde = gaussian_kde(data)
        ax.plot(xs, kde(xs), color=color, lw=2.0, zorder=3)

    # 4. Validation threshold (migrated from validation set)
    ax.axvline(threshold, color=THRESH_COLOR, ls="--", lw=1.5, zorder=4)

    # 5/6. Cosmetics
    ax.set_xlim(xlim)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#999999", alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)

    # Variant name, top-left corner
    ax.text(0.035, 0.95, variant_name, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="#222222")


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)

plot_distribution(ax_l, scores_a1, labels_a1, thr_a1, "A1 (w/o Freq Branch)", XLIM)
plot_distribution(ax_r, scores_fd, labels_fd, thr_fd, "FreqDAR (full)",        XLIM)

fig.suptitle("Anomaly Score Distribution on CICIDS2017 Test Set",
             fontsize=14, fontweight="bold", y=0.97)

legend_handles = [
    Patch(facecolor=NORMAL_COLOR, alpha=HIST_ALPHA, label="Normal"),
    Patch(facecolor=ANOM_COLOR,   alpha=HIST_ALPHA, label="Anomalous"),
    Line2D([0], [0], color=THRESH_COLOR, ls="--", lw=1.5, label="Val Threshold"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=11)

fig.subplots_adjust(left=0.07, right=0.975, top=0.86, bottom=0.18, wspace=0.16)

fig.savefig("freqdar_score_dist.png", dpi=300, facecolor="white")
fig.savefig("freqdar_score_dist.pdf", facecolor="white")
print("saved. XLIM =", XLIM)