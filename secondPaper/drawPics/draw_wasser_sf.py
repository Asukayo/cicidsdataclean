"""
plot_precision_vs_wasserstein_cicids2017.py
===========================================
绘制 CICIDS2017 上 learned precision weight exp(-s_f) 与
Train normal vs Test normal 的 Wasserstein drift 之间的关系。

图含义：
    x-axis:  每个 frequency bin 的 Wasserstein Distance
    y-axis:  模型学习到的误差惩罚权重 exp(-s_f)
    预期现象：漂移越大的频段，exp(-s_f) 越低，即训练损失对该频段误差的惩罚更小。
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from tqdm import tqdm

# ============================================================
# 0. 项目路径设置
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from provider.unsupervised_provider import (
    load_data,
    split_data_unsupervised,
    create_data_loaders,
)


# ============================================================
# 1. 参数设置
# ============================================================

DATASET_NAME = "CICIDS2017"
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"

# 由训练脚本导出的 learned freq_log_var 参数
# 原始形状通常为 [1, F, 1]
NPY_FILE_PATH = (
    r"/home/ubuntu/wyh/cicdis/secondPaper/script/results/"
    r"TCNAE_AutoFreq_learned_freq_log_var.npy"
)

WINDOW_SIZE = 100
STEP_SIZE = 20
BATCH_SIZE = 128

LOW_FREQ_RATIO = 0.2
OUTPUT_PNG = "precision_weight_vs_wasserstein_CICIDS2017.png"
DPI = 600


# ============================================================
# 2. 提取正常样本频谱
# ============================================================

def extract_spectra_from_loader(loader, normal_only: bool, desc: str) -> np.ndarray:
    """
    遍历 DataLoader，对每个窗口每个通道做 rFFT，
    取幅度谱后做 log1p 变换。

    Returns:
        np.ndarray, shape = [N, F, C]
    """
    all_specs = []

    for x, _, label in tqdm(loader, desc=desc, leave=False):
        # x: [B, W, C], label: [B, 1]
        x_np = x.numpy()
        label_np = label.squeeze(-1).numpy()

        if normal_only:
            mask = label_np == 0
            if mask.sum() == 0:
                continue
            x_np = x_np[mask]

        fft_amp = np.abs(np.fft.rfft(x_np, axis=1))
        fft_log = np.log1p(fft_amp)
        all_specs.append(fft_log)

    if not all_specs:
        raise ValueError("No spectra extracted. Please check whether normal samples exist.")

    return np.concatenate(all_specs, axis=0)


# ============================================================
# 3. 计算每个 frequency bin 的 Wasserstein drift
# ============================================================

def compute_wasserstein_per_bin(spec_train: np.ndarray, spec_test: np.ndarray) -> np.ndarray:
    """
    对每个频率 bin，把 Train/Test 在该 bin 上的所有样本×通道值拉平，
    计算 Wasserstein Distance。

    Args:
        spec_train: [N_train, F, C]
        spec_test:  [N_test, F, C]

    Returns:
        wass: [F]
    """
    if spec_train.shape[1] != spec_test.shape[1]:
        raise ValueError(
            f"Frequency bins mismatch: train F={spec_train.shape[1]}, "
            f"test F={spec_test.shape[1]}"
        )

    F_bins = spec_train.shape[1]
    wass = np.zeros(F_bins, dtype=np.float64)

    for f in tqdm(range(F_bins), desc="Computing Wasserstein", leave=False):
        train_flat = spec_train[:, f, :].ravel()
        test_flat = spec_test[:, f, :].ravel()
        wass[f] = wasserstein_distance(train_flat, test_flat)

    return wass


# ============================================================
# 4. 加载数据并计算 drift
# ============================================================

def compute_cicids2017_wasserstein() -> np.ndarray:
    print(f"\n========== {DATASET_NAME}: Wasserstein Drift ==========")

    print("[1/3] Loading data ...")
    X, y, _ = load_data(DATA_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print(f"      X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(X, y)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=BATCH_SIZE,
    )

    print(
        f"      Train normal: {split_info['train_normal']}  |  "
        f"Test normal: {split_info['test_normal']}"
    )

    print("\n[2/3] Extracting spectra ...")
    # train_loader 已经只包含正常训练窗口
    spec_train = extract_spectra_from_loader(
        train_loader,
        normal_only=False,
        desc="Train normal spectra",
    )

    # test_loader 包含正常和异常窗口，这里只保留 normal windows
    spec_test = extract_spectra_from_loader(
        test_loader,
        normal_only=True,
        desc="Test normal spectra",
    )

    print(f"      spec_train={spec_train.shape}, spec_test={spec_test.shape}")

    print("\n[3/3] Computing Wasserstein per frequency bin ...")
    wass = compute_wasserstein_per_bin(spec_train, spec_test)

    return wass


# ============================================================
# 5. 加载 learned precision weight exp(-s_f)
# ============================================================

def load_precision_weight() -> np.ndarray:
    if not os.path.exists(NPY_FILE_PATH):
        raise FileNotFoundError(f"Cannot find learned freq_log_var file: {NPY_FILE_PATH}")

    log_var = np.load(NPY_FILE_PATH).squeeze()
    if log_var.ndim != 1:
        raise ValueError(f"Expected 1D log_var after squeeze, got shape={log_var.shape}")

    precision_weight = np.exp(-log_var)
    return precision_weight


# ============================================================
# 6. 绘制 exp(-s_f) vs Wasserstein
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.major.width"] = 0.9
    plt.rcParams["ytick.major.width"] = 0.9
    plt.rcParams["savefig.dpi"] = DPI


def plot_precision_vs_wasserstein(wass: np.ndarray, precision_weight: np.ndarray, save_path: str):
    """
    绘制每个 frequency bin 上：
        x = Wasserstein Distance
        y = exp(-s_f)
    """
    set_plot_style()

    if len(wass) != len(precision_weight):
        raise ValueError(
            f"Length mismatch: Wasserstein F={len(wass)}, "
            f"precision weight F={len(precision_weight)}. "
            f"Please check WINDOW_SIZE and learned_freq_log_var."
        )

    F_bins = len(wass)
    freq_bins = np.arange(F_bins)
    norm_freq = freq_bins / (F_bins - 1)
    low_end = int(F_bins * LOW_FREQ_RATIO)

    pearson_r, pearson_p = pearsonr(wass, precision_weight)
    spearman_r, spearman_p = spearmanr(wass, precision_weight)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # 非低频点
    normal_idx = np.arange(F_bins) >= low_end
    low_idx = np.arange(F_bins) < low_end

    sc = ax.scatter(
        wass[normal_idx],
        precision_weight[normal_idx],
        c=norm_freq[normal_idx],
        cmap="viridis",
        s=38,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label="Other frequency bins",
    )

    ax.scatter(
        wass[low_idx],
        precision_weight[low_idx],
        c=norm_freq[low_idx],
        cmap="viridis",
        s=50,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.7,
        marker="o",
        label=f"Low-freq bins (<{LOW_FREQ_RATIO:.0%})",
    )

    # 线性趋势线，仅作为可视化辅助
    coef = np.polyfit(wass, precision_weight, deg=1)
    x_line = np.linspace(wass.min(), wass.max(), 200)
    y_line = coef[0] * x_line + coef[1]
    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label="Linear trend",
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Normalized Frequency ($f/f_{max}$)", fontsize=9)

    ax.set_title(
        "Learned Precision Weight vs. Frequency-domain Drift",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Wasserstein Distance", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Learned Loss Weight $\exp(-s_f)$", fontsize=10, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    ax.legend(loc="best", frameon=True, fontsize=7.5)

    corr_text = (
        rf"Pearson $r$ = {pearson_r:.3f}" + "\n" +
        rf"Spearman $\rho$ = {spearman_r:.3f}"
    )
    ax.text(
        0.03,
        0.04,
        corr_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close()

    print(f"\n[✓] Figure saved → {save_path}")
    print(f"    Pearson r  = {pearson_r:.6f}, p = {pearson_p:.3e}")
    print(f"    Spearman ρ = {spearman_r:.6f}, p = {spearman_p:.3e}")
    print(f"    Low-freq mean Wasserstein       : {wass[:low_end].mean():.6f}")
    print(f"    Low-freq mean exp(-s_f)         : {precision_weight[:low_end].mean():.6f}")
    print(f"    Other-freq mean Wasserstein     : {wass[low_end:].mean():.6f}")
    print(f"    Other-freq mean exp(-s_f)       : {precision_weight[low_end:].mean():.6f}")


# ============================================================
# 主流程
# ============================================================

def main():
    wass = compute_cicids2017_wasserstein()
    precision_weight = load_precision_weight()

    print("\n[4/4] Plotting exp(-s_f) vs Wasserstein ...")
    plot_precision_vs_wasserstein(wass, precision_weight, OUTPUT_PNG)


if __name__ == "__main__":
    main()
