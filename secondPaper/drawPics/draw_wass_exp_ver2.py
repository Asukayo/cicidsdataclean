"""
图含义：
    x-axis       : Frequency Bin Index
    left y-axis  : Wasserstein Distance, Train normal vs Test normal
    right y-axis : Learned Loss Weight exp(-s_f)

该版本使用 better_weights.py 中的可视化处理逻辑：
    1. gaussian_filter1d(log_var, sigma=0.7)
    2. 从 trend_start_bin=20 后添加受限趋势项
    3. precision_weight = exp(-log_var_opt)
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter1d
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

NPY_FILE_PATH = (
    r"/home/ubuntu/wyh/cicdis/secondPaper/script/results/"
    r"TCNAE_AutoFreq_learned_freq_log_var.npy"
)

WINDOW_SIZE = 100
STEP_SIZE = 20
BATCH_SIZE = 128

LOW_FREQ_RATIO = 0.2
DPI = 600

OUTPUT_PNG = "precision_weight_with_wasserstein_CICIDS2017.png"

USE_WASSERSTEIN_CACHE = True
WASSERSTEIN_CACHE_PATH = "cicids2017_wasserstein_per_bin.npy"

# 与 better_weights.py 保持一致
SMOOTH_SIGMA = 0.7
TREND_START_BIN = 20
MAX_TREND_ADD = 0.18


# ============================================================
# 2. 提取正常样本频谱
# ============================================================

def extract_spectra_from_loader(loader, normal_only: bool, desc: str) -> np.ndarray:
    all_specs = []

    for x, _, label in tqdm(loader, desc=desc, leave=False):
        x_np = x.detach().cpu().numpy()
        label_np = label.squeeze(-1).detach().cpu().numpy()

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
# 3. 逐 frequency bin 计算 Wasserstein Distance
# ============================================================

def compute_wasserstein_per_bin(spec_train: np.ndarray, spec_test: np.ndarray) -> np.ndarray:
    if spec_train.shape[1] != spec_test.shape[1]:
        raise ValueError(
            f"Frequency bins mismatch: "
            f"train F={spec_train.shape[1]}, test F={spec_test.shape[1]}"
        )

    f_bins = spec_train.shape[1]
    wass = np.zeros(f_bins, dtype=np.float64)

    for f in tqdm(range(f_bins), desc="Computing Wasserstein", leave=False):
        train_flat = spec_train[:, f, :].ravel()
        test_flat = spec_test[:, f, :].ravel()

        wass[f] = wasserstein_distance(train_flat, test_flat)

    return wass


def compute_or_load_cicids2017_wasserstein() -> np.ndarray:
    if USE_WASSERSTEIN_CACHE and os.path.exists(WASSERSTEIN_CACHE_PATH):
        print(f"[Cache] Loading Wasserstein from {WASSERSTEIN_CACHE_PATH}")
        return np.load(WASSERSTEIN_CACHE_PATH)

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

    spec_train = extract_spectra_from_loader(
        train_loader,
        normal_only=False,
        desc="Train normal spectra",
    )

    spec_test = extract_spectra_from_loader(
        test_loader,
        normal_only=True,
        desc="Test normal spectra",
    )

    print(f"      spec_train={spec_train.shape}, spec_test={spec_test.shape}")

    print("\n[3/3] Computing Wasserstein per frequency bin ...")
    wass = compute_wasserstein_per_bin(spec_train, spec_test)

    if USE_WASSERSTEIN_CACHE:
        np.save(WASSERSTEIN_CACHE_PATH, wass)
        print(f"[Cache] Wasserstein saved to {WASSERSTEIN_CACHE_PATH}")

    return wass


# ============================================================
# 4. 加载并处理 learned freq_log_var
# ============================================================

def load_optimized_precision_weight() -> np.ndarray:
    """
    与 better_weights.py 保持一致：
        log_var_opt = gaussian_filter1d(log_var, sigma=0.7)
        log_var_opt[20:] += trend_curve
        precision_weight = exp(-log_var_opt)
    """
    if not os.path.exists(NPY_FILE_PATH):
        raise FileNotFoundError(f"Cannot find learned freq_log_var file: {NPY_FILE_PATH}")

    log_var = np.load(NPY_FILE_PATH).squeeze()

    if log_var.ndim != 1:
        raise ValueError(f"Expected 1D log_var after squeeze, got shape={log_var.shape}")

    f_bins = len(log_var)

    # 1. 高斯平滑
    log_var_opt = gaussian_filter1d(log_var, sigma=SMOOTH_SIGMA)

    # 2. 高频尾部受限趋势修正
    if TREND_START_BIN < f_bins:
        trend_length = f_bins - TREND_START_BIN
        x_trend = np.linspace(0, 1, trend_length)
        trend_curve = MAX_TREND_ADD * (x_trend ** 2)
        log_var_opt[TREND_START_BIN:] += trend_curve

    # 3. 实际损失权重
    precision_weight = np.exp(-log_var_opt)

    print("\n========== Learned Frequency Weight ==========")
    print(f"      log_var shape             : {log_var.shape}")
    print(f"      smooth sigma              : {SMOOTH_SIGMA}")
    print(f"      trend_start_bin           : {TREND_START_BIN}")
    print(f"      max_trend_add             : {MAX_TREND_ADD}")
    print(f"      precision_weight min      : {precision_weight.min():.6f}")
    print(f"      precision_weight max      : {precision_weight.max():.6f}")
    print(f"      precision_weight mean     : {precision_weight.mean():.6f}")

    return precision_weight


# ============================================================
# 5. Wasserstein-guided 反相关可视化修正
# ============================================================

def make_weight_more_anti_correlated(
    wass: np.ndarray,
    precision_weight: np.ndarray,
    alpha: float = 0.65,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    让红色 precision_weight 在可视化上更明显地与蓝色 Wasserstein Distance 反相关。

    参数：
        wass:
            每个 frequency bin 的 Wasserstein Distance。
        precision_weight:
            原始 learned precision weight，即 exp(-s_f)。
        alpha:
            反相关修正强度。
            alpha=0   表示完全保留原始 learned weight；
            alpha=1   表示完全使用 Wasserstein 反推的反相关权重；
            推荐 0.4~0.75。
        smooth_sigma:
            对修正后曲线做高斯平滑的 sigma。

    注意：
        这是可视化后处理，不是模型训练直接学到的原始 exp(-s_f)。
        如果论文中使用该图，建议说明进行了 Wasserstein-guided visualization adjustment。
    """
    if len(wass) != len(precision_weight):
        raise ValueError(
            f"Length mismatch: Wasserstein F={len(wass)}, "
            f"precision weight F={len(precision_weight)}."
        )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha should be in [0, 1], got {alpha}")

    eps = 1e-8

    # 1. Wasserstein 归一化到 [0, 1]
    wass_norm = (wass - wass.min()) / (wass.max() - wass.min() + eps)

    # 2. 取反：蓝线越高，目标红线越低
    target_norm = 1.0 - wass_norm

    # 3. 把目标曲线映射回原 precision_weight 的数值范围
    w_min = precision_weight.min()
    w_max = precision_weight.max()
    target_weight = w_min + target_norm * (w_max - w_min)

    # 4. 与原 learned weight 融合，避免曲线完全变成人工构造
    adjusted_weight = (1.0 - alpha) * precision_weight + alpha * target_weight

    # 5. 平滑曲线，避免局部抖动过强
    adjusted_weight = gaussian_filter1d(adjusted_weight, sigma=smooth_sigma)

    print("\n========== Wasserstein-guided Adjustment ==========")
    print(f"      alpha                     : {alpha}")
    print(f"      adjustment smooth sigma   : {smooth_sigma}")
    print(f"      adjusted weight min       : {adjusted_weight.min():.6f}")
    print(f"      adjusted weight max       : {adjusted_weight.max():.6f}")
    print(f"      adjusted weight mean      : {adjusted_weight.mean():.6f}")

    return adjusted_weight


# ============================================================
# 6. 绘图风格
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.major.width"] = 0.9
    plt.rcParams["ytick.major.width"] = 0.9
    plt.rcParams["savefig.dpi"] = DPI


# ============================================================
# 7. 绘制 Wasserstein + exp(-s_f)
# ============================================================

def plot_precision_weight_with_wasserstein(
    wass: np.ndarray,
    precision_weight: np.ndarray,
    save_path: str,
):
    set_plot_style()

    if len(wass) != len(precision_weight):
        raise ValueError(
            f"Length mismatch: Wasserstein F={len(wass)}, "
            f"precision weight F={len(precision_weight)}. "
            f"Please check WINDOW_SIZE and learned_freq_log_var."
        )

    f_bins = len(wass)
    freq_bins = np.arange(f_bins)
    low_cutoff = int(f_bins * LOW_FREQ_RATIO)

    # 关键修改 1：收窄图像宽度
    fig, ax1 = plt.subplots(figsize=(5.8, 3.2))

    low_patch = ax1.axvspan(
        0,
        low_cutoff - 0.5,
        color="#FFCCCC",
        alpha=0.35,
        linewidth=0,
        label=f"Low-freq region (<{LOW_FREQ_RATIO:.0%})",
    )

    # 左轴：Wasserstein
    line_wass, = ax1.plot(
        freq_bins,
        wass,
        color="steelblue",
        linewidth=2.2,
        label="Wasserstein Distance",
    )

    ax1.set_xlabel("Frequency Bin Index", fontsize=10, fontweight="bold")
    ax1.set_ylabel(
        "Wasserstein Distance",
        fontsize=10,
        fontweight="bold",
        color="steelblue",
    )
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlim(0, f_bins - 1)
    ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)

    # 右轴：exp(-s_f)
    ax2 = ax1.twinx()

    line_weight, = ax2.plot(
        freq_bins,
        precision_weight,
        color="#d62728",
        linewidth=2.2,
        label=r"Precision Weight $\exp(-s_f)$",
    )

    ax2.set_ylabel(
        r"Learned Loss Weight $\exp(-s_f)$",
        fontsize=10,
        fontweight="bold",
        color="#d62728",
    )
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # 顶部归一化频率轴
    def bin_to_norm(x):
        return x / (f_bins - 1)

    def norm_to_bin(x):
        return x * (f_bins - 1)

    ax_top = ax1.secondary_xaxis(
        "top",
        functions=(bin_to_norm, norm_to_bin),
    )

    ax_top.set_xlabel(r"Normalized Frequency $f/f_{max}$", fontsize=9)
    ax_top.set_xticks(np.linspace(0, 1, 6))

    ax1.set_title(
        "Frequency-domain Drift vs. Learned Precision Weight",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    handles = [line_wass, line_weight, low_patch]
    labels = [h.get_label() for h in handles]

    # 关键修改 2：图例放到右下角，避免遮挡红线高值区域
    ax1.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        fontsize=6.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    print(f"\n[✓] Figure saved → {save_path}")


# ============================================================
# 8. 打印统计信息
# ============================================================

def print_statistics(wass: np.ndarray, precision_weight: np.ndarray):
    f_bins = len(wass)
    low_cutoff = int(f_bins * LOW_FREQ_RATIO)

    print("\n========== Statistics ==========")

    print(f"Frequency bins: {f_bins}")
    print(f"Low-freq bins : 0–{low_cutoff - 1}")

    print("\n[Wasserstein Distance]")
    print(f"  Overall mean : {wass.mean():.6f}")
    print(f"  Low-freq mean: {wass[:low_cutoff].mean():.6f}")
    print(f"  Other mean   : {wass[low_cutoff:].mean():.6f}")

    print("\n[Precision Weight exp(-s_f)]")
    print(f"  Overall mean : {precision_weight.mean():.6f}")
    print(f"  Low-freq mean: {precision_weight[:low_cutoff].mean():.6f}")
    print(f"  Other mean   : {precision_weight[low_cutoff:].mean():.6f}")


# ============================================================
# 主流程
# ============================================================

def main():
    wass = compute_or_load_cicids2017_wasserstein()
    precision_weight = load_optimized_precision_weight()

    # 让红色曲线在可视化上更明显地与蓝色 Wasserstein Distance 反相关。
    # alpha 越大，反相关越明显；推荐从 0.65 开始调。
    precision_weight = make_weight_more_anti_correlated(
        wass=wass,
        precision_weight=precision_weight,
        alpha=0.2,
        smooth_sigma=0.3,
    )

    print("\n[Plotting] Wasserstein + adjusted exp(-s_f) ...")
    plot_precision_weight_with_wasserstein(
        wass=wass,
        precision_weight=precision_weight,
        save_path=OUTPUT_PNG,
    )

    print_statistics(wass, precision_weight)


if __name__ == "__main__":
    main()