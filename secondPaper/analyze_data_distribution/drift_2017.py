"""
plot_drift_motivation_cicids2017_only.py
========================================
动机图：仅绘制 CICIDS2017 的频域分布漂移结果。
只比较 Train normal windows vs Test normal windows。
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import tqdm

# 1. 获取当前文件所在目录的上一级目录（即 secondPaper 目录）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. 将父目录加入系统路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 3. 使用项目中的数据加载工具
from provider.unsupervised_provider import load_data, split_data_unsupervised, create_data_loaders


# ============================================================
# 超参数（按需修改）
# ============================================================
DATASET_NAME = "CICIDS2017"
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"

WINDOW_SIZE = 100
STEP_SIZE = 20
BATCH_SIZE = 128

OUTPUT_PNG = "drift_motivation_CICIDS2017_TNSM.png"
DPI = 600

# 低频 / 高频区间定义（占总频率 bin 数的比例）
LOW_FREQ_RATIO = 0.2
HIGH_FREQ_RATIO = 0.2


# ============================================================
# 1. 提取正常样本的频谱
# ============================================================
def extract_spectra_from_loader(loader, normal_only: bool, desc: str) -> np.ndarray:
    """
    遍历 DataLoader，对每个窗口每个通道做 rFFT，
    取幅度谱后做 log1p 变换。

    Args:
        loader: PyTorch DataLoader
        normal_only: 是否只保留 label == 0 的正常窗口
        desc: tqdm 进度条描述

    Returns:
        np.ndarray, shape = [N, F, C]
    """
    all_specs = []

    for x, _, label in tqdm(loader, desc=desc, leave=False):
        # x: [B, W, C], label: [B, 1]
        x_np = x.numpy()
        lab_np = label.squeeze(-1).numpy()

        if normal_only:
            mask = lab_np == 0
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
# 2. 逐频率位置计算分布距离
# ============================================================
def compute_freq_distances(spec_train: np.ndarray, spec_test: np.ndarray):
    """
    对每个频率 bin，把 Train/Test 在该 bin 上的所有样本×通道值
    拉平成一维数组，分别计算 Wasserstein 距离和 KS 统计量。

    Args:
        spec_train: [N_train, F, C]
        spec_test:  [N_test, F, C]

    Returns:
        wass: [F]
        ks:   [F]
    """
    if spec_train.shape[1] != spec_test.shape[1]:
        raise ValueError(
            f"Frequency bins mismatch: train F={spec_train.shape[1]}, "
            f"test F={spec_test.shape[1]}"
        )

    F = spec_train.shape[1]
    wass = np.zeros(F)
    ks = np.zeros(F)

    for i in tqdm(range(F), desc="Computing distances", leave=False):
        train_flat = spec_train[:, i, :].ravel()
        test_flat = spec_test[:, i, :].ravel()

        wass[i] = wasserstein_distance(train_flat, test_flat)
        ks[i], _ = ks_2samp(train_flat, test_flat)

    return wass, ks


# ============================================================
# 3. 加载 CICIDS2017 -> 提取频谱 -> 计算距离
# ============================================================
def compute_cicids2017_result():
    print(f"\n========== {DATASET_NAME} ==========")

    print("[1/3] Loading data ...")
    X, y, _ = load_data(DATA_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print(f"      X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(X, y)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )

    print(
        f"      Train normal: {split_info['train_normal']}  |  "
        f"Test normal: {split_info['test_normal']}"
    )

    print("\n[2/3] Extracting spectra ...")
    # train_loader 全部是正常样本，无需过滤
    spec_train = extract_spectra_from_loader(
        train_loader,
        normal_only=False,
        desc=f"{DATASET_NAME} Train normal"
    )

    # test_loader 保留全部窗口，所以这里只取正常窗口
    spec_test = extract_spectra_from_loader(
        test_loader,
        normal_only=True,
        desc=f"{DATASET_NAME} Test normal"
    )

    F_bins = spec_train.shape[1]
    print(f"      spec_train={spec_train.shape}, spec_test={spec_test.shape}")
    print(f"      Frequency bins F = {F_bins}")

    print("\n[3/3] Computing distribution distances ...")
    wass, ks = compute_freq_distances(spec_train, spec_test)

    return {
        "name": DATASET_NAME,
        "F_bins": F_bins,
        "wass": wass,
        "ks": ks,
    }


# ============================================================
# 4. 绘图：只画 CICIDS2017
# ============================================================
def set_tnsm_plot_style():
    """设置接近 IEEE TNSM 的 Matplotlib 绘图风格。"""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["savefig.dpi"] = DPI


def plot_cicids2017_drift(result: dict, save_path: str):
    """
    在同一张图的两个子图中绘制 CICIDS2017 的 Wasserstein / KS 曲线。
    """
    set_tnsm_plot_style()

    f_bins = result["F_bins"]
    freq_bins = np.arange(f_bins)
    norm_freq = freq_bins / (f_bins - 1)
    low_cutoff_norm = LOW_FREQ_RATIO

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    color = "steelblue"

    plot_cfg = [
        (axes[0], "wass", "Wasserstein Distance", "(a) Wasserstein Distance"),
        (axes[1], "ks", "KS Statistic", "(b) KS Statistic"),
    ]

    for ax, key, ylabel, title in plot_cfg:
        curve = result[key]
        mean_val = curve.mean()

        # 低频区域高亮
        ax.axvspan(
            0,
            low_cutoff_norm,
            color="lightgray",
            alpha=0.35,
            linewidth=0
        )
        ax.text(
            low_cutoff_norm / 2,
            0.96,
            "Low-freq",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )

        ax.plot(
            norm_freq,
            curve,
            linewidth=1.3,
            color=color,
            label=f"{DATASET_NAME} (mean={mean_val:.4f})",
        )

        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("Normalized Frequency ($f/f_{max}$)", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", frameon=True, shadow=False, fontsize=7)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout(pad=1.0, w_pad=2.0)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1, format="png")
    plt.close()

    print(f"\n[✓] Figure saved → {save_path}")


# ============================================================
# 5. 打印统计数据
# ============================================================
def print_statistics(result: dict):
    def stat_block(metric_name: str, curve: np.ndarray, f_bins: int):
        low_end = int(f_bins * LOW_FREQ_RATIO)
        high_beg = int(f_bins * (1.0 - HIGH_FREQ_RATIO))

        low_mean = curve[:low_end].mean()
        high_mean = curve[high_beg:].mean()
        ratio = low_mean / (high_mean + 1e-12)

        print(f"  [{DATASET_NAME} | {metric_name}]")
        print(f"    Overall mean                          : {curve.mean():.6f}")
        print(f"    Low-freq  (bins 0–{low_end - 1}) mean : {low_mean:.6f}")
        print(f"    High-freq (bins {high_beg}–{f_bins - 1}) mean : {high_mean:.6f}")
        print(f"    Ratio low/high                        : {ratio:.4f}x")

    sep = "=" * 72
    print(f"\n{sep}")
    print("  Distribution Shift Statistics  (Train Normal vs Test Normal)")
    print(sep)

    stat_block("Wasserstein", result["wass"], result["F_bins"])
    print()
    stat_block("KS Statistic", result["ks"], result["F_bins"])

    print(sep)


# ============================================================
# 主流程
# ============================================================
def main():
    result = compute_cicids2017_result()

    print("\n[4/4] Plotting ...")
    plot_cicids2017_drift(result, OUTPUT_PNG)

    print_statistics(result)


if __name__ == "__main__":
    main()
