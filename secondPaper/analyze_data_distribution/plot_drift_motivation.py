"""
plot_drift_motivation.py
========================
动机图：频率 vs 分布距离（Wasserstein + KS）
证明正常流量在频域存在协变量漂移，且低频区漂移更为严重。
"""

import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import tqdm


import sys
import os

# 1. 获取当前文件所在目录的上一级目录（即 secondPaper 目录）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. 将父目录加入系统路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 3. 将相对导入改为绝对导入（去掉 '..'）
from provider.unsupervised_provider import load_data, split_data_unsupervised, create_data_loaders


# ============================================================
# 超参数（按需修改）
# ============================================================
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"  # 数据目录
WINDOW_SIZE = 100  # 窗口大小
STEP_SIZE = 20  # 滑动步长
BATCH_SIZE = 128  # DataLoader batch size
OUTPUT_PNG = "drift_motivation.png"  # 输出文件名
DPI = 300

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
        loader      : DataLoader，返回 (x, x_mark, label)
        normal_only : 若为 True，仅保留 label==0 的样本
        desc        : tqdm 进度条描述
    Returns:
        spectra: np.ndarray, shape (N, F_bins, C)
    """
    all_specs = []
    for x, _, label in tqdm(loader, desc=desc, leave=False):
        # x: [B, W, C], label: [B, 1]
        x_np = x.numpy()  # [B, W, C]
        lab_np = label.squeeze(-1).numpy()  # [B]

        if normal_only:
            mask = lab_np == 0
            if mask.sum() == 0:
                continue
            x_np = x_np[mask]

        # rFFT 沿时间轴（axis=1），结果形状 [B, F_bins, C]
        fft_amp = np.abs(np.fft.rfft(x_np, axis=1))  # [B, F_bins, C]
        fft_log = np.log1p(fft_amp)  # log1p 压缩动态范围
        all_specs.append(fft_log)

    return np.concatenate(all_specs, axis=0)  # [N, F_bins, C]


# ============================================================
# 2. 逐频率位置计算分布距离
# ============================================================

def compute_freq_distances(spec_a: np.ndarray, spec_b: np.ndarray):
    """
    对每个频率 bin，把两组数据在该 bin 上的所有样本×通道值
    拉平成一维数组，分别计算 Wasserstein 距离和 KS 统计量。

    Args:
        spec_a, spec_b: shape (N_a, F, C) 和 (N_b, F, C)
    Returns:
        wass: np.ndarray [F]  Wasserstein 距离
        ks  : np.ndarray [F]  KS 统计量
    """
    F = spec_a.shape[1]
    wass = np.zeros(F)
    ks = np.zeros(F)

    for i in tqdm(range(F), desc="  Computing distances", leave=False):
        # 拉平：(N, C) -> (N*C,)
        a_flat = spec_a[:, i, :].ravel()
        b_flat = spec_b[:, i, :].ravel()
        wass[i] = wasserstein_distance(a_flat, b_flat)
        ks[i], _ = ks_2samp(a_flat, b_flat)

    return wass, ks


# ============================================================
# 3. 画图
# ============================================================

def plot_drift(wass_tt, ks_tt, F_bins: int, save_path: str):
    """
    绘制"频率 vs 分布距离"动机图，含 Wasserstein 和 KS 两个子图。
    低频区用浅红色背景高亮，均值基线用水平虚线标注。

    Args:
        wass_tt : train vs test 的 Wasserstein 曲线 [F]
        ks_tt   : train vs test 的 KS 曲线 [F]
        F_bins  : 频率 bin 总数
        save_path: 输出路径
    """
    freq_bins = np.arange(F_bins)
    norm_freq = freq_bins / (F_bins - 1)  # 归一化频率 [0, 1]
    low_cutoff = int(F_bins * LOW_FREQ_RATIO)  # 低频截止 bin

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        "Frequency-Domain Covariate Shift of Normal Traffic\n"
        "(CICIDS2017: Train vs. Test, Normal Windows Only)",
        fontsize=13, fontweight='bold', y=1.01
    )

    plot_cfg = [
        (axes[0], wass_tt, "Wasserstein Distance", "Wasserstein"),
        (axes[1], ks_tt, "KS Statistic", "KS"),
    ]

    for ax, curve_tt, ylabel, metric in plot_cfg:
        # 低频背景高亮
        ax.axvspan(0, low_cutoff - 0.5, color='#FFCCCC', alpha=0.45,
                   label=f"Low-freq region (f < {LOW_FREQ_RATIO:.0%}·F_max)")

        # 均值基线（以 train vs test 为主线基准）
        mean_tt = curve_tt.mean()
        ax.axhline(mean_tt, color='#CC0000', linestyle=':', linewidth=1.2,
                   label=f"Mean (train–test): {mean_tt:.4f}")

        # 主线：train vs test（红色实线）
        ax.plot(freq_bins, curve_tt, color='#CC0000', linewidth=1.6,
                label="Train vs Test (normal)")

        # 双 x 轴：下方 bin index，上方归一化频率
        ax.set_xlabel("Frequency Bin Index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, F_bins - 1)

        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Normalized Frequency  f / f$_{max}$", fontsize=9)

        # 将图例移至右下角
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(f"{metric}: Distribution Shift across Frequency", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Figure saved → {save_path}")


# ============================================================
# 4. 打印统计数据
# ============================================================

def print_statistics(wass_tt, ks_tt, F_bins: int):
    """
    打印低频 / 高频区的平均分布距离及比值，供论文正文引用。
    """
    low_end = int(F_bins * LOW_FREQ_RATIO)
    high_beg = int(F_bins * (1.0 - HIGH_FREQ_RATIO))

    def stat_block(name, curve):
        low_mean = curve[:low_end].mean()
        high_mean = curve[high_beg:].mean()
        ratio = low_mean / (high_mean + 1e-12)
        print(f"  [{name}]")
        print(f"    Low-freq  (bins 0–{low_end - 1})  mean : {low_mean:.6f}")
        print(f"    High-freq (bins {high_beg}–{F_bins - 1}) mean : {high_mean:.6f}")
        print(f"    Ratio low/high                       : {ratio:.4f}x")

    sep = "=" * 58
    print(f"\n{sep}")
    print("  Distribution Shift Statistics  (Train vs Test, Normal)")
    print(sep)
    stat_block("Wasserstein", wass_tt)
    print()
    stat_block("KS Statistic", ks_tt)
    print(sep)


# ============================================================
# 主流程
# ============================================================

def main():
    # ---------- 加载数据 ----------
    print("[1/3] Loading data ...")
    X, y, _ = load_data(DATA_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print(f"      X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(X, y)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    print(f"      Train normal: {split_info['train_normal']}  |  "
          f"Test normal: {split_info['test_normal']}")

    # ---------- 提取频谱 ----------
    print("\n[2/3] Extracting spectra ...")
    # train_loader 全部是正常样本，无需过滤
    spec_train = extract_spectra_from_loader(train_loader, normal_only=False,
                                             desc="  Train")
    spec_test = extract_spectra_from_loader(test_loader, normal_only=True,
                                            desc="  Test ")
    F_bins = spec_train.shape[1]
    print(f"      spec_train={spec_train.shape}, spec_test={spec_test.shape}")
    print(f"      Frequency bins F = {F_bins}")

    # ---------- 计算分布距离 ----------
    print("\n[3/3] Computing distribution distances ...")
    print("  Group A: Train vs Test")
    wass_tt, ks_tt = compute_freq_distances(spec_train, spec_test)

    # ---------- 画图 & 输出统计 ----------
    print("\n[4/4] Plotting ...")
    plot_drift(wass_tt, ks_tt, F_bins, OUTPUT_PNG)
    print_statistics(wass_tt, ks_tt, F_bins)


if __name__ == "__main__":
    main()