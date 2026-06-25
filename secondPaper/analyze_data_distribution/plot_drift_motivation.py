"""
plot_drift_motivation_compare.py
================================
动机图：同时绘制 CICIDS2017 / CICIDS2018 的频域分布漂移结果。
每个数据集均只比较 Train vs Test 的 normal windows。
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import tqdm

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
DATASETS = {
    "CICIDS2017": r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows",
    "CICIDS2018": r"/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows",
}

WINDOW_SIZE = 100
STEP_SIZE = 20
BATCH_SIZE = 128
OUTPUT_PNG = "drift_motivation_compare_2017_2018_TNSM.png"
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
def compute_freq_distances(spec_a: np.ndarray, spec_b: np.ndarray):
    """
    对每个频率 bin，把两组数据在该 bin 上的所有样本×通道值
    拉平成一维数组，分别计算 Wasserstein 距离和 KS 统计量。
    """
    F = spec_a.shape[1]
    wass = np.zeros(F)
    ks = np.zeros(F)

    for i in tqdm(range(F), desc="  Computing distances", leave=False):
        a_flat = spec_a[:, i, :].ravel()
        b_flat = spec_b[:, i, :].ravel()
        wass[i] = wasserstein_distance(a_flat, b_flat)
        ks[i], _ = ks_2samp(a_flat, b_flat)

    return wass, ks


# ============================================================
# 3. 单个数据集：加载数据 -> 提取频谱 -> 计算距离
# ============================================================
def compute_dataset_result(dataset_name: str, data_dir: str):
    print(f"\n========== {dataset_name} ==========")
    print("[1/3] Loading data ...")
    X, y, _ = load_data(data_dir, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print(f"      X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(X, y)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    print(f"      Train normal: {split_info['train_normal']}  |  "
          f"Test normal: {split_info['test_normal']}")

    print("\n[2/3] Extracting spectra ...")
    # train_loader 全部是正常样本，无需过滤；test_loader 只保留正常样本
    spec_train = extract_spectra_from_loader(
        train_loader, normal_only=False, desc=f"  {dataset_name} Train"
    )
    spec_test = extract_spectra_from_loader(
        test_loader, normal_only=True, desc=f"  {dataset_name} Test "
    )
    F_bins = spec_train.shape[1]
    print(f"      spec_train={spec_train.shape}, spec_test={spec_test.shape}")
    print(f"      Frequency bins F = {F_bins}")

    print("\n[3/3] Computing distribution distances ...")
    wass_tt, ks_tt = compute_freq_distances(spec_train, spec_test)

    return {
        "name": dataset_name,
        "F_bins": F_bins,
        "wass": wass_tt,
        "ks": ks_tt,
    }


# ============================================================
# 4. 画图：一张图中同时包含两个数据集
# ============================================================
def set_tnsm_plot_style():
    """设置接近 IEEE TNSM 的 Matplotlib 绘图风格。"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['savefig.dpi'] = DPI


def plot_drift_compare(results, save_path: str):
    """
    在同一张图的两个子图中绘制两个数据集的 Wasserstein / KS 曲线。
    风格对齐 DrawE&SinTNSM.py：Times 字体、小尺寸、粗体标题/坐标轴、细线、浅网格、高 DPI。
    """
    if not results:
        raise ValueError("results is empty")

    set_tnsm_plot_style()

    # 保守处理：如果两个数据集 F_bins 不一致，只绘制共同长度，避免横轴错位。
    min_f_bins = min(r["F_bins"] for r in results)
    freq_bins = np.arange(min_f_bins)
    norm_freq = freq_bins / (min_f_bins - 1)
    low_cutoff_norm = LOW_FREQ_RATIO

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # 颜色参考 TNSM 示例脚本：steelblue + orangered
    color_map = {
        "CICIDS2017": "steelblue",
        "CICIDS2018": "orangered",
    }
    fallback_colors = ["steelblue", "orangered", "seagreen", "purple"]

    plot_cfg = [
        (axes[0], "wass", "Wasserstein Distance", "(a) Wasserstein Distance"),
        (axes[1], "ks", "KS Statistic", "(b) KS Statistic"),
    ]

    for ax, key, ylabel, title in plot_cfg:
        # 低频区域高亮：保留论文含义，但弱化视觉权重，避免干扰主曲线。
        ax.axvspan(0, low_cutoff_norm, color='lightgray', alpha=0.35, linewidth=0)
        ax.text(
            low_cutoff_norm / 2,
            0.96,
            "Low-freq",
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=7,
        )

        for idx, result in enumerate(results):
            name = result["name"]
            curve = result[key][:min_f_bins]
            mean_val = curve.mean()
            color = color_map.get(name, fallback_colors[idx % len(fallback_colors)])

            ax.plot(
                norm_freq,
                curve,
                linewidth=1.2,
                color=color,
                label=f"{name} (mean={mean_val:.4f})",
            )

        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
        ax.set_xlabel("Normalized Frequency ($f/f_{max}$)", fontsize=9, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', frameon=True, shadow=False, fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout(pad=1.0, w_pad=2.0)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()
    print(f"\n[✓] Figure saved → {save_path}")


# ============================================================
# 5. 打印统计数据
# ============================================================
def print_statistics_compare(results):
    def stat_block(dataset_name, metric_name, curve, f_bins):
        low_end = int(f_bins * LOW_FREQ_RATIO)
        high_beg = int(f_bins * (1.0 - HIGH_FREQ_RATIO))
        low_mean = curve[:low_end].mean()
        high_mean = curve[high_beg:].mean()
        ratio = low_mean / (high_mean + 1e-12)
        print(f"  [{dataset_name} | {metric_name}]")
        print(f"    Overall mean                         : {curve.mean():.6f}")
        print(f"    Low-freq  (bins 0–{low_end - 1})  mean : {low_mean:.6f}")
        print(f"    High-freq (bins {high_beg}–{f_bins - 1}) mean : {high_mean:.6f}")
        print(f"    Ratio low/high                       : {ratio:.4f}x")

    sep = "=" * 72
    print(f"\n{sep}")
    print("  Distribution Shift Statistics  (Train vs Test, Normal)")
    print(sep)
    for result in results:
        stat_block(result["name"], "Wasserstein", result["wass"], result["F_bins"])
        print()
        stat_block(result["name"], "KS Statistic", result["ks"], result["F_bins"])
        print("-" * 72)
    print(sep)


# ============================================================
# 主流程
# ============================================================
def main():
    results = []
    for dataset_name, data_dir in DATASETS.items():
        result = compute_dataset_result(dataset_name, data_dir)
        results.append(result)

    print("\n[4/4] Plotting ...")
    plot_drift_compare(results, OUTPUT_PNG)
    print_statistics_compare(results)


if __name__ == "__main__":
    main()
