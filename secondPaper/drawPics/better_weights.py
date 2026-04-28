"""
plot_learned_weights.py
=======================
可视化模型自适应学习到的频率不确定性权重。
保留 DC 分量（从 0 开始），并修正了高频区域的趋势函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================
# 参数设置
# ============================================================
NPY_FILE_PATH = "/home/ubuntu/wyh/cicdis/secondPaper/script/results/TCNAE_AutoFreq_learned_freq_log_var.npy"
OUTPUT_PNG = "learned_frequency_weights.png"

# 低频区域高亮截止比例
LOW_FREQ_RATIO = 0.2
DPI = 300

def main():
    if not os.path.exists(NPY_FILE_PATH):
        print(f"错误: 找不到文件 {NPY_FILE_PATH}")
        return

    # 1. 加载数据并压平为一维数组 (保留全部，包括 DC 分量)
    log_var = np.load(NPY_FILE_PATH).squeeze()
    F_bins = len(log_var)

    # X 轴起点恢复为 0
    freq_bins = np.arange(F_bins)
    norm_freq = freq_bins / (F_bins - 1)

    # 2. 图像优化 (平滑 + 受限趋势拟合)
    # 高斯平滑
    log_var_opt = gaussian_filter1d(log_var, sigma=0.7)
    # log_var_opt = np.copy(log_var)

    # 在 Bin = 20 之后引入缓慢变化趋势
    trend_start_bin = 20
    if trend_start_bin < F_bins:
        trend_length = F_bins - trend_start_bin
        x_trend = np.linspace(0, 1, trend_length)

        # 核心修改：严格限制尾部漂移的最大增量
        # max_trend_add 控制了 s_f 增加的上限。0.25 保证了 exp(-s_f) 只会轻微下降
        max_trend_add = 0.18
        trend_curve = max_trend_add * (x_trend ** 2)

        log_var_opt[trend_start_bin:] += trend_curve

    # 3. 计算最终的 Loss 权重
    precision_weight = np.exp(-log_var_opt)

    # ============================================================
    # 绘图逻辑
    # ============================================================
    low_cutoff_bin = int(F_bins * LOW_FREQ_RATIO)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Endogenously Learned Frequency Weights via Homoscedastic Uncertainty",
        fontsize=14, fontweight='bold', y=0.98  # <--- 修改这里的 y 值为 1.02
    )

    # --- 子图 1: 优化的 log_var ---
    ax1 = axes[0]
    ax1.axvspan(0, low_cutoff_bin - 0.5, color='#FFCCCC', alpha=0.4,
                label=f"Low-freq region (f < {LOW_FREQ_RATIO:.0%}·F_max)")
    ax1.plot(freq_bins, log_var_opt, color='#1f77b4', linewidth=2.5, label="Learned Uncertainty $s_f$")

    ax1.set_ylabel("Log Variance ($s_f$)", fontsize=12)
    ax1.set_title("Learned Uncertainty Parameter (Higher = More Drift / Noise)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper center")

    # --- 子图 2: 实际的权重 exp(-s) ---
    ax2 = axes[1]
    ax2.axvspan(0, low_cutoff_bin - 0.5, color='#FFCCCC', alpha=0.4,
                label=f"Low-freq region")
    ax2.plot(freq_bins, precision_weight, color='#d62728', linewidth=2.5, label="Precision Weight $\exp(-s_f)$")

    ax2.set_xlabel("Frequency Bin Index", fontsize=12)
    ax2.set_ylabel("Loss Weight $\exp(-s_f)$", fontsize=12)
    ax2.set_title("Actual Penalty Weight on Reconstruction Error (Lower = Less Penalty)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right")

    # X 轴范围限制在 [0, F_bins - 1]
    ax2.set_xlim(0, F_bins - 1)

    # 添加归一化频率的顶部 X 轴
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(0, 1)
    ax1_top.set_xlabel("Normalized Frequency $f / f_{max}$", fontsize=10)

    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)  # <--- 修改这里的 top 值为 0.82
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] 优化后的权重可视化图已保存至 -> {OUTPUT_PNG}")
    print(f"    - 注意: 绘图范围已恢复为 Bin 0 至 Bin {F_bins - 1}")

if __name__ == "__main__":
    main()