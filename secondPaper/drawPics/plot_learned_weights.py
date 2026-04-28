"""
plot_learned_weights.py
=======================
可视化模型自适应学习到的频率不确定性权重。
读取由 train_auto_freq.py 导出的 learned_freq_log_var.npy 文件。
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 参数设置
# ============================================================
# 请替换为你实际保存的 npy 文件路径
NPY_FILE_PATH = "/home/ubuntu/wyh/cicdis/secondPaper/script/results/TCNAE_AutoFreq_learned_freq_log_var.npy"
OUTPUT_PNG = "learned_frequency_weights.png"

# 低频区域高亮比例 (与 Motivation 图保持一致，默认 20%)
LOW_FREQ_RATIO = 0.2
DPI = 300


def main():
    if not os.path.exists(NPY_FILE_PATH):
        print(f"错误: 找不到文件 {NPY_FILE_PATH}")
        return

    # 1. 加载数据
    # 原始形状应该是 (1, F, 1)，使用 squeeze 压平为一维数组 (F,)
    log_var = np.load(NPY_FILE_PATH).squeeze()
    F_bins = len(log_var)

    # 2. 计算实际的加权系数 (Precision)
    # 公式：Weight = exp(-s)
    precision_weight = np.exp(-log_var)

    # 3. 绘图准备
    freq_bins = np.arange(F_bins)
    norm_freq = freq_bins / (F_bins - 1)
    low_cutoff = int(F_bins * LOW_FREQ_RATIO)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Endogenously Learned Frequency Weights via Homoscedastic Uncertainty",
        fontsize=14, fontweight='bold', y=0.96
    )

    # ====================
    # 子图 1: 学习到的不确定性参数 s (log_var)
    # ====================
    ax1 = axes[0]
    ax1.axvspan(0, low_cutoff - 0.5, color='#FFCCCC', alpha=0.4,
                label=f"Low-freq region (f < {LOW_FREQ_RATIO:.0%}·F_max)")
    ax1.plot(freq_bins, log_var, color='#1f77b4', linewidth=2, label="Learned Uncertainty $s_f$")

    ax1.set_ylabel("Log Variance ($s_f$)", fontsize=12)
    ax1.set_title("Learned Uncertainty Parameter (Higher = More Drift / Noise)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right")

    # ====================
    # 子图 2: 实际作用在误差上的权重 exp(-s)
    # ====================
    ax2 = axes[1]
    ax2.axvspan(0, low_cutoff - 0.5, color='#FFCCCC', alpha=0.4,
                label=f"Low-freq region")
    ax2.plot(freq_bins, precision_weight, color='#d62728', linewidth=2, label="Precision Weight $exp(-s_f)$")

    ax2.set_xlabel("Frequency Bin Index", fontsize=12)
    ax2.set_ylabel("Loss Weight $\exp(-s_f)$", fontsize=12)
    ax2.set_title("Actual Penalty Weight on Reconstruction Error (Lower = Less Penalty)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right")

    # 添加归一化频率的顶部 X 轴
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(0, 1)
    ax1_top.set_xlabel("Normalized Frequency $f / f_{max}$", fontsize=10)

    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] 权重可视化图已保存至 -> {OUTPUT_PNG}")
    print(f"    - 低频区平均权重: {precision_weight[:low_cutoff].mean():.4f}")
    print(f"    - 高频区平均权重: {precision_weight[low_cutoff:].mean():.4f}")


if __name__ == "__main__":
    main()