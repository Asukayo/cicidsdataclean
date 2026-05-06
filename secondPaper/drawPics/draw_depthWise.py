"""
低通卷积核可视化：时域核形状 + 频率响应曲线
用法：python plot_trend_filter.py --path ./results/FullModel_trend_filter.npy
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(path):
    W = np.load(path)  # (C, 1, kernel_size)
    W = W.squeeze(1)   # (C, kernel_size)
    C, ks = W.shape
    print(f"Trend filter: C={C}, kernel_size={ks}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- 左图：时域卷积核（选 8 个代表通道）----
    ax = axes[0]
    show_idx = np.linspace(0, C - 1, min(8, C), dtype=int)
    for i in show_idx:
        ax.plot(W[i], label=f'ch {i}', alpha=0.8, linewidth=1.2)
    ax.axhline(y=1.0 / ks, color='gray', linestyle='--', alpha=0.5, label=f'init (1/{ks})')
    ax.set_xlabel('Kernel Position')
    ax.set_ylabel('Weight')
    ax.set_title('Learned Low-Pass Kernels (time domain)', fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ---- 右图：所有通道的频率响应 ----
    ax = axes[1]
    n_fft = 256
    freqs = np.linspace(0, 0.5, n_fft // 2 + 1)  # 归一化频率 [0, 0.5]

    all_responses = []
    for i in range(C):
        H = np.abs(np.fft.rfft(W[i], n=n_fft))
        H_db = 20 * np.log10(H + 1e-10)
        all_responses.append(H_db)
        ax.plot(freqs, H_db, alpha=0.15, color='steelblue', linewidth=0.5)

    # 均值响应加粗
    mean_resp = np.mean(all_responses, axis=0)
    ax.plot(freqs, mean_resp, color='darkred', linewidth=2.0, label='Mean response')
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Frequency Response (all {C} channels)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 打印统计：-3dB 截止频率估计
    half_power = mean_resp[0] - 3.0
    cutoff_idx = np.where(mean_resp < half_power)[0]
    if len(cutoff_idx) > 0:
        cutoff_freq = freqs[cutoff_idx[0]]
        print(f"Mean -3dB cutoff frequency: {cutoff_freq:.4f} (normalized)")
    else:
        print("Mean response never drops below -3dB — very broad low-pass")

    plt.tight_layout()
    out = path.replace('.npy', '_vis.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/ubuntu/wyh/cicdis/secondPaper/script/results/FullModel_trend_filter.npy')
    main(parser.parse_args().path)