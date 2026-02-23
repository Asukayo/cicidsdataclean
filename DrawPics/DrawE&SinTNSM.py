import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.mymodel.STLDECOMP.STL_Decompose import EMA, moving_avg


def visualize_ema_from_csv(csv_path, alpha=0.3):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    x_sample = df['value'].values
    print(f"Data shape: {x_sample.shape}")
    print(f"Data range: [{x_sample.min():.2e}, {x_sample.max():.2e}]")

    ema_module = EMA(alpha=alpha)
    sma_module = moving_avg(kernel_size=25, stride=1)

    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).unsqueeze(-1).to('cpu')

    with torch.no_grad():
        x_ema = ema_module(x_tensor).cpu().numpy()[0, :, 0]
        x_sma = sma_module(x_tensor).cpu().numpy()[0, :, 0]

    # 数据归一化
    scale = 1e7
    x_sample_scaled = x_sample / scale
    x_ema_scaled = x_ema / scale
    x_sma_scaled = x_sma / scale

    # ==================== IEEE TNSM 格式设置 ====================


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['font.size'] = 8

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    time_steps = np.arange(len(x_sample))

    # ==================== 左子图: 原始数据 ====================
    axes[0].plot(time_steps, x_sample_scaled, linewidth=1.2, color='steelblue',
                 label='Original Data')
    axes[0].set_title('(a) Original Data', fontsize=9, fontweight='bold', pad=8)
    axes[0].set_xlabel('Time Step', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Value ($\\times 10^7$)', fontsize=9, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True, shadow=False, fontsize=7)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].tick_params(axis='both', labelsize=7)

    # ==================== 右子图: SMA Trend vs EMA Trend ====================
    axes[1].plot(time_steps, x_sma_scaled, linewidth=1.2, color='steelblue',
                 label='SMA Trend')
    axes[1].plot(time_steps, x_ema_scaled, linewidth=1.2, color='orangered',
                 label=f'EMA Trend ($\\alpha$={alpha})')
    axes[1].set_title('(b) Trend Comparison: SMA vs EMA',
                      fontsize=9, fontweight='bold', pad=8)
    axes[1].set_xlabel('Time Step', fontsize=9, fontweight='bold')
    axes[1].set_ylabel('Value ($\\times 10^7$)', fontsize=9, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, shadow=False, fontsize=7)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1].tick_params(axis='both', labelsize=7)

    plt.tight_layout(pad=1.0, w_pad=2.0)

    output_filename = 'sma_ema_trend_comparison_IEEE.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight', pad_inches=0.1,format = 'png')
    print("Saved:", output_filename)
    plt.show()


if __name__ == "__main__":
    CSV_PATH = "network_traffic_data3.csv"
    ALPHA = 0.3

    visualize_ema_from_csv(CSV_PATH, ALPHA)