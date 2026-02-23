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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    time_steps = np.arange(len(x_sample))

    # 左子图: 原始数据
    axes[0].plot(time_steps, x_sample, linewidth=1.5, color='steelblue', label='Data')
    axes[0].set_title('Data', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # 右子图: SMA Trend vs EMA Trend
    axes[1].plot(time_steps, x_sma, linewidth=1.5, color='steelblue', label='SMA Trend')
    axes[1].plot(time_steps, x_ema, linewidth=1.5, color='orangered', label=f'EMA Trend (α={alpha})')
    axes[1].set_title('Trend Comparison: SMA vs EMA', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Step')
    # axes[1].set_ylabel('Value')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    if x_sample.max() > 1e6:
        for ax in axes:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('sma_ema_trend_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: sma_ema_trend_comparison.png")
    plt.show()


if __name__ == "__main__":
    CSV_PATH = "network_traffic_data3.csv"
    ALPHA = 0.3

    visualize_ema_from_csv(CSV_PATH, ALPHA)