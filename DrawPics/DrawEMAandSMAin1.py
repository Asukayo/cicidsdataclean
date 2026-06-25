import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.mymodel.STLDECOMP.STL_Decompose import EMA, moving_avg

# ---- Global style (Times-like serif to match Wiley/IJCS body text) ----
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


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

    # Left: raw data
    axes[0].plot(time_steps, x_sample, linewidth=1.5, color='steelblue', label='Data')
    axes[0].set_title('Data', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, color='0.85', linestyle='-', linewidth=0.8)
    axes[0].set_axisbelow(True)

    # Right: SMA vs EMA trend
    axes[1].plot(time_steps, x_sma, linewidth=1.5, color='steelblue', label='SMA Trend')
    axes[1].plot(time_steps, x_ema, linewidth=1.5, color='orangered',
                 label=f'EMA Trend ($\\alpha$={alpha})')
    axes[1].set_title('Trend Comparison: SMA vs EMA', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Step')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, color='0.85', linestyle='-', linewidth=0.8)
    axes[1].set_axisbelow(True)

    if x_sample.max() > 1e6:
        for ax in axes:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    # ---- Export: EPS (vector, primary) + 600 dpi LZW TIFF (backup) ----
    fig.savefig('Fig2.eps', format='eps', bbox_inches='tight', facecolor='white')
    fig.savefig('Fig2.tiff', dpi=600, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print("Fig2 saved (EPS + TIFF).")


if __name__ == "__main__":
    CSV_PATH = "network_traffic_data3.csv"
    ALPHA = 0.3
    visualize_ema_from_csv(CSV_PATH, ALPHA)