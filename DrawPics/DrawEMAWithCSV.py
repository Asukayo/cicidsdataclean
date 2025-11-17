import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.mymodel.STLDECOMP.STL_Decompose import EMA,moving_avg,DEMA


def visualize_ema_from_csv(csv_path, alpha=0.3):
    """从CSV文件读取数据并可视化EMA分解（三子图布局）

    左图: 原始数据 (Data)
    中图: EMA趋势 (Trend)
    右图: 周期性成分 (Seasonality = Data - Trend)

    Args:
        csv_path: CSV文件路径
        alpha: EMA平滑系数
    """

    # 1. 读取CSV文件
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. 提取value列作为时间序列数据
    x_sample = df['value'].values  # Shape: [seq_len]
    print(f"Data shape: {x_sample.shape}")
    print(f"Data range: [{x_sample.min():.2e}, {x_sample.max():.2e}]")

    # 3. 应用EMA
    # ema_module = EMA(alpha=alpha)
    # 引用SMA
    # sma_module = moving_avg(kernel_size=25,stride=1)
    # 应用DEMA
    dema_module = DEMA(alpha=alpha)

    # 转换为tensor: [1, seq_len, 1] (batch_size=1, seq_len, num_features=1)
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).unsqueeze(-1).to('cuda')

    # with torch.no_grad():
    #     x_ema = ema_module(x_tensor).cpu().numpy()[0, :, 0]  # [seq_len]

    # with torch.no_grad():
    #     x_sma = sma_module(x_tensor).cpu().numpy()[0, :, 0]  # [seq_len]

    with torch.no_grad():
        x_dema = dema_module(x_tensor).cpu().numpy()[0, :, 0]  # [seq_len]

    # 4. 计算Seasonality（周期性成分）
    seasonality = x_sample - x_dema

    # 5. 绘图 - 三个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    time_steps = np.arange(len(x_sample))

    # 子图1: 原始数据
    axes[0].plot(time_steps, x_sample,
                 linewidth=1.5, color='steelblue', label='Data')
    axes[0].set_title('Data', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # 子图2: EMA趋势
    axes[1].plot(time_steps, x_dema,
                 linewidth=1.5, color='steelblue', label='Trend')
    axes[1].set_title(f'DEMA Decomposition (alpha = {alpha})', fontsize=12, fontweight='bold')
    # axes[1].set_title(f'SMA Decomposition', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Value')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # 子图3: 周期性成分
    axes[2].plot(time_steps, seasonality,
                 linewidth=1.5, color='orangered', label='Seasonality')
    axes[2].set_title(f'DEMA Decomposition (alpha = {alpha})', fontsize=12, fontweight='bold')
    # axes[2].set_title(f'SMA Decomposition', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Value')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

    # 设置科学计数法（如果数值较大）
    if x_sample.max() > 1e6:
        axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('ema_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: ema_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 配置参数
    CSV_PATH = "network_traffic_data3.csv"  # 修改为你的CSV文件路径
    ALPHA = 0.3  # EMA平滑系数

    visualize_ema_from_csv(CSV_PATH, ALPHA)