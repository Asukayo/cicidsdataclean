import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.mymodel.STLDECOMP.STL_Decompose import EMA, EnergyBasedDFTFilter


def generate_synthetic_timeseries(seq_len=150, seed=42):
    """生成合成时间序列数据（论文展示版）"""
    np.random.seed(seed)
    t = np.arange(seq_len)

    # 趋势分量
    trend = 0.03 * t + 50

    # 低频季节性（两个明显周期）
    seasonal = 8.0 * np.sin(2 * np.pi * t / 30) + 4.0 * np.sin(2 * np.pi * t / 60)

    # 高频噪声
    noise = 2.0 * np.random.randn(seq_len)

    return trend + seasonal + noise


def paper_visualization(alpha=0.25, energy_threshold=0.65):
    """
    论文展示专用：生成清晰的DFT分解可视化

    参数:
        alpha: EMA平滑系数
        energy_threshold: DFT能量阈值（越低分离越激进）
    """

    print("Generating synthetic time series for paper...")
    x_sample = generate_synthetic_timeseries(seq_len=150)

    # 保存数据
    df = pd.DataFrame({'time': np.arange(len(x_sample)), 'value': x_sample})
    df.to_csv('paper_synthetic_data.csv', index=False)
    print(f"✓ Data saved: paper_synthetic_data.csv")
    print(f"  - Sequence length: {len(x_sample)}")
    print(f"  - Value range: [{x_sample.min():.2f}, {x_sample.max():.2f}]")

    # EMA趋势提取
    ema_module = EMA(alpha=alpha)
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).unsqueeze(-1).to('cuda')

    with torch.no_grad():
        # 提取趋势
        x_trend = ema_module(x_tensor)

        # 计算Seasonality
        seasonality_tensor = x_tensor - x_trend

        # DFT分解
        dft_filter = EnergyBasedDFTFilter(
            top_k=3,
            low_freq_ratio=0.25,
            energy_threshold=energy_threshold
        ).to('cuda')

        x_seasonal = dft_filter(seasonality_tensor)  # 低频季节性
        x_residual = seasonality_tensor - x_seasonal  # 高频残差

        # 转换为numpy
        seasonality = seasonality_tensor.cpu().numpy()[0, :, 0]
        x_seasonal_np = x_seasonal.cpu().numpy()[0, :, 0]
        x_residual_np = x_residual.cpu().numpy()[0, :, 0]

    # ============ 论文展示图：三子图布局 ============
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time_steps = np.arange(len(x_sample))

    # 配色方案（专业论文配色）
    colors = {
        'seasonality': '#2E86AB',  # 深蓝
        'low_freq': '#06A77D',  # 绿
        'high_freq': '#D4552C'  # 橙红
    }

    # 子图1: 原始Seasonality（去趋势后）
    axes[0].plot(time_steps, seasonality,
                 linewidth=2.5, color=colors['seasonality'],
                 label='Seasonality', alpha=0.9)
    axes[0].set_title('(a) Seasonality\n(Data - Trend)',
                      fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Time Step', fontsize=13)
    axes[0].set_ylabel('Value', fontsize=13)
    # axes[0].legend(fontsize=11, framealpha=0.95, loc='upper right')
    # axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[0].tick_params(labelsize=11)

    # 子图2: 低频季节性成分
    axes[1].plot(time_steps, x_seasonal_np,
                 linewidth=2.5, color=colors['low_freq'],
                 label='Low-Freq Component', alpha=0.9)
    axes[1].set_title(f'(b) Low-Frequency\n(Energy ≤ {energy_threshold})',
                      fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Time Step', fontsize=13)
    axes[1].set_ylabel('Value', fontsize=13)
    # axes[1].legend(fontsize=11, framealpha=0.95, loc='upper right')
    # axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[1].tick_params(labelsize=11)

    # 子图3: 高频残差（噪声）
    axes[2].plot(time_steps, x_residual_np,
                 linewidth=2, color=colors['high_freq'],
                 label='High-Freq Residual', alpha=0.85)
    axes[2].set_title('(c) High-Frequency\n(Noise)',
                      fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel('Time Step', fontsize=13)
    axes[2].set_ylabel('Value', fontsize=13)
    # axes[2].legend(fontsize=11, framealpha=0.95, loc='upper right')
    # axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[2].tick_params(labelsize=11)

    plt.tight_layout()

    # 保存高分辨率图像（适合论文）
    output_file = 'paper_dft_decomposition.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_file}")
    print(f"  - Resolution: 300 DPI (publication quality)")
    print(f"  - Format: PNG with white background")

    # 统计信息
    print("\n" + "=" * 60)
    print("Decomposition Statistics")
    print("=" * 60)
    print(f"Seasonality:      mean={seasonality.mean():>7.4f}, std={seasonality.std():>7.4f}")
    print(f"Low-Freq (x_s):   mean={x_seasonal_np.mean():>7.4f}, std={x_seasonal_np.std():>7.4f}")
    print(f"High-Freq (x_r):  mean={x_residual_np.mean():>7.4f}, std={x_residual_np.std():>7.4f}")

    # 能量分布
    seasonal_energy = (x_seasonal_np.std() ** 2 / seasonality.std() ** 2) * 100
    residual_energy = (x_residual_np.std() ** 2 / seasonality.std() ** 2) * 100
    print(f"\nEnergy Distribution:")
    print(f"  Low-Freq:  {seasonal_energy:>6.2f}%")
    print(f"  High-Freq: {residual_energy:>6.2f}%")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DFT-Based Seasonality Decomposition")
    print("Paper Visualization Mode")
    print("=" * 60 + "\n")

    # 论文展示参数（经过优化）
    ALPHA = 0.25  # EMA平滑系数
    ENERGY_THRESHOLD = 0.65  # 能量阈值（控制分离程度）

    paper_visualization(
        alpha=ALPHA,
        energy_threshold=ENERGY_THRESHOLD
    )

    print("\n✓ Paper-ready visualization generated!")
    print("  Recommended: Use 'paper_dft_decomposition.png' in your manuscript\n")