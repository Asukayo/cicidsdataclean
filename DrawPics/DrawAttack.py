import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.mymodel.STLDECOMP.STL_Decompose import EMA, EnergyBasedDFTFilter


def generate_synthetic_timeseries(seq_len=100, seed=42):
    """生成类似网络流量的合成数据：波动大、变化密集、量级为1e7"""
    np.random.seed(seed)
    t = np.arange(seq_len)

    # 趋势分量（下降趋势，模拟网络流量衰减）
    trend = -3e5 * t + 2e7  # 从2000万逐渐下降

    # 低频季节性分量（多个周期叠加，幅度更大）
    seasonal = (3e6 * np.sin(2 * np.pi * t / 15) +  # 周期15
                2e6 * np.sin(2 * np.pi * t / 25) +  # 周期25
                1.5e6 * np.sin(2 * np.pi * t / 40))  # 周期40

    # 高频密集波动（短周期，模拟网络流量的快速变化）
    high_freq = (1e6 * np.sin(2 * np.pi * t / 5) +  # 周期5（密集）
                 0.8e6 * np.sin(2 * np.pi * t / 8) +  # 周期8
                 0.5e6 * np.sin(2 * np.pi * t / 3))  # 周期3（非常密集）

    # 高频噪声（幅度更大）
    noise = 1.5e6 * np.random.randn(seq_len)

    # 添加随机突发尖峰（模拟网络流量突发）
    spike_indices = np.random.choice(seq_len, size=int(seq_len * 0.1), replace=False)
    spikes = np.zeros(seq_len)
    spikes[spike_indices] = np.random.uniform(2e6, 5e6, size=len(spike_indices))

    # 组合所有分量
    time_series = trend + seasonal + high_freq + noise + spikes

    # 确保数值非负（网络流量不能为负）
    time_series = np.maximum(time_series, 1e6)

    # 返回真实分量（用于验证）
    true_seasonal = seasonal + high_freq  # 将高频也归入季节性
    true_noise = noise + spikes  # 噪声包含随机波动和突发

    return time_series, trend, true_seasonal, true_noise

    # 组合所有分量
    time_series = trend + seasonal + noise

    return time_series, trend, seasonal, noise


def visualize_dft_decomposition_synthetic(
        alpha=0.3,
        top_k=5,
        low_freq_ratio=0.3,
        energy_threshold=0.70
):
    """
    生成合成数据并使用DFT滤波器分解seasonality成分

    Args:
        alpha: EMA平滑系数
        top_k: DFT保留的最小频率分量数
        low_freq_ratio: 保留的低频比例
        energy_threshold: 能量累积阈值（调低以更明显地分离低频和高频）
    """

    # 1. 生成合成数据（网络流量风格：高波动、密集变化）
    print("Generating synthetic network traffic data...")
    x_sample, true_trend, true_seasonal, true_noise = generate_synthetic_timeseries(seq_len=100)

    # 保存为CSV（可选）
    df = pd.DataFrame({'time': np.arange(len(x_sample)), 'value': x_sample})
    df.to_csv('synthetic_timeseries.csv', index=False)
    print(f"Synthetic data saved to: synthetic_timeseries.csv")
    print(f"Data shape: {x_sample.shape}")
    print(f"Data range: [{x_sample.min():.2e}, {x_sample.max():.2e}]")
    print(f"Data mean: {x_sample.mean():.2e}, std: {x_sample.std():.2e}")

    # 2. 应用EMA提取趋势
    ema_module = EMA(alpha=alpha)

    # 转换为tensor
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).unsqueeze(-1).to('cuda')

    with torch.no_grad():
        x_trend = ema_module(x_tensor)  # [1, seq_len, 1]

        # 3. 计算Seasonality
        seasonality_tensor = x_tensor - x_trend

        # 4. 使用EnergyBasedDFTFilter分解seasonality
        dft_filter = EnergyBasedDFTFilter(
            top_k=top_k,
            low_freq_ratio=low_freq_ratio,
            energy_threshold=energy_threshold
        ).to('cuda')

        x_seasonal = dft_filter(seasonality_tensor)  # 低频季节性
        x_residual = seasonality_tensor - x_seasonal  # 高频残差

        # 转换回numpy
        seasonality = seasonality_tensor.cpu().numpy()[0, :, 0]
        x_seasonal_np = x_seasonal.cpu().numpy()[0, :, 0]
        x_residual_np = x_residual.cpu().numpy()[0, :, 0]

    # 5. 绘制分解结果（三子图）- 论文展示版
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time_steps = np.arange(len(x_sample))

    # 子图1: 原始Seasonality
    axes[0].plot(time_steps, seasonality, linewidth=2.5, color='#2E86AB', label='Seasonality')
    axes[0].set_title('Seasonality (Data - Trend)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Step', fontsize=13)
    axes[0].set_ylabel('Value', fontsize=13)
    axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[0].tick_params(labelsize=11)

    # 子图2: DFT滤波后的低频季节性
    axes[1].plot(time_steps, x_seasonal_np, linewidth=2.5, color='#06A77D', label='Low-Freq Seasonal')
    axes[1].set_title(f'DFT Filtered Seasonal (Energy≤{energy_threshold})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Step', fontsize=13)
    axes[1].set_ylabel('Value', fontsize=13)
    axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[1].tick_params(labelsize=11)

    # 子图3: 高频残差噪声
    axes[2].plot(time_steps, x_residual_np, linewidth=2, color='#D4552C', alpha=0.8, label='High-Freq Residual')
    axes[2].set_title('High-Freq Residual (Noise)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Step', fontsize=13)
    axes[2].set_ylabel('Value', fontsize=13)
    axes[2].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[2].tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig('dft_decomposition_clear.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: dft_decomposition_clear.png")

    # 6. 打印统计信息
    print("\n=== Decomposition Statistics ===")
    print(f"Seasonality: mean={seasonality.mean():.2e}, std={seasonality.std():.2e}")
    print(f"x_seasonal:  mean={x_seasonal_np.mean():.2e}, std={x_seasonal_np.std():.2e}")
    print(f"x_residual:  mean={x_residual_np.mean():.2e}, std={x_residual_np.std():.2e}")

    # 计算能量保留比例
    seasonal_energy_ratio = (x_seasonal_np.std() ** 2 / seasonality.std() ** 2) * 100
    residual_energy_ratio = (x_residual_np.std() ** 2 / seasonality.std() ** 2) * 100
    print(f"\nEnergy distribution:")
    print(f"  Low-freq seasonal: {seasonal_energy_ratio:.2f}%")
    print(f"  High-freq residual: {residual_energy_ratio:.2f}%")
    print(f"  Total: {seasonal_energy_ratio + residual_energy_ratio:.2f}%")

    # 7. 绘制真实分量对比（验证分解效果）- 论文展示版
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 9))

    # 真实vs提取的季节性
    axes2[0, 0].plot(time_steps, true_seasonal, linewidth=2.5, color='#0A6847', label='True Seasonal', alpha=0.8)
    axes2[0, 0].plot(time_steps, x_seasonal_np, linewidth=2.5, color='#06A77D', label='Extracted Seasonal',
                     linestyle='--', alpha=0.9)
    axes2[0, 0].set_title('True vs Extracted Seasonal', fontsize=14, fontweight='bold')
    axes2[0, 0].set_xlabel('Time Step', fontsize=13)
    axes2[0, 0].set_ylabel('Value', fontsize=13)
    axes2[0, 0].legend(fontsize=11, framealpha=0.9)
    axes2[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes2[0, 0].tick_params(labelsize=11)

    # 真实vs提取的噪声
    axes2[0, 1].plot(time_steps, true_noise, linewidth=1.5, color='#8B0000', label='True Noise', alpha=0.6)
    axes2[0, 1].plot(time_steps, x_residual_np, linewidth=1.5, color='#D4552C', label='Extracted Residual',
                     linestyle='--', alpha=0.8)
    axes2[0, 1].set_title('True vs Extracted Noise', fontsize=14, fontweight='bold')
    axes2[0, 1].set_xlabel('Time Step', fontsize=13)
    axes2[0, 1].set_ylabel('Value', fontsize=13)
    axes2[0, 1].legend(fontsize=11, framealpha=0.9)
    axes2[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes2[0, 1].tick_params(labelsize=11)

    # 频谱分析：Seasonality
    freq_seasonality = np.fft.rfft(seasonality)
    freq_axis = np.fft.rfftfreq(len(seasonality))
    axes2[1, 0].plot(freq_axis, np.abs(freq_seasonality), linewidth=2.5, color='#2E86AB')
    axes2[1, 0].set_title('Frequency Spectrum: Seasonality', fontsize=14, fontweight='bold')
    axes2[1, 0].set_xlabel('Frequency', fontsize=13)
    axes2[1, 0].set_ylabel('Amplitude', fontsize=13)
    axes2[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes2[1, 0].set_xlim([0, 0.25])  # 只显示低频部分
    axes2[1, 0].tick_params(labelsize=11)

    # 频谱分析：x_seasonal vs x_residual
    freq_seasonal = np.fft.rfft(x_seasonal_np)
    freq_residual = np.fft.rfft(x_residual_np)
    axes2[1, 1].plot(freq_axis, np.abs(freq_seasonal), linewidth=2.5, color='#06A77D', label='Low-Freq Seasonal',
                     alpha=0.9)
    axes2[1, 1].plot(freq_axis, np.abs(freq_residual), linewidth=2.5, color='#D4552C', label='High-Freq Residual',
                     alpha=0.9)
    axes2[1, 1].set_title('Frequency Spectrum: Separated Components', fontsize=14, fontweight='bold')
    axes2[1, 1].set_xlabel('Frequency', fontsize=13)
    axes2[1, 1].set_ylabel('Amplitude', fontsize=13)
    axes2[1, 1].legend(fontsize=11, framealpha=0.9)
    axes2[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes2[1, 1].set_xlim([0, 0.25])
    axes2[1, 1].tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig('dft_decomposition_verification.png', dpi=300, bbox_inches='tight')
    print("Verification plot saved to: dft_decomposition_verification.png")

    plt.show()


if __name__ == "__main__":
    # 配置参数 - 适配高频密集网络流量数据
    ALPHA = 0.2  # EMA平滑系数（降低以应对剧烈波动）
    TOP_K = 5  # 最小保留频率数（增加以保留更多低频）
    LOW_FREQ_RATIO = 0.35  # 低频比例（提高以应对密集高频）
    ENERGY_THRESHOLD = 0.55  # 能量阈值（降低以更激进地过滤高频噪声）

    print("=" * 60)
    print("DFT-Based Seasonality Decomposition - Network Traffic Style")
    print("=" * 60)

    visualize_dft_decomposition_synthetic(
        alpha=ALPHA,
        top_k=TOP_K,
        low_freq_ratio=LOW_FREQ_RATIO,
        energy_threshold=ENERGY_THRESHOLD
    )