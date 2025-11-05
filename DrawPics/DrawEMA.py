import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# 添加路径以导入模块
sys.path.append('/mnt/user-data/uploads')
from ..dataprovider.provider_6_1_3 import load_data, split_data_chronologically
from ..models.mymodel.STLDECOMP.STL_Decompose import EMA


def select_sample_with_anomaly(X, y, sample_type='malicious'):
    """选择包含异常的样本"""
    if sample_type == 'malicious':
        # 找到包含恶意流量的窗口
        malicious_indices = np.where(np.any(y > 0, axis=1))[0]
        if len(malicious_indices) > 0:
            return malicious_indices[len(malicious_indices) // 2]  # 选择中间的恶意样本
    return 0


def find_top_varying_features(x, top_k=3):
    """找出方差最大的特征（突变显著）"""
    # 计算每个特征的标准差
    feature_std = np.std(x, axis=0)
    # 返回方差最大的top_k个特征的索引
    top_indices = np.argsort(feature_std)[-top_k:][::-1]
    return top_indices


def visualize_ema_comparison(data_dir, window_size=100, step_size=20, alpha=0.3, top_k=3):
    """可视化原始序列与EMA趋势对比"""

    # 1. 加载数据
    print("Loading data...")
    X, y, metadata = load_data(data_dir, window_size, step_size)

    # 2. 选择一个包含异常的样本
    sample_idx = select_sample_with_anomaly(X, y, sample_type='malicious')
    x_sample = X[sample_idx]  # Shape: [seq_len, num_features]
    y_sample = y[sample_idx]

    print(f"Selected sample {sample_idx}: shape={x_sample.shape}")
    print(f"Label distribution: {np.bincount(y_sample.astype(int))}")

    # 3. 找出突变显著的特征
    top_features = find_top_varying_features(x_sample, top_k=top_k)
    print(f"Top {top_k} varying features: {top_features}")

    # 4. 应用EMA
    ema_module = EMA(alpha=alpha)
    # 转换为tensor: [1, seq_len, num_features]
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).to('cuda')

    with torch.no_grad():
        x_ema = ema_module(x_tensor).cpu().numpy()[0]  # [seq_len, num_features]

    # 5. 绘图
    fig, axes = plt.subplots(top_k, 2, figsize=(14, 4 * top_k))
    if top_k == 1:
        axes = axes.reshape(1, -1)

    time_steps = np.arange(len(x_sample))

    for i, feat_idx in enumerate(top_features):
        # 原始序列
        axes[i, 0].plot(time_steps, x_sample[:, feat_idx],
                        linewidth=1, alpha=0.7, color='steelblue')
        axes[i, 0].set_title(f'Original - Feature {feat_idx}', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)

        # EMA趋势
        axes[i, 1].plot(time_steps, x_sample[:, feat_idx],
                        linewidth=1, alpha=0.3, color='lightgray', label='Original')
        axes[i, 1].plot(time_steps, x_ema[:, feat_idx],
                        linewidth=2, color='orangered', label=f'EMA (α={alpha})')
        axes[i, 1].set_title(f'EMA Trend - Feature {feat_idx}', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # 标记异常区域（如果有标签）
        if np.any(y_sample > 0):
            anomaly_regions = np.where(y_sample > 0)[0]
            if len(anomaly_regions) > 0:
                axes[i, 0].axvspan(anomaly_regions[0], anomaly_regions[-1],
                                   alpha=0.2, color='red', label='Anomaly')
                axes[i, 1].axvspan(anomaly_regions[0], anomaly_regions[-1],
                                   alpha=0.2, color='red')

    plt.tight_layout()
    plt.savefig('ema_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: ema_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "../cicids2017/selected_features/"  # 修改为实际数据路径
    WINDOW_SIZE = 100
    STEP_SIZE = 20
    ALPHA = 0.3  # EMA平滑系数
    TOP_K = 3  # 显示前K个变化最显著的特征

    visualize_ema_comparison(DATA_DIR, WINDOW_SIZE, STEP_SIZE, ALPHA, TOP_K)