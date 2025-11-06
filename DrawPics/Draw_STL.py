import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# 添加路径以导入模块
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically
from models.mymodel.STLDECOMP.STL_Decompose import EMA


def select_sample_with_anomaly(X, y, sample_type='malicious', random_select=True, seed=None):
    """选择包含异常的样本

    Args:
        X: 特征数据
        y: 标签数据
        sample_type: 样本类型，'malicious' 或 'normal'
        random_select: 是否随机选择样本（True=每次不同，False=固定选择中间样本）
        seed: 随机种子，用于复现结果（仅在random_select=True时有效）
    """
    if sample_type == 'malicious':
        # 找到包含恶意流量的窗口
        malicious_indices = np.where(np.any(y > 0, axis=1))[0]
        if len(malicious_indices) > 0:
            if random_select:
                if seed is not None:
                    np.random.seed(seed)
                return np.random.choice(malicious_indices)
            else:
                return malicious_indices[len(malicious_indices) // 2]  # 选择中间的恶意样本
    return 0


def find_top_varying_features(x, top_k=3, selection_mode='trend'):
    """找出最适合展示的特征

    Args:
        x: [seq_len, num_features]
        top_k: 返回前K个特征
        selection_mode: 'variance' - 仅考虑方差
                       'trend' - 考虑趋势性和平滑度（推荐）
    """
    if selection_mode == 'variance':
        # 原始方法：仅考虑方差
        feature_std = np.std(x, axis=0)
        top_indices = np.argsort(feature_std)[-top_k:][::-1]
        return top_indices

    elif selection_mode == 'trend':
        # 新方法：综合考虑趋势性、平滑度和方差
        seq_len, num_features = x.shape
        time_steps = np.arange(seq_len)

        scores = []
        for feat_idx in range(num_features):
            feature_data = x[:, feat_idx]

            # 1. 计算方差（归一化到0-1）
            variance = np.var(feature_data)

            # 2. 计算线性趋势强度（R²值）
            if variance > 1e-10:  # 避免除零
                # 线性回归
                coeffs = np.polyfit(time_steps, feature_data, 1)
                trend_line = np.polyval(coeffs, time_steps)
                # R² = 1 - (SS_res / SS_tot)
                ss_res = np.sum((feature_data - trend_line) ** 2)
                ss_tot = np.sum((feature_data - np.mean(feature_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r_squared = max(0, r_squared)  # R²可能为负，取max
            else:
                r_squared = 0

            # 3. 计算平滑度（通过一阶差分的标准差衡量，值越小越平滑）
            diff = np.diff(feature_data)
            diff_std = np.std(diff)
            # 归一化：平滑度分数 = 1 / (1 + diff_std/variance)
            smoothness = 1.0 / (1.0 + diff_std / (np.sqrt(variance) + 1e-10))

            # 4. 综合得分（权重可调）
            # 高方差 + 高趋势性 + 适度平滑 = 好的可视化特征
            score = (
                    0.1 * (variance / (np.max(np.var(x, axis=0)) + 1e-10)) +  # 方差权重30%
                    0.5 * r_squared +  # 趋势性权重50%
                    0.2 * smoothness  # 平滑度权重20%
            )
            scores.append(score)

        # 返回得分最高的top_k个特征
        scores = np.array(scores)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return top_indices

    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")


def visualize_stl_decomposition(data_dir, window_size=100, step_size=20, alpha=0.3,
                                top_k=3, random_select=True, seed=None,
                                normalize=False, selection_mode='trend'):
    """可视化原始数据与STL分解（Trend + Seasonal）

    Args:
        data_dir: 数据目录
        window_size: 窗口大小
        step_size: 步长
        alpha: EMA平滑系数
        top_k: 显示前K个变化最显著的特征
        random_select: 是否随机选择样本
        seed: 随机种子
        normalize: 是否对数据进行标准化
        selection_mode: 特征选择模式 ('variance' 或 'trend')
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. 加载数据
    print("Loading data...")
    X, y, metadata = load_data(data_dir, window_size, step_size)

    # 2. 选择一个包含异常的样本
    sample_idx = select_sample_with_anomaly(X, y, sample_type='malicious',
                                            random_select=random_select, seed=seed)
    x_sample = X[sample_idx]  # Shape: [seq_len, num_features]
    y_sample = y[sample_idx]

    print(f"Selected sample {sample_idx}: shape={x_sample.shape}")
    print(f"Label distribution: {np.bincount(y_sample.astype(int))}")

    # 2.5 数据标准化（如果启用）
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_sample_original = x_sample.copy()  # 保存原始数据用于显示
        x_sample = scaler.fit_transform(x_sample)
        print("Data normalized using StandardScaler (Z-score)")

    # 3. 找出最适合展示的特征
    top_features = find_top_varying_features(x_sample, top_k=top_k,
                                             selection_mode=selection_mode)
    print(f"Top {top_k} features (mode={selection_mode}): {top_features}")

    # 4. 使用EMA提取趋势
    ema_module = EMA(alpha=alpha)
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).to('cuda')

    with torch.no_grad():
        x_trend = ema_module(x_tensor).cpu().numpy()[0]  # [seq_len, num_features]

    # 5. 计算季节性分量：seasonal = original - trend
    x_seasonal = x_sample - x_trend

    # 6. 绘图：左边显示原始数据，右边显示Trend + Seasonal
    fig, axes = plt.subplots(top_k, 2, figsize=(14, 4 * top_k))

    # 处理单个特征的情况
    if top_k == 1:
        axes = axes.reshape(1, -1)

    time_steps = np.arange(len(x_sample))

    for i, feat_idx in enumerate(top_features):
        # 左图：原始数据
        axes[i, 0].plot(time_steps, x_sample[:, feat_idx],
                        linewidth=1.5, color='steelblue', label='Data')
        axes[i, 0].set_title('Data', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend(loc='upper left')
        axes[i, 0].grid(True, alpha=0.3)

        # 右图：Trend + Seasonal
        axes[i, 1].plot(time_steps, x_trend[:, feat_idx],
                        linewidth=2, color='steelblue', label='Trend')
        axes[i, 1].plot(time_steps, x_seasonal[:, feat_idx],
                        linewidth=1.5, color='orangered', label='Seasonality')
        axes[i, 1].set_title(f'EMA Decomposition (alpha = {alpha})', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stl_decomposition.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: stl_decomposition.png")
    plt.show()


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "../cicids2017/selected_features/"  # 修改为实际数据路径
    WINDOW_SIZE = 100
    STEP_SIZE = 20
    ALPHA = 0.3  # EMA平滑系数
    TOP_K = 3  # 显示前K个变化最显著的特征

    # 随机选择设置
    RANDOM_SELECT = True  # True=每次选择不同样本，False=固定选择中间样本
    SEED = None  # 设置为整数（如42）可复现结果，None则完全随机

    # 数据处理设置
    NORMALIZE = False  # True=标准化数据，False=使用原始数据

    # 特征选择模式
    # 'variance' - 仅考虑方差（原始方法，可能选到脉冲式特征）
    # 'trend' - 综合考虑趋势性和平滑度（推荐，选择有明显变化趋势的特征）
    SELECTION_MODE = 'trend'

    visualize_stl_decomposition(DATA_DIR, WINDOW_SIZE, STEP_SIZE, ALPHA, TOP_K,
                                random_select=RANDOM_SELECT, seed=SEED,
                                normalize=NORMALIZE, selection_mode=SELECTION_MODE)