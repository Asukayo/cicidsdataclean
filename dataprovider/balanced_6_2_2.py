import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader



# 平衡得到的结果为6：1：3




class NetworkTrafficDataset(Dataset):
    """CICIDS2017 网络流量数据集"""

    def __init__(self, X, y, scaler=None, is_training=True):
        """
        Args:
            X: 形状为 [num_windows, window_size, num_features] 的数据
            y: 形状为 [num_windows, window_size] 的标签
            scaler: 特征缩放器
            is_training: 是否为训练模式
        """
        self.X = X
        self.y = y
        self.is_training = is_training
        self.scaler = scaler

        # 对每个窗口进行标签聚合（多数投票）
        self.window_labels = self._aggregate_window_labels()

        # 标准化特征
        if self.scaler is not None:
            self.X = self._normalize_features()

    def _aggregate_window_labels(self):
        """将窗口内的标签聚合为单个标签（存在恶意流量即标记为恶意）"""
        window_labels = []
        for window_y in self.y:
            # 只要窗口中存在恶意流量就标记为恶意
            has_malicious = np.any(window_y > 0)
            window_labels.append(1 if has_malicious else 0)
        return np.array(window_labels)

    def _normalize_features(self):
        """标准化特征"""
        original_shape = self.X.shape
        # 重塑为 [num_samples, num_features] 进行标准化
        X_reshaped = self.X.reshape(-1, original_shape[-1])

        if self.is_training:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)

        # 重塑回原始形状
        return X_normalized.reshape(original_shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])  # [seq_len, features]
        y = torch.LongTensor([self.window_labels[idx]])  # [1]

        # 创建时间标记（用于padding mask）
        x_mark = torch.ones(x.shape[0])  # [seq_len]

        return x, x_mark, y

def load_data(data_dir, window_size=100, step_size=20):
    """加载处理后的数据"""
    X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(data_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    X = np.load(X_file)
    y = np.load(y_file)

    with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)



    return X, y, metadata


def split_data_chronologically(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    按异常比例平衡的时间顺序分割数据
    保持函数签名不变，但内部实现动态分割点调整
    """
    total_samples = len(X)
    target_abnormal_ratio = 0.25  # 目标异常比例25%
    tolerance = 0.03  # 3%的容差范围

    # 计算每个窗口的异常标签（窗口级别聚合）
    window_labels = []
    for window_y in y:
        has_malicious = np.any(window_y > 0)
        window_labels.append(1 if has_malicious else 0)
    window_labels = np.array(window_labels)

    # 根据CICIDS2017的日期分布信息定义时间段边界
    # 基于您提供的window_distribution_analysis.txt
    daily_boundaries = [
        25128,  # Monday end
        46205,  # Tuesday end
        76723,  # Wednesday end
        84926,  # Thursday-AM end
        97561,  # Thursday-PM end
        106759,  # Friday-AM end
        117443,  # Friday-PM-1 end
        128593  # Friday-PM-2 end (total)
    ]

    def calculate_abnormal_ratio(start_idx, end_idx):
        """计算指定范围内的异常比例"""
        if start_idx >= end_idx:
            return 0.0
        segment_labels = window_labels[start_idx:end_idx]
        return np.sum(segment_labels) / len(segment_labels)

    def find_optimal_split_points():
        """寻找最优分割点以平衡异常比例"""
        best_split1, best_split2 = None, None
        best_score = float('inf')

        # 在日期边界附近搜索最优分割点
        for i, boundary1 in enumerate(daily_boundaries[1:-1], 1):
            for j, boundary2 in enumerate(daily_boundaries[i + 1:], i + 2):
                if boundary2 >= total_samples * 0.9:  # 确保测试集有足够样本
                    break

                # 计算三个数据集的异常比例
                train_ratio_actual = calculate_abnormal_ratio(0, boundary1)
                val_ratio_actual = calculate_abnormal_ratio(boundary1, boundary2)
                test_ratio_actual = calculate_abnormal_ratio(boundary2, total_samples)

                # 计算与目标比例的偏差
                train_deviation = abs(train_ratio_actual - target_abnormal_ratio)
                val_deviation = abs(val_ratio_actual - target_abnormal_ratio)
                test_deviation = abs(test_ratio_actual - target_abnormal_ratio)

                # 总体偏差分数（加权验证集偏差，因为这是主要问题）
                total_score = train_deviation + 2 * val_deviation + test_deviation

                if total_score < best_score:
                    best_score = total_score
                    best_split1, best_split2 = boundary1, boundary2

        return best_split1, best_split2

    # 寻找最优分割点
    optimal_split1, optimal_split2 = find_optimal_split_points()

    # 如果没有找到合适的分割点，回退到传统方法但进行微调
    if optimal_split1 is None or optimal_split2 is None:
        print("未找到最优分割点，使用微调后的传统分割")
        # 微调传统分割点以改善验证集异常比例
        train_end = int(total_samples * 0.65)  # 增加训练集比例
        val_end = int(total_samples * 0.80)  # 减少验证集比例
    else:
        train_end = optimal_split1
        val_end = optimal_split2

    # 分割数据
    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    # 计算实际的异常比例
    train_abnormal_ratio = calculate_abnormal_ratio(0, train_end)
    val_abnormal_ratio = calculate_abnormal_ratio(train_end, val_end)
    test_abnormal_ratio = calculate_abnormal_ratio(val_end, total_samples)

    print(f"优化后的数据分割:")
    print(
        f"  训练集: {len(X_train)} ({len(X_train) / total_samples * 100:.1f}%) - 异常比例: {train_abnormal_ratio:.1%}")
    print(f"  验证集: {len(X_val)} ({len(X_val) / total_samples * 100:.1f}%) - 异常比例: {val_abnormal_ratio:.1%}")
    print(f"  测试集: {len(X_test)} ({len(X_test) / total_samples * 100:.1f}%) - 异常比例: {test_abnormal_ratio:.1%}")

    # 检查是否达到平衡目标
    if abs(val_abnormal_ratio - target_abnormal_ratio) > tolerance:
        print(f"⚠️ 警告: 验证集异常比例({val_abnormal_ratio:.1%})偏离目标值({target_abnormal_ratio:.1%})超过容差")
    else:
        print(f"✅ 成功: 验证集异常比例已平衡")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """创建数据加载器"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # 创建标准化器
    scaler = StandardScaler()

    # 创建数据集
    train_dataset = NetworkTrafficDataset(X_train, y_train, scaler=scaler, is_training=True)
    val_dataset = NetworkTrafficDataset(X_val, y_val, scaler=scaler, is_training=False)
    test_dataset = NetworkTrafficDataset(X_test, y_test, scaler=scaler, is_training=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler

