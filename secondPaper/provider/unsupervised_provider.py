"""
无监督范式数据加载器
====================
与原始 provider_6_1_3.py 的区别：
  - 训练集：仅保留正常窗口（y_i = 0），用于学习正常模式
  - 验证集：保留全部窗口，用于调节重构误差阈值
  - 测试集：保留全部窗口，用于最终评估
  - StandardScaler 仅在正常训练数据上 fit
"""

import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class UnsupervisedTrafficDataset(Dataset):
    """无监督范式的网络流量数据集"""

    def __init__(self, X, y, scaler=None, fit_scaler=False):
        """
        Args:
            X: [num_windows, window_size, num_features]
            y: [num_windows, window_size] 流级别标签
            scaler: StandardScaler 实例
            fit_scaler: 是否在当前数据上 fit scaler（仅训练集为 True）
        """
        self.y_raw = y  # 保留原始流级别标签，方便后续分析
        self.scaler = scaler

        # 窗口级标签：窗口内存在恶意流即为 1
        self.window_labels = (np.any(y > 0, axis=1)).astype(int)

        # 标准化
        if self.scaler is not None:
            original_shape = X.shape
            X_flat = X.reshape(-1, original_shape[-1])

            if fit_scaler:
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)

            self.X = X_flat.reshape(original_shape)
        else:
            self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])              # [W, F]
        label = torch.LongTensor([self.window_labels[idx]])  # [1]
        x_mark = torch.ones(x.shape[0])                 # [W]
        return x, x_mark, label


def load_data(data_dir, window_size=100, step_size=20):
    """加载 .npy 数据文件"""
    X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(data_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    X = np.load(X_file)
    y = np.load(y_file)

    metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

    return X, y, metadata


def split_data_unsupervised(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    按时间顺序划分，训练集仅保留正常窗口

    Returns:
        train_data: (X_train_normal, y_train_normal) 仅正常窗口
        val_data:   (X_val, y_val)                   全部窗口
        test_data:  (X_test, y_test)                  全部窗口
        split_info: dict 包含划分的统计信息
    """
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    # 按时间划分
    X_train_all, y_train_all = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 过滤训练集：仅保留正常窗口
    train_window_labels = np.any(y_train_all > 0, axis=1).astype(int)
    normal_mask = train_window_labels == 0

    X_train_normal = X_train_all[normal_mask]
    y_train_normal = y_train_all[normal_mask]

    # 统计信息
    val_labels = np.any(y_val > 0, axis=1).astype(int)
    test_labels = np.any(y_test > 0, axis=1).astype(int)

    split_info = {
        'train_before_filter': len(X_train_all),
        'train_removed': int((~normal_mask).sum()),
        'train_normal': len(X_train_normal),
        'val_total': len(X_val),
        'val_normal': int((val_labels == 0).sum()),
        'val_anomalous': int((val_labels == 1).sum()),
        'test_total': len(X_test),
        'test_normal': int((test_labels == 0).sum()),
        'test_anomalous': int((test_labels == 1).sum()),
    }

    return (X_train_normal, y_train_normal), (X_val, y_val), (X_test, y_test), split_info


def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """
    创建数据加载器

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # scaler 仅在正常训练数据上 fit
    scaler = StandardScaler()

    train_dataset = UnsupervisedTrafficDataset(
        X_train, y_train, scaler=scaler, fit_scaler=True
    )
    val_dataset = UnsupervisedTrafficDataset(
        X_val, y_val, scaler=scaler, fit_scaler=False
    )
    test_dataset = UnsupervisedTrafficDataset(
        X_test, y_test, scaler=scaler, fit_scaler=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def print_split_info(split_info):
    """打印划分统计信息"""
    print("=" * 55)
    print("  Unsupervised Data Split Summary")
    print("=" * 55)
    print(f"  Train (normal only): {split_info['train_normal']} windows")
    print(f"    (filtered out {split_info['train_removed']} anomalous windows "
          f"from {split_info['train_before_filter']})")
    print(f"  Val:   {split_info['val_total']} windows "
          f"(normal: {split_info['val_normal']}, "
          f"anomalous: {split_info['val_anomalous']})")
    print(f"  Test:  {split_info['test_total']} windows "
          f"(normal: {split_info['test_normal']}, "
          f"anomalous: {split_info['test_anomalous']})")
    print("=" * 55)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/selected_features"  # <-- 修改为你的实际路径
    WINDOW_SIZE = 100
    STEP_SIZE = 20
    BATCH_SIZE = 128

    # 1. 加载原始数据
    X, y, metadata = load_data(DATA_DIR, WINDOW_SIZE, STEP_SIZE)
    print(f"Loaded: X={X.shape}, y={y.shape}")

    # 2. 划分（训练集自动过滤异常窗口）
    train_data, val_data, test_data, split_info = split_data_unsupervised(
        X, y, train_ratio=0.6, val_ratio=0.2
    )
    print_split_info(split_info)

    # 3. 创建 DataLoader
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )

    # 4. 验证
    for batch_x, batch_mark, batch_y in train_loader:
        print(f"\nTrain batch check:")
        print(f"  x:     {batch_x.shape}")       # [B, W, F]
        print(f"  mark:  {batch_mark.shape}")     # [B, W]
        print(f"  label: {batch_y.shape}")        # [B, 1]
        print(f"  labels in batch: {batch_y.squeeze().unique().tolist()}")
        # 训练集应该全部是 0
        assert (batch_y == 0).all(), "ERROR: anomalous window found in training set!"
        print(f"  All normal (label=0): PASS")
        break

    for batch_x, batch_mark, batch_y in test_loader:
        print(f"\nTest batch check:")
        print(f"  x:     {batch_x.shape}")
        print(f"  labels in batch: {batch_y.squeeze().unique().tolist()}")
        # 测试集应该包含 0 和 1
        break

    print("\nData pipeline ready.")