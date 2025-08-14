import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
from collections import defaultdict


class WindowDataset(Dataset):
    """
    用于窗口数据的PyTorch数据集
    每个样本是一个窗口 (window_size, num_features)
    """

    def __init__(self, X_path, y_path, metadata_path, use_mmap=True):
        """
        参数:
        - X_path: X数据文件路径
        - y_path: y数据文件路径
        - metadata_path: 元数据文件路径
        - use_mmap: 是否使用内存映射（节省内存）
        """
        # 使用内存映射加载数据，避免全部加载到内存
        if use_mmap:
            self.X = np.load(X_path, mmap_mode='r')
            self.y = np.load(y_path, mmap_mode='r')
        else:
            self.X = np.load(X_path)
            self.y = np.load(y_path)

        # 加载元数据
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # 获取窗口级别的标签
        self.window_labels = np.array([w['is_malicious'] for w in self.metadata['window_metadata']])

        self.num_windows = self.X.shape[0]
        self.window_size = self.X.shape[1]
        self.num_features = self.X.shape[2]

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        返回:
        - X_window: (window_size, num_features)
        - y_window: (window_size,) - 窗口内每个流的标签
        - window_label: 标量 - 窗口级别标签（0或1）
        """
        # 复制数据以避免内存映射问题
        X_window = np.array(self.X[idx])
        y_window = np.array(self.y[idx])
        window_label = self.window_labels[idx]

        return X_window, y_window, window_label


class IncrementalRandomForest:
    """
    支持批量增量训练的随机森林
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # 初始化基础模型
        self.base_rf = RandomForestClassifier(
            n_estimators=1,  # 每次训练一棵树
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            warm_start=True
        )

        self.trees = []
        self.feature_importances_list = []
        self.is_fitted = False

    def partial_fit(self, X_batch, y_batch, classes=None):
        """
        增量训练

        参数:
        - X_batch: (batch_size * window_size, num_features) 或其他形状
        - y_batch: (batch_size,) 批量标签
        - classes: 所有可能的类别
        """
        if classes is None:
            classes = np.array([0, 1])

        # 为这个批次训练多棵树
        trees_per_batch = max(1, self.n_estimators // 20)  # 分20批训练

        for i in range(trees_per_batch):
            # 克隆基础模型
            rf = clone(self.base_rf)
            rf.random_state = self.random_state + len(self.trees) + i

            # 使用bootstrap采样
            n_samples = len(X_batch)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_batch[indices]
            y_bootstrap = y_batch[indices]

            # 训练单棵树
            rf.fit(X_bootstrap, y_bootstrap)

            # 保存树
            self.trees.extend(rf.estimators_)

            if len(self.trees) >= self.n_estimators:
                break

        self.is_fitted = True

    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型还未训练")

        # 使用所有树进行预测
        predictions = np.zeros((len(X), 2))  # 二分类

        for tree in self.trees:
            pred_proba = tree.predict_proba(X)
            predictions += pred_proba

        predictions /= len(self.trees)

        return np.argmax(predictions, axis=1)

    def get_feature_importances(self):
        """获取特征重要性"""
        if not self.is_fitted:
            return None

        # 平均所有树的特征重要性
        importances = np.zeros(self.trees[0].n_features_in_)

        for tree in self.trees:
            importances += tree.feature_importances_

        importances /= len(self.trees)

        return importances


def train_rf_with_dataloader(
        dataset_path_dict,
        feature_names,
        batch_size=32,
        n_estimators=100,
        aggregation='mean',
        use_flow_labels=False,
        sample_ratio=1.0
):
    """
    使用DataLoader训练随机森林

    参数:
    - dataset_path_dict: 包含X_path, y_path, metadata_path的字典
    - feature_names: 特征名称列表
    - batch_size: 批量大小
    - n_estimators: 树的数量
    - aggregation: 聚合方法 ('mean', 'max', 'std', 'all')
    - use_flow_labels: 是否使用流级别标签（True）还是窗口级别标签（False）
    - sample_ratio: 采样比例（用于大数据集）
    """

    # 创建数据集和数据加载器
    dataset = WindowDataset(
        dataset_path_dict['X_path'],
        dataset_path_dict['y_path'],
        dataset_path_dict['metadata_path'],
        use_mmap=True
    )

    # 如果需要采样
    if sample_ratio < 1.0:
        n_samples = int(len(dataset) * sample_ratio)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 使用0避免多进程问题
        pin_memory=False
    )

    print(f"数据集大小: {len(dataset)} 个窗口")
    print(f"批量大小: {batch_size}")
    print(f"特征聚合方法: {aggregation}")
    print(f"使用标签: {'流级别' if use_flow_labels else '窗口级别'}")

    # 初始化随机森林
    if use_flow_labels:
        # 使用增量学习方法
        rf_model = IncrementalRandomForest(n_estimators=n_estimators)
    else:
        # 使用标准随机森林（需要先收集所有数据）
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

    # 收集训练数据
    X_train_list = []
    y_train_list = []

    print("\n处理数据批次...")
    for batch_idx, (X_batch, y_batch, window_labels) in enumerate(tqdm(dataloader)):
        # X_batch: (batch_size, window_size, num_features)
        # y_batch: (batch_size, window_size)
        # window_labels: (batch_size,)

        X_batch = X_batch.numpy()
        y_batch = y_batch.numpy()
        window_labels = window_labels.numpy()

        if use_flow_labels:
            # 使用流级别标签：展平窗口
            X_flat = X_batch.reshape(-1, X_batch.shape[-1])  # (batch_size * window_size, num_features)
            y_flat = y_batch.reshape(-1)  # (batch_size * window_size,)

            # 增量训练
            if isinstance(rf_model, IncrementalRandomForest):
                rf_model.partial_fit(X_flat, y_flat)
            else:
                X_train_list.append(X_flat)
                y_train_list.append(y_flat)

        else:
            # 使用窗口级别标签：聚合窗口特征
            if aggregation == 'mean':
                X_agg = np.mean(X_batch, axis=1)  # (batch_size, num_features)
            elif aggregation == 'max':
                X_agg = np.max(X_batch, axis=1)
            elif aggregation == 'std':
                X_agg = np.std(X_batch, axis=1)
            elif aggregation == 'all':
                # 组合多个统计量
                X_mean = np.mean(X_batch, axis=1)
                X_std = np.std(X_batch, axis=1)
                X_max = np.max(X_batch, axis=1)
                X_min = np.min(X_batch, axis=1)
                X_agg = np.hstack([X_mean, X_std, X_max, X_min])
            else:
                raise ValueError(f"未知的聚合方法: {aggregation}")

            X_train_list.append(X_agg)
            y_train_list.append(window_labels)

        # 定期清理内存
        if batch_idx % 100 == 0:
            gc.collect()

    # 如果使用标准随机森林，现在训练
    if not isinstance(rf_model, IncrementalRandomForest) and X_train_list:
        print("\n训练随机森林...")
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")

        rf_model.fit(X_train, y_train)

        # 清理内存
        del X_train_list, y_train_list
        gc.collect()

    # 获取特征重要性
    print("\n计算特征重要性...")
    if isinstance(rf_model, IncrementalRandomForest):
        feature_importances = rf_model.get_feature_importances()
    else:
        feature_importances = rf_model.feature_importances_

    # 如果使用了特征组合，需要调整特征名
    if aggregation == 'all' and not use_flow_labels:
        # 扩展特征名
        extended_features = []
        for suffix in ['_mean', '_std', '_max', '_min']:
            extended_features.extend([f + suffix for f in feature_names])
        feature_names_used = extended_features
    else:
        feature_names_used = feature_names

    # 获取特征重要性排序
    indices = np.argsort(feature_importances)[::-1]

    print(f"\nTop 20 最重要的特征:")
    print("-" * 50)

    selected_features = []
    selected_indices = []

    for i in range(min(20, len(feature_names_used))):
        idx = indices[i]
        selected_features.append(feature_names_used[idx])
        selected_indices.append(idx)
        print(f"{i + 1}. {feature_names_used[idx]}: {feature_importances[idx]:.4f}")

    return {
        'model': rf_model,
        'feature_importances': feature_importances,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'feature_names_used': feature_names_used
    }


def evaluate_rf_with_dataloader(rf_model, dataset_path_dict, batch_size=32, aggregation='mean'):
    """
    使用DataLoader评估随机森林
    """
    dataset = WindowDataset(
        dataset_path_dict['X_path'],
        dataset_path_dict['y_path'],
        dataset_path_dict['metadata_path'],
        use_mmap=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []

    print("评估模型...")
    for X_batch, y_batch, window_labels in tqdm(dataloader):
        X_batch = X_batch.numpy()
        window_labels = window_labels.numpy()

        # 聚合特征
        if aggregation == 'mean':
            X_agg = np.mean(X_batch, axis=1)
        elif aggregation == 'max':
            X_agg = np.max(X_batch, axis=1)
        elif aggregation == 'std':
            X_agg = np.std(X_batch, axis=1)
        elif aggregation == 'all':
            X_mean = np.mean(X_batch, axis=1)
            X_std = np.std(X_batch, axis=1)
            X_max = np.max(X_batch, axis=1)
            X_min = np.min(X_batch, axis=1)
            X_agg = np.hstack([X_mean, X_std, X_max, X_min])

        # 预测
        if hasattr(rf_model, 'predict'):
            batch_pred = rf_model.predict(X_agg)
        else:
            # 对于IncrementalRandomForest
            batch_pred = rf_model.predict(X_agg)

        y_true.extend(window_labels)
        y_pred.extend(batch_pred)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# 使用示例
if __name__ == "__main__":
    # 路径配置
    integrated_dir = "../cicids2017/integrated_windows"
    features_info_path = "../cicids2017/flow_windows/features_info.pkl"

    dataset_paths = {
        'X_path': f"{integrated_dir}/integrated_X_w200_s50.npy",
        'y_path': f"{integrated_dir}/integrated_y_w200_s50.npy",
        'metadata_path': f"{integrated_dir}/integrated_metadata_w200_s50.pkl"
    }

    # 加载特征名称
    with open(features_info_path, 'rb') as f:
        features_info = pickle.load(f)
    feature_names = features_info['features']

    print(f"总特征数: {len(feature_names)}")

    # 使用DataLoader训练随机森林
    result = train_rf_with_dataloader(
        dataset_paths,
        feature_names,
        batch_size=64,
        n_estimators=100,
        aggregation='all',  # 使用所有统计特征
        use_flow_labels=False,  # 使用窗口级别标签
        sample_ratio=0.1  # 只使用10%的数据进行演示
    )

    # 评估模型
    eval_result = evaluate_rf_with_dataloader(
        result['model'],
        dataset_paths,
        batch_size=64,
        aggregation='all'
    )

    # 保存结果
    save_path = f"{integrated_dir}/rf_dataloader_results.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'feature_importances': result['feature_importances'],
            'selected_features': result['selected_features'],
            'selected_indices': result['selected_indices'],
            'evaluation': eval_result
        }, f)

    print(f"\n结果已保存到: {save_path}")