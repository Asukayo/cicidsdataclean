import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve, auc, confusion_matrix)
import time
import json
import os

# 导入数据加载模块
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


class RFConfig:
    """Random Forest分类器配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 38  # 特征数量

        # Random Forest配置
        self.n_estimators = 50  # 树的数量
        self.max_depth = None  # 树的最大深度（限制深度可以加速训练和防止过拟合）
        self.min_samples_split = 5  # 分裂内部节点所需的最小样本数
        self.min_samples_leaf = 2  # 叶节点所需的最小样本数
        self.max_features = 'sqrt'  # 寻找最佳分裂时考虑的特征数量
        self.class_weight = 'balanced'  # 类别权重平衡
        self.n_jobs = -1  # 并行作业数（-1表示使用所有CPU核心）

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.batch_size = 128

        # 其他
        self.save_dir = 'checkpoints_rf'


def flatten_time_series(X):
    """
    将3D时间序列数据展平为2D
    输入: (samples, seq_len, features)
    输出: (samples, seq_len * features)
    """
    return X.reshape(X.shape[0], -1)


def prepare_data_for_rf(data_loader):
    """从PyTorch DataLoader中提取数据并展平"""
    X_list = []
    y_list = []

    for batch_X, batch_X_mark, batch_y in data_loader:
        X_list.append(batch_X.numpy())
        y_list.append(batch_y.squeeze().numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # 展平时间序列维度
    X_flat = flatten_time_series(X)

    return X_flat, y


def train_random_forest(configs):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    print("=" * 60)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. Loading data...")
    data_dir = "../../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    configs.enc_in = len(metadata['feature_names'])
    print(f"Feature count: {configs.enc_in}")
    print(f"Original data shape: {X.shape}")

    # 2. 分割数据
    print("\n2. Splitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 3. 创建数据加载器（用于标准化）
    print("\n3. Creating data loaders and scaling...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 4. 准备Random Forest所需的2D数据
    print("\n4. Preparing flattened data for Random Forest...")
    X_train, y_train = prepare_data_for_rf(train_loader)
    X_val, y_val = prepare_data_for_rf(val_loader)
    X_test, y_test = prepare_data_for_rf(test_loader)

    print(f"Training set shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Flattened feature dimension: {X_train.shape[1]}")

    # 5. 创建并训练Random Forest模型
    print("\n5. Training Random Forest model...")
    print(f"n_estimators: {configs.n_estimators}, max_depth: {configs.max_depth}")
    print(f"max_features: {configs.max_features}, n_jobs: {configs.n_jobs}")

    model = RandomForestClassifier(
        n_estimators=configs.n_estimators,
        max_depth=configs.max_depth,
        min_samples_split=configs.min_samples_split,
        min_samples_leaf=configs.min_samples_leaf,
        max_features=configs.max_features,
        class_weight=configs.class_weight,
        n_jobs=configs.n_jobs,
        random_state=42,
        verbose=1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    # 6. 验证集评估
    print("\n6. Validation set evaluation...")

    # 获取正类概率，直接预测类别
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]


    val_acc = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, average='macro')
    val_recall = recall_score(y_val, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)

    precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_probs)
    val_pr_auc = auc(recall_curve, precision_curve)

    print(f"Validation Accuracy:  {val_acc:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall:    {val_recall:.4f}")
    print(f"Validation F1:        {val_f1:.4f}")
    print(f"Validation PR-AUC:    {val_pr_auc:.4f}")

    # 7. 测试集评估
    print("\n7. Test set evaluation...")
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, )
    test_recall = recall_score(y_test, test_preds,  zero_division=0)
    test_f1 = f1_score(y_test, test_preds,zero_division=0)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = auc(recall_curve, precision_curve)

    test_cm = confusion_matrix(y_test, test_preds)

    # 8. 特征重要性分析（可选）
    print("\n8. Feature importance analysis...")
    feature_importance = model.feature_importances_
    # 获取Top 10重要特征
    top_k = 10
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    print(f"\nTop {top_k} Most Important Features (by flattened index):")
    for i, idx in enumerate(top_indices, 1):
        # 计算原始特征索引和时间步
        feature_idx = idx % configs.enc_in
        time_step = idx // configs.enc_in
        print(f"  {i}. Index {idx} (time={time_step}, feature={feature_idx}): {feature_importance[idx]:.6f}")

    # 9. 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")
    print(f"Test PR-AUC:        {test_pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print(f"  TN: {test_cm[0, 0]}  FP: {test_cm[0, 1]}")
    print(f"  FN: {test_cm[1, 0]}  TP: {test_cm[1, 1]}")

    # 10. 保存结果
    results = {
        'model': 'Random Forest',
        'config': {
            'n_estimators': configs.n_estimators,
            'max_depth': configs.max_depth,
            'max_features': configs.max_features,
            'seq_len': configs.seq_len,
            'features': configs.enc_in,
            'flattened_dim': X_train.shape[1]
        },
        'training_time': train_time,
        'validation': {
            'accuracy': float(val_acc),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'pr_auc': float(val_pr_auc)
        },
        'test': {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'pr_auc': float(test_pr_auc),
            'confusion_matrix': test_cm.tolist()
        },
        'feature_importance': {
            'top_10_indices': top_indices.tolist(),
            'top_10_values': [float(feature_importance[i]) for i in top_indices]
        }
    }

    with open(os.path.join(configs.save_dir, 'rf_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {os.path.join(configs.save_dir, 'rf_results.json')}")

    return model, results


if __name__ == "__main__":
    # 创建配置
    configs = RFConfig()

    # 开始训练
    model, results = train_random_forest(configs)

    print("\nRandom Forest training completed!")