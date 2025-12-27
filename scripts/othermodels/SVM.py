import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve, auc, confusion_matrix)
import time
import json
import os

# 导入数据加载模块
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


class SVMConfig:
    """SVM分类器配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 38  # 特征数量

        # SVM配置
        self.kernel = 'linear'  # 核函数: 'linear'（对高维数据速度快且效果好）
        self.C = 1.0  # 正则化参数
        self.probability = True  # 启用概率估计（用于计算PR-AUC）
        self.class_weight = 'balanced'  # 类别权重平衡
        self.max_iter = 1000  # 最大迭代次数

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.batch_size = 128

        # 其他
        self.save_dir = 'checkpoints_svm'


def flatten_time_series(X):
    """
    将3D时间序列数据展平为2D
    输入: (samples, seq_len, features)
    输出: (samples, seq_len * features)
    """
    return X.reshape(X.shape[0], -1)


# 采用窗口级标签进行训练的
def prepare_data_for_svm(data_loader):
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


def train_svm(configs):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    print("=" * 60)
    print("SVM TRAINING PIPELINE")
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

    # 4. 准备SVM所需的2D数据
    print("\n4. Preparing flattened data for SVM...")
    X_train, y_train = prepare_data_for_svm(train_loader)
    X_val, y_val = prepare_data_for_svm(val_loader)
    X_test, y_test = prepare_data_for_svm(test_loader)

    print(f"Training set shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Flattened feature dimension: {X_train.shape[1]}")

    # 5. 创建并训练SVM模型
    print("\n5. Training SVM model...")
    print(f"Kernel: {configs.kernel}, C: {configs.C}")
    print(f"Note: Linear kernel is much faster for high-dimensional data!")

    model = SVC(
        kernel=configs.kernel,
        C=configs.C,
        probability=configs.probability,
        class_weight=configs.class_weight,
        max_iter=configs.max_iter,
        verbose=True, # 启用概率估计
        cache_size=2000  # 增加缓存以加速训练
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Number of support vectors: {model.n_support_}")

    # 6. 验证集评估
    print("\n6. Validation set evaluation...")

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

    #获取概率以及对应的预测
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='macro')
    test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = auc(recall_curve, precision_curve)

    test_cm = confusion_matrix(y_test, test_preds)

    # 8. 打印最终结果
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

    # 9. 保存结果
    results = {
        'model': 'SVM',
        'config': {
            'kernel': configs.kernel,
            'C': configs.C,
            'max_iter': configs.max_iter,
            'seq_len': configs.seq_len,
            'features': configs.enc_in,
            'flattened_dim': X_train.shape[1]
        },
        'training_time': train_time,
        'n_support_vectors': int(model.n_support_[0] + model.n_support_[1]),
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
        }
    }

    with open(os.path.join(configs.save_dir, 'svm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {os.path.join(configs.save_dir, 'svm_results.json')}")

    return model, results


if __name__ == "__main__":
    # 创建配置
    configs = SVMConfig()

    # 开始训练
    model, results = train_svm(configs)

    print("\nSVM training completed!")