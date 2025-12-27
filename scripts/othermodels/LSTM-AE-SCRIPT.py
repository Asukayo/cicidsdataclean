import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve, auc
from tqdm import tqdm

# 导入LSTM自编码器模型
from LSTM_AE import LSTM_Autoencoder

from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


class Config:
    """LSTM-AE弱监督异常检测配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.input_dim = 38  # 特征数量（等于enc_in）

        # LSTM-AE模型配置
        self.hidden_dim = 128  # LSTM隐藏层维度
        self.latent_dim = 64  # 潜在空间维度
        self.num_layers = 2  # LSTM层数
        self.dropout = 0.2

        # 弱监督配置
        self.weak_supervised = True  # 是否使用弱监督（利用窗口标签）
        self.loss_strategy = 'weighted'  # 'weighted', 'normal_only', 'contrastive'
        self.anomaly_weight = 0.5  # 异常窗口损失的权重
        self.margin = 0.1  # 对比学习的margin（仅在contrastive模式下）

        # 训练配置
        self.epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.patience = 10  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # 异常检测配置
        self.threshold_method = 'percentile'  # 'f1', 'percentile'
        self.percentile = 99  # 如果使用percentile方法

        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch_ae_weakly_supervised(model, train_loader, optimizer, configs, device='cuda'):
    """弱监督训练一个epoch"""
    model.train()
    total_loss = 0

    # 统计信息
    normal_loss_sum = 0
    anomaly_loss_sum = 0
    normal_count = 0
    anomaly_count = 0

    criterion_none = nn.MSELoss(reduction='none')
    criterion_mean = nn.MSELoss()

    for batch_X, batch_X_mark, batch_y in tqdm(train_loader, desc="Training"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.squeeze().to(device)

        # 前向传播
        reconstructed, latent = model(batch_X)

        if configs.weak_supervised:
            if configs.loss_strategy == 'weighted':
                mse = criterion_none(reconstructed, batch_X).mean(dim=(1, 2))
                normal_mask = (batch_y == 0)
                anomaly_mask = (batch_y == 1)

                if normal_mask.sum() > 0:
                    loss_normal = mse[normal_mask].mean()
                    normal_loss_sum += loss_normal.item() * normal_mask.sum().item()
                    normal_count += normal_mask.sum().item()
                else:
                    loss_normal = torch.tensor(0.0, device=device)

                if anomaly_mask.sum() > 0:
                    loss_anomaly = -torch.log(mse[anomaly_mask] + 1e-8).mean()
                    anomaly_loss_sum += mse[anomaly_mask].mean().item() * anomaly_mask.sum().item()
                    anomaly_count += anomaly_mask.sum().item()
                else:
                    loss_anomaly = torch.tensor(0.0, device=device)

                loss = loss_normal + configs.anomaly_weight * loss_anomaly

            elif configs.loss_strategy == 'normal_only':
                normal_mask = (batch_y == 0)
                if normal_mask.sum() == 0:
                    continue
                batch_X_normal = batch_X[normal_mask]
                reconstructed_normal = reconstructed[normal_mask]
                loss = criterion_mean(reconstructed_normal, batch_X_normal)
                normal_loss_sum += loss.item() * normal_mask.sum().item()
                normal_count += normal_mask.sum().item()

            elif configs.loss_strategy == 'contrastive':
                mse = criterion_none(reconstructed, batch_X).mean(dim=(1, 2))
                normal_mask = (batch_y == 0)
                anomaly_mask = (batch_y == 1)

                if normal_mask.sum() > 0:
                    loss_normal = mse[normal_mask].mean()
                    normal_loss_sum += loss_normal.item() * normal_mask.sum().item()
                    normal_count += normal_mask.sum().item()
                else:
                    loss_normal = torch.tensor(0.0, device=device)

                if anomaly_mask.sum() > 0:
                    loss_anomaly = torch.clamp(configs.margin - mse[anomaly_mask], min=0).mean()
                    anomaly_loss_sum += mse[anomaly_mask].mean().item() * anomaly_mask.sum().item()
                    anomaly_count += anomaly_mask.sum().item()
                else:
                    loss_anomaly = torch.tensor(0.0, device=device)

                loss = loss_normal + loss_anomaly

            else:
                raise ValueError(f"Unknown loss strategy: {configs.loss_strategy}")

        else:
            loss = criterion_mean(reconstructed, batch_X)
            with torch.no_grad():
                mse = criterion_none(reconstructed, batch_X).mean(dim=(1, 2))
                normal_mask = (batch_y == 0)
                anomaly_mask = (batch_y == 1)

                if normal_mask.sum() > 0:
                    normal_loss_sum += mse[normal_mask].mean().item() * normal_mask.sum().item()
                    normal_count += normal_mask.sum().item()

                if anomaly_mask.sum() > 0:
                    anomaly_loss_sum += mse[anomaly_mask].mean().item() * anomaly_mask.sum().item()
                    anomaly_count += anomaly_mask.sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    stats = {
        'avg_loss': avg_loss,
        'normal_avg_recon_error': normal_loss_sum / normal_count if normal_count > 0 else 0,
        'anomaly_avg_recon_error': anomaly_loss_sum / anomaly_count if anomaly_count > 0 else 0,
        'normal_count': normal_count,
        'anomaly_count': anomaly_count
    }

    return avg_loss, stats


def val_epoch_ae(model, val_loader, device='cuda'):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    reconstruction_errors = []
    labels = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.squeeze()

            reconstructed, latent = model(batch_X)

            loss = criterion(reconstructed, batch_X)
            total_loss += loss.item()

            # 计算重构误差
            batch_errors = torch.mean((reconstructed - batch_X) ** 2, dim=(1, 2)).cpu().numpy()
            reconstruction_errors.extend(batch_errors)
            labels.extend(batch_y.numpy())

    avg_loss = total_loss / len(val_loader)
    reconstruction_errors = np.array(reconstruction_errors)
    labels = np.array(labels)

    return avg_loss, reconstruction_errors, labels


def find_optimal_threshold(reconstruction_errors, labels, method='f1', percentile=95):
    """找到最优的异常检测阈值"""
    if method == 'f1':
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}

        thresholds = np.percentile(reconstruction_errors, np.linspace(50, 99.9, 100))

        for threshold in thresholds:
            predictions = (reconstruction_errors > threshold).astype(int)
            f1 = f1_score(labels, predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'accuracy': accuracy_score(labels, predictions),
                    'precision': precision_score(labels, predictions, zero_division=0),
                    'recall': recall_score(labels, predictions, zero_division=0),
                    'f1': f1
                }

        return best_threshold, best_metrics

    elif method == 'percentile':
        threshold = np.percentile(reconstruction_errors, percentile)
        predictions = (reconstruction_errors > threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

        return threshold, metrics

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def test_with_threshold(model, test_loader, threshold, device='cuda'):
    """使用给定阈值在测试集上评估"""
    model.eval()
    reconstruction_errors = []
    labels = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.squeeze()

            reconstructed, latent = model(batch_X)

            batch_errors = torch.mean((reconstructed - batch_X) ** 2, dim=(1, 2)).cpu().numpy()
            reconstruction_errors.extend(batch_errors)
            labels.extend(batch_y.numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    labels = np.array(labels)

    predictions = (reconstruction_errors > threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, reconstruction_errors)
    pr_auc = auc(recall_curve, precision_curve)

    cm = confusion_matrix(labels, predictions)

    return accuracy, precision, recall, f1, pr_auc, cm


def train_model(configs):
    """主训练函数"""
    # 加载数据
    data_dir = "../../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    configs.input_dim = len(metadata['feature_names'])

    # 分割数据
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 创建模型
    model = LSTM_Autoencoder(
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        latent_dim=configs.latent_dim,
        num_layers=configs.num_layers,
        dropout=configs.dropout
    ).to(configs.device)

    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    best_val_f1 = 0
    best_threshold = None
    best_model_state = None
    patience_counter = 0

    for epoch in range(configs.epochs):
        # 训练
        train_loss, train_stats = train_epoch_ae_weakly_supervised(
            model, train_loader, optimizer, configs, configs.device
        )

        # 验证
        val_loss, val_errors, val_labels = val_epoch_ae(model, val_loader, configs.device)

        # 寻找最优阈值
        threshold, val_metrics = find_optimal_threshold(
            val_errors, val_labels, method=configs.threshold_method
        )

        # 学习率调整
        scheduler.step(val_loss)

        # 打印训练信息
        print(f"\nEpoch {epoch + 1:3d}/{configs.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        if configs.weak_supervised:
            print(f"    ├─ Normal Recon Error:  {train_stats['normal_avg_recon_error']:.6f}")
            print(f"    └─ Anomaly Recon Error: {train_stats['anomaly_avg_recon_error']:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Val Metrics: Acc={val_metrics['accuracy']:.4f} | Pre={val_metrics['precision']:.4f} | "
              f"Rec={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_threshold = threshold
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best Val F1: {val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience: {patience_counter}/{configs.patience})")

        # 早停
        if patience_counter >= configs.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型并测试
    model.load_state_dict(best_model_state)
    test_acc, test_precision, test_recall, test_f1, test_pr_auc, test_cm = test_with_threshold(
        model, test_loader, best_threshold, configs.device
    )

    # 打印最终测试结果
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:   {test_acc:.4f}")
    print(f"Precision:  {test_precision:.4f}")
    print(f"Recall:     {test_recall:.4f}")
    print(f"F1:         {test_f1:.4f}")
    print(f"PR-AUC:     {test_pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print("=" * 60)

    return model, best_threshold


if __name__ == "__main__":
    # 创建配置
    configs = Config()

    # 开始训练
    model, threshold = train_model(configs)