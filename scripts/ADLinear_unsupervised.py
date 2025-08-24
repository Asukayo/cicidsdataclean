import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from tqdm import tqdm

from dataprovider.provider_unsupervised import load_data, split_data_by_chronological_strategy, create_data_loaders
from models.Dlinear import Model


class Config:
    """ADLinear_unsupervised配置类"""

    def __init__(self):
        # 模型参数
        self.seq_len = CICIDS_WINDOW_SIZE  # 输入序列长度
        self.pred_len = CICIDS_WINDOW_SIZE  # 重构长度（与seq_len相同）
        self.enc_in = 45  # CICIDS2017特征数量
        self.individual = True  # 是否为每个特征单独建模

        # 训练参数
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 100
        self.patience = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for x, x_mark, y in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device)  # [batch_size, seq_len, features]

        optimizer.zero_grad()

        # 前向传播（重构）
        x_recon = model(x)  # [batch_size, seq_len, features]

        # 计算重构误差
        loss = criterion(x_recon, x)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x, x_mark, y in tqdm(val_loader, desc="Validation", leave=False):
            x = x.to(device)

            # 前向传播（重构）
            x_recon = model(x)

            # 计算重构误差
            loss = criterion(x_recon, x)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def calculate_threshold(model, val_loader, device, percentile=80):
    """基于验证集计算异常检测阈值"""
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for x, x_mark, y in val_loader:
            x = x.to(device)
            x_recon = model(x)

            # 计算每个样本的重构误差
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2))  # [batch_size]
            reconstruction_errors.extend(mse.cpu().numpy())

    # 使用百分位数作为阈值
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold, reconstruction_errors


def evaluate_model(model, test_loader, threshold, device):
    """评估模型异常检测性能"""
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for x, x_mark, y in test_loader:
            x = x.to(device)
            y_true.extend(y.cpu().numpy().flatten())

            # 计算重构误差作为异常分数
            x_recon = model(x)
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2))
            y_scores.extend(mse.cpu().numpy())

    # 基于阈值预测
    y_pred = (np.array(y_scores) > threshold).astype(int)

    # 计算评估指标
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_scores)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': threshold
    }


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('ADLinear Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主训练函数"""
    # 配置
    config = Config()
    print(f"使用设备: {config.device}")

    # 数据加载
    print("加载数据...")
    data_dir = "../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, window_size=CICIDS_WINDOW_SIZE, step_size=CICIDS_WINDOW_STEP)

    # 数据分割
    train_data, val_data, test_data = split_data_by_chronological_strategy(X, y, metadata)

    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, batch_size=config.batch_size
    )

    # 创建模型
    print("创建模型...")
    model = Model(config).to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("开始训练...")
    for epoch in range(config.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)

        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, config.device)

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'best_adlinear_model.pth')
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Best Val Loss: {best_val_loss:.6f}")

        if patience_counter >= config.patience:
            print(f"早停触发！在第 {epoch + 1} 轮停止训练")
            break

    # 加载最佳模型
    print("加载最佳模型进行评估...")
    checkpoint = torch.load('best_adlinear_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 计算阈值
    print("计算异常检测阈值...")
    threshold, val_errors = calculate_threshold(model, val_loader, config.device)
    #
    # threshold_90 = np.percentile(reconstruction_errors, 90)
    print(f"异常检测阈值 (95th percentile): {threshold:.6f}")

    # 测试集评估
    print("评估模型性能...")
    metrics = evaluate_model(model, test_loader, threshold, config.device)

    print("=== 最终结果 ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses)

    # 保存最终结果
    final_results = {
        'metrics': metrics,
        'threshold': threshold,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(final_results, 'adlinear_results.pth')

    print("训练完成！模型和结果已保存。")


if __name__ == "__main__":
    main()