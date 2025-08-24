import json
import os

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from torch import nn, optim
from tqdm import tqdm

from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from models.ADLinear_supervised import Model
from units.trainer_valder import train_epoch,val_epoch

class Config:
    """DLinear分类器配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 45  # 特征数量（important_features_90.txt）

        # 模型配置
        self.pred_len = 38  # 特征提取维度
        self.num_classes = 2  # 二分类
        self.individual = False  # 共享权重

        # 训练配置
        self.epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 10  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'checkpoints'





def train_model(configs):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    # 打印操作
    print("=" * 60)
    print("DLinear-Classifier Training for CICIDS2017")
    print("=" * 60)
    print(f"Device: {configs.device}")
    print(f"Window size: {configs.seq_len}, Features: {configs.enc_in}")


    # 加载数据
    print("\n1. Loading data...")
    data_dir = "../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    configs.enc_in = len(metadata['feature_names'])
    print(f"Updated feature count: {configs.enc_in}")

    # 分割数据
    print("\n2. Splitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 创建数据加载器
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 创建模型
    print("\n4. Creating model...")
    model = Model(configs).to(configs.device)
    # sum(p.numel() for p in model.parameters() if p.requires_grad)：
    # 统计所有需要梯度的参数数量
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 创建Adam优化器，lr时学习率,weight_decay是L2正则系数，防止过拟合
    optimizer = optim.Adam(
        model.parameters(), lr=configs.learning_rate,
        weight_decay=configs.weight_decay)
    # 创建学习率调度器
    # 策略：当验证损失停止下降时自动降低学习率
    # mode='min'：监控指标需要最小化
    # factor=0.5：每次将学习率乘以0.5
    # patience=5：容忍5个epoch无改善
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)



    # 开启训练循环
    # 训练循环
    print("\n5. Training...")
    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(configs.epochs):
        # 训练模型
        train_loss,train_accuracy = (
            train_epoch(model,train_loader,criterion,optimizer,configs.device))

        # 验证模型
        val_loss,val_accuracy,val_precision,val_recall,val_f1 = (
            val_epoch(model,val_loader,criterion,configs.device))

        # 根据val_loss进行学习率调整
        scheduler.step(val_loss)

        # 记录历史
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_stats)

        # 打印进度
        print(f"Epoch {epoch + 1:3d}/{configs.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'configs': configs,
                'scaler': scaler,
                'best_val_f1': best_val_f1,
                'feature_names': metadata['feature_names']
            }
            torch.save(checkpoint, os.path.join(configs.save_dir, 'best_model.pth'))
            print(f"  → New best F1: {best_val_f1:.4f}, model saved!")
        else:
            patience_counter += 1

        # 早停，即模型已达到最佳训练效果
        # 早停
        if patience_counter >= configs.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # 保存训练历史
        with open(os.path.join(configs.save_dir, 'train_history.json'), 'w') as f:
            json.dump(train_history, f, indent=2)

        best_checkpoint = torch.load(os.path.join(configs.save_dir, 'best_model.pth'))
        model.load_state_dict(best_checkpoint['model_state_dict'])

        test_loss, test_acc, test_precision, test_recall, test_f1 = val_epoch(
            model, test_loader, criterion, configs.device
        )
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Best Validation F1: {best_val_f1:.4f}")
        print(f"Test Accuracy:      {test_acc:.4f}")
        print(f"Test Precision:     {test_precision:.4f}")
        print(f"Test Recall:        {test_recall:.4f}")
        print(f"Test F1:            {test_f1:.4f}")

        return model, train_history


if __name__ == "__main__":
    # 创建配置
    configs = Config()

    # 开始训练
    model, history = train_model(configs)

    print("\nTraining completed!")
