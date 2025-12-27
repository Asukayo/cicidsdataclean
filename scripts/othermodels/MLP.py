import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

# 导入MLP模型
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MLP模型用于时间序列异常检测
    将时间序列展平后通过多层感知机进行分类
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 100
        self.enc_in = configs.enc_in  # 38
        self.num_classes = configs.num_classes  # 2

        # 展平后的输入维度
        self.input_dim = self.seq_len * self.enc_in  # 100 * 38 = 3800

        # MLP结构：3层全连接网络
        self.mlp = nn.Sequential(
            # 第一层: 3800 -> 512
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 第二层: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 输出层: 256 -> 2
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, enc_in)
        Returns:
            output: (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # 展平时间序列
        # (batch_size, seq_len, enc_in) -> (batch_size, seq_len * enc_in)
        x = x.reshape(batch_size, -1)

        # MLP分类
        output = self.mlp(x)

        return output

from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from scripts.units.trainer_valder import train_epoch, val_epoch


class Config:
    """MLP分类器配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 38  # 特征数量

        # 模型配置
        self.num_classes = 2  # 二分类

        # 训练配置
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.001  # MLP使用标准学习率
        self.weight_decay = 1e-5
        self.patience = 10  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.1

        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'checkpoints_mlp'


def train_model(configs):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    # 加载数据
    data_dir = "../../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    configs.enc_in = len(metadata['feature_names'])
    print(f"Updated feature count: {configs.enc_in}")

    # 分割数据
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 创建数据加载器
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 创建模型
    print("\n4. Creating MLP model...")
    model = Model(configs).to(configs.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    print("\n5. Training...")
    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(configs.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, configs.device)

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_epoch(
            model, val_loader, criterion, configs.device
        )

        # 学习率调整
        scheduler.step(val_loss)

        # 记录历史
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_stats)

        # 打印进度
        print(f"Epoch {epoch + 1:3d}/{configs.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

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

        # 早停
        if patience_counter >= configs.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 保存训练历史
    with open(os.path.join(configs.save_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    # 测试最佳模型
    print("\n6. Testing best model...")
    best_checkpoint = torch.load(os.path.join(configs.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    from scripts.units.trainer_valder import test_with_detailed_metrics

    test_loss, test_acc, test_precision, test_recall, test_f1, test_pr_auc, test_cm = test_with_detailed_metrics(
        model, test_loader, criterion, configs.device
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS (MLP)")
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

    return model, train_history


if __name__ == "__main__":
    # 创建配置
    configs = Config()

    # 开始训练
    model, history = train_model(configs)

    print("\nMLP Training completed!")