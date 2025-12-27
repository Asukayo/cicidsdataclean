import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """裁剪模块，用于因果卷积"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN的基本时间块，包含两层因果卷积"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 第一层卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将两层卷积组合成一个序列
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 残差连接的1x1卷积（当输入输出通道数不同时）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, seq_len)
        Returns:
            out: (batch_size, channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN网络，由多个TemporalBlock堆叠而成"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: 输入通道数（特征维度）
            num_channels: 每层的通道数列表，例如 [64, 64, 128, 128]
            kernel_size: 卷积核大小
            dropout: dropout概率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长的膨胀率: 1, 2, 4, 8, ...
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # 计算padding以保持序列长度
            padding = (kernel_size - 1) * dilation_size

            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        Returns:
            out: (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class Model(nn.Module):
    """
    TCN模型用于时间序列异常检测
    使用因果卷积和膨胀卷积捕获长期时序依赖
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 100
        self.enc_in = configs.enc_in  # 38
        self.num_classes = configs.num_classes  # 2

        # TCN配置
        # 通道数逐层增加: [64, 64, 128, 128]
        # 4层结构，感受野 = (kernel_size - 1) * sum(2^i) + 1
        # 对于kernel_size=3: 感受野 = 2 * (1+2+4+8) + 1 = 31
        self.num_channels = [64, 64, 128, 256]
        self.kernel_size = 3
        self.dropout = 0.2

        # TCN网络
        self.tcn = TemporalConvNet(
            num_inputs=self.enc_in,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )

        # 全局平均池化 + 分类层
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.num_channels[-1], self.num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, enc_in)
        Returns:
            output: (batch_size, num_classes)
        """
        # 转换维度以适配TCN: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len)

        # TCN前向传播
        # tcn_out: (batch_size, num_channels[-1], seq_len)
        tcn_out = self.tcn(x)

        # 全局平均池化: 对时间维度取平均
        # pooled: (batch_size, num_channels[-1])
        pooled = torch.mean(tcn_out, dim=2)

        # 分类
        output = self.fc(pooled)

        return output




from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from scripts.units.trainer_valder import train_epoch, val_epoch


class Config:
    """TCN分类器配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 38  # 特征数量

        # 模型配置
        self.num_classes = 2  # 二分类

        # 训练配置
        self.epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.001
        # TCN使用标准学习率
        self.weight_decay = 1e-5
        self.patience = 10  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.1

        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'checkpoints_tcn'


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
    print("\n4. Creating TCN model...")
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
    print("FINAL RESULTS (TCN)")
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

    print("\nTCN Training completed!")