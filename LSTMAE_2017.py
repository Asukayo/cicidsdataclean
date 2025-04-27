import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
from tqdm import tqdm
import gc  # 添加垃圾回收模块

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
BATCH_SIZE = 8  # 减小批量大小以减少内存使用
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# 自定义Dataset类，避免一次性加载所有数据到内存
class CICIDSDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# 时序自编码器模型定义
class TSAutoEncoder(nn.Module):
    def __init__(self, seq_len, feature_dim, hidden_dim=64, latent_dim=32):
        super(TSAutoEncoder, self).__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 编码器 - 使用LSTM处理时序数据
        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 编码器 - 将LSTM输出转换为潜在表示
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        # 解码器 - 从潜在表示开始
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 解码器 - 使用LSTM生成序列
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 最终输出层 - 恢复原始特征维度
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, feature_dim]
        batch_size = x.size(0)

        # 编码
        _, (h_n, _) = self.encoder_lstm(x)
        # 使用最后一层LSTM的隐藏状态
        h_n = h_n[-1]  # 形状: [batch_size, hidden_dim]

        # 转换为潜在表示
        z = self.encoder_fc(h_n)  # 形状: [batch_size, latent_dim]

        # 从潜在表示解码
        h_decoded = self.decoder_fc(z)  # 形状: [batch_size, hidden_dim]

        # 重复潜在表示以匹配序列长度
        h_decoded = h_decoded.unsqueeze(1).repeat(1, self.seq_len, 1)  # 形状: [batch_size, seq_len, hidden_dim]

        # 解码序列
        decoded_seq, _ = self.decoder_lstm(h_decoded)  # 形状: [batch_size, seq_len, hidden_dim]

        # 输出层 - 恢复原始特征维度
        output = self.output_layer(decoded_seq)  # 形状: [batch_size, seq_len, feature_dim]

        return output

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1]
        z = self.encoder_fc(h_n)
        return z


# 实用函数：计算重构误差
def compute_reconstruction_error(model, data_loader, device):
    model.eval()
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 计算每个样本的MSE重构误差
            errors = torch.mean(torch.mean((outputs - inputs) ** 2, dim=2), dim=1).cpu().numpy()
            all_errors.extend(errors)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_errors), np.array(all_labels)


# 加载数据函数
def load_cicids_data(data_dir, first_day_name, second_day_name, window_size=1000, step_size=100):
    """
    加载CICIDS数据集的两天数据

    参数:
    - data_dir: 包含处理后窗口数据的目录
    - first_day_name: 第一天的数据目录名
    - second_day_name: 第二天的数据目录名

    返回:
    - train_data: 第一天的窗口特征数据
    - train_labels: 第一天的窗口标签（0=正常，1=异常）
    - test_data: 第二天的窗口特征数据
    - test_labels: 第二天的窗口标签（0=正常，1=异常）
    - feature_names: 特征名称列表
    """
    # 加载第一天数据
    day1_dir = os.path.join(data_dir, first_day_name)
    X_train = np.load(os.path.join(day1_dir, f'X_windows_w{window_size}_s{step_size}.npy'))

    # 加载元数据以获取特征名称和窗口标签信息
    with open(os.path.join(day1_dir, f'metadata_w{window_size}_s{step_size}.pkl'), 'rb') as f:
        day1_metadata = pickle.load(f)

    # 获取特征名称
    feature_names = day1_metadata['feature_names']

    # 从元数据中提取窗口级别的标签（0=正常，1=异常）
    train_labels = np.array([m['is_malicious'] for m in day1_metadata['window_metadata']])

    # 加载第二天数据
    day2_dir = os.path.join(data_dir, second_day_name)
    X_test = np.load(os.path.join(day2_dir, f'X_windows_w{window_size}_s{step_size}.npy'))

    with open(os.path.join(day2_dir, f'metadata_w{window_size}_s{step_size}.pkl'), 'rb') as f:
        day2_metadata = pickle.load(f)

    # 从元数据中提取窗口级别的标签
    test_labels = np.array([m['is_malicious'] for m in day2_metadata['window_metadata']])

    print(
        f"加载完成: {first_day_name} 包含 {len(train_labels)} 个窗口, {second_day_name} 包含 {len(test_labels)} 个窗口")
    print(f"特征维度: {X_train.shape}")

    return X_train, train_labels, X_test, test_labels, feature_names


# 训练函数
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    训练自编码器模型

    参数:
    - model: 自编码器模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - epochs: 训练轮数
    - learning_rate: 学习率
    - device: 训练设备

    返回:
    - 训练历史
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            data = data.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, data)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 显式清理缓存，减少内存占用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item()

                # 显式清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss /= len(val_loader)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 打印进度
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 在每个epoch后进行一次强制垃圾回收
        gc.collect()

    return history


# 评估函数
def evaluate_model(model, test_loader, y_test, device):
    """
    评估模型性能

    参数:
    - model: 训练好的自编码器模型
    - test_loader: 测试数据加载器
    - y_test: 测试标签
    - device: 设备
    """
    # 计算重构误差
    reconstruction_errors, _ = compute_reconstruction_error(model, test_loader, device)

    # 使用重构误差作为异常分数
    y_score = reconstruction_errors

    # 计算ROC和PR曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # 绘制ROC曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, 'g', label=f'AP = {pr_auc:.2f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.show()

    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")

    # 找出最佳阈值（使用F1分数）
    thresholds = np.linspace(min(y_score), max(y_score), 100)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 使用最佳阈值进行预测
    y_pred = (y_score >= best_threshold).astype(int)

    # 计算混淆矩阵
    TP = np.sum((y_pred == 1) & (y_test == 1))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))

    # 计算各种评估指标
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"\n最佳阈值: {best_threshold:.6f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 分析错误分类的窗口
    error_indices = np.where(y_pred != y_test)[0]
    print(f"\n错误分类的窗口数量: {len(error_indices)}")

    if len(error_indices) > 0:
        # 显示一些错误分类窗口的重构误差
        print("\n错误分类窗口的重构误差:")
        for i in range(min(5, len(error_indices))):
            idx = error_indices[i]
            print(f"窗口 {idx}: 真实标签 = {y_test[idx]}, 预测标签 = {y_pred[idx]}, 重构误差 = {y_score[idx]:.6f}")

    # 绘制重构误差分布
    plt.figure(figsize=(10, 6))

    benign_errors = y_score[y_test == 0]
    malicious_errors = y_score[y_test == 1]

    plt.hist(benign_errors, bins=50, alpha=0.7, label='正常', density=True)
    plt.hist(malicious_errors, bins=50, alpha=0.7, label='异常', density=True)
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'阈值 = {best_threshold:.4f}')

    plt.xlabel('重构误差')
    plt.ylabel('密度')
    plt.title('重构误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reconstruction_error_distribution.png')
    plt.show()

    return y_score, best_threshold


def main():
    # 数据目录设置
    data_dir = "./cicids2017/flow_windows_w1000_s100"  # 修改为你的数据目录
    first_day = "Monday-WorkingHours"
    second_day = "Tuesday-WorkingHours"
    window_size = 1000
    step_size = 100

    # 加载数据
    X_train, y_train, X_test, y_test, feature_names = load_cicids_data(
        data_dir, first_day, second_day,
        window_size=window_size, step_size=step_size
    )

    # 分离正常样本进行训练 (仅使用正常样本进行训练)
    benign_indices = np.where(y_train == 0)[0]
    X_train_benign = X_train[benign_indices]

    # 获取序列长度和特征维度
    _, seq_len, feature_dim = X_train.shape
    print(f"序列长度: {seq_len}, 特征维度: {feature_dim}")

    # 准备数据加载器 - 使用自定义Dataset类来减少内存使用
    # 训练集（只使用正常样本）
    train_dataset = CICIDSDataset(X_train_benign, np.zeros(len(X_train_benign)))

    # 将训练集拆分为训练和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建数据加载器，添加num_workers以加速数据加载，pin_memory=True以加速GPU传输
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 测试集（包含正常和异常样本）
    test_dataset = CICIDSDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 初始化模型 - 减小模型复杂度以减少内存使用
    model = TSAutoEncoder(seq_len=seq_len, feature_dim=feature_dim, hidden_dim=32, latent_dim=16).to(DEVICE)
    print(model)

    # 尝试输出模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")

    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png')
    plt.show()

    # 评估模型
    y_score, threshold = evaluate_model(model, test_loader, y_test, DEVICE)

    # 保存模型和阈值
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_len': seq_len,
        'feature_dim': feature_dim,
        'threshold': threshold
    }, 'cicids_ts_autoencoder.pth')

    print("模型已保存至 'cicids_ts_autoencoder.pth'")


if __name__ == "__main__":
    main()