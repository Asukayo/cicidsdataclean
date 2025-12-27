"""
TreeMIL for CICIDS2017 Network Traffic Anomaly Detection
基于弱监督学习的时间序列异常检测

训练策略:
- 训练阶段:使用序列级标签(窗口级)
- 测试阶段:评估序列级和点级性能

BUG FIX: 修复了refer_points函数中的索引越界问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)

# 导入数据加载模块
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


# ==================== TreeMIL模型组件 ====================

def get_mask(seq_len, ary_size, inner_size):
    """
    构建N叉树的mask和尺寸信息
    """
    # 计算实际可以池化的最大深度
    depth = 1
    current_size = seq_len
    while current_size >= ary_size:
        current_size = current_size // ary_size
        depth += 1

    all_size = []
    effe_size = []

    # 计算每层的节点数
    current_size = seq_len
    for i in range(depth):
        effe_size.append(current_size)
        all_size.append(current_size)
        current_size = current_size // ary_size

    # 创建mask
    total_nodes = sum(all_size)
    mask = torch.ones(1, 1, total_nodes, total_nodes)

    return mask, all_size, effe_size


def refer_points(effe_size, ary_size):
    """
    生成每个时间点对应的多尺度节点索引

    Args:
        effe_size: 每层的有效节点数 [L_S, L_{S-1}, ..., L_1]
        ary_size: N叉树分支数

    Returns:
        indexes: [T, depth] 每个时间点在各层的节点索引

    FIX: 添加了索引边界检查,确保parent_idx不超出effe_size[level]的范围
    """
    T = effe_size[0]  # 叶节点数(时间步数)
    depth = len(effe_size)

    indexes = []
    for t in range(T):
        point_indexes = [t]  # 叶节点索引

        # 计算在每一层的父节点索引
        for level in range(1, depth):
            parent_idx = int(t / (ary_size ** level))

            # *** FIX: 确保parent_idx在有效范围内 ***
            # 由于整数除法的特性,对于某些时间点,parent_idx可能超出该层的节点数
            parent_idx = min(parent_idx, effe_size[level] - 1)

            # 加上前面层的节点总数作为偏移
            offset = sum(effe_size[:level])
            point_indexes.append(offset + parent_idx)

        indexes.append(point_indexes)

    return torch.LongTensor(indexes)  # [T, depth]


class MaxPooling_Construct(nn.Module):
    """使用MaxPooling构建N叉树"""

    def __init__(self, d_model, depth, ary_size):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.ary_size = ary_size

        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=ary_size, stride=ary_size)
            for _ in range(depth - 1)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            out: [B, T+T/N+T/N^2+..., D] 多尺度特征
        """
        x = x.transpose(1, 2)  # [B, D, T]

        outputs = [x.transpose(1, 2)]  # 存储每层的输出

        for pool in self.pools:
            if x.size(2) < self.ary_size:
                break
            x = pool(x)
            outputs.append(x.transpose(1, 2))

        # 拼接所有层的特征
        return torch.cat(outputs, dim=1)  # [B, sum(L_i), D]


class DataEmbedding(nn.Module):
    """数据嵌入层:1D卷积 + 位置编码"""

    def __init__(self, input_size, d_model, seq_len):
        super().__init__()
        self.value_embedding = nn.Conv1d(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )

        # 位置编码
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, T, D_in]
        Returns:
            [B, T, d_model]
        """
        x = x.transpose(1, 2)  # [B, D_in, T]
        x = self.value_embedding(x)  # [B, d_model, T]
        x = x.transpose(1, 2)  # [B, T, d_model]

        # 添加位置编码
        x = x + self.pe[:x.size(1), :]

        return x


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner_hid, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D]
            mask: attention mask
        Returns:
            x: [B, N, D]
            attn: attention weights
        """
        # Self-attention
        residual = x
        x, attn = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(residual + x)

        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)

        return x, attn


class TreeMIL(nn.Module):
    """
    TreeMIL: Tree-based Multi-Instance Learning for Time Series Anomaly Detection
    """

    def __init__(self, config):
        super().__init__()

        # 配置参数
        self.input_size = config.enc_in
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.ary_size = config.ary_size
        self.num_classes = config.num_classes

        # 构建N叉树结构
        self.mask, self.all_size, self.effe_size = get_mask(
            self.seq_len, self.ary_size, config.inner_size
        )
        self.depth = len(self.all_size)
        self.indexes = refer_points(self.effe_size, self.ary_size)

        # 数据嵌入
        self.embedding = DataEmbedding(self.input_size, self.d_model, self.seq_len)

        # N叉树构建(使用MaxPooling)
        self.conv_layers = MaxPooling_Construct(
            self.d_model, self.depth, self.ary_size
        )

        # Transformer编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                d_inner_hid=config.d_inner_hid,
                n_head=config.n_head,
                d_k=config.d_k,
                d_v=config.d_v,
                dropout=config.dropout
            ) for _ in range(config.n_layer)
        ])

        # 异常分数计算器
        self.scorenet = nn.Linear(self.d_model, 1)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, D_in]
        Returns:
            out: [B, L1+L2+...+Ls, d_model] 多尺度特征
        """
        # 1. 数据嵌入
        seq_enc = self.embedding(x)  # [B, T, d_model]

        # 2. 构建N叉树
        seq_enc = self.conv_layers(seq_enc)  # [B, sum(all_size), d_model]

        # 3. Transformer特征提取
        mask = self.mask.to(x.device)
        for layer in self.layers:
            seq_enc, _ = layer(seq_enc, mask=None)

        # 4. 提取有效节点特征
        out_list = []
        for i in range(len(self.all_size)):
            start_idx = sum(self.all_size[:i])
            end_idx = start_idx + self.effe_size[i]
            out_list.append(seq_enc[:, start_idx:end_idx, :])

        out = torch.cat(out_list, dim=1)  # [B, sum(effe_size), d_model]

        return out

    def get_scores(self, x):
        """
        计算序列级和点级异常分数

        Args:
            x: [B, T, D_in]
        Returns:
            ret: dict包含wscore(序列级)和dscore(点级)
        """
        ret = {}
        out = self.forward(x)  # [B, sum(L_i), d_model]

        # 序列级分数(全局池化)
        pooled = torch.max(out, dim=1)[0]  # [B, d_model]
        wscore = torch.sigmoid(self.scorenet(pooled).squeeze(dim=1))  # [B]
        ret['wscore'] = wscore

        # 点级分数(时间维度池化)
        B, _, D = out.size()
        T = self.effe_size[0]

        # 为每个时间点聚合多尺度特征
        indexes = self.indexes.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, D).to(x.device)  # [B, T, depth, D]
        indexes = indexes.view(B, -1, D)  # [B, T*depth, D]

        all_enc = torch.gather(out, 1, indexes)  # [B, T*depth, D]
        all_enc = all_enc.view(B, T, -1, D)  # [B, T, depth, D]
        h = torch.mean(all_enc, dim=2)  # [B, T, D]

        dscore = torch.sigmoid(self.scorenet(h).squeeze(dim=2))  # [B, T]
        ret['dscore'] = dscore

        return ret


# ==================== 配置类 ====================

class Config:
    """TreeMIL配置"""

    def __init__(self):
        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.enc_in = 38  # 特征数量

        # TreeMIL树结构配置
        self.ary_size = 2  # 二叉树
        self.inner_size = 3  # 邻居节点数

        # 模型配置
        self.d_model = 128  # 隐藏维度
        self.d_k = 128  # Key维度
        self.d_v = 128  # Value维度
        self.d_inner_hid = 256  # FFN隐藏维度
        self.n_head = 4  # 注意力头数
        self.n_layer = 2  # Transformer层数
        self.dropout = 0.1

        # 分类配置
        self.num_classes = 2  # 二分类

        # 训练配置
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.00001
        self.weight_decay = 1e-5
        self.patience = 15  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.15

        # 阈值配置(用于二分类)
        self.seq_threshold = 0.3  # 序列级阈值
        self.point_threshold = 0.5  # 点级阈值

        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'checkpoints_treemil'


# ==================== 训练和评估函数 ====================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_X_mark, batch_y in tqdm(train_loader, desc="Training"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.squeeze().to(device)  # [B]

        optimizer.zero_grad()

        # 获取序列级分数
        ret = model.get_scores(batch_X)
        wscore = ret['wscore']  # [B]

        # 计算损失(序列级)
        loss = criterion(wscore, batch_y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 预测
        preds = (wscore >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def val_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.squeeze().to(device)

            # 获取分数
            ret = model.get_scores(batch_X)
            wscore = ret['wscore']  # [B]

            # 计算损失
            loss = criterion(wscore, batch_y.float())
            total_loss += loss.item()

            # 预测
            preds = (wscore >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
            all_scores.extend(wscore.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


def test_with_detailed_metrics(model, test_loader, criterion, device):
    """测试并返回详细指标"""
    model.eval()
    total_loss = 0

    # 序列级指标
    all_seq_preds = []
    all_seq_labels = []
    all_seq_scores = []

    # 点级指标(可选,展示TreeMIL的点级预测能力)
    all_point_preds = []
    all_point_labels = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.squeeze().to(device)

            # 获取分数
            ret = model.get_scores(batch_X)
            wscore = ret['wscore']  # [B]
            dscore = ret['dscore']  # [B, T]

            # 序列级
            loss = criterion(wscore, batch_y.float())
            total_loss += loss.item()

            seq_preds = (wscore >= 0.5).long().cpu().numpy()
            all_seq_preds.extend(seq_preds)
            all_seq_labels.extend(batch_y.cpu().numpy())
            all_seq_scores.extend(wscore.cpu().numpy())

            # 点级(用于展示,不用于主要评估)
            point_preds = (dscore >= 0.5).long().cpu().numpy()
            all_point_preds.append(point_preds)

    # 转换为numpy数组
    all_seq_labels = np.array(all_seq_labels)
    all_seq_preds = np.array(all_seq_preds)
    all_seq_scores = np.array(all_seq_scores)

    # 序列级指标
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_seq_labels, all_seq_preds)
    precision = precision_score(all_seq_labels, all_seq_preds,  zero_division=0)
    recall = recall_score(all_seq_labels, all_seq_preds,  zero_division=0)
    f1 = f1_score(all_seq_labels, all_seq_preds,zero_division=0)

    # AUC和PR-AUC
    try:
        roc_auc = roc_auc_score(all_seq_labels, all_seq_scores)
    except:
        roc_auc = 0.0

    precision_curve, recall_curve, _ = precision_recall_curve(all_seq_labels, all_seq_scores)
    pr_auc = auc(recall_curve, precision_curve)

    # 混淆矩阵
    cm = confusion_matrix(all_seq_labels, all_seq_preds)

    return avg_loss, accuracy, precision, recall, f1, roc_auc, pr_auc, cm


# ==================== 主训练函数 ====================

def train_model(config):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    print("=" * 70)
    print("TreeMIL for CICIDS2017 Network Traffic Anomaly Detection")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. Loading data...")
    data_dir = "../../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    config.enc_in = len(metadata['feature_names'])
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Feature count: {config.enc_in}")
    print(f"   Window size: {config.seq_len}")

    # 2. 分割数据
    print("\n2. Splitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, config.train_ratio, config.val_ratio
    )
    print(f"   Train: {train_data[0].shape[0]} windows")
    print(f"   Val:   {val_data[0].shape[0]} windows")
    print(f"   Test:  {test_data[0].shape[0]} windows")

    # 3. 创建数据加载器
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, config.batch_size
    )

    # 4. 创建模型
    print("\n4. Creating TreeMIL model...")
    model = TreeMIL(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {total_params:,}")
    print(f"   Tree depth: {model.depth}")
    print(f"   Effective sizes per level: {model.effe_size}")

    # 5. 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 6. 训练循环
    print("\n5. Training...")
    print("-" * 70)

    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(config.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_epoch(
            model, val_loader, criterion, config.device
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
        print(f"Epoch {epoch + 1:3d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'scaler': scaler,
                'best_val_f1': best_val_f1,
                'feature_names': metadata['feature_names']
            }
            torch.save(checkpoint, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"  → New best F1: {best_val_f1:.4f}, model saved!")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 保存训练历史
    with open(os.path.join(config.save_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    # 7. 测试最佳模型
    print("\n" + "=" * 70)
    print("6. Testing best model...")
    print("=" * 70)

    best_checkpoint = torch.load(os.path.join(config.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, test_cm = \
        test_with_detailed_metrics(model, test_loader, criterion, config.device)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Sequence-Level)")
    print("=" * 70)
    print(f"Test Loss:          {test_loss:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")
    print(f"Test ROC-AUC:       {test_roc_auc:.4f}")
    print(f"Test PR-AUC:        {test_pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print(f"  TN: {test_cm[0, 0]:6d}  FP: {test_cm[0, 1]:6d}")
    print(f"  FN: {test_cm[1, 0]:6d}  TP: {test_cm[1, 1]:6d}")
    print("=" * 70)

    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc,
        'confusion_matrix': test_cm.tolist()
    }

    with open(os.path.join(config.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    return model, train_history


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 创建配置
    config = Config()

    print("\n配置信息:")
    print(f"  序列长度: {config.seq_len}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  N叉树分支数: {config.ary_size}")
    print(f"  模型维度: {config.d_model}")
    print(f"  设备: {config.device}")

    # 开始训练
    model, history = train_model(config)

    print("\n训练完成!")