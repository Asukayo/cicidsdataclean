import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """去除卷积右侧padding，确保因果性"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    单个TCN Block：膨胀因果卷积 + 残差连接
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 主路径
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 残差投影（维度匹配）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class LightweightTCN(nn.Module):
    """
    轻量级TCN：多层膨胀卷积
    输入/输出: [Batch, Channels, Num_seq]
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_inputs: 输入通道数
            num_channels: list，每层的输出通道数，例如 [64, 64]
            kernel_size: 卷积核大小
            dropout: dropout概率
        """
        super(LightweightTCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长：1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding, dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [Batch, Channels, Num_seq]
        输出: [Batch, Channels, Num_seq]
        """
        return self.network(x)


class CosineSimilarityGate(nn.Module):
    """
    基于余弦相似度的自注意力门控

    核心思想：
    - 如果 Fused 和 Temporal 特征相似（cosine ≈ 1）→ gate ≈ 0（不需要TCN）
    - 如果特征互补/正交（cosine ≈ 0）→ gate ≈ 1（需要TCN）
    """

    def __init__(self, channels, reduction=16, temperature=1.0):
        """
        Args:
            channels: 特征通道数
            reduction: 降维比例（用于门控网络）
            temperature: 温度参数（控制门控的锐度）
        """
        super(CosineSimilarityGate, self).__init__()

        self.temperature = temperature

        # 确保维度至少为1
        hidden_dim = max(1, channels // reduction)
        gate_hidden_dim = max(1, hidden_dim // 2)

        # 特征提取网络（全局池化后的特征）
        self.fc_fused = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU()
        )

        self.fc_temporal = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU()
        )

        # 门控决策网络
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1)
        )

    def forward(self, fused, temporal):
        """
        Args:
            fused: [Batch, Channels, Num_seq] - 简单融合的特征
            temporal: [Batch, Channels, Num_seq] - TCN提取的特征

        Returns:
            gate_weight: [Batch, 1, 1] - 门控权重
            cosine_sim: [Batch] - 余弦相似度（用于分析）
        """
        # 获取批次大小
        batch_size = fused.size(0)

        # 全局平均池化 [B, C, T] → [B, C]
        fused_global = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        temporal_global = F.adaptive_avg_pool1d(temporal, 1).squeeze(-1)

        # 计算余弦相似度
        # cosine_sim = 1:两个特征高度对齐,TCN提取的时序信息冗余
        # cosine_sim = 0:两个特征正交，TCN提供互补信息
        cosine_sim = F.cosine_similarity(
            fused_global, temporal_global, dim=1
        )  # [B]

        # 相似度 → 互补度（1 - cosine）
        # cosine = 1 (高度相似) → complementarity = 0 (低互补)
        # cosine = 0 (正交) → complementarity = 1 (高互补)
        complementarity = 1.0 - cosine_sim  # [B]

        # 特征提取
        fused_feat = self.fc_fused(fused_global)  # [B, C//r]
        temporal_feat = self.fc_temporal(temporal_global)  # [B, C//r]

        # 加权融合（基于互补度）
        # 互补度高的样本，temporal特征权重更大
        complementarity_expanded = complementarity.unsqueeze(-1)  # [B, 1]

        joint_feat = (
                complementarity_expanded * temporal_feat +
                (1 - complementarity_expanded) * fused_feat
        )  # [B, C//r]

        # 计算最终门控权重
        gate_weight = self.gate_fc(joint_feat)  # [B, 1]

        # 温度缩放（可选，控制门控的锐度）
        gate_weight = torch.sigmoid(
            gate_weight / self.temperature
        )

        gate_weight = gate_weight.unsqueeze(-1)  # [B, 1, 1]

        return gate_weight, cosine_sim


class TCNWithSelfAttentionGate(nn.Module):
    """
    完整模块：TCN + 余弦相似度自注意力门控

    输入：attended 和 residual（都是 [B, C, T]）
    输出：refined features [B, C, T]
    """

    def __init__(self, channels, tcn_channels=[64, 64], kernel_size=3,
                 dropout=0.1, gate_reduction=16, temperature=1.0):
        """
        Args:
            channels: 输入特征通道数
            tcn_channels: TCN每层的通道数列表
            kernel_size: TCN卷积核大小
            dropout: Dropout概率
            gate_reduction: 门控网络降维比例
            temperature: 门控温度参数
        """
        super(TCNWithSelfAttentionGate, self).__init__()

        # TCN模块
        self.tcn = LightweightTCN(
            num_inputs=channels,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 维度对齐（如果TCN改变了通道数）
        final_tcn_channels = tcn_channels[-1]
        self.channel_align = None
        if final_tcn_channels != channels:
            self.channel_align = nn.Conv1d(final_tcn_channels, channels, 1)

        # 余弦相似度门控
        self.gate = CosineSimilarityGate(
            channels=channels,
            reduction=gate_reduction,
            temperature=temperature
        )

        # 可学习的缩放因子（可选）
        self.scale_tcn = nn.Parameter(torch.ones(1))
        self.scale_fused = nn.Parameter(torch.ones(1))

    def forward(self, attended, residual):
        """
        Args:
            attended: [Batch, Channels, Num_seq] - Channel Attention输出
            residual: [Batch, Channels, Num_seq] - 残差分量

        Returns:
            output: [Batch, Channels, Num_seq] - 精炼后的特征
            gate_weight: [Batch, 1, 1] - 门控权重
            cosine_sim: [Batch] - 余弦相似度
        """
        # 步骤1: ResNet融合
        fused = attended + residual  # [B, C, T]

        # 步骤2: TCN提取时序特征
        temporal_features = self.tcn(fused)  # [B, C', T]

        # 步骤3: 通道对齐
        if self.channel_align is not None:
            temporal_features = self.channel_align(temporal_features)  # [B, C, T]

        # 步骤4: 基于余弦相似度的门控
        gate_weight, cosine_sim = self.gate(fused, temporal_features)
        # gate_weight: [B, 1, 1], cosine_sim: [B]

        # 步骤5: 门控融合
        output = (
                gate_weight * (self.scale_tcn * temporal_features) +
                (1 - gate_weight) * (self.scale_fused * fused)
        )

        return output, gate_weight, cosine_sim


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置
    batch_size = 64
    channels = 38  # 特征通道数
    num_seq = 128  # 时间序列长度

    # 创建模型
    model = TCNWithSelfAttentionGate(
        channels=channels,
        tcn_channels=[64, 64],  # 2层TCN，每层64通道
        kernel_size=3,
        dropout=0.1,
        gate_reduction=16,
        temperature=1.0
    )

    # 模拟输入
    attended = torch.randn(batch_size, channels, num_seq)  # Channel Attention输出
    residual = torch.randn(batch_size, channels, num_seq)  # 残差分量

    # 前向传播
    output, gate_weight, cosine_sim = model(attended, residual)

    # 输出形状
    print("=" * 60)
    print("TCN with Cosine Similarity Self-Attention Gate")
    print("=" * 60)
    print(f"Input shapes:")
    print(f"  attended:  {attended.shape}")
    print(f"  residual:  {residual.shape}")
    print(f"\nOutput shapes:")
    print(f"  output:       {output.shape}")
    print(f"  gate_weight:  {gate_weight.shape}")
    print(f"  cosine_sim:   {cosine_sim.shape}")
    print(f"\nGate statistics:")
    print(f"  mean: {gate_weight.mean().item():.4f}")
    print(f"  std:  {gate_weight.std().item():.4f}")
    print(f"  min:  {gate_weight.min().item():.4f}")
    print(f"  max:  {gate_weight.max().item():.4f}")
    print(f"\nCosine similarity statistics:")
    print(f"  mean: {cosine_sim.mean().item():.4f}")
    print(f"  std:  {cosine_sim.std().item():.4f}")
    print("=" * 60)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    print("=" * 60)