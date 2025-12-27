import torch
from torch import nn


class LSTM_Baseline(nn.Module):
    """
    用于消融实验的LSTM基线模块，替代DDI
    保持与DDI相同的输入输出维度
    """

    def __init__(self, input_shape, dropout=0.2, hidden_size=64, num_layers=2):
        """
        Args:
            input_shape: 输入形状 (seq_len, feature_num)
            dropout: dropout率
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super(LSTM_Baseline, self).__init__()
        self.seq_len = input_shape[0]  # 100
        self.feature_num = input_shape[1]  # 38
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层：输入维度为feature_num，输出维度为hidden_size
        self.lstm = nn.LSTM(
            input_size=self.feature_num,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 投影层：将LSTM输出映射回原始特征维度
        self.projection = nn.Linear(self.hidden_size, self.feature_num)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, feature_num, seq_len]
        Returns:
            output: [batch_size, feature_num, seq_len]
        """
        batch_size = x.shape[0]

        # 转置：[batch_size, feature_num, seq_len] -> [batch_size, seq_len, feature_num]
        x = x.permute(0, 2, 1)

        # LSTM前向传播
        # lstm_out: [batch_size, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 投影回原始特征维度
        # output: [batch_size, seq_len, feature_num]
        output = self.projection(lstm_out)

        # 转置回原始格式：[batch_size, seq_len, feature_num] -> [batch_size, feature_num, seq_len]
        output = output.permute(0, 2, 1)

        # 残差连接（与DDI保持一致）
        output = output + x.permute(0, 2, 1)

        return output