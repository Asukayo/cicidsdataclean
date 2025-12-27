import torch
import torch.nn as nn


class LSTM_Autoencoder(nn.Module):
    """
    标准LSTM自编码器用于网络流量异常检测

    架构:
    - Encoder: 多层LSTM将时序特征压缩到潜在空间
    - Decoder: 多层LSTM从潜在空间重构原始输入
    - 异常检测: 通过重构误差识别异常流量
    """

    def __init__(self,
                 input_dim,  # 输入特征维度 (CICIDS2017通常为77-83维)
                 hidden_dim=128,  # LSTM隐藏层维度
                 latent_dim=64,  # 潜在空间维度 (瓶颈层)
                 num_layers=2,  # LSTM层数
                 dropout=0.2):  # Dropout比率
        super(LSTM_Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ============ Encoder ============
        # 将输入序列编码为潜在表示
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 从LSTM隐藏状态到潜在空间的映射
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # ============ Decoder ============
        # 从潜在空间初始化解码器
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)

        # 解码器LSTM重构序列
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层：重构原始特征
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        编码器前向传播

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            latent: (batch_size, latent_dim) - 潜在表示
        """
        # LSTM编码
        lstm_out, (hidden, cell) = self.encoder_lstm(x)

        # 使用最后一个时间步的隐藏状态
        # hidden: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # 映射到潜在空间
        latent = self.encoder_fc(last_hidden)  # (batch_size, latent_dim)

        return latent

    def decode(self, latent, seq_len):
        """
        解码器前向传播

        Args:
            latent: (batch_size, latent_dim) - 潜在表示
            seq_len: int - 需要重构的序列长度

        Returns:
            reconstructed: (batch_size, seq_len, input_dim) - 重构的序列
        """
        batch_size = latent.size(0)

        # 从潜在空间映射回隐藏维度
        hidden = self.decoder_fc(latent)  # (batch_size, hidden_dim)

        # 准备LSTM初始隐藏状态
        # 复制到所有层
        h_0 = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        # 解码器输入：重复隐藏状态作为每个时间步的输入
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # LSTM解码
        lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))

        # 重构输出
        reconstructed = self.output_fc(lstm_out)  # (batch_size, seq_len, input_dim)

        return reconstructed

    def forward(self, x):
        """
        完整前向传播: 编码 -> 解码

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            reconstructed: (batch_size, seq_len, input_dim) - 重构的输入
            latent: (batch_size, latent_dim) - 潜在表示
        """
        seq_len = x.size(1)

        # 编码
        latent = self.encode(x)

        # 解码
        reconstructed = self.decode(latent, seq_len)

        return reconstructed, latent

    def get_reconstruction_error(self, x, reduction='mean'):
        """
        计算重构误差（用于异常检测）

        Args:
            x: (batch_size, seq_len, input_dim)
            reduction: 'mean', 'sum', 或 'none'

        Returns:
            error: 重构误差
        """
        reconstructed, _ = self.forward(x)

        # MSE重构误差
        mse = nn.MSE(reduction='none')(reconstructed, x)

        if reduction == 'mean':
            # 对序列和特征维度取平均
            error = mse.mean(dim=(1, 2))  # (batch_size,)
        elif reduction == 'sum':
            error = mse.sum(dim=(1, 2))
        else:
            error = mse

        return error