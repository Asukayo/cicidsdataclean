from torch import nn
import torch


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        [batch, seq_len, features]
        return: [batch, seq_len, features]
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    用于提取时间序列的趋势分量
    """

    def __init__(self, alpha=0.3):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        # 将alpha作为固定超参数，论文中使用0.3
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        """
        输入序列x的格式为 x: [Batch, Input, Channel]
        Batch:批次大小
        Input:时间序列长度
        Channel:特征维度（对于单变量是1）
        """

        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        # 提取时间序列长度t
        _, t, _ = x.shape
        # torch.arange(t, dtype=torch.double)：生成序列[0，1，2，....，t-1],使用double避免数值下溢
        # torch.flip(..., dims=(0,))：反转序列[t-1,t-2,....,1,0],因为EMA权重需要从最大次幂到0
        # 示例(t=5):powers = [4,3,2,1,0]
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        # 计算权重基础（1-α）^{powers}
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        # 创建归一化除数,复制权重作为除数，用于后续归一化（确保权重和为1）
        divisor = weights.clone()
        # 调整权重，（除了首项外所有权重乘以α）
        weights[1:] = weights[1:] * self.alpha
        # 调整张量形状，将形状从[t]变为[1,t,1]
        # 目的为了与输入[Batch,Input,Channel]进行广播运算
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        # x * weights：逐元素相乘（广播机制）,对于每一个时间步t，数据乘以对应权重
        # torch.cumsum(..., dim=1)：沿时间维度累积求和
        x = torch.cumsum(x * weights, dim=1)
        # 归一化，雏裔累积权重和，保证数值稳定性，相当于标准化操作
        x = torch.div(x, divisor)
        # 返回平滑后的序列，即趋势项
        return x.to(torch.float32)


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5, low_freq_ratio=0.4):
        super(DFT_series_decomp, self).__init__()

        # 保留频谱中最大的前五个频率分量作为季节性成分
        self.top_k = top_k
        self.low_freq_ratio = low_freq_ratio

    def forward(self, x):
        """
        Batch, Input, Channel
        """
        # 传入的x:[batch_size,seq_len,features]
        # 在内部进行维度交换
        x = x.permute(0, 2, 1)
        # 假设传入的x.shape为[[64, 38, 100]],batch_size=64, features=38, seq_len=100

        # 对最后一个维度（时间维度）进行实数FFT
        # xf 是复数张量，包含频域信息
        xf = torch.fft.rfft(x)  # xf.shape = [64, 38, 51](F = T//2 +1)
        # 计算频率幅度谱
        freq = abs(xf)  # freq.shape = [64, 38, 51]，包含各频率分量的幅度

        # 将DC分量（零频率）设为0，去除常数项影响
        freq[:, :, 0] = 0

        # 动态确定低频范围，避免高频噪声
        total_freq_bins = freq.shape[-1]
        low_freq_cutoff = max(self.top_k, int(total_freq_bins * self.low_freq_ratio))
        low_freq_cutoff = min(low_freq_cutoff, total_freq_bins)

        # 在低频范围内选择top_k分量
        freq_low = freq[:, :, 1:low_freq_cutoff]  # 从1开始为了排除DC分量

        if freq_low.shape[-1] < self.top_k:
            # 如果低频分量不足，使用所有可用分量
            actual_k = freq_low.shape[-1]
            top_k_freq, top_list = torch.topk(freq_low, actual_k)
            top_list = top_list + 1  # 偏移1，因为排除了DC分量
        else:
            top_k_freq, top_list = torch.topk(freq_low, self.top_k)
            top_list = top_list + 1  # 偏移1，因为排除了DC分量

            # 创建频域滤波器
        xf_filtered = torch.zeros_like(xf)

        # 向量化索引操作
        batch_idx = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1)
        channel_idx = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)

        # 保留选中的频率分量
        xf_filtered[batch_idx, channel_idx, top_list] = xf[batch_idx, channel_idx, top_list]

        # 逆FFT重构季节性成分
        x_season = torch.fft.irfft(xf_filtered, n=x.shape[-1])

        # 转回[batch, seq_len, features]
        return x_season.permute(0, 2, 1)


class STL_Decompose(nn.Module):
    """
    混合分解：使用移动平均提取趋势信息，使用DFT提取季节性信息
    """

    def __init__(self, kernel_size=25, top_k=5, low_freq_ratio=0.5,ema_alpha=0.3,ma_type='ema'):
        super(STL_Decompose, self).__init__()
        if ma_type=='ema':
            self.moving_avg = EMA(alpha=ema_alpha)
        elif ma_type=='averagePooling':
            self.moving_avg = moving_avg(kernel_size=kernel_size, stride=1)
        self.dft_decomp = DFT_series_decomp(top_k=top_k, low_freq_ratio=low_freq_ratio)

    def forward(self, x):
        """
        x: [batch, seq_len, features]
        Returns:
            x_trend: [batch, seq_len, features] - 趋势分量
            x_seasonal: [batch, seq_len, features] - 季节性分量
            x_residual: [batch, seq_len, features] - 残差分量
        """
        x_trend = self.moving_avg(x)

        x_detrended = x - x_trend

        x_seasonal = self.dft_decomp(x_detrended)

        x_residual = x_detrended - x_seasonal

        return x_trend, x_seasonal, x_residual
