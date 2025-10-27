from torch import nn
import torch

from models.test.test_STL_Decompse import x_seasonal


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

    def __init__(self, top_k=5, low_freq_ratio=0.4,energy_threshold=0.95):
        super(DFT_series_decomp, self).__init__()

        # 保留频谱中最大的前五个频率分量作为季节性成分(用于最小保留)
        self.top_k = top_k
        self.low_freq_ratio = low_freq_ratio

        # 能量阈值：保留累积能量到达该比例的频率分量（如0.95表示保留95%的能量）

    def forward(self, x):
        """
        Batch, Input, Channel
        返回去除高频噪声后的季节性成分
        """
        # 维度变换：[batch,input,channels] -> [batch,channels,input]
        x = x.permute(0, 2, 1)

        # 对时间维度进行实数FFT
        xf = torch.fft.rfft(x) #[batch,channels,freq_bins]

        # 计算频率幅度谱
        freq = abs(xf)  # shape:[batch,channels,freq_bins]

        # 将DC(零频率)设为0，去除常数项影响
        freq[:,:,0] = 0

        # ==== 采用能量累积方法识别高频噪声 ====
        # 1. 计算每个频率分量的能量（幅度的平方）
        energy = freq ** 2  # shape: [batch, channels, freq_bins]

        # 2.计算总能量
        total_energy = energy.sum(dim=-1,keepdim=True) # shape: [batch, channels, 1]

        # 3.沿频率维度计算累积能量
        cumsum_energy = torch.cumsum(energy, dim=-1) # shape: [batch, channels, freq_bins]

        # 4.计算累积能量比例
        energy_ratio = cumsum_energy / (total_energy + 1e-8) # 避免除零

        # 5.创建掩码：保留累积能量小于等于阈值的频率分量
        # True表示保留该频率分量，False表示视为高频噪声需要去除
        energy_mask = energy_ratio <= self.energy_threshold

        # ====== 确保最小频率保留量 ======
        total_freq_bins = freq.shape[-1]
        min_keep_bins = max(self.top_k,int(total_freq_bins * self.low_freq_ratio))
        min_keep_bins = min(min_keep_bins, total_freq_bins)

        # 强制保留前min_keep_bins个低频分量（避免过度滤波）
        energy_mask[:,:,:min_keep_bins] = True

        # ===== 应用掩码去除高频噪声 =====
        # 将高频噪声对应的频率分量置零
        xf_filtered = xf * energy_mask

        # 逆FFT重构季节性成分(已去除高频噪声)
        x_season = torch.fft.irfft(xf_filtered, n=x.shape[-1])

        # 转回原始维度：[batch, features, seq_len] -> [batch, seq_len, features]
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
