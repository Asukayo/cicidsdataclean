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

    def __init__(self, kernel_size = 25, top_k=5, low_freq_ratio=0.5):
        super(STL_Decompose, self).__init__()
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
