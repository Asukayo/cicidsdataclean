from torch import nn
import torch

class EnhancedAdaptiveMDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(EnhancedAdaptiveMDM, self).__init__()
        self.seq_len = input_shape[0]
        self.feature_num = input_shape[1]
        self.k = k

        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]

            self.avg_pools = nn.ModuleList([
                nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list
            ])
            self.max_pools = nn.ModuleList([
                nn.MaxPool1d(kernel_size=k, stride=k) for k in self.k_list
            ])

            # 为每个尺度创建选择器
            self.pool_selectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_num * 3, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                for _ in self.k_list
            ])

            self.linears = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seq_len // k, self.seq_len // k),
                    nn.GELU(),
                    nn.Linear(self.seq_len // k, self.seq_len * c // k),
                )
                for k in self.k_list
            ])

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        """
        :param x: [batch_size, feature_num, seq_len]
        """
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        if self.k == 0:
            return x

        sample_x = []

        for i, k in enumerate(self.k_list):
            # 先进行池化操作
            avg_pooled = self.avg_pools[i](x)  # [B, C, L//k]
            max_pooled = self.max_pools[i](x)  # [B, C, L//k]

            # 对池化后的数据计算统计特征来判断该尺度的特性
            # 这里我们综合考虑两种池化结果的统计特征
            pooled_for_stats = (avg_pooled + max_pooled) / 2  # 先取平均作为代表

            # 计算该尺度的统计特征
            mean = pooled_for_stats.mean(dim=-1)  # [B, C]
            std = pooled_for_stats.std(dim=-1)  # [B, C]
            max_val = pooled_for_stats.max(dim=-1)[0]  # [B, C]

            # 拼接统计特征
            stats = torch.cat([mean, std, max_val], dim=1)  # [B, 3C]

            # 计算该尺度的选择权重
            weight = self.pool_selectors[i](stats)  # [B, 1]
            weight = weight.unsqueeze(-1)  # [B, 1, 1]

            # 自适应混合
            pooled = weight * avg_pooled + (1 - weight) * max_pooled
            sample_x.append(pooled)

        sample_x.append(x)
        n = len(sample_x)

        # 自底向上混合
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)

        return sample_x[n - 1]

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    用于提取时间序列的趋势分量
    """

    def __init__(self, alpha=0.3):
        super(EMA, self).__init__()
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
        # 归一化，除以累积权重和，保证数值稳定性，相当于标准化操作
        x = torch.div(x, divisor)
        # 返回平滑后的序列，即趋势项
        return x.to(torch.float32)


class EnergyBasedDFTFilter(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5, low_freq_ratio=0.4,energy_threshold=0.95):
        super(EnergyBasedDFTFilter, self).__init__()

        # 保留频谱中最大的前五个频率分量作为季节性成分(用于最小保留)
        self.top_k = top_k
        self.low_freq_ratio = low_freq_ratio

        # 能量阈值：保留累积能量到达该比例的频率分量（如0.95表示保留95%的能量）
        self.energy_threshold = energy_threshold

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


class HybridSeriesDecompose(nn.Module):
    """
    混合分解：使用移动平均提取趋势信息，使用DFT提取季节性信息
    """

    def __init__(self, kernel_size=25,
                 top_k=5, low_freq_ratio=0.5,energy_threshold=0.97,ema_alpha=0.3,ma_type='ema',
                 seq_len = None,features = None):
        super(HybridSeriesDecompose, self).__init__()
        if ma_type=='ema':
            self.moving_avg = EMA(alpha=ema_alpha)
        self.dft_decomp = EnergyBasedDFTFilter(
            top_k=top_k,
            low_freq_ratio=low_freq_ratio,
            energy_threshold=energy_threshold)
        # 对MDM进行初始化操作
        if seq_len is not None and features is not None:
            trend_shape = [seq_len, features]
            self.MDM = EnhancedAdaptiveMDM(trend_shape)


    def forward(self, x):
        """
        x: [batch, seq_len, features]
        Returns:
            x_trend: [batch, seq_len, features] - 趋势分量
            x_seasonal: [batch, seq_len, features] - 季节性分量
            x_residual: [batch, seq_len, features] - 残差分量
        """
        x_trend = self.moving_avg(x)


        strengthened_trend = self.MDM(x_trend.permute(0, 2, 1)).permute(0, 2, 1)
        # 从原始x中去除trend分量
        x_detrended = x - x_trend

        x_seasonal = self.dft_decomp(x_detrended)

        x_residual = x_detrended - x_seasonal

        return strengthened_trend, x_seasonal, x_residual
