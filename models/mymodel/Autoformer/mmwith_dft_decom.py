import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mymodel.Autoformer.STL_Decompose import STL_Decompose


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

"""
    分解模块，将原始时序数据分解为趋势和周期变量，使用移动平均来实现
"""

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        # 修改moving_mean的维度为[batch_size,channels,seq_len]
        return res, moving_mean.permute(0, 2, 1)




class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5,low_freq_ratio = 0.4):
        super(DFT_series_decomp, self).__init__()
        # 保留频谱中最大的前五个频率分量作为季节性成分
        self.top_k = top_k
        self.low_freq_ratio = low_freq_ratio

    def forward(self, x):
        # 在内部进行维度交换
        x = x.permute(0, 2, 1)
        # 假设传入的x.shape为[[64, 38, 100]],batch_size=64, features=38, seq_len=100

        # 对最后一个维度（时间维度）进行实数FFT
        # xf 是复数张量，包含频域信息
        xf = torch.fft.rfft(x) # xf.shape = [64, 38, 51](F = T//2 +1)
        # 计算频率幅度谱
        freq = abs(xf) # freq.shape = [64, 38, 51]，包含各频率分量的幅度

        # 将DC分量（零频率）设为0，去除常数项影响
        freq[:,:,0] = 0

        # 动态确定低频范围，避免高频噪声
        total_freq_bins = freq.shape[-1]
        low_freq_cutoff = max(self.top_k,int(total_freq_bins*self.low_freq_ratio))
        low_freq_cutoff = min(low_freq_cutoff, total_freq_bins)

        # 在低频范围内选择top_k分量
        freq_low = freq[:,:,1:low_freq_cutoff] # 从1开始为了排除DC分量

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

        # 计算趋势成分
        x_trend = x - x_season

        return x_season, x_trend


class Model(nn.Module):
    """
    主体模型。目前使用MLP架构
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 即输入的时间序列长度
        self.seq_len = configs.seq_len
        # 将pred_len重新定义为特征提取维度
        self.feature_dim = getattr(configs, 'pred_len', 32)  # 默认38维特征
        # 默认二分类
        self.num_classes = 2

        # 定义分解周期/趋势变量的核大小
        kernel_size = 25

        # 定义DFT分解块
        self.stl_decomp = STL_Decompose(top_k=5,low_freq_ratio=0.4)

        self.decompsition = series_decomp(kernel_size)
        # 是否对每一个维度的变量使用独立的线性层
        # 即默认通道独立或者通道相关
        self.individual = configs.individual
         # 通道数，即输入数据维度大小
        self.channels = configs.enc_in


        # self.resnet_output = 64

        # 特征提取模块，即线性层
        # 运用通道独立时
        if self.individual:
            # 季节性变量的线性层
            self.Linear_Seasonal = nn.ModuleList()
            # 趋势变量的线性层
            self.Linear_Trend = nn.ModuleList()
            # 对每个变量使用独立的线性层
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.feature_dim))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.feature_dim))
        else:
            # 每个变量功用一个线性层
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.feature_dim)
            self.Linear_Trend = nn.Linear(self.seq_len, self.feature_dim)

            # 残差变量
            # self.Linear_Redusial = nn.Linear(self.seq_len, self.feature_dim)



        # 添加分类头：将分解后的特征映射到分类结果
        self.classifier = nn.Sequential(
            # 直接进行展平拼接为[,]二维张量
            nn.Linear(self.channels * self.feature_dim * 2, self.num_classes),# *2因为有seasonal+trend
            # nn.Conv1d(64,32,kernel_size=3,padding=1),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(64, self.num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # # nn.BatchNorm1d(32),
            # 最终输出二分类结果
            # nn.Linear(32, self.num_classes)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 使用分解模块，将原始的序列分解为季节和趋势变量

        trend_init, seasonal_init,_ = self.stl_decomp(x) # stl 输出的形状为[batch_size,channels,seq_len]

        # 交换Input length,Channel维度，使得x变为
        # x :[Batch,Channel,Input length]
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.permute(0, 2, 1)

        # 使用线性层
        if self.individual:
            # 如果使用独立的线性层，先定义一个空的output张量数组用来存储线性层结果
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.feature_dim],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.feature_dim],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # 对每个维度使用独立的先行层
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # 所有维度共享一个线性层
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            # redusial_output = self.Linear_Redusial(redusial)

        # 将seasonal和trend特征展平并拼接
        seasonal_flat = seasonal_output.flatten(1)  # [Batch, channels * feature_dim]
        trend_flat = trend_output.flatten(1)  # [Batch, channels * feature_dim]

        # redusial_flat = redusial_output.flatten(1)

        combined_features = torch.cat([trend_flat, seasonal_flat], dim=1)  # [Batch, channels * feature_dim * 2]

        # 分类输出
        output = self.classifier(combined_features)  # [Batch, num_classes]
        return output