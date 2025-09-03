import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mymodel.millet.resnet import ResNetFeatureExtractor


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
        return res, moving_mean

class Residual(nn.Module):
    """
    自行编写的残差块
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



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
        self.decompsition = series_decomp(kernel_size)
        # 是否对每一个维度的变量使用独立的线性层
        # 即默认通道独立或者通道相关
        self.individual = configs.individual
         # 通道数，即输入数据维度大小
        self.channels = configs.enc_in


        # 定义ResNet特征提取器属性
        # 定义季节趋势特征提取器
        self.resnet_seasonal_feature_extractor = ResNetFeatureExtractor(self.channels,padding_mode="replicate").instance_encoder
        # 定义趋势特征提取器
        self.resnet_trend_feature_extractor = ResNetFeatureExtractor(self.channels,
                                                                        padding_mode="replicate").instance_encoder

        # 如果趋势变量和季节变量使用同一个特征提取装置呢？
        # self.common_feature_extractor = ResNetFeatureExtractor(self.channels,padding_mode="replicate").instance_encoder

        self.resnet_output = 64

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

        # 在classifier前加一个残差块(失败)
        # self.residual_block = Residual(self.channels,self.channels)

        # 添加分类头：将分解后的特征映射到分类结果
        self.classifier = nn.Sequential(
            # 直接进行展平拼接为[,]二维张量
            nn.Linear(self.channels * self.resnet_output * 2, 64),# *2因为有seasonal+trend
            # nn.Conv1d(64,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.BatchNorm1d(64),
            # nn.Linear(64,32),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.BatchNorm1d(32),
            # 最终输出二分类结果
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 使用分解模块，将原始的序列分解为季节和趋势变量
        seasonal_init, trend_init = self.decompsition(x)
        # 交换Input length,Channel维度，使得x变为
        # x :[Batch,Channel,Input length]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)


        # 在线性层之前分别对seasonal和trend使用resnet特征提取
        seasonal_init, trend_init = (self.resnet_seasonal_feature_extractor.forward(seasonal_init),
                                     self.resnet_trend_feature_extractor.forward(trend_init))

        # 使用同一个resnet_block对特征进行提取
        # seasonal_init = self.common_feature_extractor(seasonal_init)
        # trend_init = self.common_feature_extractor(trend_init)

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

        # 在展平前加一个残差块
        # seasonal_output = self.residual_block(seasonal_output)
        # trend_output = self.residual_block(trend_output)

        # 将seasonal和trend特征展平并拼接
        seasonal_flat = seasonal_output.flatten(1)  # [Batch, channels * feature_dim]
        trend_flat = trend_output.flatten(1)  # [Batch, channels * feature_dim]
        combined_features = torch.cat([seasonal_flat, trend_flat], dim=1)  # [Batch, channels * feature_dim * 2]

        # 分类输出
        output = self.classifier(combined_features)  # [Batch, num_classes]
        return output