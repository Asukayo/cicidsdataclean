import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mymodel.catch.channel_mask import channel_mask_generator
from models.mymodel.catch.cross_channel_Transformer import Trans_C


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




class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        # 将pred_len重新定义为特征提取维度
        self.feature_dim = getattr(configs, 'pred_len', 32)  # 默认32维特征
        self.num_classes = getattr(configs, 'num_classes', 2)  # 默认二分类

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        # 构建掩码注意力模块
        self.seasonal_channel_mask_generator = channel_mask_generator(num_seq=100,n_vars=38)
        self.trend_channel_mask_generator = channel_mask_generator(num_seq=100, n_vars=38)
        # 构建通道注意力模块
        self.seasonal_cross_channel_transformer = Trans_C(dim = 128,depth=4,heads=8,mlp_dim=256,dim_head=64,dropout=0.7,patch_dim=100,temperature=0.5,d_model=self.feature_dim)
        self.trend_cross_channel_transformer = Trans_C(dim = 128,depth=4,heads=8,mlp_dim=256,dim_head=64,dropout=0.7,patch_dim=100,temperature=0.5,d_model=self.feature_dim)

        # 添加分类头：将分解后的特征映射到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(self.channels * self.feature_dim * 2, 64),  # *2因为有seasonal+trend
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        # [Batch_size,Channel,Input_length]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_mask = self.seasonal_channel_mask_generator(seasonal_init)
        trend_mask = self.trend_channel_mask_generator(trend_init)

        seasonal_transC,seasonal_dcloss = self.seasonal_cross_channel_transformer(seasonal_init, seasonal_mask)
        trend_transC,trend_dcloss = self.trend_cross_channel_transformer(trend_init, trend_mask)

        # 合并对比损失
        total_dcloss = seasonal_dcloss + trend_dcloss

        # 将seasonal和trend特征展平并拼接
        seasonal_flat = seasonal_transC.flatten(1)  # [Batch, channels * feature_dim]
        trend_flat = trend_transC.flatten(1)  # [Batch, channels * feature_dim]
        combined_features = torch.cat([seasonal_flat, trend_flat], dim=1)  # [Batch, channels * feature_dim * 2]

        # 分类输出
        output = self.classifier(combined_features)  # [Batch, num_classes]
        return output,total_dcloss,seasonal_dcloss,trend_dcloss