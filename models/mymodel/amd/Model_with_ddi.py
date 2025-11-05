import torch
import torch.nn as nn
from models.mymodel.STLDECOMP.STL_Decompose import HybridSeriesDecompose
from .DDI import DDI


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

        # 定义DFT分解块
        self.stl_decomp = (
            HybridSeriesDecompose(
                top_k=5,low_freq_ratio=0.4,energy_threshold=0.95,ma_type='dema',
                ema_alpha=0.4,
            seq_len=100,features=38))
         # 通道数，即输入数据维度大小
        self.channels = configs.enc_in

        input_shape = [100,38]
        self.DDI_trend = DDI(input_shape=input_shape,patch=20,dropout=0.4)

        self.linear_before_classifier = nn.Linear(100, self.feature_dim)

        # 添加分类头：将分解后的特征映射到分类结果
        self.classifier = nn.Sequential(
            # 直接进行展平拼接为[,]二维张量
            nn.Linear(self.channels * self.feature_dim, self.num_classes),
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 使用分解模块，将原始的序列分解为季节和趋势变量

        trend_init, seasonal_init,_ = self.stl_decomp(x) # stl 输出的形状为[batch_size,channels,seq_len]

        # 交换Input length,Channel维度，使得x变为
        # x :[Batch,Channel,Input length]
        trend_init = trend_init.permute(0, 2, 1)

        ddi_output = self.DDI_trend(trend_init)

        trend_output = self.linear_before_classifier(ddi_output)

        # 将seasonal和trend特征展平并拼接
        trend_flat = trend_output.flatten(1)  # [Batch, channels * feature_dim]

        combined_features = torch.cat([trend_flat], dim=1)  # [Batch, channels * feature_dim * 2]

        # 分类输出
        output = self.classifier(combined_features)  # [Batch, num_classes]
        return output