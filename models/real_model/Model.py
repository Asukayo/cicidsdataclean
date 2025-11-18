import torch
import torch.nn as nn
from models.mymodel.STLDECOMP.STL_Decompose import HybridSeriesDecompose
from models.mymodel.cct.just_cct import Trans_C
from .DDI import DDI
from models.layers.revin import RevIN
from ...layers.TCN_GATE import TCNWithSelfAttentionGate
from ...layers.redusial_seq_len_adapter import SeqLenAdapter


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
        # 定义seasonal分支网络中seq_len的维度
        self.cct_output_len = 128

        # 初始化 RevIN 模块用于数据归一化
        self.revin = RevIN(num_features=self.feature_dim, affine=True)

        # 定义DFT分解块
        self.stl_decomp = (
            HybridSeriesDecompose(
                top_k=5,low_freq_ratio=0.4,energy_threshold=0.8,ma_type='sma',
                ema_alpha=0.2,
            seq_len=100,features=38))
         # 通道数，即输入数据维度大小
        self.channels = configs.enc_in

        input_shape = [100,38]
        self.DDI_trend = DDI(input_shape=input_shape,patch=20,dropout=0.4)

        # 添加通道注意力机制
        self.crosschannelTransformer = (
            Trans_C(dim = 128,
                    depth=4,
                    heads=10,
                    mlp_dim=256,
                    dim_head=64,
                    dropout=0.6,
                    seq_len=100,
                    d_model=self.cct_output_len))

        # 调整Redusial的seq_len维度与cross_channel_Transformer的seq_len维度相匹配
        self.redusial_adapter = SeqLenAdapter(channels=self.channels,
                                              new_seq_len=self.cct_output_len)

        # 添加TCN+门控Network
        self.tcn_gate = TCNWithSelfAttentionGate(
            channels=self.channels,
            tcn_channels= [64,64],  #两层TCN，每层64通道
            kernel_size=5,
            dropout=0.4,
            gate_reduction=16,
            temperature=1.0
        )

        # 用于在分类器前调整linear和seasonal维度大小
        self.trend_linear_before_classifier = nn.Linear(100, self.feature_dim)
        self.seasonal_linear_before_classifier = nn.Linear(self.cct_output_len, self.feature_dim)

        # 添加分类头：将分解后的特征映射到分类结果
        self.classifier = nn.Sequential(
            # 直接进行展平拼接为[,]二维张量
            nn.Linear(self.channels * self.feature_dim * 2
                      , self.num_classes),
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 使用分解模块，将原始的序列分解为季节和趋势变量

        # 使用 RevIN 进行归一化
        # x_normalized = self.revin(x, mode='norm')

        # 使用分解模块，将归一化后的序列分解为季节和趋势变量
        trend_init, seasonal_init, redusial_init = self.stl_decomp(x) # stl 输出的形状为[batch_size,channels,seq_len]

        # 交换Input length,Channel维度，使得x变为
        # x :[Batch,Channel,Input length]
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        redusial_init = redusial_init.permute(0, 2, 1)

        ddi_output = self.DDI_trend(trend_init)
        cross_output = self.crosschannelTransformer(seasonal_init)

        # 进行cross_output与redusial的融合
        adapted_redusial = self.redusial_adapter(redusial_init)
        fused_output,gate_weight,cosine_sim = self.tcn_gate(cross_output,adapted_redusial)

        trend_output = self.trend_linear_before_classifier(ddi_output)
        seasonal_output = self.seasonal_linear_before_classifier(fused_output)

        # 将seasonal和trend特征展平并拼接
        trend_flat = trend_output.flatten(1)  # [Batch, channels * feature_dim]
        seasonal_flat = seasonal_output.flatten(1)

        combined_features = torch.cat([seasonal_flat,trend_flat], dim=1)  # [Batch, channels * feature_dim * 2]

        # 分类输出
        output = self.classifier(combined_features)  # [Batch, num_classes]
        return output