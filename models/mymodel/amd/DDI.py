import torch
from torch import nn

class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, layernorm=True):
        """
        input_shape:输入形状(seq_len,feature_num)，其中seq_len是序列长度,feature_num是特征数量
        patch:patch的大小，将序列分割成多个patch进行处理
        alpha:0.0通道混合的缩放因子，控制通道依赖的权重，alpha=0表示只使用时间依赖
        layernorm：是否使用层归一化
        """
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.input_shape = input_shape

        # 历史窗口数量,表示使用一个历史patch来预测下一个patch
        self.n_history = 1

        # 保存patch大小
        self.patch = patch

        # 如果启用层归一化，创建输入归一化层
        self.layernorm = layernorm
        if self.layernorm:
            # BatchNorm1d需要的维度是seq_len × feature_num
            # 需要先将输入展平(flatten)后再进行归一化
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])

        # 用于归一化历史patch的数据
        # 维度：n_history × patch × feature_num = 1 × 12 × feature_num
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])

        # self.agg:线性层，用于时间维度的聚合
        # 将历史信息(n_history × patch)个时间步聚合为当前patch(patch个时间步)
        self.agg = nn.Linear(self.n_history * self.patch, self.patch)

        # 用于时间混合后的dropout正则化
        self.dropout_t = nn.Dropout(dropout)


    def forward(self, x):
        # [batch_size, feature_num, seq_len]
        if self.layernorm:
            # torch.flatten(x, 1, -1): 将维度1到最后展平，变成[batch_size, feature_num × seq_len]
            # .reshape(x.shape): 归一化后恢复原始形状
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        # 创建一个与输入x形状相同的零张量作为输出
        output = torch.zeros_like(x)
        # 将前n_history × patch个时间步直接从输入复制到输出
        # 这些初始时间步作为"启动"数据，因为没有足够的历史信息来预测它们
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()

        # 从第n_history × patch个时间步开始，每次跳跃patch个时间步
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            # 提取历史patch，从output中提取当前位置之前的历史patch
            # 形状：[batch_size, feature_num, n_history × patch]
            input = output[:, :, i - self.n_history * self.patch: i]

            # input [batch_size, feature_num, self.n_history * patch]
            # 对历史patch进行归一化操作，展平之后进行归一化，在恢复原状
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)

            # 时间聚合（Time-Mixing）,将历史信息(n_history × patch)个时间步聚合为当前patch(patch个时间步)
            # 这是时间依赖建模的核心：将历史信息聚合成对当前patch的预测
            # aggregation
            # 输出形状为：[batch_size, feature_num, patch]
            # 应用dropout
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)


            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            # 这一行代码是残差链接
            # input：从历史信息预测的当前patch
            # x[:, :, i: i + self.patch]是当前patch的原始输入
            # 残差连接有助于梯度流动和信息保留
            tmp = input + x[:, :, i: i + self.patch]
            #res保存这个结果作为备份
            res = tmp

            # res为之间依赖的结果，去除了后面通道依赖的结果
            output[:, :, i: i + self.patch] = res

        # [batch_size, feature_num, seq_len]
        return output

