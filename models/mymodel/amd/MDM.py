import torch
from torch import nn


class HybridMDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, pool_mode='hybrid', beta=0.5, layernorm=True):
        """
        :param input_shape: 输入形状(seq_len,feature_num)
        :param k:下采样层数（downsampling层数），控制分解的尺度数量
        :param c:下采样率(downsampling rate)，每次下采样的倍数
        :param pool_mode='hybrid': 'avg','max','hybrid'
        :param beta=0.5 :混合池化的权重（仅当pool_mode='hybrid'）时启用
        :layernrom=True:是否使用归一化
        """
        super(HybridMDM, self).__init__()
        # 保存输入序列的长度（时间倍数）
        # 保存下采样层数
        self.seq_len = input_shape[0]
        self.k = k

        # 构建多尺度结构
        if self.k > 0:
            # 计算每一层的下采样核大小(kernel_size)
            # 计算逻辑，从粗粒度到细粒度
            self.k_list = [c ** i for i in range(k, 0, -1)]

            if pool_mode in ['avg', 'hybrid']:
                # 创建平均池化层，为每个下采样因子创建一个平均池化层
                # kernel_size 和 stride均为k，实现无重叠的下采样
                self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])

            if pool_mode in ['max', 'hybrid']:
                # 创建最大池化层，为每一个下采样因子创建一个最大池化层
                # 为了解决过度平滑的问题
                self.max_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=k, stride=k) for k in self.k_list])

            # 创建线性变换层，为每个尺度创建一个两层的前馈网络
            # 网络结构：1.第一个线性层，维度不变 2.GELU激活函数 3.第二个线性层，self.seq_len // k → self.seq_len * c // k 上采样
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )

        # 归一化层设置，归一化维度为seq_len × feature_num
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        """
        : param x:[batch_size, feature_num, seq_len]
        """
        # 对输入进行归一化操作
        if self.layernorm:
            # 展平后归一化，再恢复原状
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        # 对特殊情况进行处理，如果k=0，直接返回输入
        if self.k == 0:
            return x

        # 进行多尺度分解操作
        # x [batch_size, feature_num, seq_len]
        # sample_x：列表，用于存储不同尺度的时间模式
        sample_x = []

        # 根据模式选择池化策略
        # 循环操作：对每个下采样因子，应用平均池化提取对应尺度的模式
        for i, k in enumerate(self.k_list):
            if self.pool_mode == 'avg':
                pooled = self.avg_pools[i](x)
            elif self.pool_mode == 'max':
                pooled = self.max_pools[i](x)
            elif self.pool_mode == 'hybrid':
                avg_pooled = self.avg_pools[i](x)
                max_pooled = self.max_pools[i](x)
                pooled = self.beta * avg_pooled + (1 - self.beta) * max_pooled

            sample_x.append(pooled)


        # 最后添加原始数据，保留最细粒度的信息
        sample_x.append(x)

        # 计算尺度数量n = k + 1
        n = len(sample_x)

        # 自底向上的混合过程（Mixing）
        # 循环范围，从粗粒度到细粒度依次处理
        for i in range(n - 1):
            # 对当前尺度上运用线性变换并上采样
            tmp = self.linears[i](sample_x[i])
            # 将变换后的结果加到下一层
            # 此处的aplha可以控制残差连接的强度，其值本身没有其他含义
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        return sample_x[n - 1]
