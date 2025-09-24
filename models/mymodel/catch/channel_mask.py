'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:32
*
* @description: 
'''
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax
# einops.rearrange：用于张量维度重排的工具，比传统的reshape更加直观
# gumbel_softmax: Gumbel-softmax技巧，用于在反向传播中近似离散采样


class channel_mask_generator(torch.nn.Module):
    def __init__(self, num_seq, n_vars):
        super(channel_mask_generator, self).__init__()


        # 核心生成器网络
        # 线性层
        self.generator = nn.Sequential(
            # 输出维度是通道数量，每个通道对应一个选择概率
            # bias=False，不适用偏置项，简化模型
            # sigmoid激活：将输出限制在[0,1]范围内，表示每个通道的选择概率
            torch.nn.Linear(num_seq, n_vars, bias=False), nn.Sigmoid())


        # 零初始化，将线性层权重初始化为0
        # 确保模型开始时所有通道被平等对待
        with torch.no_grad():
            self.generator[0].weight.zero_()
        # 保存通道数量
        self.n_vars = n_vars


    #
    def forward(self, x):  # x: [bs  x n_vars x num_seq]

        # 生成概率分布矩阵
        # 输入数据通过生成器网络，得到每个通道的选择概率
        # 输出形状[batch_size x n_vars x n_vars]
        distribution_matrix = self.generator(x)

        # 将连续概率转换为近似的离散选择
        # 在保证可微性的同时实现离散采样
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        # 构造掩码矩阵
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device) # 反单位矩阵除了对角线为0，其他位置为1的矩阵
        diag = torch.eye(self.n_vars).to(x.device)  # 单位矩阵

        # 最终掩码计算，Einstein求和：einsum实现逐元素乘法
        # 对角线元素强制为1（自己总是关注自己）
        # 非对角线元素通过重采样来决定是否关注其他通道
        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag

        return resample_matrix



    def _bernoulli_gumbel_rsample(self, distribution_matrix):

        b, c, d = distribution_matrix.shape   # 获取维度信息

        # 张量展平，将3D张量重排为2D张量
        # 新形状[(b*c*d) x 1] 每个概率值独立处理
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')

        # 计算互补概率
        # 用于构造二项分布的两个状态
        r_flatten_matrix = 1 - flatten_matrix

        # 计算对数几率，将概率转换为对数几率形式，便于Gumbel-Softmax处理
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        # 拼接：创建形状为[(b*c*d) X 2] 的张量
        # 每一行包含了选择/不选择的对数几率
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        # Gumbel-Softmax采样 hard = True，使用硬Gumbel-Softmax，产生one-hot向量
        # 效果，每个位置要么完全选择[1]，要么完全不选择[0]
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        # 提取选择结果
        # [...,0] 提取第一列(选择概率)
        # rearrange:重新排列会原始3D形状
        # 结果：二进制掩码矩阵，1表示选择，0表示不选择
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)

        return resample_matrix
