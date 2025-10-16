import torch


class DynamicalContrastiveLoss(torch.nn.Module):

    def __init__(self, temperature=0.5, k=0.3):
        """



        Parameters
        ----------
        temperature  温度参数，默认0.5，控制softmax的锐度
        k            正则化系数，默认0.3，平衡聚类损失和正则化损失
        """
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature  # 用于缩放余弦相似度
        self.k = k  # 保存正则化权重，用于平衡不同损失项
    def _stable_scores(self, scores):
        """
        辅助方法，进行数值稳定化处理（虽然在当前实现中尚未使用）
        Parameters
        ----------
        scores
        Returns
        -------
        """
        # 先沿着最后一个维度找最大值，[0]用于取数值（不要索引）
        # unsqueeze(-1)： 在最后添加一个维度，用于广播
        # 招到每行的最大值，用于数值稳定化
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        # 每个分数减去对应行的最大值，防止exp运算时数值溢出
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        """
        核心实现
        Parameters
        ----------
        scores      :TODO:（可能有误，需要看论文）注意力分数矩阵，形状[batch_size * patch_num,heads,n_vars,n_vars]
        attn_mask   :TODO（可能有误，需要看论文）注意力掩码，形状[batch_szie * patch_num,n_vars,n_vars]
        norm_matrix :归一化矩阵，Q和K的范数乘积,形状 [(batch_size * patch_num), heads, n_vars, n_vars]
        Returns
        -------
        """
        b = scores.shape[0]  # batch_size * patch_num的乘积
        n_vars = scores.shape[-1]  # 变量数量（通道数），从scores的最后一个维度进行获取
        # 获取跨头的余弦相似度矩阵
        # score通道间注意力分数，norm_matrix通道间QK范数乘积
        # 进行逐元素除法，计算通道间余弦相似度，随后在维度1heads上求平均
        # cosine矩阵的第[i,j]个元素表示第i个特征和第j个特征的相似度
        cosine = (scores / norm_matrix).mean(1)  # 输出[b,n_vars,n_vars] 平均的通道间余弦相似度矩阵
        # 计算正样本分数
        # 温度缩放（cosine/self.temperature）控制相似度分布的锐度，小温度突出最相似的通道对，大温度平滑所有通道间的关系
        # torch.exp指数变换，将相似度转为正值概率，范围为(0,+inf)
        # 应用掩码 * attn_mask 只保留通道掩码生成器认为重要的通道连接，被掩码的通道对分数为0，有效通道保留其原分数
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        # 计算所有样本分数，没有应用attn_mask
        # 所有可能的通道对关系
        # 作为对比学习的归一化分母
        # 形状为[b,n_vars,n_vars]
        all_scores = torch.exp(cosine / self.temperature)

        # 计算聚类损失
        # pos_scores.sum(dim=-1) 对每个通道i，计算它与所有被掩码选中通道的相似度总和
        # all_scores.sum(dim=-1) 对每个通道i，计算它与所有通道的相似度总和
        # 比值含义，每个通道与被选中通道的相似度占总相似度的比例
        # 负对数：最大化这个比例（最小化负对数损失）
        # 鼓励重要通道间相似度高，与不重要通道相似度低

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))
        # 构造单位矩阵
        # torch.eye(attn_mask.shape[-1])：创建n_vars x n_vars单位矩阵
        # unsqueeze(0):[n_vars,n_vars] → [1,n_vars,n_vars]
        # repeat(b, 1, 1):[1,n_vars,n_vars] → [b,n_vars,n_vars]
        # 单位矩阵，理想的稀疏通道连接模式，对角线=1，每个通道关注自己，非对角线=0，表示不关注其他通道

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)

        # 计算正则化损失
        # (n_vars * (n_vars - 1))：非对角线元素的总数
        # 归一化系数 1 / (n_vars * (n_vars - 1)) 作用：将损失标准化到[0,1]范围
        # eye.reshape(b, -1),attn_mask.reshape((b, -1) ：[b,n_vars * n_vars]   便于计算：L1范数需要向量输入
        # 两者差值含义：实际掩码与理想稀疏模式的偏差，鼓励掩码接近单位矩阵（稀疏连接）
        # L1范数
        # torch.norm(..., p=1, dim=-1)  # 形状: [b]
        # dim=-1，在特征维度上计算，输出每个样本的稀疏性惩罚
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        # 组合最终损失，clustering_loss.mean(1)：在通道维度上平均聚类损失，形状变化[b,n_vars] → [b]
        # self.k * regular_loss加权的稀疏性损失
        # 加法组合：平衡两个目标
        # 聚类目标：重要通道相似，不重要通道不相似
        # 稀疏目标：减少通道间连接数量
        loss = clustering_loss.mean(1) + self.k * regular_loss
        # 最终在batch维度上求平均，返回单个损失值用于反向传播
        mean_loss = loss.mean()
        return mean_loss
