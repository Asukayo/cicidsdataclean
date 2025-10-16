from torch import nn, einsum
from einops import rearrange
import math, torch
from .ch_discover_loss import DynamicalContrastiveLoss
# einsum: 爱因斯坦求和约定，用于复杂的张量运算

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)# 层归一化
        self.fn = fn  # 要包装的函数/模块

    # Pre-Norm结构，先运用归一化再去应用函数，比Post-Norm更加稳定
    # kwargs：传递额外参数给被包装的函数
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 前馈网络
# 维度变化dim->hidden dim->dim
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # 比ReLu更平滑的激活函数，在Transformer中表现更好
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 通道注意力机制
class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        '''

        Parameters
        ----------
        dim   输入的特征维数
        heads   多头注意力的头数
        dim_head    每个注意力头的维数
        dropout     dropout率：默认0.8（较高的dropout率用于正则化）
        regular_lambda  对比学习正则化系数，控制对比损失的权重
        temperature     温度参数，用于对比学习中的softmax锐化
        '''
        super().__init__()

        self.dim_head = dim_head # 每个注意力头的维度
        self.heads = heads  # 注意力头数量，用于张量重排和计算

        self.d_k = math.sqrt(self.dim_head) #缩放因子，防止注意力分数过大导致softmax饱和

        inner_dim = dim_head * heads    #总的内部维度=每个头维度 x 头数   ，用于QKV线性投影层的输出维度

        self.attend = nn.Softmax(dim=-1)    # Softmax函数，dim=-1，在最后一个维度上应用softmax，将注意力分数转为概率分布
        # 将dim->inner_dim
        self.to_q = nn.Linear(dim, inner_dim)   # Query投影，将输入映射为查询向量
        self.to_k = nn.Linear(dim, inner_dim)   # Key投影，将输入映射为键向量
        self.to_v = nn.Linear(dim, inner_dim)   # Value投影，将输入映射为值向量

        self.to_out = nn.Sequential(            # 输出投影
            nn.Linear(inner_dim, dim),          # 将多头输出重新投影回原始维度
            nn.Dropout(dropout)                 # 应用Dropout正则化防止过拟合
        )
        # regular_lambda正则化强度 temperature：温度参数，空值对比学习的锐度
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature) # k对比学习的正则化强度 #temperature：控制对比学习中softmax的锐度

    # 前向传播方法
    def forward(self, x, attn_mask=None):
        '''

        Parameters
        ----------
        x            x输入张量，形状通常为[batch_size,seq_len,dim]
        attn_mask   可选的注意力掩码，用于屏蔽某些位置的注意力

        Returns
        -------

        '''

        h = self.heads  # 将头数赋值给局部变量，便于后续使用
        # qkv计算
        q = self.to_q(x) # Query: [b, seq_len, inner_dim]
        k = self.to_k(x) # Key: [b, seq_len, inner_dim]
        v = self.to_v(x) # Value: [b, seq_len, inner_dim]
        scale = 1 / self.d_k    # 缩放因子

        # 多头重排
        # 将拼接的头分离为独立的头维度
        # h：头数量，d：每头维度
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)   # [b,h,n,d]     n为seq_len
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)   # [b,h,n,d]
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)   # [b,h,n,d]

        # 初始化对比损失,设置为None，后续根据条件进行更新
        dynamical_contrastive_loss = None

        # 计算注意力分数
        # Einstein求和：计算Q和K的点积
        # 结果注意力分数[b,h,n,n]
        # b ： batch维度保持
        # h ： heads维度保持
        # i,j 序列位置两两组合
        # d: 在该维度上求点积(消除)
        # 输出形状 [batch,heads,seq_len,seq_len]
        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        # 计算归一化矩阵
        q_norm = torch.norm(q, dim=-1, keepdim=True)    #Q的L2范数,在最后一个维度dim_head上计算范数，保持原有维度结构,避免广播错误
        k_norm = torch.norm(k, dim=-1, keepdim=True)    # K的L2范数
        # 计算Q范数和K范数的外积，为对比学习损失提供归一化信息
        # 形状[batch,heads,seq_len,seq_len]
        norm_matrix = torch.einsum('b h i d,b h j d->b h i j', q_norm, k_norm) # 范数乘积的矩阵

        # 掩码处理,掩码作用，屏蔽不应该参与注意力计算的位置
        if attn_mask is not None:
            # 将掩码为0的位置设为大负数
            # 保持掩码为1的位置不变

            # 定义掩码函数
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)    #大负数，近似负无穷
                # 将掩码为0的位置设置为大负数，掩码为1的位置设置为0，被掩码位置的注意力权重趋近于0
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                # 应用掩码
                # unsqueeze(1)：在第一维度增加维度，适配heads维度
                # 乘法掩码：保留有效位置的分数
                # 加法掩码:将无效位置设为大负数
                # 组合效果：在有效位置保持原分数，无效位置变为大负数
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores
            # 引用掩码分数
            masked_scores = _mask(scores, attn_mask)
            # 只在有掩码时进行计算
            # 原始注意力分数，注意力掩码，范数矩阵

            # 之前：训练和验证都计算
            # 计算动态学习对比损失，增强特征判别性
            # dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)

            # 之后：只在训练时计算
            # 之后：只在训练时计算
            if self.training:
                dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
            else:
                dynamical_contrastive_loss = torch.tensor(0.0, device=scores.device, requires_grad=False)
        else:
            masked_scores = scores

        #注意力计算和输出
        # 先应用缩放因子防止梯度消失
        attn = self.attend(masked_scores * scale)   #Softmax注意力权重，将分数转换为概率分布
        # 加权求和计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)    # 使用注意力权重对value进行加权平均
        out = rearrange(out, 'b h n d -> b n (h d)')    #[batch,heads,seq_len,dim_head] -> [batch,seq_len,heads*dim_head]
        # self.to_out(out)：应用输出投影层，包含线性变换和dropout
        # 投影后的输出：最终的特征表示  attn：注意力权重矩阵，用于可视化和分析   dynamical_contrastive_loss：对比学习损失，用于训练优化
        return self.to_out(out), attn, dynamical_contrastive_loss


#完整Transformer类
# 构建完整的多层Transformer架构
class c_Transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        """

        Parameters
        ----------
        dim: 模型的主要特征维度
        depth: Transformer的层数（深度）
        heads： 多头注意力的头数
        dim_head： 每个注意力头的维度
        mlp_dim：  前馈网络的隐藏层维度
        dropout：  Dopout率，默认0.8（相对较高的正则化）
        regular_lambda： 对比学习正则化系数
        temperature： 对比学习温度参数
        """

        super().__init__()
        # 存储多个Transformer层，自动注册为模型参数
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            # 循环次数：depth次，创建指定深度的Transformer
            self.layers.append(nn.ModuleList([
                # 单层Transformer的创建
                # 每一层都包含预归一化的注意力模块+预归一化的前馈模块
                # PreNorm比PostNorm训练起来更加稳定
                PreNorm(dim,
                        c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda,
                                    temperature=temperature)),
                # 同样使用预归一化
                # 多层感知机模块，输入输出维度都是dim，中间维度是mlp_dim
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))


    def forward(self, x, attn_mask=None):
        """

        Parameters
        ----------
        x： 输入张量，通常形状为[batch_size,seq_len,dim]
        attn_mask   : 可选的注意力掩码

        Returns     : 返回处理后的特征，注意力权重、对比学习损失
        -------

        """
        # 设置总对比学习损失为0
        # 累积策略：将所有层的损失相加后平均
        total_loss = 0
        # 逐层处理循环
        # 每层包含有注意力模块attn和前馈网络ff
        # 先处理注意力，再处理前馈网络
        for attn, ff in self.layers:
            # x_n：注意力处理后的特征
            # attn: 注意力权重矩阵
            # dcloss: 当前层的动态对比损失学习
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)    #注意力结算
            # 将当前层的损失加到总损失中
            # 用于后续计算平均损失
            total_loss += dcloss    # 累计损失
            # 将注意力输出与原始输入增加
            # 缓解梯度消失问题，允许信息直接流动，提高训练稳定性
            x = x_n + x             # 残差连接
            # 执行预归一化的前馈网络，同样使用残差连接
            # x现在包含当前层的完整输出
            x = ff(x) + x           # 前馈加残差链接
        # 计算所有层损失的平均值，避免损失随层数线性增长
        dcloss = total_loss / len(self.layers)  # 平均损失
        # x：经过所有Transformer层处理的最终特征
        # attn: 最后一层的注意力权重(用于可视化分析)
        # dcloss：平均的动态对比学习损失
        return x, attn, dcloss






# 跨通道Transformer主接口
# 完整的跨通道Transformer模块，包含输入处理和输出投影
class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, d_model,
                 regular_lambda=0.3, temperature=0.1):

        """
        完整的跨通道Transformer模块
        Parameters
        ----------
        dim     :Transformer内部特征维度
        depth   :Transfromer层数
        heads   :多头注意力头数
        mlp_dim : 前馈网络隐藏维度
        dim_head: 每个注意力头的维度
        dropout： Dropout概率
        patch_dim：输入分片的维度
        d_model： 模型输出维度
        regular_lambda： 对比学习正则化系数
        temperature： 对比学习温度参数
        """

        super().__init__()
        # 初始化操作
        self.dim = dim      # 保存内部特征维度，供后续使用
        self.patch_dim = patch_dim # 保存输入分片维度，用于输入验证

        # 输入嵌入和Transformer
        # 将输入分片嵌入到Transformer空间

        self.to_patch_embedding = nn.Sequential(
            # 维度变换patch_dim -> dim
            # 将输入分片投影到transformer的内部特征空间
            nn.Linear(patch_dim, dim), nn.Dropout(dropout))

        # 随机置零部分神经元，防止过拟合，提高泛化能力
        self.dropout = nn.Dropout(dropout)

        # 创建c_Transformer实例，将所有相关参数传递给Transformer
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)
        # 输出头
        # 维度变换dim->d_model,适配后续模块的输入要求
        self.mlp_head = nn.Linear(dim, d_model)



    def forward(self, x, attn_mask=None):
        # x：输入张量，通常来自频域分片数据
        # 进行处理，线性投影，patch_dim -> dim
        # [batch,seq_len.patch_dim] -> [batch,seq_len,dim]
        x = self.to_patch_embedding(x)  #嵌入投影
        # 执行多层Transformer处理
        # 输入：嵌入后的特征和注意力掩码
        x, attn, dcloss = self.transformer(x, attn_mask) # Transformer处理
        # 在输出投影前应用Dropout，进一步防止过拟合
        x = self.dropout(x)                             # Dropout
        # 维度变换 dim->d_model
        # 移除大小为1的维度，有时线性层会产生额外的单维度
        x = self.mlp_head(x).squeeze()                  # 最终投影
        # x的最终特征表示，用于后续重构
        # dcloss：对比学习损失，用于训练优化
        return x, dcloss  # ,attn
