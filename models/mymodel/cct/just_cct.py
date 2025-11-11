from torch import nn, einsum
from einops import rearrange
import math, torch


# einsum: 爱因斯坦求和约定,用于复杂的张量运算

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn  # 要包装的函数/模块

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """前馈网络，维度变化 dim -> hidden_dim -> dim"""

    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # 比ReLU更平滑的激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    """通道注意力机制 - 简化版"""

    def __init__(self, dim, heads, dim_head, dropout=0.5):
        """
        Parameters
        ----------
        dim : 输入的特征维度
        heads : 多头注意力的头数
        dim_head : 每个注意力头的维度
        dropout : dropout率
        """
        super().__init__()

        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)  # 缩放因子

        inner_dim = dim_head * heads  # 总的内部维度

        self.attend = nn.Softmax(dim=-1)

        # QKV投影层
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        # 输出投影层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : 输入张量，形状 [batch_size, seq_len, dim]

        Returns
        -------
        out : 输出特征，形状 [batch_size, seq_len, dim]
        attn : 注意力权重矩阵，形状 [batch_size, heads, seq_len, seq_len]
        """
        h = self.heads

        # QKV计算
        q = self.to_q(x)  # [b, seq_len, inner_dim]
        k = self.to_k(x)
        v = self.to_v(x)

        scale = 1 / self.d_k  # 缩放因子

        # 多头重排
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # 计算注意力分数：Q @ K^T
        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        # 应用缩放并计算注意力权重
        attn = self.attend(scores * scale)

        # 加权求和：Attention @ V
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 重排并投影回原始维度
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class c_Transformer(nn.Module):
    """完整的多层Transformer架构 - 简化版"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.5):
        """
        Parameters
        ----------
        dim : 模型的主要特征维度
        depth : Transformer的层数
        heads : 多头注意力的头数
        dim_head : 每个注意力头的维度
        mlp_dim : 前馈网络的隐藏层维度
        dropout : Dropout率
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Parameters
        ----------
        x : 输入张量，形状 [batch_size, seq_len, dim]

        Returns
        -------
        x : 处理后的特征
        attn : 最后一层的注意力权重
        """
        for attn, ff in self.layers:
            x_n, attn_weights = attn(x)  # 注意力计算
            x = x_n + x  # 残差连接
            x = ff(x) + x  # 前馈网络 + 残差连接

        return x, attn_weights


class Trans_C(nn.Module):
    """跨通道Transformer主接口 - 简化版"""

    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, seq_len, d_model):
        """
        Parameters
        ----------
        dim : Transformer内部特征维度
        depth : Transformer层数
        heads : 多头注意力头数
        mlp_dim : 前馈网络隐藏维度
        dim_head : 每个注意力头的维度
        dropout : Dropout概率
        patch_dim : 输入分片的维度
        d_model : 模型输出维度
        """
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # 输入嵌入层
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(seq_len, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer主体
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 输出投影层
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : 输入张量，形状 [batch_size, channels,seq_len]

        Returns
        -------
        x : 最终特征表示
        """
        # 输入嵌入
        x = self.to_patch_embedding(x)  # [batch, seq_len, dim]

        # Transformer处理
        x, attn = self.transformer(x)

        # 输出投影
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()

        return x