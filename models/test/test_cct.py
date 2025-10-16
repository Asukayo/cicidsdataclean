import torch

from models.mymodel.catch_backup.cross_channel_Transformer import Trans_C
from models.mymodel.catch_backup.channel_mask import channel_mask_generator

x = torch.randn(64,38,100).to("cuda")

# 掩码生成器模块
channel_mask = channel_mask_generator(num_seq=100, n_vars=38)
channel_mask = channel_mask.to("cuda")  # 添加这一行，将模型移动到GPU
masked = channel_mask(x)

# 创建通道注意力模块
cross_channel_Transformer = (
    Trans_C(dim = 128,depth=2,heads=10,mlp_dim=256,dim_head=64,dropout=0.6,patch_dim=100,d_model=128))
cross_channel_Transformer = cross_channel_Transformer.to("cuda")  # 添加这一行，将模型移动到GPU

# 传入原始数据x的同时传入掩码注意力矩阵
var,dc_loss = cross_channel_Transformer(x,attn_mask=masked)

# 打印所有参数的名称和形状
for name, param in cross_channel_Transformer.named_parameters():
    print(f"{name}: {param.shape}")

print("=======================result=================")
print("dc_loss:",dc_loss.item())    # 累计损失
print("masked:",masked.shape)       # torch.Size([64, 38, 38])
print("var:",var.shape)  # torch.Size([64, 38, 48])