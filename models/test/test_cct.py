import torch

from models.mymodel.cct.just_cct import Trans_C


x = torch.randn(64,38,100).to("cuda")



# 创建通道注意力模块
cross_channel_Transformer = (
    Trans_C(dim = 128,depth=2,heads=10,mlp_dim=256,dim_head=64,dropout=0.6,seq_len=100,d_model=128))
cross_channel_Transformer = cross_channel_Transformer.to("cuda")  # 添加这一行，将模型移动到GPU

# 传入原始数据x的同时传入掩码注意力矩阵
var = cross_channel_Transformer(x)

# 打印所有参数的名称和形状
for name, param in cross_channel_Transformer.named_parameters():
    print(f"{name}: {param.shape}")

print("=======================result=================")
print("var:",var.shape)  # torch.Size([64, 38, 48])