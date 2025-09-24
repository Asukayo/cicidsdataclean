import torch

from models.mymodel.catch.channel_mask import channel_mask_generator


x = torch.randn(64,38,100).to("cuda")
x.shape

channel_mask = channel_mask_generator(num_seq=100, n_vars=38)
channel_mask = channel_mask.to("cuda")  # 添加这一行，将模型移动到GPU
var = channel_mask(x)



# 生成对角线全为1，剩余元素为0或1的矩阵
print(var.shape)  # torch.Size([64, 38, 38])
