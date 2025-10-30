import torch
from models.mymodel.amd.MDM import HybridMDM
# [batch_size,channels,seq_len]
x = torch.randn(64,38,100).to("cuda")

input_shape = [x.shape[-1],x.shape[1]]

MDM = HybridMDM(input_shape=input_shape)

hybrid_mdm = MDM.to("cuda")

var = hybrid_mdm(x)

print(var.shape) # torch.Size([64, 38, 100])