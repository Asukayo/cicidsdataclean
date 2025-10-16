from models.mymodel.Autoformer.STL_Decompose import STL_Decompose
import torch

decomp = STL_Decompose(
    kernel_size=25,      # 移动平均窗口大小
    top_k=5,            # 保留的主要频率分量数
    low_freq_ratio=0.4  # 低频范围比例
)

# 输入数据: [batch_size, seq_len, features]
x = torch.randn(64, 100, 38)  # 例如: 64个样本, 100时间步, 38个特征

# 分解
x_trend, x_seasonal, x_residual = decomp(x)

print(f"原始信号: {x.shape}") # torch.Size([64, 100, 38])
print(f"趋势分量: {x_trend.shape}") # torch.Size([64, 100, 38])
print(f"季节性分量: {x_seasonal.shape}") # torch.Size([64, 100, 38])
print(f"残差分量: {x_residual.shape}")

# 验证重构
x_reconstructed = x_trend + x_seasonal + x_residual
reconstruction_error = torch.mean((x - x_reconstructed)**2)
print(f"重构误差: {reconstruction_error.item():.2e}")