from torch import nn


class ConvBlock(nn.Module):
    """
    单层卷积块：一个Conv1D + BatchNorm + （可选）ReLu
    """
    def __init__(
            self,
            n_in_channels,
            n_out_channels,
            kernel_size,
            padding,
            include_relu= True,
                 ):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                # Conv1d 层使用了 "same" 填充模式，意味着输出的宽度与输入相同。这通过对输入进行合适的填充来实现。
                padding="same",
                # 如果你设置了 padding_mode='replicate'，那卷积层会在输入序列的两端进行填充，填充的大小是卷积核大小减去 1，
                # 即 kernel_size - 1 = 3 - 1 = 2。因为复制填充会在两边各填充一部分，所以需要把这 2 个单位的填充分配到两边。
                # padding_mode 的作用：
                # padding_mode="replicate" 表示使用输入序列的边缘值进行填充。
                # 也就是说，在输入序列的两端，会分别使用第一个值和最后一个值进行填充。
                # padding_mode="constant" 是使用常数值进行填充（通常是 0，但可以手动指定填充值）。
                # padding_mode="reflect" 则表示使用反射填充，即输入序列会从边缘反射自己进行填充。
                padding_mode=padding,
            ),
            # BatchNorm1d 会在卷积后进行归一化处理，避免训练过程中出现梯度消失或者爆炸的问题。
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out
