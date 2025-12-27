import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqLenAdapter(nn.Module):
    def __init__(self, channels = 38,new_seq_len = None):
        super(SeqLenAdapter, self).__init__()
        self.channels = channels
        self.new_seq_len = new_seq_len
        # 使用1x1卷积保持通道数不变
        self.conv = nn.Conv1d(
            channels,channels,kernel_size = 1,
        )

    def forward(self,x):
        # x:[batch_size,channels,seq_len] = [64,38,100]
        x = self.conv(x) #[64,38,100]

        # 调整序列长度
        if self.new_seq_len is not None:
            x = F.interpolate(x,
                              size=self.new_seq_len,
                              mode='linear',
                              align_corners=False)
        return x  # [64,38,new_seq_len]