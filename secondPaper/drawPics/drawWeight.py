import matplotlib.pyplot as plt
import torch

import sys
import os
# 1. 获取当前文件所在目录的上一级目录（即 secondPaper 目录）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. 将父目录加入系统路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir
                    )
from utils.FrequencyMasking import get_train_freq_weight


w = get_train_freq_weight(51, device='cpu').squeeze().numpy()
plt.figure(figsize=(8, 3), dpi=300)
plt.plot(range(51), w, color='steelblue', linewidth=2, label='Frequency weight')
plt.xlabel('Frequency Bin Index')
plt.ylabel('Loss Weight')
plt.title('Stability-Aware Frequency Weighting')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('freq_weight.png')