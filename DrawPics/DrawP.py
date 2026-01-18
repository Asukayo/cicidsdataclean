import matplotlib.pyplot as plt
import numpy as np

# 数据
patch_sizes = [10, 20, 25, 50]
num_patches = [10, 5, 4, 2]
f1_scores = [0.849, 0.864, 0.855, 0.847]

# 创建图形和双Y轴
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# 绘制F1-Score曲线（左Y轴）
color1 = '#2E86AB'
ax1.set_xlabel(r'Patch Size ($p$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1-Score', fontsize=14, fontweight='bold', color=color1)
line1 = ax1.plot(patch_sizes, f1_scores, marker='o', linewidth=2.5,
                 markersize=8, color=color1, label='F1-Score')
ax1.tick_params(axis='y', labelcolor=color1)

# 标注基准点 (p=20)
baseline_idx = 1
ax1.scatter(patch_sizes[baseline_idx], f1_scores[baseline_idx],
           s=200, color='#A23B72', marker='*',
           zorder=5, label='Baseline (p=20)')

# 添加F1-Score数值标签
for i, (p, f1) in enumerate(zip(patch_sizes, f1_scores)):
    ax1.text(p, f1 + 0.0015, f'{f1:.3f}',
            ha='center', va='bottom', fontsize=10, color=color1)

# 创建第二个Y轴（右侧）显示Patches数量
ax2 = ax1.twinx()
color2 = '#F18F01'
ax2.set_ylabel('Number of Patches', fontsize=14, fontweight='bold', color=color2)
line2 = ax2.plot(patch_sizes, num_patches, marker='s', linewidth=2.5,
                 markersize=8, color=color2, linestyle='--',
                 label='Number of Patches')
ax2.tick_params(axis='y', labelcolor=color2)

# 添加Patches数量标签
for i, (p, n) in enumerate(zip(patch_sizes, num_patches)):
    ax2.text(p, n + 0.3, f'{n}',
            ha='center', va='bottom', fontsize=10, color=color2)

# 设置坐标轴范围
ax1.set_ylim(0.840, 0.870)
ax2.set_ylim(0, 12)
ax1.set_xticks(patch_sizes)
ax1.grid(True, alpha=0.3, linestyle='--')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
          loc='upper right', frameon=True, shadow=True, fontsize=11)

plt.title('Model Performance under Different Patch Sizes',
         fontsize=15, fontweight='bold', pad=20)

fig.tight_layout()

# 保存图形
plt.savefig('patch_size_performance.png',
            dpi=300, bbox_inches='tight')
print("图形已保存至: patch_size_performance.png")

plt.show()