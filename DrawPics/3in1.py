import matplotlib.pyplot as plt
import numpy as np

# 创建大图，包含3个子图
fig = plt.figure(figsize=(18, 5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ==================== 子图1：Window Size (左) ====================
ax1 = plt.subplot(1, 3, 1)

# 数据
window_sizes = [50, 100, 150, 200]
f1_scores_w = [0.815, 0.864, 0.845, 0.829]
annotation_counts = [56000, 28000, 19000, 14000]

# 绘制F1-Score曲线（左Y轴）
color1 = '#2E86AB'
ax1.set_xlabel(r'Window Size ($W$)', fontsize=13, fontweight='bold')
ax1.set_ylabel('F1-Score', fontsize=13, fontweight='bold', color=color1)
line1 = ax1.plot(window_sizes, f1_scores_w, marker='o', linewidth=2.5,
                 markersize=8, color=color1, label='F1-Score')
ax1.tick_params(axis='y', labelcolor=color1)

# 标注基准点 (W=100)
baseline_idx = 1
ax1.scatter(window_sizes[baseline_idx], f1_scores_w[baseline_idx],
           s=180, color='#A23B72', marker='*',
           zorder=5, label='Baseline')

# 不显示F1-Score数值标签

# 创建第二个Y轴（右侧）显示标注数量
ax1_right = ax1.twinx()
color2 = '#E63946'
ax1_right.set_ylabel('Annotations', fontsize=13, fontweight='bold', color=color2)
line2 = ax1_right.plot(window_sizes, annotation_counts, marker='s', linewidth=2.5,
                 markersize=8, color=color2, linestyle='--',
                 label='Annotation Count')
ax1_right.tick_params(axis='y', labelcolor=color2)

# 不显示标注数量标签

# 设置坐标轴范围
ax1.set_ylim(0.805, 0.875)
ax1_right.set_ylim(10000, 60000)
ax1.set_xticks(window_sizes)
ax1.grid(True, alpha=0.3, linestyle='--')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
          loc='upper right', frameon=True, shadow=True, fontsize=9)

ax1.set_title('(a) Window Size Analysis',
         fontsize=13, fontweight='bold', pad=12)

# ==================== 子图2：Alpha Values (中) ====================
ax2 = plt.subplot(1, 3, 2)

# 数据
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
f1_scores_a = [0.853, 0.852, 0.864, 0.857, 0.855, 0.842]

# 绘制曲线
ax2.plot(alpha_values, f1_scores_a, marker='o', linewidth=2.5,
         markersize=8, color='#2E86AB', label='F1-Score')

# 标注基准点 (α=0.3)
baseline_idx = 2
ax2.scatter(alpha_values[baseline_idx], f1_scores_a[baseline_idx],
           s=180, color='#A23B72', marker='*',
           zorder=5, label='Baseline')

# 不显示数值标签

# 图形美化
ax2.set_xlabel(r'$\alpha$', fontsize=13, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
ax2.set_title(r'(b) $\alpha$ Parameter Analysis',
         fontsize=13, fontweight='bold', pad=12)

ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', frameon=True, shadow=True, fontsize=9)

# 设置坐标轴范围
ax2.set_xlim(0.05, 0.65)
ax2.set_ylim(0.835, 0.870)
ax2.set_xticks(alpha_values)

# ==================== 子图3：Patch Size (右) ====================
ax3 = plt.subplot(1, 3, 3)

# 数据
patch_sizes = [10, 20, 25, 50]
num_patches = [10, 5, 4, 2]
f1_scores_p = [0.849, 0.864, 0.855, 0.847]

# 绘制F1-Score曲线（左Y轴）
color1 = '#2E86AB'
ax3.set_xlabel(r'Patch Size ($p$)', fontsize=13, fontweight='bold')
ax3.set_ylabel('F1-Score', fontsize=13, fontweight='bold', color=color1)
line1 = ax3.plot(patch_sizes, f1_scores_p, marker='o', linewidth=2.5,
                 markersize=8, color=color1, label='F1-Score')
ax3.tick_params(axis='y', labelcolor=color1)

# 标注基准点 (p=20)
baseline_idx = 1
ax3.scatter(patch_sizes[baseline_idx], f1_scores_p[baseline_idx],
           s=180, color='#A23B72', marker='*',
           zorder=5, label='Baseline')

# 不显示F1-Score数值标签

# 创建第二个Y轴（右侧）显示Patches数量
ax3_right = ax3.twinx()
color2 = '#F18F01'
ax3_right.set_ylabel('Num Patches', fontsize=13, fontweight='bold', color=color2)
line2 = ax3_right.plot(patch_sizes, num_patches, marker='s', linewidth=2.5,
                 markersize=8, color=color2, linestyle='--',
                 label='Number of Patches')
ax3_right.tick_params(axis='y', labelcolor=color2)

# 不显示Patches数量标签

# 设置坐标轴范围
ax3.set_ylim(0.840, 0.870)
ax3_right.set_ylim(0, 12)
ax3.set_xticks(patch_sizes)
ax3.grid(True, alpha=0.3, linestyle='--')

# 合并图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2,
          loc='upper right', frameon=True, shadow=True, fontsize=9)

ax3.set_title('(c) Patch Size Analysis',
         fontsize=13, fontweight='bold', pad=12)

# 调整整体布局
plt.tight_layout()

# 保存图形
plt.savefig('combined_ablation_study.png',
            dpi=500, bbox_inches='tight')
print("合并图形已保存至: combined_ablation_study.png")

plt.show()