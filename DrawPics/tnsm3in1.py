import matplotlib.pyplot as plt
import numpy as np

# ==================== 全局美化配置 ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5  # 加厚边框

# 提高基本字号，确保缩小到 Word 后依然清晰
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 14
FONT_SIZE_TITLE = 18

# ==================== 创建大画布 ====================
# 使用更大的尺寸 (15x5.5)，产生更开阔的视觉效果
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))

# 公用样式设置
def beauty_plot(ax, x_label, y_label):
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# ==================== 子图1：Window Size ====================
window_sizes = [50, 100, 150, 200]
f1_scores_w = [0.815, 0.864, 0.845, 0.829]
annotation_counts = [56000, 28000, 19000, 14000]

color_main, color_sub = '#1F77B4', '#D62728' # 经典学术深蓝与深红
line1 = ax1.plot(window_sizes, f1_scores_w, marker='o', linewidth=3, markersize=8, color=color_main, label='F1-Score')
ax1.scatter(window_sizes[1], f1_scores_w[1], s=250, color='#FFD700', marker='*', edgecolors='black', zorder=5, label='Baseline')

ax1_right = ax1.twinx()
ax1_right.plot(window_sizes, annotation_counts, marker='s', linewidth=3, markersize=8, color=color_sub, linestyle='--', label='Annotations')
ax1_right.set_ylabel('Annotations', fontsize=FONT_SIZE_LABEL, color=color_sub, fontweight='bold')
ax1_right.tick_params(axis='y', labelcolor=color_sub, labelsize=FONT_SIZE_TICK)

beauty_plot(ax1, 'Window Size $W$', 'F1-Score')
ax1.set_ylim(0.75, 0.90)
ax1.set_xticks(window_sizes)
ax1.set_title('(a)', y=-0.25, fontsize=FONT_SIZE_TITLE)

# ==================== 子图2：Alpha Values ====================
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
f1_scores_a = [0.853, 0.852, 0.864, 0.857, 0.855, 0.842]

ax2.plot(alpha_values, f1_scores_a, marker='o', linewidth=3, markersize=8, color=color_main)
ax2.scatter(alpha_values[2], f1_scores_a[2], s=250, color='#FFD700', marker='*', edgecolors='black', zorder=5)

beauty_plot(ax2, r'Parameter $\alpha$', 'F1-Score')
ax2.set_ylim(0.75, 0.90)
ax2.set_xticks(alpha_values)
ax2.set_title('(b)', y=-0.25, fontsize=FONT_SIZE_TITLE)

# ==================== 子图3：Patch Size ====================
patch_sizes = [10, 20, 25, 50]
num_patches = [10, 5, 4, 2]
f1_scores_p = [0.849, 0.864, 0.855, 0.847]

color_patch = '#2CA02C' # 深绿
ax3.plot(patch_sizes, f1_scores_p, marker='o', linewidth=3, markersize=8, color=color_main)
ax3.scatter(patch_sizes[1], f1_scores_p[1], s=250, color='#FFD700', marker='*', edgecolors='black', zorder=5)

ax3_right = ax3.twinx()
ax3_right.plot(patch_sizes, num_patches, marker='s', linewidth=3, markersize=8, color=color_patch, linestyle='--')
ax3_right.set_ylabel('Num Patches', fontsize=FONT_SIZE_LABEL, color=color_patch, fontweight='bold')
ax3_right.tick_params(axis='y', labelcolor=color_patch, labelsize=FONT_SIZE_TICK)

beauty_plot(ax3, 'Patch Size $p$', 'F1-Score')
ax3.set_ylim(0.75, 0.90)
ax3.set_xticks(patch_sizes)
ax3.set_title('(c)', y=-0.25, fontsize=FONT_SIZE_TITLE)

# 自动调整布局，增加间距
plt.tight_layout(pad=3.0)

# 保存为超高分辨率 PNG，方便在 Word 中随意缩小而不失真
plt.savefig('aesthetic_ablation_study.png', dpi=600, bbox_inches='tight')
print("美化版图形已保存，请在 Word 中插入后调整至合适宽度。")
plt.show()