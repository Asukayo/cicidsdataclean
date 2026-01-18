import matplotlib.pyplot as plt
import numpy as np

# 数据
window_sizes = [50, 100, 150, 200]
f1_scores = [0.815, 0.864, 0.845, 0.829]
annotation_counts = [56000, 28000, 19000, 14000]  # 单位：个

# 创建图形和双Y轴
fig, ax1 = plt.subplots(figsize=(11, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# 绘制F1-Score曲线（左Y轴）
color1 = '#2E86AB'
ax1.set_xlabel(r'Window Size ($W$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1-Score', fontsize=14, fontweight='bold', color=color1)
line1 = ax1.plot(window_sizes, f1_scores, marker='o', linewidth=2.5,
                 markersize=8, color=color1, label='F1-Score')
ax1.tick_params(axis='y', labelcolor=color1)

# 标注基准点 (W=100)
baseline_idx = 1
ax1.scatter(window_sizes[baseline_idx], f1_scores[baseline_idx],
           s=200, color='#A23B72', marker='*',
           zorder=5, label='Baseline (W=100)')

# 添加F1-Score数值标签
for i, (w, f1) in enumerate(zip(window_sizes, f1_scores)):
    ax1.text(w, f1 + 0.0025, f'{f1:.3f}',
            ha='center', va='bottom', fontsize=10, color=color1, fontweight='bold')

# 创建第二个Y轴（右侧）显示标注数量
ax2 = ax1.twinx()
color2 = '#E63946'
ax2.set_ylabel('Number of Annotations', fontsize=14, fontweight='bold', color=color2)
line2 = ax2.plot(window_sizes, annotation_counts, marker='s', linewidth=2.5,
                 markersize=8, color=color2, linestyle='--',
                 label='Annotation Count')
ax2.tick_params(axis='y', labelcolor=color2)

# 添加标注数量标签（以千为单位显示）
for i, (w, count) in enumerate(zip(window_sizes, annotation_counts)):
    label = f'~{count/1000:.1f}k'
    ax2.text(w, count + 1800, label,
            ha='center', va='bottom', fontsize=10, color=color2, fontweight='bold')

# 设置坐标轴范围
ax1.set_ylim(0.805, 0.875)
ax2.set_ylim(10000, 60000)
ax1.set_xticks(window_sizes)
ax1.grid(True, alpha=0.3, linestyle='--')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
          loc='upper right', frameon=True, shadow=True, fontsize=11)

plt.title('Model Performance under Different Window Sizes',
         fontsize=15, fontweight='bold', pad=20)

# 添加注释说明标注数量的计算方式
fig.text(0.5, 0.02, r'Note: Annotation Count $\approx$ Total Flows / $W$',
         ha='center', fontsize=10, style='italic', color='gray')

fig.tight_layout(rect=[0, 0.03, 1, 1])

# 保存图形
plt.savefig('window_size_performance.png',
            dpi=300, bbox_inches='tight')
print("图形已保存至: window_size_performance.png")

plt.show()