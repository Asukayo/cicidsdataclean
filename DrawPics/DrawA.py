import matplotlib.pyplot as plt
import numpy as np

# 数据
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
f1_scores = [0.853, 0.852, 0.864, 0.857, 0.855, 0.842]

# 设置图形样式
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# 绘制曲线
plt.plot(alpha_values, f1_scores, marker='o', linewidth=2.5,
         markersize=8, color='#2E86AB', label='F1-Score')

# 标注基准点 (α=0.3)
baseline_idx = 2
plt.scatter(alpha_values[baseline_idx], f1_scores[baseline_idx],
           s=200, color='#A23B72', marker='*',
           zorder=5, label='Baseline (α=0.3)')

# 添加数值标签
for i, (alpha, f1) in enumerate(zip(alpha_values, f1_scores)):
    plt.text(alpha, f1 + 0.0015, f'{f1:.3f}',
            ha='center', va='bottom', fontsize=10)

# 图形美化
plt.xlabel(r'$\alpha$', fontsize=14, fontweight='bold')
plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
plt.title('Model Performance under Different α Values',
         fontsize=15, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', frameon=True, shadow=True, fontsize=11)

# 设置坐标轴范围
plt.xlim(0.05, 0.65)
plt.ylim(0.835, 0.870)
plt.xticks(alpha_values)

plt.tight_layout()

# 保存图形
plt.savefig('alpha_f1_performance.png',
            dpi=400, bbox_inches='tight')
print("图形已保存至: alpha_f1_performance.png")

plt.show()