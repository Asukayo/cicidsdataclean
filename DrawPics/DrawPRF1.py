import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 消融实验数据
ablation_data = {
    'w/o HSTD': {'P': 0.800, 'R': 0.813, 'F1': 0.806},
    'w/o MSTE': {'P': 0.859, 'R': 0.803, 'F1': 0.830},
    'Replace PTA with LSTM': {'P': 0.836, 'R': 0.813, 'F1': 0.824},
    'Fixed weight fusion ($g=0.5$)': {'P': 0.795, 'R': 0.838, 'F1': 0.816},
    'w/o residual': {'P': 0.764, 'R': 0.796, 'F1': 0.780},
    'Full Model': {'P': 0.872, 'R': 0.855, 'F1': 0.864}
}

# 按F1分数排序
sorted_configs = sorted(ablation_data.items(), key=lambda x: x[1]['F1'])
config_names = [item[0] for item in sorted_configs]
p_values = [item[1]['P'] for item in sorted_configs]
r_values = [item[1]['R'] for item in sorted_configs]
f1_values = [item[1]['F1'] for item in sorted_configs]

# 设置绘图参数
x = np.arange(len(config_names))
width = 0.25

# 创建图形
fig, ax = plt.subplots(figsize=(14, 6))

# 绘制柱状图
bars1 = ax.bar(x - width, p_values, width, label='Precision', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, r_values, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, f1_values, width, label='F1-Score', color='#e74c3c', alpha=0.8)

# 在柱状图上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# 设置图表属性
ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study Results (Sorted by F1-Score)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(config_names, rotation=15, ha='right')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim([0.70, 0.90])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 高亮完整模型
full_model_idx = config_names.index('Full Model')
ax.axvspan(full_model_idx - 0.5, full_model_idx + 0.5, alpha=0.1, color='gold')

plt.tight_layout()

# 先保存图片
plt.savefig('./ablation_study.png', dpi=300, bbox_inches='tight')
plt.savefig('./ablation_study.pdf', bbox_inches='tight')
print("图表已保存至 ./ablation_study.png 和 .pdf")

# 然后在PyCharm中显示
plt.show()  # 这会在PyCharm的SciView中显示图片