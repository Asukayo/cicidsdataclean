import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== TNSM字体设置 ==========
# 使用Liberation Serif (Times New Roman的开源替代)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
# 也可以使用Arial/Helvetica作为备选（IEEE也接受）
# plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# MyModel的混淆矩阵数据
cm_mymodel = np.array([[18426, 1696],
                        [1968, 11631]])

# MyModel的指标
p_mymodel = 0.872
r_mymodel = 0.855
f1_mymodel = 0.864

# TreeMIL的指标
p_treemil = 0.828
r_treemil = 0.781
f1_treemil = 0.804

# 根据TreeMIL的P和R推算混淆矩阵
# 假设正负样本数与MyModel相同
total_positive = 13599  # TP + FN
total_negative = 20122  # TN + FP

tp_treemil = int(r_treemil * total_positive)
fn_treemil = total_positive - tp_treemil
fp_treemil = int(tp_treemil / p_treemil - tp_treemil)
tn_treemil = total_negative - fp_treemil

cm_treemil = np.array([[tn_treemil, fp_treemil],
                        [fn_treemil, tp_treemil]])

# 创建图形：调整为TNSM整页宽度 (7.16 inches)
# 对于双图并排，使用整页宽度
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))

# 左图：WSTD
sns.heatmap(cm_mymodel, annot=True, fmt='d', cmap='Blues',
            cbar=False, square=True, linewidths=1.5, linecolor='black',
            annot_kws={'size': 12, 'weight': 'bold'},
            cbar_kws={'label': 'Count'}, ax=axes[0])

axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[0].set_title(f'WSTD\nP={p_mymodel:.3f}  R={r_mymodel:.3f}  F1={f1_mymodel:.3f}',
                  fontsize=10, fontweight='bold', pad=10)
axes[0].set_xticklabels(['Normal', 'Attack'], fontsize=10)
axes[0].set_yticklabels(['Normal', 'Attack'], fontsize=10, rotation=0)

# 右图：TreeMIL
sns.heatmap(cm_treemil, annot=True, fmt='d', cmap='Oranges',
            cbar=False, square=True, linewidths=1.5, linecolor='black',
            annot_kws={'size': 12, 'weight': 'bold'},
            cbar_kws={'label': 'Count'}, ax=axes[1])

axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[1].set_title(f'TreeMIL\nP={p_treemil:.3f}  R={r_treemil:.3f}  F1={f1_treemil:.3f}',
                  fontsize=10, fontweight='bold', pad=10)
axes[1].set_xticklabels(['Normal', 'Attack'], fontsize=10)
axes[1].set_yticklabels(['Normal', 'Attack'], fontsize=10, rotation=0)

plt.tight_layout(pad=2.0)

# 保存为符合TNSM要求的高分辨率图片
# 彩色/灰度图要求≥300 dpi，这里使用600 dpi以确保质量
output_file = 'confusion_matrix_comparison_tnsm.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight',
            pad_inches=0.1, facecolor='white')
print(f"✓ 混淆矩阵对比图已保存至: {output_file}")

# 同时保存TIFF格式（备选）
output_tiff = 'confusion_matrix_comparison_tnsm.tiff'
plt.savefig(output_tiff, dpi=600, bbox_inches='tight',
            pad_inches=0.1, facecolor='white')
print(f"✓ TIFF格式已保存至: {output_tiff}")

print(f"✓ 尺寸: 7.16 × 3.5 inches (TNSM整页宽度)")
print(f"✓ 分辨率: 600 dpi")
print(f"✓ 字体: Liberation Serif (Times New Roman替代)")

# 打印两个模型的详细信息
print("\n" + "="*50)
print("WSTD:")
print("="*50)
TN, FP = cm_mymodel[0]
FN, TP = cm_mymodel[1]
print(f"混淆矩阵: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"Precision = {TP}/{TP+FP} = {TP/(TP+FP):.4f}")
print(f"Recall = {TP}/{TP+FN} = {TP/(TP+FN):.4f}")
print(f"F1-Score = {2*p_mymodel*r_mymodel/(p_mymodel+r_mymodel):.4f}")

print("\n" + "="*50)
print("TreeMIL:")
print("="*50)
TN, FP = cm_treemil[0]
FN, TP = cm_treemil[1]
print(f"混淆矩阵: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"Precision = {TP}/{TP+FP} = {TP/(TP+FP):.4f}")
print(f"Recall = {TP}/{TP+FN} = {TP/(TP+FN):.4f}")
print(f"F1-Score = {2*p_treemil*r_treemil/(p_treemil+r_treemil):.4f}")

# 显示实际使用的字体
import matplotlib.font_manager as fm
font_prop = fm.FontProperties(family='serif')
actual_font = fm.findfont(font_prop)
print(f"\n✓ 实际使用字体: {actual_font}")

# 显示文件大小
import os
png_size = os.path.getsize(output_file) / 1024
tiff_size = os.path.getsize(output_tiff) / 1024
print(f"\n文件大小:")
print(f"  PNG:  {png_size:.1f} KB")
print(f"  TIFF: {tiff_size:.1f} KB")

plt.close()