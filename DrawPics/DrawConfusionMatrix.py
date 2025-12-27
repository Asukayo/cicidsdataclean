import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 创建图形：包含两个子图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：MyModel
sns.heatmap(cm_mymodel, annot=True, fmt='d', cmap='Blues',
            cbar=True, square=True, linewidths=2, linecolor='black',
            annot_kws={'size': 16, 'weight': 'bold'},
            cbar_kws={'label': 'Count'}, ax=axes[0])

axes[0].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=13, fontweight='bold')
axes[0].set_title(f'MyModel\nP={p_mymodel:.3f}  R={r_mymodel:.3f}  F1={f1_mymodel:.3f}',
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xticklabels(['Normal', 'Attack'], fontsize=11)
axes[0].set_yticklabels(['Normal', 'Attack'], fontsize=11, rotation=0)

# 右图：TreeMIL
sns.heatmap(cm_treemil, annot=True, fmt='d', cmap='Oranges',
            cbar=True, square=True, linewidths=2, linecolor='black',
            annot_kws={'size': 16, 'weight': 'bold'},
            cbar_kws={'label': 'Count'}, ax=axes[1])

axes[1].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=13, fontweight='bold')
axes[1].set_title(f'TreeMIL\nP={p_treemil:.3f}  R={r_treemil:.3f}  F1={f1_treemil:.3f}',
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xticklabels(['Normal', 'Attack'], fontsize=11)
axes[1].set_yticklabels(['Normal', 'Attack'], fontsize=11, rotation=0)

plt.tight_layout()
plt.savefig('./confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("混淆矩阵对比图已保存至: confusion_matrix_comparison.png")

# 打印两个模型的详细信息
print("\n" + "="*50)
print("MyModel:")
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

plt.show()