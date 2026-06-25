#!/usr/bin/env python3
"""
Generate TNSM-compliant comparison table - CICIDS2017 vs CICIDS2018
- Double column width: 7.16 inches
- Resolution: >=600 dpi
- Font: Liberation Serif (Times New Roman alternative)
- Format: Professional IEEE style with compact spacing
"""

import matplotlib.pyplot as plt
from matplotlib.table import Table

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 10

# ========== 数据 ==========
# 统一方法名和顺序: (Method, [2017: AUC-ROC, F1, F1*], [2018: AUC-ROC, F1, F1*])
data = [
    ['Isolation Forest', '0.561', '0.451', '0.604',  '0.500', '0.459', '0.498'],
    ['LSTM-AE',          '0.672', '0.554', '0.673',  '0.511', '0.496', '0.532'],
    ['OmniAnomaly',      '0.787', '0.664', '0.772',  '0.669', '0.594', '0.689'],
    ['USAD',             '0.712', '0.586', '0.694',  '0.547', '0.504', '0.535'],
    ['MemAE',            '0.756', '0.637', '0.714',  '0.657', '0.573', '0.639'],
    ['TranAD',           '0.791', '0.653', '0.716',  '0.668', '0.577', '0.655'],
    ['DTAAD',            '0.796', '0.674', '0.745',  '0.711', '0.618', '0.681'],
    ['TransDe',         '0.824', '0.697', '0.759',  '0.746', '0.649', '0.703'],
    ['STFT-TCAN',        '0.798', '0.669', '0.737',  '0.714', '0.609', '0.674'],
    ['FreqDAR',         '0.817', '0.721', '0.772',  '0.753', '0.685', '0.728'],
]

# ========== 表头 ==========
# Row 0: merged headers  (Method, CICIDS2017 x3, CICIDS2018 x3)
# Row 1: sub-headers      (     , AUC-ROC, F1, F1*, AUC-ROC, F1, F1*)
sub_headers = ['Method', 'AUC-ROC', 'F1', 'F1*', 'AUC-ROC', 'F1', 'F1*']

n_data_rows = len(data)
n_rows = n_data_rows + 2  # 2 header rows

# ========== 布局 ==========
fig_width = 7.16   # TNSM double-column
fig_height = 2.8
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# 列宽比例：
# 1. 缩窄 Method 列
# 2. AUC-ROC 列略宽，避免表头拥挤
# 3. F1 / F1* 列保持紧凑
col_w = [0.18, 0.14, 0.12, 0.12, 0.14, 0.12, 0.12]
s = sum(col_w)
col_w = [w / s for w in col_w]

table_width = 0.96
table_height = 0.93
table_left = 0.02
table_bottom = 0.02
cell_h = table_height / n_rows

table = Table(ax, bbox=[table_left, table_bottom, table_width, table_height])

# ---------- Row 0: 合并表头行 ----------
# Method 列（占位，跨 Row0 和 Row1 视觉上合并）
cell = table.add_cell(0, 0, width=col_w[0] * table_width, height=cell_h,
                       text='', loc='center', facecolor='white')
cell.set_edgecolor('black')
cell.set_linewidth(1.2)
cell.visible_edges = 'T'
cell.PAD = 0.02

# CICIDS2017 标题横跨 col 1-3
for j in range(1, 4):
    text = 'CICIDS2017' if j == 2 else ''
    cell = table.add_cell(0, j, width=col_w[j] * table_width, height=cell_h,
                           text=text, loc='center', facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black')
    cell.set_linewidth(1.2)
    cell.visible_edges = 'T'
    cell.PAD = 0.02

# CICIDS2018 标题横跨 col 4-6
for j in range(4, 7):
    text = 'CICIDS2018' if j == 5 else ''
    cell = table.add_cell(0, j, width=col_w[j] * table_width, height=cell_h,
                           text=text, loc='center', facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black')
    cell.set_linewidth(1.2)
    cell.visible_edges = 'T'
    cell.PAD = 0.02

# ---------- Row 1: 子表头行 ----------
for j, h in enumerate(sub_headers):
    cell = table.add_cell(1, j, width=col_w[j] * table_width, height=cell_h,
                           text=h, loc='center', facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black')
    cell.set_linewidth(0.8)
    cell.visible_edges = 'B'
    cell.PAD = 0.02

# ---------- 数据行 ----------
for i, row in enumerate(data):
    row_idx = i + 2  # 表格行号（前 2 行是表头）
    is_last = (i == n_data_rows - 1)

    for j, val in enumerate(row):
        # Method 列和数值列都居中，避免左偏
        cell = table.add_cell(row_idx, j, width=col_w[j] * table_width,
                              height=cell_h, text=val, loc='center',
                              facecolor='white')

        # 与原脚本一致：最后一行加粗，强调本文方法
        weight = 'bold' if is_last else 'normal'
        cell.set_text_props(weight=weight, fontsize=10)
        cell.set_edgecolor('black')
        cell.PAD = 0.02

        if is_last:
            cell.set_linewidth(1.2)
            cell.visible_edges = 'B'
        else:
            cell.visible_edges = ''

ax.add_table(table)
plt.tight_layout(pad=0.05)

# ========== 保存 ==========
output_file = './table_screenshot_metrics_centered.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)

print(f"✓ 表格已生成: {output_file}")
print(f"✓ 尺寸: {fig_width} × {fig_height} inches")
print("✓ 分辨率: 600 dpi")
print("✓ 已缩窄 Method 列，并将 Method 列内容居中")

plt.close()
