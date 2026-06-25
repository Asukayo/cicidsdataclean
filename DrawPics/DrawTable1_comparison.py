#!/usr/bin/env python3
"""
Generate TNSM-compliant comparison table - CICIDS2017 vs CICIDS2018
- Double column width: 7.16 inches
- Resolution: ≥600 dpi
- Font: Liberation Serif (Times New Roman替代字体)
- Format: Professional IEEE style with compact spacing
"""

import matplotlib.pyplot as plt
from matplotlib.table import Table
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 10

# ========== 数据 ==========
# 统一方法名和顺序: (Method, [2017: P, R, F1], [2018: P, R, F1])
data = [
    ['Random Forest', '0.661','0.596','0.627',  '0.565','0.460','0.507'],
    ['OCSVM',         '0.676','0.579','0.624',  '0.642','0.498','0.561'],
    ['MLP',           '0.758','0.740','0.749',  '0.697','0.612','0.652'],
    ['LSTM',          '0.767','0.743','0.755',  '0.719','0.633','0.673'],
    ['Transformer',   '0.772','0.756','0.764',  '0.735','0.606','0.664'],
    ['TCN',           '0.785','0.754','0.769',  '0.749','0.658','0.701'],
    ['LSTM-AE',       '0.786','0.741','0.763',  '0.744','0.609','0.670'],
    ['FEDformer',     '0.791','0.762','0.776',  '0.755','0.647','0.697'],
    ['TreeMIL',       '0.828','0.781','0.804',  '0.778','0.662','0.715'],
    ['WSTD',          '0.872','0.855','0.864',  '0.824','0.712','0.764'],
]

# ========== 表头 ==========
# Row 0: merged headers  (Method, CICIDS2017 x3, CICIDS2018 x3)
# Row 1: sub-headers      (     , P, R, F1,       P, R, F1)
sub_headers = ['Method', 'P', 'R', 'F1', 'P', 'R', 'F1']

n_cols = 7
n_data_rows = len(data)
n_rows = n_data_rows + 2  # 2 header rows

# ========== 布局 ==========
fig_width = 7.16   # TNSM double-column
fig_height = 2.8
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# 列宽比例 (总和=1.0)
col_w = [0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
# 归一化
s = sum(col_w)
col_w = [w / s for w in col_w]

table_width = 0.96
table_height = 0.93
table_left = 0.02
table_bottom = 0.02
cell_h = table_height / n_rows

table = Table(ax, bbox=[table_left, table_bottom, table_width, table_height])

# ---------- Row 0: 合并表头行 ----------
# Method 列（占位，跨Row0和Row1视觉上合并）
cell = table.add_cell(0, 0, width=col_w[0]*table_width, height=cell_h,
                       text='', loc='center', facecolor='white')
cell.set_edgecolor('black'); cell.set_linewidth(1.2)
cell.visible_edges = 'T'; cell.PAD = 0.03

# CICIDS2017 标题横跨 col 1-3
for j in range(1, 4):
    text = 'CICIDS2017' if j == 2 else ''
    cell = table.add_cell(0, j, width=col_w[j]*table_width, height=cell_h,
                           text=text, loc='center', facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black'); cell.set_linewidth(1.2)
    cell.visible_edges = 'T'; cell.PAD = 0.03

# CICIDS2018 标题横跨 col 4-6
for j in range(4, 7):
    text = 'CICIDS2018' if j == 5 else ''
    cell = table.add_cell(0, j, width=col_w[j]*table_width, height=cell_h,
                           text=text, loc='center', facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black'); cell.set_linewidth(1.2)
    cell.visible_edges = 'T'; cell.PAD = 0.03

# ---------- Row 1: 子表头行 ----------
for j, h in enumerate(sub_headers):
    loc = 'center'
    cell = table.add_cell(1, j, width=col_w[j]*table_width, height=cell_h,
                           text=h, loc=loc, facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    cell.set_edgecolor('black'); cell.set_linewidth(0.8)
    cell.visible_edges = 'B'; cell.PAD = 0.03

# ---------- 数据行 ----------
for i, row in enumerate(data):
    row_idx = i + 2  # 表格行号（前2行是表头）
    is_last = (i == n_data_rows - 1)
    for j, val in enumerate(row):
        loc = 'left' if j == 0 else 'center'
        cell = table.add_cell(row_idx, j, width=col_w[j]*table_width,
                               height=cell_h, text=val, loc=loc,
                               facecolor='white')
        weight = 'bold' if is_last else 'normal'
        cell.set_text_props(weight=weight, fontsize=10)
        cell.set_edgecolor('black')
        cell.PAD = 0.03

        if is_last:
            cell.set_linewidth(1.2); cell.visible_edges = 'B'
        else:
            cell.visible_edges = ''

ax.add_table(table)
plt.tight_layout(pad=0.05)

# ========== 保存 ==========
output_file = './table_comparison_two_datasets.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)
print(f"✓ 表格已生成: {output_file}")
print(f"✓ 尺寸: {fig_width} × {fig_height} inches")
print(f"✓ 分辨率: 600 dpi")

plt.close()