#!/usr/bin/env python3
"""
Generate TNSM-compliant efficiency comparison table
"""

import matplotlib.pyplot as plt
from matplotlib.table import Table

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 10

# ========== 数据（按 Latency 排序，WSTD 放最后） ==========
rows = [
    ['MLP',         '1.04M',  '0.04',  '325.33M', '19.7'],
    ['LSTM-AE',     '1.03M',  '1.91',  '6.69M',   '170.9'],
    ['TCN',         '1.75M',  '2.22',  '5.78M',   '141.2'],
    ['LSTM',        '2.18M',  '2.45',  '5.23M',   '243.0'],
    ['WSTD',        '1.76M',  '3.58',  '3.57M',   '107.1'],
    ['Transformer', '2.12M',  '3.98',  '3.22M',   '125.7'],
    ['FEDformer',   '0.94M',  '4.43',  '2.89M',   '95.0'],
    ['TreeMIL',     '0.61M',  '15.40', '0.83M',   '253.2'],
]

headers = ['Method', 'Params', 'Latency (ms)', 'Throughput (flows/s)', 'GPU Mem (MB)']
n_cols = len(headers)
n_data = len(rows)
n_rows = n_data + 1

# ========== 布局 ==========
fig_width = 7.16
fig_height = 2.7
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

col_w = [0.18, 0.14, 0.18, 0.28, 0.18]
s = sum(col_w)
col_w = [w / s for w in col_w]

tw, th = 0.96, 0.93
tl, tb = 0.02, 0.02
cell_h = th / n_rows

table = Table(ax, bbox=[tl, tb, tw, th])

# ---------- 表头 ----------
for j, h in enumerate(headers):
    c = table.add_cell(0, j, width=col_w[j]*tw, height=cell_h,
                       text=h, loc='center', facecolor='white')
    c.set_text_props(weight='bold', fontsize=10)
    c.set_edgecolor('black'); c.set_linewidth(1.2)
    c.visible_edges = 'TB'; c.PAD = 0.03

# ---------- 数据行 ----------
for i, row in enumerate(rows):
    row_idx = i + 1
    is_last = (i == n_data - 1)
    for j, val in enumerate(row):
        loc = 'left' if j == 0 else 'center'
        c = table.add_cell(row_idx, j, width=col_w[j]*tw, height=cell_h,
                           text=val, loc=loc, facecolor='white')
        is_ours = (row[0] == 'WSTD')
        weight = 'bold' if is_ours else 'normal'
        c.set_text_props(weight=weight, fontsize=10)
        c.set_edgecolor('black'); c.PAD = 0.03
        if is_last:
            c.set_linewidth(1.2); c.visible_edges = 'B'
        else:
            c.visible_edges = ''

ax.add_table(table)
plt.tight_layout(pad=0.05)

output = './table_efficiency.png'
plt.savefig(output, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)
plt.close()
print(f"✓ Done: {output}")