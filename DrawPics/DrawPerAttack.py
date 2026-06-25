#!/usr/bin/env python3
"""
Generate TNSM-compliant per-attack detection results table
- CICIDS2017 and CICIDS2018 grouped in one table
- Double column width: 7.16 inches
- Resolution: ≥600 dpi
"""

import matplotlib.pyplot as plt
from matplotlib.table import Table

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 10

# ========== 数据 ==========
# (Category, Attack Type, Count, TP, FN, Detection Rate)
rows = [
    # CICIDS2017
    ('CICIDS2017', 'DDoS',             '7,047',  '6,496',  '551',   '0.922'),
    ('',           'PortScan',         '5,356',  '4,419',  '937',   '0.825'),
    ('',           'Bot',              '1,194',  '716',    '478',   '0.600'),
    ('',           'Benign',           '20,122', '18,426', '1,696', '0.916'),
    # CICIDS2018
    ('CICIDS2018', 'Brute Force-Web',  '889',    '772',    '117',   '0.868'),
    ('',           'Brute Force-XSS',  '426',    '355',    '71',    '0.833'),
    ('',           'Bot',              '28,309', '21,260', '7,049', '0.751'),
    ('',           'Infiltration',     '10,400', '6,140',  '4,260', '0.590'),
    ('',           'SQL Injection',    '188',    '98',     '90',    '0.521'),
    ('',           'Benign',           '80,854', '74,754', '6,100', '0.925'),
]

headers = ['Dataset', 'Category', 'Count', 'TP', 'FN', 'DR']
n_cols = len(headers)
n_data = len(rows)
n_rows = n_data + 1  # +1 header

# ========== 布局 ==========
fig_width = 7.16
fig_height = 3.2
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

col_w = [0.15, 0.20, 0.14, 0.14, 0.14, 0.14]
s = sum(col_w)
col_w = [w / s for w in col_w]

tw, th = 0.96, 0.94
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
# 分组分隔线位置：CICIDS2017最后一行（index 3）底部画线, CICIDS2018最后一行底部画线
sep_after = {3}  # 0-indexed data row，第4行(Benign of 2017)后加分隔线

for i, row in enumerate(rows):
    row_idx = i + 1
    is_last = (i == n_data - 1)
    is_sep = (i in sep_after)

    for j, val in enumerate(row):
        loc = 'left' if j <= 1 else 'center'
        c = table.add_cell(row_idx, j, width=col_w[j]*tw, height=cell_h,
                           text=val, loc=loc, facecolor='white')

        # Dataset列(j==0)有文字时加粗
        if j == 0 and val:
            c.set_text_props(weight='bold', fontsize=10, style='italic')
        else:
            c.set_text_props(weight='normal', fontsize=10)

        c.set_edgecolor('black'); c.PAD = 0.03

        if is_last:
            c.set_linewidth(1.2); c.visible_edges = 'B'
        elif is_sep:
            c.set_linewidth(0.6); c.visible_edges = 'B'
        else:
            c.visible_edges = ''

ax.add_table(table)
plt.tight_layout(pad=0.05)

output = './ppngs/table_per_attack.png'
plt.savefig(output, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)
plt.close()
print(f"✓ Done: {output}")