#!/usr/bin/env python3
"""
Generate TNSM-compliant Ablation Study Table - Ultra Compact Version
Changes:
- Larger font for data rows (11 → 12)
- Even more compact layout (height reduced further)
- Tighter spacing overall
"""

import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np

# ========== TNSM字体设置 ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 14  # 基础字体增大到12

# 表格数据
data = [
    ['w/o HSTD', '0.800', '0.813', '0.806'],
    ['w/o MSTE', '0.859', '0.803', '0.830'],
    ['Replace PTA with LSTM', '0.836', '0.813', '0.824'],
    ['Fixed $g=0.5$', '0.795', '0.838', '0.816'],
    ['w/o residual', '0.764', '0.796', '0.780'],
    ['WSTD', '0.872', '0.855', '0.864']  # 完整模型，最后一行加粗
]

# 表头
headers = ['Model Configuration', 'Precision', 'Recall', 'F1-Score']

# 创建图形 - 进一步减小高度使更紧凑
fig_width = 3.5  # inches (TNSM单栏宽度)
fig_height = 2.0  # inches (从2.2进一步减少到2.0，更紧凑)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# 列宽设置（相对比例）
col_widths = [0.42, 0.20, 0.19, 0.19]  # Model Configuration列更宽

# 计算表格位置和大小
table_height = 0.96  # 进一步增加占比
table_width = 0.95
table_left = 0.025
table_bottom = 0.015  # 进一步减小底部边距

# 创建表格
table = Table(ax, bbox=[table_left, table_bottom, table_width, table_height])

# 行数和列数
n_rows = len(data) + 1  # +1 for header
n_cols = len(headers)

# 单元格高度
cell_height = table_height / n_rows

# 添加表头
for j, header in enumerate(headers):
    cell = table.add_cell(0, j,
                          width=col_widths[j] * table_width,
                          height=cell_height,
                          text=header,
                          loc='center',
                          facecolor='white')
    cell.set_text_props(weight='bold', fontsize=11)  # 表头保持11号
    # 只在顶部和底部添加边框
    cell.set_edgecolor('black')
    cell.set_linewidth(1.2)
    cell.visible_edges = 'TB'  # Top and Bottom
    cell.PAD = 0.015  # 进一步减小内边距

# 添加数据行
for i, row in enumerate(data, start=1):
    for j, cell_text in enumerate(row):
        # 判断是否是最后一行（WSTD - 完整模型）
        is_last_row = (i == len(data))

        # 文本对齐：Model Configuration列左对齐，数值列居中
        loc = 'left' if j == 0 else 'center'

        cell = table.add_cell(i, j,
                              width=col_widths[j] * table_width,
                              height=cell_height,
                              text=cell_text,
                              loc=loc,
                              facecolor='white')

        # 设置字体 - 数据行使用12号字体
        if is_last_row:
            # 最后一行加粗（完整模型WSTD）
            cell.set_text_props(weight='bold', fontsize=12)  # 增大到12号
        else:
            cell.set_text_props(weight='normal', fontsize=12)  # 增大到12号

        # 进一步减小单元格内边距，更紧凑
        cell.PAD = 0.015  # 从0.02进一步减少到0.015

        # 设置边框
        cell.set_edgecolor('black')
        if i == 1:
            # 第一行数据：顶部线
            cell.set_linewidth(0.8)
            cell.visible_edges = 'T'
        elif is_last_row:
            # 最后一行：底部线
            cell.set_linewidth(1.2)
            cell.visible_edges = 'B'
        else:
            # 中间行：无边框
            cell.visible_edges = ''

# 添加表格到坐标系
ax.add_table(table)

# 调整布局 - 更紧凑
plt.tight_layout(pad=0.02)  # 进一步减小

# 保存为高分辨率图片
output_file = 'table_ablation_ultra_compact.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.01)

print(f"✓ 超紧凑版消融实验表格已生成: {output_file}")
print(f"✓ 尺寸: {fig_width} × {fig_height} inches (超紧凑)")
print(f"✓ 分辨率: 600 dpi")
print(f"✓ 表头字体: 11号")
print(f"✓ 数据行字体: 12号 (更大更清晰)")
print(f"✓ 布局: 超紧凑 (高度2.0英寸)")

# 同时保存为TIFF格式（备选）
output_tiff = 'table_ablation_ultra_compact.tiff'
plt.savefig(output_tiff, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.01)
print(f"✓ TIFF格式已保存: {output_tiff}")

plt.close()

# 显示实际使用的字体
import matplotlib.font_manager as fm

font_prop = fm.FontProperties(family='serif')
actual_font = fm.findfont(font_prop)
print(f"\n✓ 实际使用字体: {actual_font}")

# 显示文件信息
import os

png_size = os.path.getsize(output_file) / 1024  # KB
tiff_size = os.path.getsize(output_tiff) / 1024  # KB
print(f"\n文件大小:")
print(f"  PNG:  {png_size:.1f} KB")
print(f"  TIFF: {tiff_size:.1f} KB")

print("\n改进说明:")
print("  ✅ 表头字体: 11号")
print("  ✅ 数据行字体: 12号 (从11增大)")
print("  ✅ 高度: 2.0英寸 (从2.2减小)")
print("  ✅ 单元格内边距: 0.015 (从0.02减小)")
print("  ✅ 字体更大 + 布局超紧凑")