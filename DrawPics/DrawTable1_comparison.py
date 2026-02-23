#!/usr/bin/env python3
"""
Generate TNSM-compliant table image - Improved Version
Requirements:
- Single column width: 3.5 inches
- Resolution: ≥600 dpi
- Font: Liberation Serif (Times New Roman替代字体)
- Format: Professional IEEE style with compact spacing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np

# 字体设置 - Liberation Serif是Times New Roman的最佳开源替代
# 系统中没有Times New Roman，使用Liberation Serif（度量完全兼容）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['font.size'] = 10

# 表格数据
data = [
    ['Random Forest', '0.661', '0.596', '0.627'],
    ['OCSVM', '0.676', '0.579', '0.624'],
    ['LSTM', '0.767', '0.743', '0.755'],
    ['MLP', '0.758', '0.740', '0.749'],
    ['Transformer', '0.772', '0.756', '0.764'],
    ['TCN', '0.785', '0.754', '0.769'],
    ['LSTM-AE', '0.786', '0.741', '0.763'],
    ['TreeMIL', '0.828', '0.781', '0.804'],
    ['WSTD', '0.872', '0.855', '0.864']  # 最后一行（提出的方法）
]

# 表头
headers = ['Method', 'Precision', 'Recall', 'F1-Score']

# 创建图形 - 单栏宽度3.5英寸，高度减小使行间距更紧凑
fig_width = 3.5  # inches (TNSM单栏宽度)
fig_height = 2.5  # inches (减小高度使表格更紧凑，原来是3.0)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')

# 列宽设置（相对比例）
col_widths = [0.35, 0.22, 0.22, 0.21]  # Method列宽一些，其他均匀分布

# 计算表格位置和大小 - 调整以获得更紧凑的布局
table_height = 0.92  # 增加表格占比，使其更紧凑
table_width = 0.95
table_left = 0.025
table_bottom = 0.03  # 减小底部边距

# 创建表格
table = Table(ax, bbox=[table_left, table_bottom, table_width, table_height])

# 行数和列数
n_rows = len(data) + 1  # +1 for header
n_cols = len(headers)

# 单元格高度 - 更紧凑的间距
cell_height = table_height / n_rows

# 添加表头
for j, header in enumerate(headers):
    cell = table.add_cell(0, j,
                          width=col_widths[j] * table_width,
                          height=cell_height,
                          text=header,
                          loc='center',
                          facecolor='white')
    cell.set_text_props(weight='bold', fontsize=10)
    # 只在顶部和底部添加边框
    cell.set_edgecolor('black')
    cell.set_linewidth(1.2)
    cell.visible_edges = 'TB'  # Top and Bottom
    # 减小单元格内边距使内容更紧凑
    cell.PAD = 0.03  # 默认是0.1，减小到0.03

# 添加数据行
for i, row in enumerate(data, start=1):
    for j, cell_text in enumerate(row):
        # 判断是否是最后一行（WSTD）
        is_last_row = (i == len(data))

        # 文本对齐：Method列左对齐，数值列居中
        loc = 'left' if j == 0 else 'center'

        cell = table.add_cell(i, j,
                              width=col_widths[j] * table_width,
                              height=cell_height,
                              text=cell_text,
                              loc=loc,
                              facecolor='white')

        # 设置字体
        if is_last_row:
            # 最后一行加粗
            cell.set_text_props(weight='bold', fontsize=10)
        else:
            cell.set_text_props(weight='normal', fontsize=10)

        # 减小单元格内边距
        cell.PAD = 0.03  # 默认是0.1，减小到0.03使行间距更紧凑

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

# 调整布局
plt.tight_layout(pad=0.05)  # 减小padding使整体更紧凑

# 保存为高分辨率图片
# PNG格式，600 dpi（符合TNSM线条图/表格要求）
output_file = 'table_comparison_compact.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)

print(f"✓ 表格已生成: {output_file}")
print(f"✓ 尺寸: {fig_width} × {fig_height} inches (更紧凑)")
print(f"✓ 分辨率: 600 dpi")
print(f"✓ 符合TNSM单栏宽度要求 (3.5 inches)")

# 同时保存为TIFF格式（备选）
output_tiff = 'table_comparison_compact.tiff'
plt.savefig(output_tiff, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.02)
print(f"✓ TIFF格式已生成: {output_tiff}")

# 显示使用的字体
import matplotlib.font_manager as fm
font_prop = fm.FontProperties(family='serif')
actual_font = fm.findfont(font_prop)
print(f"\n✓ 实际使用字体: {actual_font}")

plt.close()

# 显示文件信息
import os
png_size = os.path.getsize(output_file) / 1024  # KB
tiff_size = os.path.getsize(output_tiff) / 1024  # KB
print(f"\n文件大小:")
print(f"  PNG:  {png_size:.1f} KB")
print(f"  TIFF: {tiff_size:.1f} KB")

print("\n📝 字体说明:")
print("  - 使用 Liberation Serif (Times New Roman的开源替代)")
print("  - 度量完全兼容，适合IEEE期刊投稿")
print("  - 如需在Windows上使用真正的Times New Roman，请在Windows系统运行此脚本")