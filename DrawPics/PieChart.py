import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib参数解决乱码问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = 'monospace'  # 使用更基础的等宽字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12  # 增大全局字体大小
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染

# 读取数据
df = pd.read_parquet('../cicids2017/clean/all_data.parquet')

# 统计各类别数量
label_counts = df['Label'].value_counts()

# 设置图形大小 - 增大以容纳更大的图例
plt.figure(figsize=(18, 10))

# 定义专业的调色板 - 使用渐变色和对比色
# 为网络流量分析设计的配色方案
colors = [
    '#2E86AB',  # 深蓝色 - 正常流量
    '#A23B72',  # 深紫红 - 高危攻击
    '#F18F01',  # 橙色 - 中等威胁
    '#C73E1D',  # 红色 - 严重攻击
    '#4ECDC4',  # 青绿色 - 扫描类
    '#45B7D1',  # 浅蓝色 - 探测类
    '#96CEB4',  # 浅绿色 - 轻微异常
    '#FFEAA7',  # 浅黄色 - 其他类型
    '#DDA0DD',  # 淡紫色 - 补充颜色1
    '#F0E68C',  # 卡其色 - 补充颜色2
    '#FFB6C1',  # 浅粉色 - 补充颜色3
    '#98FB98',  # 浅绿色 - 补充颜色4
    '#87CEEB',  # 天空蓝 - 补充颜色5
    '#DEB887',  # 淡棕色 - 补充颜色6
    '#F5DEB3'   # 米色 - 补充颜色15
]

# 确保颜色数量足够
while len(colors) < len(label_counts):
    colors.extend(sns.color_palette("husl", len(label_counts) - len(colors)))

# 创建饼图，使用自定义颜色
wedges, texts = plt.pie(label_counts.values,
                       startangle=90,
                       labels=None,
                       autopct=None,
                       colors=colors[:len(label_counts)],
                       wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1.5))

# 在左侧添加图例，显示类别名称和百分比
total = sum(label_counts.values)
legend_labels = []
for label, count in label_counts.items():
    # 彻底清理特殊字符，替换所有非ASCII字符
    clean_label = str(label)
    # 替换常见的特殊破折号字符
    clean_label = clean_label.replace('–', '-').replace('—', '-').replace('−', '-')
    # 移除或替换其他特殊字符
    clean_label = clean_label.replace('→', '->').replace('←', '<-')
    # 只保留ASCII字符
    clean_label = ''.join(char if ord(char) < 128 else '' for char in clean_label)
    # 如果清理后为空，使用简化标签
    if not clean_label.strip():
        clean_label = f'Attack_{len(legend_labels)+1}'
    legend_labels.append(f'{clean_label}: {count:,} ({count/total*100:.1f}%)')

# 优化图例样式 - 放大图例
plt.legend(wedges, legend_labels,
          title="Traffic Categories",
          loc="center left",
          bbox_to_anchor=(1.05, 0, 0.5, 1),
          fontsize=12,
          title_fontsize=14,
          frameon=True,
          fancybox=True,
          shadow=True,
          framealpha=0.9,
          markerscale=2.0,  # 放大图例标记
          markerfirst=True,
          borderpad=1.0,    # 增加图例内边距
          columnspacing=1.0,
          handlelength=2.0,  # 增加图例标记长度
          handletextpad=0.8) # 增加标记与文本间距

# 优化标题样式 - 放大标题
plt.title('CICIDS2017 Dataset - Traffic Categories Distribution',
          fontsize=18, fontweight='bold', pad=25)

# 确保饼图为圆形
plt.axis('equal')

# 调整布局以适应图例
plt.tight_layout()

# 保存图形
plt.savefig('cicids2017_traffic_categories.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')

plt.savefig('cicids2017_traffic_categories.pdf',
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')

# 显示图形
plt.show()

print("已放大图例的优化饼图已保存为 'cicids2017_traffic_categories.png' 和 'cicids2017_traffic_categories.pdf'")