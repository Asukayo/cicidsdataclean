import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib参数以支持更好的显示
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.family'] = 'Arial'


def load_and_preprocess_data(file_path):
    """
    加载并预处理CICIDS2017数据

    参数:
    - file_path: parquet文件路径

    返回:
    - df: 预处理后的DataFrame
    """
    print("Loading data...")

    # 加载parquet文件
    df = pd.read_parquet(file_path)
    print(f"Original data shape: {df.shape}")

    # 检查并处理Timestamp列
    if 'Timestamp' not in df.columns:
        raise ValueError("Timestamp column not found in the data")

    # 转换时间戳为datetime类型
    print("Processing timestamps...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # 删除无效时间戳的行
    before_drop = len(df)
    df = df.dropna(subset=['Timestamp'])
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows with invalid timestamps")

    # 按时间戳排序
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # 显示时间范围
    print(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Total duration: {df['Timestamp'].max() - df['Timestamp'].min()}")

    return df


def create_time_windows(df, window_size_seconds=60, overlap_seconds=0):
    """
    创建固定时间窗口并统计每个窗口中的流量数量

    参数:
    - df: 输入DataFrame
    - window_size_seconds: 窗口大小（秒）
    - overlap_seconds: 窗口重叠时间（秒）

    返回:
    - window_stats: 包含窗口统计信息的DataFrame
    """
    print(f"Creating time windows of {window_size_seconds} seconds...")

    # 获取时间范围
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    # 计算步长
    step_seconds = window_size_seconds - overlap_seconds

    # 生成窗口
    windows = []
    current_time = start_time
    window_id = 0

    while current_time + timedelta(seconds=window_size_seconds) <= end_time:
        window_start = current_time
        window_end = current_time + timedelta(seconds=window_size_seconds)

        # 统计当前窗口内的流量
        window_flows = df[(df['Timestamp'] >= window_start) &
                          (df['Timestamp'] < window_end)]

        # 统计标签分布
        label_counts = window_flows['Label'].value_counts().to_dict() if 'Label' in df.columns else {}
        benign_count = label_counts.get('Benign', 0) + label_counts.get('BENIGN', 0)
        malicious_count = len(window_flows) - benign_count

        window_info = {
            'window_id': window_id,
            'start_time': window_start,
            'end_time': window_end,
            'flow_count': len(window_flows),
            'benign_flows': benign_count,
            'malicious_flows': malicious_count,
            'malicious_ratio': malicious_count / len(window_flows) if len(window_flows) > 0 else 0
        }

        # 添加攻击类型统计
        if 'Label' in df.columns:
            attack_types = {}
            for label, count in label_counts.items():
                if label not in ['Benign', 'BENIGN']:
                    attack_types[label] = count
            window_info['attack_types'] = attack_types
            window_info['primary_attack'] = max(attack_types, key=attack_types.get) if attack_types else 'None'

        windows.append(window_info)

        # 移动到下一个窗口
        current_time += timedelta(seconds=step_seconds)
        window_id += 1

    window_stats = pd.DataFrame(windows)
    print(f"Created {len(window_stats)} time windows")

    return window_stats


def plot_flow_distribution(window_stats, save_path=None):
    """
    绘制流量分布图表

    参数:
    - window_stats: 窗口统计DataFrame
    - save_path: 保存路径（可选）
    """

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Flow Distribution Analysis in 60-Second Time Windows\n(Friday Morning - CICIDS2017)',
                 fontsize=16, fontweight='bold')

    # 1. 流量数量时间序列图
    ax1 = axes[0, 0]
    ax1.plot(window_stats['start_time'], window_stats['flow_count'],
             linewidth=1.5, color='steelblue', alpha=0.8)
    ax1.fill_between(window_stats['start_time'], window_stats['flow_count'],
                     alpha=0.3, color='steelblue')
    ax1.set_title('Flow Count Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Number of Flows', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 添加统计信息
    mean_flows = window_stats['flow_count'].mean()
    max_flows = window_stats['flow_count'].max()
    ax1.axhline(y=mean_flows, color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_flows:.1f}')
    ax1.axhline(y=max_flows, color='orange', linestyle='--', alpha=0.7,
                label=f'Max: {max_flows}')
    ax1.legend()

    # 2. 流量数量分布直方图
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(window_stats['flow_count'], bins=30,
                                color='lightblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Flow Counts', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Flows per Window', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Mean: {window_stats["flow_count"].mean():.1f}\n'
    stats_text += f'Std: {window_stats["flow_count"].std():.1f}\n'
    stats_text += f'Min: {window_stats["flow_count"].min()}\n'
    stats_text += f'Max: {window_stats["flow_count"].max()}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. 恶意vs正常流量堆叠图
    ax3 = axes[1, 0]
    ax3.stackplot(window_stats['start_time'],
                  window_stats['benign_flows'],
                  window_stats['malicious_flows'],
                  labels=['Benign Flows', 'Malicious Flows'],
                  colors=['lightgreen', 'lightcoral'], alpha=0.8)
    ax3.set_title('Benign vs Malicious Flows Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Number of Flows', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 4. 恶意流量比例散点图
    ax4 = axes[1, 1]
    # 创建颜色映射
    colors = ['green' if ratio == 0 else 'red' for ratio in window_stats['malicious_ratio']]
    scatter = ax4.scatter(window_stats['start_time'], window_stats['malicious_ratio'],
                          c=colors, alpha=0.6, s=30)
    ax4.set_title('Malicious Flow Ratio Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('Malicious Flow Ratio', fontsize=12)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # 添加水平线表示不同的威胁级别
    ax4.axhline(y=0.1, color='yellow', linestyle='--', alpha=0.5, label='10% threshold')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def analyze_attack_patterns(window_stats):
    """
    分析攻击模式

    参数:
    - window_stats: 窗口统计DataFrame
    """
    print("\n=== Attack Pattern Analysis ===")

    # 统计有恶意流量的窗口
    malicious_windows = window_stats[window_stats['malicious_flows'] > 0]
    print(f"Windows with malicious traffic: {len(malicious_windows)} out of {len(window_stats)} "
          f"({len(malicious_windows) / len(window_stats) * 100:.1f}%)")

    if len(malicious_windows) > 0:
        # 统计攻击类型
        all_attacks = {}
        for _, window in malicious_windows.iterrows():
            if 'attack_types' in window and window['attack_types']:
                for attack, count in window['attack_types'].items():
                    if attack in all_attacks:
                        all_attacks[attack] += count
                    else:
                        all_attacks[attack] = count

        if all_attacks:
            print("\nAttack types distribution:")
            for attack, count in sorted(all_attacks.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {attack}: {count} flows")

    # 流量统计
    print(f"\nFlow statistics:")
    print(f"  - Total flows: {window_stats['flow_count'].sum()}")
    print(f"  - Benign flows: {window_stats['benign_flows'].sum()}")
    print(f"  - Malicious flows: {window_stats['malicious_flows'].sum()}")
    print(f"  - Average flows per window: {window_stats['flow_count'].mean():.1f}")
    print(f"  - Max flows in a window: {window_stats['flow_count'].max()}")
    print(f"  - Min flows in a window: {window_stats['flow_count'].min()}")


def create_summary_table(window_stats):
    """
    创建汇总统计表

    参数:
    - window_stats: 窗口统计DataFrame

    返回:
    - summary_df: 汇总统计DataFrame
    """

    # 按小时分组统计
    window_stats['hour'] = window_stats['start_time'].dt.hour
    hourly_stats = window_stats.groupby('hour').agg({
        'flow_count': ['mean', 'std', 'min', 'max', 'sum'],
        'malicious_flows': ['sum', 'mean'],
        'benign_flows': ['sum', 'mean'],
        'malicious_ratio': 'mean'
    }).round(2)

    # 重命名列
    hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns.values]

    print("\n=== Hourly Statistics ===")
    print(hourly_stats)

    return hourly_stats


def main():
    """
    主函数
    """
    # 文件路径（请根据实际路径修改）
    file_path = "../cicids2017/clean/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"

    try:
        # 1. 加载和预处理数据
        df = load_and_preprocess_data(file_path)

        # 2. 创建时间窗口
        window_stats = create_time_windows(df, window_size_seconds=60, overlap_seconds=0)

        # 3. 绘制分布图
        plot_flow_distribution(window_stats, save_path="flow_distribution_analysis.png")

        # 4. 分析攻击模式
        analyze_attack_patterns(window_stats)

        # 5. 创建汇总表
        hourly_summary = create_summary_table(window_stats)

        # 6. 保存结果
        window_stats.to_csv("window_analysis_results.csv", index=False)
        hourly_summary.to_csv("hourly_summary.csv")

        print(f"\nAnalysis complete!")
        print(f"Results saved to:")
        print(f"  - window_analysis_results.csv")
        print(f"  - hourly_summary.csv")
        print(f"  - flow_distribution_analysis.png")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure the file path is correct and the file exists.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()