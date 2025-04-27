import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_window_distribution(window_dict, window_size, step_size, output_dir):
    """绘制窗口中流量计数和时间点分布"""
    # 创建多子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. 绘制流量计数分布
    benign_counts = [w['flow_count'] for w in window_dict['benign']]
    malicious_counts = [w['flow_count'] for w in window_dict['malicious']]

    axes[0].hist(benign_counts, alpha=0.5, label='Benign', bins=30, color='green')
    axes[0].hist(malicious_counts, alpha=0.5, label='Malicious', bins=30, color='red')
    axes[0].set_title(f'Flow Count Distribution (Window: {window_size}min, Step: {step_size}min)')
    axes[0].set_xlabel('Number of Flows per Window')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 绘制唯一时间点分布
    benign_timestamps = [w['unique_timestamps'] for w in window_dict['benign']]
    malicious_timestamps = [w['unique_timestamps'] for w in window_dict['malicious']]

    axes[1].hist(benign_timestamps, alpha=0.5, label='Benign', bins=30, color='green')
    axes[1].hist(malicious_timestamps, alpha=0.5, label='Malicious', bins=30, color='red')
    axes[1].set_title(f'Unique Timestamps Distribution (Window: {window_size}min, Step: {step_size}min)')
    axes[1].set_xlabel('Number of Unique Timestamps per Window')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_w{window_size}_s{step_size}.png'))
    plt.close()

    # 额外绘制时间密度散点图
    plt.figure(figsize=(10, 6))
    benign_density = [w['time_density'] for w in window_dict['benign']]
    plt.scatter(benign_counts, benign_density, alpha=0.5, label='Benign', color='green')

    if window_dict['malicious']:
        malicious_density = [w['time_density'] for w in window_dict['malicious']]
        plt.scatter(malicious_counts, malicious_density, alpha=0.5, label='Malicious', color='red')

    plt.title(f'Flow Count vs Time Density (Window: {window_size}min, Step: {step_size}min)')
    plt.xlabel('Number of Flows per Window')
    plt.ylabel('Time Density (Unique Timestamps per Minute)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'time_density_w{window_size}_s{step_size}.png'))
    plt.close()

    # 如果有异常窗口，绘制攻击类型分布
    if window_dict['malicious']:
        # 合并所有攻击类型
        all_attacks = {}
        for window in window_dict['malicious']:
            for attack_type, count in window['attack_types'].items():
                if attack_type in all_attacks:
                    all_attacks[attack_type] += count
                else:
                    all_attacks[attack_type] = count

        # 绘制攻击类型分布
        plt.figure(figsize=(12, 6))
        attacks_df = pd.Series(all_attacks).sort_values(ascending=False)
        attacks_df.plot(kind='bar', color='crimson')
        plt.title(f'Attack Type Distribution (Window: {window_size}min, Step: {step_size}min)')
        plt.xlabel('Attack Type')
        plt.ylabel('Flow Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attacks_w{window_size}_s{step_size}.png'))
        plt.close()

def analyze_time_distribution(dataset_path):
    """分析数据集中的时间戳分布"""
    if dataset_path.endswith('.feather'):
        df = pd.read_feather(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .feather 或 .parquet 文件")

    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    print("时间戳分析:")
    print(f"记录总数: {len(df)}")
    print(f"唯一时间戳数: {df['Timestamp'].nunique()}")

    # 计算重复情况
    time_counts = df['Timestamp'].value_counts().reset_index()
    time_counts.columns = ['timestamp', 'count']

    print(f"最大重复次数: {time_counts['count'].max()}")
    print(f"平均重复次数: {time_counts['count'].mean():.2f}")

    # 绘制重复次数分布
    plt.figure(figsize=(10, 6))
    plt.hist(time_counts['count'], bins=30, color='teal', alpha=0.7)
    plt.title('Timestamp Repetition Distribution')
    plt.xlabel('Number of Records per Timestamp')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('timestamp_repetition.png')
    plt.close()

    # 绘制时间戳的时间序列
    plt.figure(figsize=(12, 6))

    # 为了可视化效果，采样一部分数据点
    sample_size = min(10000, len(df))
    df_sample = df.sample(sample_size)

    benign_df = df_sample[df_sample['Label'] == 'Benign']
    attack_df = df_sample[df_sample['Label'] != 'Benign']

    plt.scatter(benign_df['Timestamp'], [1] * len(benign_df),
                marker='|', color='green', alpha=0.5, label='Benign')
    plt.scatter(attack_df['Timestamp'], [1] * len(attack_df),
                marker='|', color='red', alpha=0.5, label='Attack')

    plt.title('Timestamp Distribution Over Time (Sample)')
    plt.xlabel('Time')
    plt.ylabel('Occurrence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('timestamp_distribution.png')
    plt.close()

    return time_counts
