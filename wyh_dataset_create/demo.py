import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from utils.draw_plots import plot_window_distribution,analyze_time_distribution
from utils.create_big_windows import create_time_windows
from utils.create_sub_windows import create_fixed_sub_windows_overlapping

# 步长可以进行修改 step_sizes=[1, 2, 5]
def extract_time_series(dataset_path, output_dir, window_sizes=[5, 10, 15], step_sizes=[1, 2, 5]):
    """
    使用不同窗口大小和步长提取时间序列数据集

    参数:
    - dataset_path: 数据集路径
    - output_dir: 输出目录
    - window_sizes: 窗口大小列表（分钟）
    - step_sizes: 滑动步长列表（分钟）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 为不同窗口大小和步长创建时间序列数据集
    for window_size in window_sizes:
        for step_size in step_sizes:
            print(f"\n处理窗口大小 {window_size} 分钟，步长 {step_size} 分钟")

            # 跳过不合理的组合（步长大于窗口大小）
            if step_size > window_size:
                continue

            result_dict = create_time_windows(
                dataset_path=dataset_path,
                window_size_minutes=window_size,
                step_size_minutes=step_size
            )

            # 保存结果
            output_path = os.path.join(
                output_dir,
                f"time_series_w{window_size}_s{step_size}.pkl"
            )

            with open(output_path, 'wb') as f:
                pickle.dump(result_dict, f)

            print(f"保存到: {output_path}")

            # 绘制窗口分布图
            plot_window_distribution(result_dict, window_size, step_size, output_dir)




def create_flattened_dataset(time_series_path, output_path):
    """
    将时间序列窗口数据转换为扁平化格式，用于机器学习模型训练

    参数:
    - time_series_path: 时间序列数据pkl文件路径
    - output_path: 输出CSV文件路径
    """
    with open(time_series_path, 'rb') as f:
        window_dict = pickle.load(f)

    # 合并正常和异常窗口
    all_windows = []

    for window in window_dict['benign']:
        flat_window = {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'flow_count': window['flow_count'],
            'unique_timestamps': window['unique_timestamps'],
            'time_density': window['time_density'],
            'is_malicious': 0
        }

        # 添加统计特征
        for stat_type, stat_dict in window['features'].items():
            for feature, value in stat_dict.items():
                flat_window[f'{feature}_{stat_type}'] = value

        all_windows.append(flat_window)

    for window in window_dict['malicious']:
        flat_window = {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'flow_count': window['flow_count'],
            'unique_timestamps': window['unique_timestamps'],
            'time_density': window['time_density'],
            'is_malicious': 1,
            'malicious_ratio': window['malicious_ratio'],
            'primary_attack': window['primary_attack']
        }

        # 添加统计特征
        for stat_type, stat_dict in window['features'].items():
            for feature, value in stat_dict.items():
                flat_window[f'{feature}_{stat_type}'] = value

        all_windows.append(flat_window)

    # 转换为DataFrame并保存
    df = pd.DataFrame(all_windows)
    df.sort_values('start_time', inplace=True)
    df.to_csv(output_path, index=False)

    print(f"扁平化数据集已保存至 {output_path}")
    print(f"数据集包含 {len(df)} 个窗口，其中 {df['is_malicious'].sum()} 个异常窗口")

    return df




if __name__ == "__main__":
    # 文件路径配置
    dataset_path = "../cicids2017/clean/Tuesday-WorkingHours.pcap_ISCX.csv.parquet"  # 清洗后的数据集路径
    output_dir = "../cicids2017/time_series/Tuesday-WorkingHours"  # 输出目录

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 首先分析时间戳分布
    print("分析数据集的时间戳分布...")
    time_counts = analyze_time_distribution(dataset_path)

    # 提取不同参数的时间序列数据集
    extract_time_series(
        dataset_path=dataset_path,
        output_dir=output_dir,
        window_sizes=[5, 10, 15],  # 窗口大小（分钟）
        step_sizes=[1, 5]  # 滑动步长（分钟）
    )

    # 为一个特定窗口参数创建扁平化数据集（机器学习用）
    time_series_path = os.path.join(output_dir, "time_series_w10_s1.pkl")
    flat_output_path = os.path.join(output_dir, "flat_dataset_w10_s1.csv")
    flat_df = create_flattened_dataset(time_series_path, flat_output_path)

    # 输出数据集基本信息
    print("\n数据集统计信息:")
    print(f"窗口总数: {len(flat_df)}")
    print(f"正常窗口数: {(flat_df['is_malicious'] == 0).sum()}")
    print(f"异常窗口数: {(flat_df['is_malicious'] == 1).sum()}")

    if 'primary_attack' in flat_df.columns:
        print("\n攻击类型分布:")
        print(flat_df[flat_df['is_malicious'] == 1]['primary_attack'].value_counts())

    # 时间密度与流量数量关系分析
    plt.figure(figsize=(10, 6))
    plt.scatter(flat_df[flat_df['is_malicious'] == 0]['flow_count'],
                flat_df[flat_df['is_malicious'] == 0]['time_density'],
                alpha=0.5, label='Benign', color='green')
    plt.scatter(flat_df[flat_df['is_malicious'] == 1]['flow_count'],
                flat_df[flat_df['is_malicious'] == 1]['time_density'],
                alpha=0.5, label='Malicious', color='red')
    plt.title('Flow Count vs Time Density')
    plt.xlabel('Number of Flows per Window')
    plt.ylabel('Time Density (Unique Timestamps per Minute)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'flow_vs_density_analysis.png'))
    plt.close()