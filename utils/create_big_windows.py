import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

def create_time_windows(dataset_path, window_size_minutes=5, step_size_minutes=1, min_flows=10):
    """
    基于时间窗口滑动方式创建时间序列数据集，处理Timestamp重复的情况

    参数:
    - dataset_path: 数据集路径，指向清洗后的 feather 文件
    - window_size_minutes: 窗口大小（分钟）
    - step_size_minutes: 窗口滑动步长（分钟）
    - min_flows: 窗口内最小流量数，小于此数量的窗口将被丢弃

    返回:
    - 包含正常窗口和异常窗口的字典
    """
    print(f"正在加载数据集: {dataset_path}")

    # 加载数据
    if dataset_path.endswith('.feather'):
        df = pd.read_feather(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .feather 或 .parquet 文件")

    # 确保时间戳列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 按时间戳排序
    df = df.sort_values('Timestamp')

    # 转换窗口参数为 timedelta
    window_size = timedelta(minutes=window_size_minutes)
    step_size = timedelta(minutes=step_size_minutes)

    # 获取数据集的起止时间
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    # 分析时间戳重复情况
    time_counts = df['Timestamp'].value_counts()
    max_duplicates = time_counts.max()
    avg_duplicates = time_counts.mean()

    print(f"数据集时间范围: {start_time} 到 {end_time}")
    print(f"时间戳重复情况: 最大重复次数 {max_duplicates}, 平均重复次数 {avg_duplicates:.2f}")
    print(f"窗口大小: {window_size_minutes} 分钟, 滑动步长: {step_size_minutes} 分钟")

    # 使用唯一时间点来计算窗口数量
    unique_timestamps = df['Timestamp'].unique()
    unique_timestamps.sort()
    total_duration = (end_time - start_time).total_seconds() / 60  # 分钟
    num_windows = int((total_duration - window_size_minutes) / step_size_minutes) + 1

    print(f"唯一时间点数量: {len(unique_timestamps)}")
    print(f"预计窗口数量: {num_windows}")

    # 初始化结果存储
    benign_windows = []
    malicious_windows = []

    # 滑动窗口
    current_time = start_time

    for _ in tqdm(range(num_windows)):
        window_end = current_time + window_size

        # 提取当前窗口内的数据
        window_data = df[(df['Timestamp'] >= current_time) & (df['Timestamp'] < window_end)]

        # 计算窗口内唯一时间点数量
        unique_times_in_window = window_data['Timestamp'].nunique()

        if len(window_data) >= min_flows:
            # 判断窗口内是否有恶意流量
            has_malicious = (window_data['Label'] != 'Benign').any()

            # 提取特征列（排除 Timestamp 和 Label）
            # 如果是要进行训练就不去除TimeStamp
            feature_cols = [col for col in window_data.columns if col not in ['Timestamp', 'Label']]
            window_features = window_data[feature_cols]

            # 计算统计特征
            window_stats = {
                'start_time': current_time,
                'end_time': window_end,
                'flow_count': len(window_data),
                'unique_timestamps': unique_times_in_window,
                'time_density': unique_times_in_window / (window_size_minutes),  # 每分钟的平均唯一时间点数
                'features': {
                    'mean': window_features.mean().to_dict(),
                    'std': window_features.std().to_dict(),
                    'min': window_features.min().to_dict(),
                    'max': window_features.max().to_dict(),
                    'q25': window_features.quantile(0.25).to_dict(),
                    'median': window_features.quantile(0.5).to_dict(),
                    'q75': window_features.quantile(0.75).to_dict()
                }
            }

            # 如果窗口中包含恶意流量，添加恶意标签信息
            if has_malicious:
                # 获取窗口内恶意流量的类型分布
                attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
                # 计算恶意流量占比
                malicious_ratio = (window_data['Label'] != 'Benign').mean()

                window_stats['attack_types'] = attack_counts
                window_stats['malicious_ratio'] = malicious_ratio
                window_stats['primary_attack'] = max(attack_counts, key=attack_counts.get)

                malicious_windows.append(window_stats)
            else:
                benign_windows.append(window_stats)

        # 滑动窗口
        current_time += step_size

    print(f"创建完成：正常窗口 {len(benign_windows)} 个，异常窗口 {len(malicious_windows)} 个")

    # 分析窗口中的时间点信息
    if benign_windows:
        benign_timestamps = [w['unique_timestamps'] for w in benign_windows]
        print(
            f"正常窗口唯一时间点统计: 最小 {min(benign_timestamps)}, 最大 {max(benign_timestamps)}, 平均 {np.mean(benign_timestamps):.2f}")

    if malicious_windows:
        malicious_timestamps = [w['unique_timestamps'] for w in malicious_windows]
        print(
            f"异常窗口唯一时间点统计: 最小 {min(malicious_timestamps)}, 最大 {max(malicious_timestamps)}, 平均 {np.mean(malicious_timestamps):.2f}")

    return {
        'benign': benign_windows,
        'malicious': malicious_windows
    }
