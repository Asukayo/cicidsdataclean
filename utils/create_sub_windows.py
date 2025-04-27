import pandas as pd
import numpy as np
import os
from datetime import timedelta
from tqdm import tqdm
import pickle  # Still useful for saving/loading intermediate results if needed


# Helper function to safely calculate std dev (returns 0 if count < 2)
def safe_std(x):
    return x.std(ddof=0) if len(x) > 0 else 0.0


# Helper function to safely calculate max (returns 0 if empty)
def safe_max(x):
    return x.max() if not x.empty else 0.0


# Helper function to safely calculate mean (returns 0 if empty)
def safe_mean(x):
    return x.mean() if not x.empty else 0.0


def create_fixed_sub_windows_overlapping(
        dataset_path,
        large_window_minutes=5,
        sub_window_seconds=1,
        step_size_minutes=1  # New parameter for large window step
):
    """
    基于固定子窗口聚合方式创建时间序列数据集，使用有重叠的大窗口滑动。

    参数:
    - dataset_path: 数据集路径 (feather or parquet)
    - large_window_minutes: 大窗口大小（分钟）
    - sub_window_seconds: 子窗口大小（秒）
    - step_size_minutes: 大窗口滑动步长（分钟），必须小于 large_window_minutes

    返回:
    - numpy array: 包含所有大窗口序列的数据 (num_windows, num_sub_windows, num_features)
    - numpy array: 每个大窗口的标签 (0 for benign, 1 for malicious)
    """
    print(f"加载数据集: {dataset_path}")
    print(f"大窗口: {large_window_minutes} 分钟, 子窗口: {sub_window_seconds} 秒, 大窗口步长: {step_size_minutes} 分钟")

    # --- 1. 数据加载与预处理 ---
    if dataset_path.endswith('.feather'):
        df = pd.read_feather(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .feather 或 .parquet 文件")

    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)  # Sort and reset index

    large_window_size = timedelta(minutes=large_window_minutes)
    sub_window_size = timedelta(seconds=sub_window_seconds)
    step_size_large = timedelta(minutes=step_size_minutes)  # Define large step size

    # 确保步长小于窗口大小以实现重叠
    if step_size_large >= large_window_size:
        raise ValueError("大窗口步长 (step_size_minutes) 必须小于大窗口大小 (large_window_minutes) 才能实现重叠。")

    num_sub_windows_per_large = int(large_window_size / sub_window_size)

    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    print(f"数据时间范围: {start_time} 到 {end_time}")

    # --- 2. 定义聚合特征与方法 ---
    core_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Avg Packet Size', 'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
        'Init Fwd Win Bytes', 'Init Bwd Win Bytes'
    ]
    # 确保所有核心特征都存在于DataFrame中
    core_features = [f for f in core_features if f in df.columns]
    print(f"使用的核心特征数量: {len(core_features)}")

    sum_features = [f for f in ['Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total',
                                'Bwd Packets Length Total', 'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count'] if
                    f in core_features]
    mms_features = [f for f in core_features if f not in sum_features]

    # 使用更安全的聚合函数处理空或单元素子窗口
    agg_funcs = {
        # 使用一个肯定存在的列 (如 Timestamp 或索引) 来做 size 计数
        'flow_count': pd.NamedAgg(column='Timestamp', aggfunc='size')
    }
    for f in sum_features:
        agg_funcs[f'{f}_sum'] = pd.NamedAgg(column=f, aggfunc='sum')
    for f in mms_features:
        agg_funcs[f'{f}_mean'] = pd.NamedAgg(column=f, aggfunc=safe_mean)
        agg_funcs[f'{f}_std'] = pd.NamedAgg(column=f, aggfunc=safe_std)
        agg_funcs[f'{f}_max'] = pd.NamedAgg(column=f, aggfunc=safe_max)

    # --- 3. 按子窗口高效聚合 ---
    print("按子窗口聚合特征...")
    # 使用索引（如果已重置）或 Timestamp 进行分组
    grouper = pd.Grouper(key='Timestamp', freq=f'{sub_window_seconds}S', origin=start_time)
    cols_to_agg = ['Timestamp', 'Label'] + core_features  # 选择需要的列
    sub_window_aggregated = df[cols_to_agg].groupby(grouper).agg(**agg_funcs)

    # 再次检查并填充NaN (主要针对聚合结果本身是NaN的情况，虽然安全函数应已处理)
    sub_window_aggregated.fillna(0, inplace=True)
    print(f"子窗口聚合完成，生成 {len(sub_window_aggregated)} 个子窗口统计数据。")

    # 获取所有聚合特征的列名，用于后续创建空数组
    aggregated_feature_names = sub_window_aggregated.columns.tolist()
    num_agg_features = len(aggregated_feature_names)

    # --- 4. 生成重叠的大窗口序列 ---
    print("生成重叠的大窗口序列...")
    all_large_window_sequences = []
    all_large_window_labels = []

    # 使用预计算的子窗口聚合结果的索引（时间戳）
    all_sub_window_times = sub_window_aggregated.index

    current_large_window_start = start_time
    # 循环条件：只要窗口的起始时间在数据结束时间之前即可开始一个窗口
    # 内部切片会处理实际数据边界
    for _ in tqdm(range(int((end_time - start_time) / step_size_large) + 1)):  # 估算迭代次数用于tqdm
        if current_large_window_start >= end_time:  # 确保起始点不超过数据终点
            break

        large_window_end = current_large_window_start + large_window_size

        # 找到落在这个大窗口内的所有【已聚合子窗口】的时间点
        # 注意：这里使用 >= start 和 < end
        sub_times_in_large = all_sub_window_times[
            (all_sub_window_times >= current_large_window_start) &
            (all_sub_window_times < large_window_end)
            ]

        # 获取这些子窗口的聚合数据
        large_window_sequence_df = sub_window_aggregated.loc[sub_times_in_large]

        # --- 5. 序列填充/截断，确保长度固定 ---
        sequence_len = len(large_window_sequence_df)
        sequence_data = np.zeros((num_sub_windows_per_large, num_agg_features))  # 初始化为0

        if sequence_len > 0:
            valid_data = large_window_sequence_df.values
            if sequence_len >= num_sub_windows_per_large:
                # 长度足够或超出，截断取前 num_sub_windows_per_large 个
                sequence_data = valid_data[:num_sub_windows_per_large, :]
            else:
                # 长度不足，在末尾填充0 (因为已经初始化为0，所以只需把有效数据放进去)
                sequence_data[:sequence_len, :] = valid_data
        # else: sequence_len == 0，保持全0序列

        all_large_window_sequences.append(sequence_data)

        # --- 6. 确定大窗口标签 ---
        # 需要查询原始数据来确定这个大窗口内是否有攻击流
        # 为了效率，可以考虑在聚合前给每个子窗口预计算一个 'has_malicious' 标志
        # 这里还是用原始方式查询：
        original_flows_mask = (
                (df['Timestamp'] >= current_large_window_start) &
                (df['Timestamp'] < large_window_end)
        )
        # 检查这个 mask 是否有任何 True 值 (即是否有数据落入窗口)
        if original_flows_mask.any():
            # 如果窗口内有数据，则检查是否有非 Benign 标签
            is_malicious = (df.loc[original_flows_mask, 'Label'] != 'Benign').any()
            all_large_window_labels.append(1 if is_malicious else 0)
        else:
            # 如果窗口内没有任何原始数据，则标记为正常 (或者根据需求处理)
            all_large_window_labels.append(0)

        # !!! --- 更新滑动逻辑 --- !!!
        # 向前滑动一个【小步长】的距离
        current_large_window_start += step_size_large

    print(f"处理完成，生成 {len(all_large_window_sequences)} 个重叠的大窗口序列")

    # --- 7. 返回结果 ---
    if not all_large_window_sequences:
        print("警告：未生成任何窗口序列，请检查数据或参数。")
        # 返回空的 numpy 数组，指定正确的维度形状 (0, num_sub_windows, num_features)
        return np.empty((0, num_sub_windows_per_large, num_agg_features)), np.empty((0,))

    return np.array(all_large_window_sequences), np.array(all_large_window_labels)