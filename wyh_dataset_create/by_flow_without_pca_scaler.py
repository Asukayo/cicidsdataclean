import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
import gc  # 添加垃圾回收模块

warnings.filterwarnings('ignore')


def compute_feature_statistics(data_files):
    """
    计算特征统计信息，但不进行归一化和PCA
    使用更节省内存的方式处理

    参数:
    - data_files: 数据文件路径列表

    返回:
    - feature_cols: 特征列名称
    """
    print("计算特征统计信息...")

    # 收集所有文件的特征数据
    common_features = None

    for file_path in tqdm(data_files, desc="读取文件"):
        try:
            # 加载数据 - 尝试只读取列名或小样本
            if file_path.endswith('.feather'):
                # 对于feather文件，读取一小部分样本
                df_sample = pd.read_feather(file_path)
                if len(df_sample) > 100:
                    df_sample = df_sample.iloc[:100]
            elif file_path.endswith('.parquet'):
                # 对于parquet文件，直接获取列名
                import pyarrow.parquet as pq
                # 先获取列名
                parquet_file = pq.ParquetFile(file_path)
                columns = parquet_file.schema.names
                # 然后读取少量行
                df_sample = pd.read_parquet(file_path)
                if len(df_sample) > 100:
                    df_sample = df_sample.iloc[:100]
            elif file_path.endswith('.csv'):
                # 对于CSV文件，只读取头部
                df_sample = pd.read_csv(file_path, nrows=100)
            else:
                print(f"跳过不支持的文件格式: {file_path}")
                continue

            # 提取特征列（排除时间戳和标签）
            file_features = [col for col in df_sample.columns if col not in ['Timestamp', 'Label', 'label']]

            # 只保留数值型特征
            numeric_features = df_sample[file_features].select_dtypes(include=['number']).columns.tolist()
            if len(numeric_features) < len(file_features):
                print(
                    f"  警告: 文件 {os.path.basename(file_path)} 中 {len(file_features) - len(numeric_features)} 个特征不是数值类型")
                file_features = numeric_features

            # 更新共同特征集
            if common_features is None:
                common_features = set(file_features)
            else:
                common_features = common_features.intersection(set(file_features))

            # 清理内存
            del df_sample
            gc.collect()

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    # 将共同特征转换为列表
    common_features = list(common_features) if common_features else []
    if not common_features:
        raise ValueError("在所有文件中没有找到共同的特征列")

    print(f"在所有文件中找到 {len(common_features)} 个共同特征")

    return common_features


def process_day_file(file_path, output_dir, window_size=100, step_size=20, common_features=None, batch_size=1000):
    """
    处理单个日期的数据文件并创建窗口序列，不进行归一化和PCA
    使用分块处理减少内存占用

    参数:
    - file_path: 数据文件路径
    - output_dir: 输出目录
    - window_size: 每个窗口包含的流量数量
    - step_size: 窗口滑动的流量数量
    - common_features: 所有文件中共同的特征列
    - batch_size: 每批处理的窗口数量，用于控制内存使用

    返回:
    - 处理结果的字典
    """
    file_name = os.path.basename(file_path)
    day_name = file_name.split('.')[0]  # 提取日期部分作为文件名

    print(f"\n正在处理: {file_name}")

    # 创建输出目录
    day_output_dir = os.path.join(output_dir, day_name)
    os.makedirs(day_output_dir, exist_ok=True)

    # 读取整个文件 (这一步可能会消耗大量内存，但我们需要完整数据来进行窗口处理)
    print(f"加载数据文件: {file_path}")
    try:
        if file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用 .feather, .parquet 或 .csv 文件")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

    print(f"数据加载完成，形状: {df.shape}")

    # 基本数据处理
    # 确保有Timestamp列
    if 'Timestamp' not in df.columns:
        print("警告: 数据中没有找到'Timestamp'列，将添加虚拟时间戳")
        df['Timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='S')

    # 确保Timestamp是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # 处理无法解析的时间戳
        if df['Timestamp'].isna().any():
            print(f"警告: {df['Timestamp'].isna().sum()} 个时间戳无法解析，将替换为索引日期")
            invalid_indices = df['Timestamp'].isna()
            df.loc[invalid_indices, 'Timestamp'] = pd.date_range(
                start='2017-01-01', periods=invalid_indices.sum(), freq='S'
            )

    # 按时间戳排序
    df = df.sort_values('Timestamp')

    # 检查并统一Label列
    if 'Label' not in df.columns and 'label' in df.columns:
        df.rename(columns={'label': 'Label'}, inplace=True)

    if 'Label' not in df.columns:
        raise ValueError("数据中没有找到'Label'列")

    # 标准化Label列: BENIGN -> Benign
    df['Label'] = df['Label'].astype(str)
    df['Label'] = df['Label'].replace({'BENIGN': 'Benign'})

    # 创建标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Label'])
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    numeric_labels = label_encoder.transform(df['Label'])
    df['label_encoded'] = numeric_labels

    # 处理特征
    # 使用所有文件共同的特征
    feature_cols = common_features if common_features else [col for col in df.columns if
                                                            col not in ['Timestamp', 'Label', 'label_encoded']]

    # 确认所有需要的特征都存在
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"在文件 {file_name} 中缺少以下特征: {missing_features}")

    # 处理无穷值和NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if df[feature_cols].isna().any().any():
        print(f"警告: 数据中包含NaN值，将使用0填充")
        df[feature_cols] = df[feature_cols].fillna(0)

    # 计算窗口数量
    total_records = len(df)
    num_windows = (total_records - window_size) // step_size + 1

    if num_windows <= 0:
        print(f"警告: 数据记录数 ({total_records}) 小于窗口大小 ({window_size})，无法创建窗口")
        return None

    print(f"总记录数: {total_records}")
    print(f"窗口大小: {window_size} 条流量记录")
    print(f"窗口步长: {step_size} 条流量记录")
    print(f"预计窗口数量: {num_windows}")

    # 分批处理窗口
    X_file = os.path.join(day_output_dir, f'X_windows_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(day_output_dir, f'y_windows_w{window_size}_s{step_size}.npy')

    # 记录窗口元数据
    all_window_metadata = []

    # 标签分布统计
    label_counts = {}

    # 分批次保存窗口，使用numpy的mmap_mode来减少内存使用
    first_batch = True
    malicious_count = 0
    benign_count = 0

    # 计算需要多少批次
    num_batches = (num_windows + batch_size - 1) // batch_size

    for batch in tqdm(range(num_batches), desc=f"处理窗口批次 (每批 {batch_size} 个窗口)"):
        batch_start = batch * batch_size
        batch_end = min((batch + 1) * batch_size, num_windows)

        batch_X = []
        batch_y = []
        batch_metadata = []

        for i in range(batch_start, batch_end):
            # 计算窗口的起始和结束索引
            start_idx = i * step_size
            end_idx = start_idx + window_size

            # 确保不超出数据范围
            if end_idx > total_records:
                break

            # 提取当前窗口的数据
            window_data = df.iloc[start_idx:end_idx].copy()

            # 判断窗口内是否有恶意流量
            has_malicious = (window_data['Label'] != 'Benign').any()

            # 更新计数器
            if has_malicious:
                malicious_count += 1
            else:
                benign_count += 1

            # 保存特征矩阵
            features = window_data[feature_cols].values

            # 保存标签（编码后的数值标签）
            labels = window_data['label_encoded'].values

            # 更新标签计数
            unique_labels, label_counts_array = np.unique(labels, return_counts=True)
            for label_idx, count in zip(unique_labels, label_counts_array):
                label_name = label_mapping[label_idx]
                if label_name in label_counts:
                    label_counts[label_name] += count
                else:
                    label_counts[label_name] = count

            # 创建窗口元数据
            metadata = {
                'window_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': window_data['Timestamp'].iloc[0],
                'end_time': window_data['Timestamp'].iloc[-1],
                'is_malicious': 1 if has_malicious else 0,  # 二分类标签
                'original_labels': window_data['Label'].tolist()  # 转换为列表以便序列化
            }

            # 如果窗口包含恶意流量，添加攻击类型信息
            if has_malicious:
                attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
                metadata['attack_types'] = attack_counts
                metadata['primary_attack'] = max(attack_counts, key=attack_counts.get)

            batch_X.append(features)
            batch_y.append(labels)
            batch_metadata.append(metadata)

        # 当批次处理完成，保存到文件
        if batch_X:
            batch_X_array = np.array(batch_X)
            batch_y_array = np.array(batch_y)

            if first_batch:
                # 第一个批次，直接创建文件
                np.save(X_file, batch_X_array)
                np.save(y_file, batch_y_array)
                first_batch = False
            else:
                # 后续批次，需要追加
                # 对于numpy数组，我们需要先读取已有数据
                try:
                    existing_X = np.load(X_file, mmap_mode='r')
                    existing_y = np.load(y_file, mmap_mode='r')

                    # 创建新的合并数组
                    merged_X = np.vstack([existing_X, batch_X_array])
                    merged_y = np.vstack([existing_y, batch_y_array])

                    # 保存合并后的数组
                    np.save(X_file, merged_X)
                    np.save(y_file, merged_y)

                    # 清理内存
                    del existing_X, existing_y, merged_X, merged_y
                except Exception as e:
                    print(f"追加数据到文件时出错: {str(e)}")
                    # 如果出错，直接覆盖保存当前批次
                    np.save(X_file, batch_X_array)
                    np.save(y_file, batch_y_array)

            # 添加到元数据列表
            all_window_metadata.extend(batch_metadata)

            # 清理批次数据
            del batch_X, batch_y, batch_X_array, batch_y_array
            gc.collect()

    # 窗口处理完成后，保存元数据
    window_count = len(all_window_metadata)
    if window_count == 0:
        print("警告: 没有成功创建任何窗口")
        return None

    # 保存窗口元数据
    with open(os.path.join(day_output_dir, f'metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        metadata_dict = {
            'window_metadata': all_window_metadata,
            'feature_names': feature_cols,
            'original_features': feature_cols.copy(),
            'label_mapping': label_mapping,  # 添加标签映射
            'label_encoder': label_encoder,  # 添加标签编码器
            'config': {
                'window_size': window_size,
                'step_size': step_size,
                'file_name': file_name
            }
        }
        pickle.dump(metadata_dict, f)

    # 打印统计信息
    print(f"创建完成：窗口总数 {window_count} 个")
    print(f"正常窗口: {benign_count} 个 ({benign_count / window_count * 100:.1f}%)")
    print(f"异常窗口: {malicious_count} 个 ({malicious_count / window_count * 100:.1f}%)")

    # 打印标签分布
    print("\n标签分布:")
    total_flows = sum(label_counts.values())
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {count} 条记录 ({count / total_flows * 100:.2f}%)")

    print(f"\n数据已保存到: {day_output_dir}")

    # 加载保存的X和y以获取形状信息
    try:
        X_shape = np.load(X_file, mmap_mode='r').shape
        y_shape = np.load(y_file, mmap_mode='r').shape
        print(f"X形状: {X_shape}, y形状: {y_shape}")
    except Exception as e:
        print(f"获取数组形状时出错: {str(e)}")

    # 清理原始数据帧以释放内存
    del df
    gc.collect()

    return {
        'metadata': metadata_dict,
        'output_dir': day_output_dir
    }


def process_all_days(data_dir, output_dir, window_size=100, step_size=20, batch_size=1000):
    """
    处理所有日期的数据文件，不进行归一化和PCA
    使用分批处理减少内存使用

    参数:
    - data_dir: 包含日期数据文件的目录
    - output_dir: 输出目录
    - window_size: 窗口大小
    - step_size: 步长
    - batch_size: 一次处理的窗口批次大小
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有parquet文件
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    if not parquet_files:
        print(f"在 {data_dir} 中没有找到parquet文件")
        # 尝试查找其他格式
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        feather_files = glob.glob(os.path.join(data_dir, "*.feather"))

        if csv_files:
            print(f"找到 {len(csv_files)} 个CSV文件")
            data_files = csv_files
        elif feather_files:
            print(f"找到 {len(feather_files)} 个Feather文件")
            data_files = feather_files
        else:
            print("没有找到支持的数据文件")
            return
    else:
        print(f"找到 {len(parquet_files)} 个Parquet文件")
        data_files = parquet_files

    data_files = sorted(data_files)

    # 计算共同特征
    common_features = compute_feature_statistics(data_files)

    # 保存特征信息
    features_path = os.path.join(output_dir, 'features_info.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump({
            'features': common_features,
            'config': {
                'window_size': window_size,
                'step_size': step_size
            }
        }, f)

    print(f"特征信息已保存到: {features_path}")

    # 处理每个文件
    results = {}

    for file_path in data_files:
        day_name = os.path.basename(file_path).split('.')[0]
        result = process_day_file(
            file_path=file_path,
            output_dir=output_dir,
            window_size=window_size,
            step_size=step_size,
            common_features=common_features,
            batch_size=batch_size
        )

        if result:
            results[day_name] = result

        # 手动触发垃圾回收
        gc.collect()

    # 创建一个汇总文件
    summary_path = os.path.join(output_dir, f'summary_w{window_size}_s{step_size}.txt')

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"CICIDS2017数据集窗口处理汇总 (窗口大小: {window_size}, 步长: {step_size})\n")
            f.write(f"=====================================\n\n")

            if not results:
                f.write("处理过程中没有成功创建结果\n")
                print(f"\n汇总信息已保存到: {summary_path}")
                return None

            # 总体统计
            total_windows = sum(len(result['metadata']['window_metadata']) for result in results.values())
            total_malicious = sum(
                sum(1 for w in result['metadata']['window_metadata'] if w['is_malicious'] == 1)
                for result in results.values()
            )
            total_benign = total_windows - total_malicious

            f.write(f"总窗口数: {total_windows}\n")
            f.write(f"正常窗口: {total_benign} ({total_benign / total_windows * 100:.1f}%)\n")
            f.write(f"异常窗口: {total_malicious} ({total_malicious / total_windows * 100:.1f}%)\n\n")

            # 记录全局标签分布需要遍历每个文件的元数据
            f.write("全局标签分布:\n")
            f.write("-------------\n")

            # 这里我们简化计算，只基于元数据中的信息而不加载完整数据
            all_label_counts = {}
            for day_name, result in results.items():
                window_metadata = result['metadata']['window_metadata']

                for window in window_metadata:
                    for label in window['original_labels']:
                        if label in all_label_counts:
                            all_label_counts[label] += 1
                        else:
                            all_label_counts[label] = 1

            total_flows = sum(all_label_counts.values())
            for label, count in sorted(all_label_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {label}: {count} 条记录 ({count / total_flows * 100:.2f}%)\n")

            f.write("\n")

            # 按日期统计
            f.write("按日期统计:\n")
            f.write("-------------\n")

            for day_name, result in results.items():
                window_metadata = result['metadata']['window_metadata']
                day_windows = len(window_metadata)
                day_malicious = sum(1 for w in window_metadata if w['is_malicious'] == 1)
                day_benign = day_windows - day_malicious

                f.write(f"\n{day_name}:\n")
                f.write(f"  窗口总数: {day_windows}\n")
                f.write(f"  正常窗口: {day_benign} ({day_benign / day_windows * 100:.1f}%)\n")
                f.write(f"  异常窗口: {day_malicious} ({day_malicious / day_windows * 100:.1f}%)\n")

                # 该日的标签分布
                day_label_counts = {}
                for window in window_metadata:
                    for label in window['original_labels']:
                        if label in day_label_counts:
                            day_label_counts[label] += 1
                        else:
                            day_label_counts[label] = 1

                f.write("  标签分布:\n")
                day_total_flows = sum(day_label_counts.values())
                for label, count in sorted(day_label_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    - {label}: {count} 条记录 ({count / day_total_flows * 100:.2f}%)\n")

        print(f"\n汇总信息已保存到: {summary_path}")
    except Exception as e:
        print(f"创建汇总文件时出错: {str(e)}")

    return results


if __name__ == "__main__":
    # 配置参数
    data_dir = "../cicids2017/clean"  # 原始数据文件目录
    output_dir = "../cicids2017/flow_windows"  # 输出目录

    # 创建窗口数据集 - 使用更小的窗口和批处理参数来减少内存压力
    results = process_all_days(
        data_dir=data_dir,
        output_dir=output_dir,
        window_size=570,  # 窗口大小
        step_size=50,  # 步长
        batch_size=500  # 每批处理的窗口数量
    )

    print("所有日期数据处理完成!")