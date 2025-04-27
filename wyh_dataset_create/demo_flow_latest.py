import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


def compute_global_transformers(data_files, apply_pca=False, pca_components=20):
    """
    计算全局的StandardScaler和PCA（如果需要）

    参数:
    - data_files: 数据文件路径列表
    - apply_pca: 是否应用PCA
    - pca_components: PCA组件数量

    返回:
    - global_scaler: 全局StandardScaler
    - global_pca: 全局PCA（如果apply_pca=True）
    - feature_cols: 特征列名称
    """
    print("计算全局特征统计信息...")

    # 收集所有文件的特征数据
    all_features_data = []
    common_features = None

    for file_path in tqdm(data_files, desc="读取文件"):
        # 加载数据
        if file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"跳过不支持的文件格式: {file_path}")
            continue

        # 提取特征列（排除时间戳和标签）
        file_features = [col for col in df.columns if col not in ['Timestamp', 'Label', 'label']]

        # 只保留数值型特征
        numeric_features = df[file_features].select_dtypes(include=['number']).columns.tolist()
        if len(numeric_features) < len(file_features):
            print(
                f"  警告: 文件 {os.path.basename(file_path)} 中 {len(file_features) - len(numeric_features)} 个特征不是数值类型")
            file_features = numeric_features

        # 更新共同特征集
        if common_features is None:
            common_features = set(file_features)
        else:
            common_features = common_features.intersection(set(file_features))

        # 处理无穷值和NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[file_features] = df[file_features].fillna(0)

        # 收集特征数据（使用子集以减少内存使用）
        # 最多取10000行样本计算统计量
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        all_features_data.append(df_sample[file_features])

    # 将共同特征转换为列表
    common_features = list(common_features)
    if not common_features:
        raise ValueError("在所有文件中没有找到共同的特征列")

    print(f"在所有文件中找到 {len(common_features)} 个共同特征")

    # 合并所有文件的特征数据
    print("合并特征数据以计算全局统计量...")
    combined_features = pd.concat([df[common_features] for df in all_features_data], ignore_index=True)

    # 创建全局scaler
    print("拟合全局StandardScaler...")
    global_scaler = StandardScaler()
    global_scaler.fit(combined_features)

    # 创建全局PCA（如果需要）
    global_pca = None
    if apply_pca:
        print(f"拟合全局PCA (components={pca_components})...")
        global_pca = PCA(n_components=pca_components)
        # 对标准化后的数据应用PCA
        global_pca.fit(global_scaler.transform(combined_features))
        print(f"PCA解释方差比例: {global_pca.explained_variance_ratio_.sum():.4f}")

    # 清理内存
    del combined_features
    del all_features_data

    return global_scaler, global_pca, common_features


def process_day_file(file_path, output_dir, window_size=100, step_size=20,
                     apply_pca=False, pca_components=20,
                     global_scaler=None, global_pca=None, common_features=None):
    """
    处理单个日期的数据文件并创建窗口序列

    参数:
    - file_path: 数据文件路径
    - output_dir: 输出目录
    - window_size: 每个窗口包含的流量数量
    - step_size: 窗口滑动的流量数量
    - apply_pca: 是否应用PCA降维
    - pca_components: PCA组件数量
    - global_scaler: 全局StandardScaler
    - global_pca: 全局PCA
    - common_features: 所有文件中共同的特征列

    返回:
    - 处理结果的字典
    """
    file_name = os.path.basename(file_path)
    day_name = file_name.split('.')[0]  # 提取日期部分作为文件名

    print(f"\n正在处理: {file_name}")

    # 加载数据
    if file_path.endswith('.feather'):
        df = pd.read_feather(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .feather, .parquet 或 .csv 文件")

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
    numeric_labels = label_encoder.fit_transform(df['Label'])
    df['label_encoded'] = numeric_labels

    # 创建标签映射字典（用于后续参考）
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    # 处理特征
    # 使用所有文件共同的特征
    feature_cols = common_features if common_features else [col for col in df.columns if
                                                            col not in ['Timestamp', 'Label', 'label_encoded']]
    original_features = feature_cols.copy()

    # 确认所有需要的特征都存在
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"在文件 {file_name} 中缺少以下特征: {missing_features}")

    # 处理无穷值和NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if df[feature_cols].isna().any().any():
        print(f"警告: 数据中包含NaN值，将使用0填充")
        df[feature_cols] = df[feature_cols].fillna(0)

    # 使用全局scaler进行标准化
    if global_scaler:
        print("使用全局StandardScaler转换特征...")
        df[feature_cols] = global_scaler.transform(df[feature_cols])
        scaler = global_scaler
    else:
        # 如果没有提供全局scaler，则创建一个新的（不推荐，但提供兼容性）
        print("警告: 没有提供全局StandardScaler，将使用本地缩放（不推荐）")
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 应用PCA降维（如果需要）
    pca_transformer = None
    if apply_pca:
        if global_pca:
            print("使用全局PCA转换特征...")
            pca_features = global_pca.transform(df[feature_cols])
            pca_transformer = global_pca
        else:
            # 如果没有提供全局PCA，则创建一个新的（不推荐，但提供兼容性）
            print("警告: 没有提供全局PCA，将使用本地PCA（不推荐）")
            pca_transformer = PCA(n_components=pca_components)
            pca_features = pca_transformer.fit_transform(df[feature_cols])
            print(f"本地PCA解释方差比例: {pca_transformer.explained_variance_ratio_.sum():.4f}")

        # 将PCA结果放回DataFrame
        pca_feature_cols = [f'pca_{i}' for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_feature_cols, index=df.index)

        # 保留原始的时间戳、标签和编码标签
        df = pd.concat([df[['Timestamp', 'Label', 'label_encoded']], pca_df], axis=1)
        feature_cols = pca_feature_cols

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

    # 创建窗口
    X_windows = []  # 特征窗口
    y_windows = []  # 原始标签窗口（形状为 [window_size]）
    window_metadata = []  # 窗口元数据

    for i in tqdm(range(num_windows), desc="创建窗口"):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        # 确保不超出数据范围
        if end_idx > total_records:
            break

        # 提取当前窗口的数据
        window_data = df.iloc[start_idx:end_idx].copy()

        # 判断窗口内是否有恶意流量
        has_malicious = (window_data['Label'] != 'Benign').any()

        # 保存特征矩阵
        features = window_data[feature_cols].values

        # 保存标签（编码后的数值标签）
        labels = window_data['label_encoded'].values

        # 创建窗口元数据
        metadata = {
            'window_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': window_data['Timestamp'].iloc[0],
            'end_time': window_data['Timestamp'].iloc[-1],
            'is_malicious': 1 if has_malicious else 0,  # 二分类标签
            'original_labels': window_data['Label'].values  # 原始字符串标签（用于参考）
        }

        # 如果窗口包含恶意流量，添加攻击类型信息
        if has_malicious:
            attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
            metadata['attack_types'] = attack_counts
            metadata['primary_attack'] = max(attack_counts, key=attack_counts.get)

        X_windows.append(features)
        y_windows.append(labels)  # 存储每个flow的编码标签
        window_metadata.append(metadata)

    # 检查是否成功创建窗口
    if not X_windows:
        print("警告: 没有成功创建任何窗口")
        return None

    # 转换为NumPy数组
    X = np.array(X_windows)
    y = np.array(y_windows)  # 形状为 [num_windows, window_size]

    # 准备保存路径
    day_output_dir = os.path.join(output_dir, day_name)
    os.makedirs(day_output_dir, exist_ok=True)

    # 保存NumPy数组
    np.save(os.path.join(day_output_dir, f'X_windows_w{window_size}_s{step_size}.npy'), X)
    np.save(os.path.join(day_output_dir, f'y_windows_w{window_size}_s{step_size}.npy'), y)

    # 保存窗口元数据
    with open(os.path.join(day_output_dir, f'metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        metadata_dict = {
            'window_metadata': window_metadata,
            'feature_names': feature_cols,
            'original_features': original_features,
            'label_mapping': label_mapping,  # 添加标签映射
            'scaler': scaler,
            'pca_transformer': pca_transformer,
            'label_encoder': label_encoder,  # 添加标签编码器
            'config': {
                'window_size': window_size,
                'step_size': step_size,
                'apply_pca': apply_pca,
                'pca_components': pca_components,
                'file_name': file_name,
                'using_global_transformers': (global_scaler is not None)
            }
        }
        pickle.dump(metadata_dict, f)

    # 统计窗口信息
    malicious_count = sum(1 for w in window_metadata if w['is_malicious'] == 1)
    benign_count = len(window_metadata) - malicious_count

    print(f"创建完成：窗口总数 {len(window_metadata)} 个")
    print(f"正常窗口: {benign_count} 个 ({benign_count / len(window_metadata) * 100:.1f}%)")
    print(f"异常窗口: {malicious_count} 个 ({malicious_count / len(window_metadata) * 100:.1f}%)")

    # 统计标签分布
    label_counts = {}
    for window_y in y:
        unique_labels, counts = np.unique(window_y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = label_mapping[label]
            if label_name in label_counts:
                label_counts[label_name] += count
            else:
                label_counts[label_name] = count

    print("\n标签分布:")
    total_flows = sum(label_counts.values())
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {count} 条记录 ({count / total_flows * 100:.2f}%)")

    print(f"\n数据已保存到: {day_output_dir}")
    print(f"X形状: {X.shape}, y形状: {y.shape}")

    return {
        'X': X,
        'y': y,
        'metadata': metadata_dict,
        'output_dir': day_output_dir
    }


def process_all_days(data_dir, output_dir, window_size=100, step_size=20, apply_pca=False, pca_components=20):
    """
    处理所有日期的数据文件，使用全局scaler和PCA

    参数:
    - data_dir: 包含日期数据文件的目录
    - output_dir: 输出目录
    - window_size: 窗口大小
    - step_size: 步长
    - apply_pca: 是否应用PCA
    - pca_components: PCA组件数
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

    # 计算全局转换器
    global_scaler, global_pca, common_features = compute_global_transformers(
        data_files, apply_pca=apply_pca, pca_components=pca_components
    )

    # 保存全局转换器
    transformers_path = os.path.join(output_dir, 'global_transformers.pkl')
    with open(transformers_path, 'wb') as f:
        pickle.dump({
            'scaler': global_scaler,
            'pca': global_pca,
            'features': common_features,
            'config': {
                'apply_pca': apply_pca,
                'pca_components': pca_components
            }
        }, f)

    print(f"全局转换器已保存到: {transformers_path}")

    # 处理每个文件
    results = {}

    for file_path in data_files:
        day_name = os.path.basename(file_path).split('.')[0]
        result = process_day_file(
            file_path=file_path,
            output_dir=output_dir,
            window_size=window_size,
            step_size=step_size,
            apply_pca=apply_pca,
            pca_components=pca_components,
            global_scaler=global_scaler,
            global_pca=global_pca,
            common_features=common_features
        )

        if result:
            results[day_name] = result

    # 创建一个汇总文件
    summary_path = os.path.join(output_dir, f'summary_w{window_size}_s{step_size}.txt')

    with open(summary_path, 'w',encoding='utf-8') as f:
        f.write(f"CICIDS2017数据集窗口处理汇总 (窗口大小: {window_size}, 步长: {step_size})\n")
        f.write(f"=====================================\n\n")

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

        # 全局标签分布
        all_label_counts = {}
        for day_name, result in results.items():
            # 从y中获取标签分布
            y_data = result['y']
            label_mapping = result['metadata']['label_mapping']

            for window_y in y_data:
                unique_labels, counts = np.unique(window_y, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_name = label_mapping[label]
                    if label_name in all_label_counts:
                        all_label_counts[label_name] += count
                    else:
                        all_label_counts[label_name] = count

        f.write("全局标签分布:\n")
        f.write("-------------\n")
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
            label_counts = {}
            y_data = result['y']
            label_mapping = result['metadata']['label_mapping']

            for window_y in y_data:
                unique_labels, counts = np.unique(window_y, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_name = label_mapping[label]
                    if label_name in label_counts:
                        label_counts[label_name] += count
                    else:
                        label_counts[label_name] = count

            f.write("  标签分布:\n")
            day_total_flows = sum(label_counts.values())
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"    - {label}: {count} 条记录 ({count / day_total_flows * 100:.2f}%)\n")

    print(f"\n汇总信息已保存到: {summary_path}")

    return results


if __name__ == "__main__":
    # 配置参数
    data_dir = "../cicids2017/clean"  # 原始数据文件目录
    output_dir = "../cicids2017/flow_windows_w1000_s100"  # 输出目录

    # 创建窗口数据集
    results = process_all_days(
        data_dir=data_dir,
        output_dir=output_dir,
        window_size=1000,  # 窗口大小
        step_size=100,  # 步长
        apply_pca=False,  # 应用PCA降维
        pca_components=20  # PCA组件数
    )

    print("所有日期数据处理完成!")