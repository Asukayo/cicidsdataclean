import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def create_flow_based_windows(dataset_path, window_size=100, step_size=20, apply_pca=False, pca_components=20):
    """
    基于固定流量数量创建窗口，保留标签信息

    参数:
    - dataset_path: 数据集路径
    - window_size: 每个窗口包含的流量数量
    - step_size: 窗口滑动的流量数量
    - apply_pca: 是否应用PCA降维
    - pca_components: PCA组件数量

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

    # 预处理特征列（排除 Timestamp 和 Label）
    feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Label']]
    original_features = feature_cols.copy()

    # 标准化特征
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 应用PCA降维（可选）
    pca_transformer = None
    if apply_pca and len(feature_cols) > pca_components:
        from sklearn.decomposition import PCA
        print(f"应用PCA降维，将特征从 {len(feature_cols)} 减少到 {pca_components}")
        pca_transformer = PCA(n_components=pca_components)
        pca_features = pca_transformer.fit_transform(df[feature_cols])

        # 将PCA结果放回DataFrame
        pca_feature_cols = [f'pca_{i}' for i in range(pca_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_feature_cols, index=df.index)

        # 保留原始的时间戳和标签
        df = pd.concat([df[['Timestamp', 'Label']], pca_df], axis=1)
        feature_cols = pca_feature_cols

        print(f"PCA解释方差比例: {np.sum(pca_transformer.explained_variance_ratio_):.4f}")

    # 计算窗口数量
    num_windows = (len(df) - window_size) // step_size + 1

    print(f"总记录数: {len(df)}")
    print(f"窗口大小: {window_size} 条流量记录")
    print(f"窗口步长: {step_size} 条流量记录")
    print(f"预计窗口数量: {num_windows}")

    # 初始化结果存储
    benign_windows = []
    malicious_windows = []

    # 创建窗口
    for i in tqdm(range(num_windows)):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        # 提取当前窗口的数据
        window_data = df.iloc[start_idx:end_idx].copy()

        # 判断窗口内是否有恶意流量
        has_malicious = (window_data['Label'] != 'Benign').any()

        # 获取窗口的时间信息
        start_time = window_data['Timestamp'].min()
        end_time = window_data['Timestamp'].max()
        time_span = (end_time - start_time).total_seconds() / 60  # 分钟

        # 获取标签信息
        labels = window_data['Label'].values

        # 创建窗口信息
        window_info = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_time,
            'end_time': end_time,
            'time_span': time_span,
            'unique_timestamps': window_data['Timestamp'].nunique(),
            'features': window_data[feature_cols].values,  # 特征数据，形状为 (window_size, n_features)
            'labels': labels,  # 保留原始标签
            'timestamps': window_data['Timestamp'].values  # 保留时间戳
        }

        # 计算窗口统计特征
        window_info['flow_rate'] = window_size / time_span if time_span > 0 else 0  # 流量率（每分钟流量数）
        window_info['time_density'] = window_info['unique_timestamps'] / time_span if time_span > 0 else 0  # 时间密度

        # 如果窗口包含恶意流量，添加相关信息
        if has_malicious:
            attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
            malicious_ratio = (window_data['Label'] != 'Benign').mean()

            window_info['attack_types'] = attack_counts
            window_info['malicious_ratio'] = malicious_ratio
            window_info['primary_attack'] = max(attack_counts, key=attack_counts.get)
            window_info['is_malicious'] = 1

            malicious_windows.append(window_info)
        else:
            window_info['is_malicious'] = 0
            benign_windows.append(window_info)

    print(f"创建完成：正常窗口 {len(benign_windows)} 个，异常窗口 {len(malicious_windows)} 个")

    return {
        'benign': benign_windows,
        'malicious': malicious_windows,
        'feature_names': feature_cols,
        'original_features': original_features,
        'scaler': scaler,
        'pca_transformer': pca_transformer,
        'config': {
            'window_size': window_size,
            'step_size': step_size,
            'apply_pca': apply_pca,
            'pca_components': pca_components
        }
    }


def extract_flow_based_windows(dataset_path, output_dir, window_sizes=[100, 200, 500],
                               step_sizes=[20, 50, 100], apply_pca=False, pca_components=20):
    """
    使用不同窗口大小和步长提取基于流量的窗口数据集

    参数:
    - dataset_path: 数据集路径
    - output_dir: 输出目录
    - window_sizes: 窗口大小列表（流量数量）
    - step_sizes: 滑动步长列表（流量数量）
    - apply_pca: 是否应用PCA降维
    - pca_components: PCA组件数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 为不同窗口大小和步长创建窗口数据集
    for window_size in window_sizes:
        for step_size in step_sizes:
            if step_size > window_size:
                continue  # 跳过不合理的组合

            print(f"\n处理窗口大小 {window_size} 条流量，步长 {step_size} 条流量")

            result_dict = create_flow_based_windows(
                dataset_path=dataset_path,
                window_size=window_size,
                step_size=step_size,
                apply_pca=apply_pca,
                pca_components=pca_components
            )

            # 保存结果
            output_path = os.path.join(
                output_dir,
                f"flow_windows_w{window_size}_s{step_size}.pkl"
            )

            with open(output_path, 'wb') as f:
                pickle.dump(result_dict, f)

            print(f"保存到: {output_path}")

            # 绘制窗口分布图
            plot_flow_window_distribution(result_dict, window_size, step_size, output_dir)


def plot_flow_window_distribution(window_dict, window_size, step_size, output_dir):
    """绘制窗口特性分布"""
    # 创建多子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 绘制窗口时间跨度分布
    benign_timespan = [w['time_span'] for w in window_dict['benign']]
    malicious_timespan = [w['time_span'] for w in window_dict['malicious']]

    axes[0, 0].hist(benign_timespan, alpha=0.5, label='Benign', bins=30, color='green')
    axes[0, 0].hist(malicious_timespan, alpha=0.5, label='Malicious', bins=30, color='red')
    axes[0, 0].set_title(f'Time Span Distribution (Window: {window_size} flows, Step: {step_size} flows)')
    axes[0, 0].set_xlabel('Window Time Span (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 绘制唯一时间点分布
    benign_timestamps = [w['unique_timestamps'] for w in window_dict['benign']]
    malicious_timestamps = [w['unique_timestamps'] for w in window_dict['malicious']]

    axes[0, 1].hist(benign_timestamps, alpha=0.5, label='Benign', bins=30, color='green')
    axes[0, 1].hist(malicious_timestamps, alpha=0.5, label='Malicious', bins=30, color='red')
    axes[0, 1].set_title(f'Unique Timestamps Distribution (Window: {window_size} flows, Step: {step_size} flows)')
    axes[0, 1].set_xlabel('Number of Unique Timestamps per Window')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 绘制流量率分布
    benign_flow_rate = [w['flow_rate'] for w in window_dict['benign']]
    malicious_flow_rate = [w['flow_rate'] for w in window_dict['malicious']]

    axes[1, 0].hist(benign_flow_rate, alpha=0.5, label='Benign', bins=30, color='green')
    axes[1, 0].hist(malicious_flow_rate, alpha=0.5, label='Malicious', bins=30, color='red')
    axes[1, 0].set_title(f'Flow Rate Distribution (Window: {window_size} flows, Step: {step_size} flows)')
    axes[1, 0].set_xlabel('Flow Rate (flows per minute)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 绘制时间密度与流量率关系
    axes[1, 1].scatter([w['flow_rate'] for w in window_dict['benign']],
                       [w['time_density'] for w in window_dict['benign']],
                       alpha=0.5, label='Benign', color='green')
    axes[1, 1].scatter([w['flow_rate'] for w in window_dict['malicious']],
                       [w['time_density'] for w in window_dict['malicious']],
                       alpha=0.5, label='Malicious', color='red')
    axes[1, 1].set_title(f'Flow Rate vs Time Density (Window: {window_size} flows, Step: {step_size} flows)')
    axes[1, 1].set_xlabel('Flow Rate (flows per minute)')
    axes[1, 1].set_ylabel('Time Density (unique timestamps per minute)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flow_distribution_w{window_size}_s{step_size}.png'))
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
        plt.title(f'Attack Type Distribution (Window: {window_size} flows, Step: {step_size} flows)')
        plt.xlabel('Attack Type')
        plt.ylabel('Flow Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'flow_attacks_w{window_size}_s{step_size}.png'))
        plt.close()


def prepare_sequence_datasets(window_dict, output_dir, window_size, step_size, test_size=0.2):
    """
    准备用于序列模型的数据集

    参数:
    - window_dict: 窗口数据字典
    - output_dir: 输出目录
    - window_size: 窗口大小
    - step_size: 窗口步长
    - test_size: 测试集比例
    """
    from sklearn.model_selection import train_test_split

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 合并正常和异常窗口
    all_windows = window_dict['benign'] + window_dict['malicious']

    # 提取特征和标签
    X = np.array([window['features'] for window in all_windows])
    y = np.array([window['is_malicious'] for window in all_windows])

    # 获取每个样本的攻击类型
    attack_types = []
    for window in all_windows:
        if window['is_malicious'] == 1:
            attack_types.append(window['primary_attack'])
        else:
            attack_types.append('Benign')

    # 将攻击类型转换为数字标签
    attack_type_map = {attack: i for i, attack in enumerate(np.unique(attack_types))}
    attack_type_labels = np.array([attack_type_map[attack] for attack in attack_types])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        X, y, attack_type_labels, test_size=test_size, random_state=42, stratify=y
    )

    # 保存数据集
    np.save(os.path.join(output_dir, f'X_train_w{window_size}_s{step_size}.npy'), X_train)
    np.save(os.path.join(output_dir, f'X_test_w{window_size}_s{step_size}.npy'), X_test)
    np.save(os.path.join(output_dir, f'y_train_w{window_size}_s{step_size}.npy'), y_train)
    np.save(os.path.join(output_dir, f'y_test_w{window_size}_s{step_size}.npy'), y_test)
    np.save(os.path.join(output_dir, f'attack_train_w{window_size}_s{step_size}.npy'), attack_train)
    np.save(os.path.join(output_dir, f'attack_test_w{window_size}_s{step_size}.npy'), attack_test)

    # 保存攻击类型映射
    with open(os.path.join(output_dir, f'attack_map_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        pickle.dump(attack_type_map, f)

    # 输出数据集信息
    print(f"\n数据集信息 (w{window_size}_s{step_size}):")
    print(f"特征形状: {X.shape}")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"正常样本: {np.sum(y == 0)} 个 ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"异常样本: {np.sum(y == 1)} 个 ({np.sum(y == 1) / len(y) * 100:.1f}%)")

    # 输出攻击类型分布
    print("\n攻击类型分布:")
    attack_counts = pd.Series(attack_types).value_counts()
    for attack, count in attack_counts.items():
        print(f"- {attack}: {count} ({count / len(attack_types) * 100:.1f}%)")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'attack_train': attack_train,
        'attack_test': attack_test,
        'attack_map': attack_type_map
    }


def create_flattened_dataset(window_dict, output_path):
    """
    将窗口数据转换为扁平化格式用于传统机器学习

    参数:
    - window_dict: 窗口数据字典
    - output_path: 输出CSV文件路径
    """
    # 合并正常和异常窗口
    all_windows = []

    # 处理正常窗口
    for window in window_dict['benign']:
        # 计算每个窗口的统计特征
        features = window['features']

        flat_window = {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'time_span': window['time_span'],
            'unique_timestamps': window['unique_timestamps'],
            'flow_rate': window['flow_rate'],
            'time_density': window['time_density'],
            'is_malicious': 0,

            # 添加统计特征
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0)
        }

        all_windows.append(flat_window)

    # 处理异常窗口
    for window in window_dict['malicious']:
        # 计算每个窗口的统计特征
        features = window['features']

        flat_window = {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'time_span': window['time_span'],
            'unique_timestamps': window['unique_timestamps'],
            'flow_rate': window['flow_rate'],
            'time_density': window['time_density'],
            'is_malicious': 1,
            'malicious_ratio': window['malicious_ratio'],
            'primary_attack': window['primary_attack'],

            # 添加统计特征
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0)
        }

        all_windows.append(flat_window)

    # 展开统计特征
    feature_names = window_dict['feature_names']
    flattened_data = []

    for window in all_windows:
        flat_dict = {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'time_span': window['time_span'],
            'unique_timestamps': window['unique_timestamps'],
            'flow_rate': window['flow_rate'],
            'time_density': window['time_density'],
            'is_malicious': window['is_malicious']
        }

        # 如果是异常窗口，添加攻击信息
        if window['is_malicious'] == 1:
            flat_dict['malicious_ratio'] = window['malicious_ratio']
            flat_dict['primary_attack'] = window['primary_attack']

        # 展开统计特征
        for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
            for i, feature in enumerate(feature_names):
                flat_dict[f'{feature}_{stat}'] = window[stat][i]

        flattened_data.append(flat_dict)

    # 转换为DataFrame并保存
    df = pd.DataFrame(flattened_data)
    df.sort_values('start_time', inplace=True)
    df.to_csv(output_path, index=False)

    print(f"扁平化数据集已保存至 {output_path}")
    print(f"数据集包含 {len(df)} 个窗口，其中 {df['is_malicious'].sum()} 个异常窗口")

    return df


if __name__ == "__main__":
    # 文件路径配置
    dataset_path = "cicids2017/clean/all_data.feather"  # 清洗后的数据集路径
    output_dir = "cicids2017/flow_windows_w10000_s1000"  # 输出目录

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 提取基于流量的窗口数据集
    extract_flow_based_windows(
        dataset_path=dataset_path,
        output_dir=output_dir,
        window_sizes=[100, 200, 500],  # 窗口大小（流量数量）
        step_sizes=[20, 100],  # 滑动步长（流量数量）
        apply_pca=True,  # 应用PCA降维
        pca_components=20  # 保留20个主成分
    )

    # 为一个特定窗口参数创建序列数据集（用于深度学习）
    window_dict_path = os.path.join(output_dir, "flow_windows_w200_s20.pkl")
    with open(window_dict_path, 'rb') as f:
        window_dict = pickle.load(f)

    sequence_dir = os.path.join(output_dir, "sequence_data")
    prepare_sequence_datasets(window_dict, sequence_dir, window_size=200, step_size=20)

    # 创建扁平化数据集（用于传统机器学习）
    flat_output_path = os.path.join(output_dir, "flat_dataset_w200_s20.csv")
    flat_df = create_flattened_dataset(window_dict, flat_output_path)