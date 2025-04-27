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


def create_time_windows(dataset_path, window_size_minutes=5, step_size_minutes=1,
                        min_flows=10, apply_pca=False, pca_components=20):
    """
    基于时间窗口滑动方式创建时间序列数据集，处理Timestamp重复的情况

    参数:
    - dataset_path: 数据集路径，指向清洗后的 feather/parquet 文件
    - window_size_minutes: 窗口大小（分钟）
    - step_size_minutes: 窗口滑动步长（分钟）
    - min_flows: 窗口内最小流量数，小于此数量的窗口将被丢弃
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
            window_features = window_data[feature_cols]

            # 计算精简的统计特征（只保留关键统计量）
            window_stats = {
                'start_time': current_time,
                'end_time': window_end,
                'flow_count': len(window_data),
                'unique_timestamps': unique_times_in_window,
                'time_density': unique_times_in_window / (window_size_minutes),  # 每分钟的平均唯一时间点数
                'stats': {
                    'mean': window_features.mean().to_dict(),
                    'std': window_features.std().to_dict(),
                    'median': window_features.quantile(0.5).to_dict()
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
        'malicious': malicious_windows,
        'feature_names': feature_cols,
        'original_features': original_features,
        'scaler': scaler,
        'pca_transformer': pca_transformer,
        'config': {
            'window_size_minutes': window_size_minutes,
            'step_size_minutes': step_size_minutes,
            'apply_pca': apply_pca,
            'pca_components': pca_components
        }
    }


def extract_time_series(dataset_path, output_dir, window_sizes=[5, 10, 15],
                        step_sizes=[1, 2, 5], apply_pca=False, pca_components=20):
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
                step_size_minutes=step_size,
                apply_pca=apply_pca,
                pca_components=pca_components
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


def create_flattened_dataset(time_series_path, output_path):
    """
    将时间序列窗口数据转换为扁平化格式用于机器学习

    参数:
    - time_series_path: 时间序列数据pkl文件路径
    - output_path: 输出CSV文件路径
    """
    with open(time_series_path, 'rb') as f:
        window_dict = pickle.load(f)

    # 合并正常和异常窗口
    all_windows = []

    # 获取特征名列表
    feature_names = window_dict['feature_names']

    # 处理正常窗口
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
        for stat_type in ['mean', 'std', 'median']:
            for feature in feature_names:
                flat_window[f'{feature}_{stat_type}'] = window['stats'][stat_type][feature]

        all_windows.append(flat_window)

    # 处理异常窗口
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
        for stat_type in ['mean', 'std', 'median']:
            for feature in feature_names:
                flat_window[f'{feature}_{stat_type}'] = window['stats'][stat_type][feature]

        all_windows.append(flat_window)

    # 转换为DataFrame并保存
    df = pd.DataFrame(all_windows)
    df.sort_values('start_time', inplace=True)
    df.to_csv(output_path, index=False)

    print(f"扁平化数据集已保存至 {output_path}")
    print(f"数据集包含 {len(df)} 个窗口，其中 {df['is_malicious'].sum()} 个异常窗口")

    # 输出特征数量
    feature_columns = [col for col in df.columns if
                       col not in ['start_time', 'end_time', 'is_malicious', 'primary_attack']]
    print(f"特征数量: {len(feature_columns)}")

    return df


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


def select_important_features(dataset_path, method='mutual_info', k=20):
    """
    特征选择：从原始数据中选择最重要的特征

    参数:
    - dataset_path: 数据集路径
    - method: 特征选择方法，可选 'mutual_info', 'chi2', 'f_classif', 'rfe'
    - k: 选择的特征数量

    返回:
    - 选择的特征名称列表
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    # 加载数据
    if dataset_path.endswith('.feather'):
        df = pd.read_feather(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .feather 或 .parquet 文件")

    # 处理标签列
    df['target'] = (df['Label'] != 'Benign').astype(int)

    # 提取特征和标签
    feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Label', 'target']]
    X = df[feature_cols]
    y = df['target']

    selected_features = []

    if method == 'mutual_info':
        # 互信息特征选择
        print("使用互信息方法选择特征...")
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]

        # 打印特征重要性分数
        scores = selector.scores_
        feature_scores = list(zip(feature_cols, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        print("\n前20个最重要特征及其分数:")
        for feature, score in feature_scores[:20]:
            print(f"{feature}: {score:.4f}")

    elif method == 'chi2':
        # 确保所有特征非负（卡方检验要求）
        X_chi = X.copy()
        for col in X_chi.columns:
            if X_chi[col].min() < 0:
                X_chi[col] = X_chi[col] - X_chi[col].min()

        # 卡方检验特征选择
        print("使用卡方检验方法选择特征...")
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_chi, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]

    elif method == 'f_classif':
        # F检验特征选择
        print("使用F检验方法选择特征...")
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]

    elif method == 'rfe':
        # 递归特征消除
        print("使用递归特征消除方法选择特征...")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        selector.fit(X, y)
        selected_indices = np.where(selector.support_)[0]
        selected_features = [feature_cols[i] for i in selected_indices]

        # 获取特征重要性排名
        feature_ranking = list(zip(feature_cols, selector.ranking_))
        feature_ranking.sort(key=lambda x: x[1])  # 排名越小越重要
        print("\n前20个最重要特征及其排名:")
        for feature, rank in feature_ranking[:20]:
            print(f"{feature}: {rank}")

    else:
        raise ValueError(f"不支持的特征选择方法: {method}")

    print(f"\n已选择 {len(selected_features)} 个特征:")
    for feature in selected_features:
        print(f"- {feature}")

    return selected_features


if __name__ == "__main__":
    # 文件路径配置
    dataset_path = "cicids2017/clean/all_data.feather"  # 清洗后的数据集路径
    output_dir = "cicids2017/time_series"  # 输出目录

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 分析时间戳分布
    print("分析数据集的时间戳分布...")
    time_counts = analyze_time_distribution(dataset_path)

    # 步骤2: 特征选择（可选）
    # selected_features = select_important_features(dataset_path, method='mutual_info', k=20)

    # 步骤3: 创建不同参数的时间窗口数据集
    extract_time_series(
        dataset_path=dataset_path,
        output_dir=output_dir,
        window_sizes=[5, 10, 15],  # 窗口大小（分钟）
        step_sizes=[1, 5],  # 滑动步长（分钟）
        apply_pca=True,  # 应用PCA降维
        pca_components=20  # 保留20个主成分
    )

    # 步骤4: 为一个特定窗口参数创建扁平化数据集（机器学习用）
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