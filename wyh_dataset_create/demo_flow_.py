import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def create_flow_based_windows(dataset_path, window_size=100, step_size=20, apply_pca=False, pca_components=20):
    """
    基于固定流量数量创建窗口，保留原始流量记录和标签

    参数:
    - dataset_path: 数据集路径
    - window_size: 每个窗口包含的流量数量
    - step_size: 窗口滑动的流量数量
    - apply_pca: 是否应用PCA降维
    - pca_components: PCA组件数量

    返回:
    - 包含窗口数据的字典
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

    # 初始化窗口列表
    windows = []

    # 创建窗口
    for i in tqdm(range(num_windows)):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        # 提取当前窗口的数据
        window_data = df.iloc[start_idx:end_idx].copy()

        # 判断窗口内是否有恶意流量
        has_malicious = (window_data['Label'] != 'Benign').any()

        # 创建窗口信息（仅包含必要内容）
        window_info = {
            'window_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': window_data['Timestamp'].min(),
            'end_time': window_data['Timestamp'].max(),
            'features': window_data[feature_cols].values,  # 特征数据
            'labels': window_data['Label'].values,  # 原始标签
            'timestamps': window_data['Timestamp'].values,  # 时间戳
            'is_malicious': 1 if has_malicious else 0  # 窗口是否包含异常流量（二分类标签）
        }

        # 如果窗口包含恶意流量，添加相关信息
        if has_malicious:
            attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
            window_info['attack_types'] = attack_counts
            window_info['primary_attack'] = max(attack_counts, key=attack_counts.get)

        windows.append(window_info)

    # 统计窗口信息
    malicious_count = sum(1 for w in windows if w['is_malicious'] == 1)
    benign_count = len(windows) - malicious_count

    print(f"创建完成：窗口总数 {len(windows)} 个")
    print(f"正常窗口: {benign_count} 个 ({benign_count / len(windows) * 100:.1f}%)")
    print(f"异常窗口: {malicious_count} 个 ({malicious_count / len(windows) * 100:.1f}%)")

    return {
        'windows': windows,
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

            # 仅分析攻击类型分布，不做其他统计
            plot_attack_distribution(result_dict, window_size, step_size, output_dir)


def plot_attack_distribution(result_dict, window_size, step_size, output_dir):
    """绘制攻击类型分布"""
    # 计算包含异常的窗口数量
    malicious_windows = [w for w in result_dict['windows'] if w['is_malicious'] == 1]

    if not malicious_windows:
        print("没有发现异常窗口，跳过绘图")
        return

    # 统计攻击类型
    all_attacks = {}
    for window in malicious_windows:
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
    plt.savefig(os.path.join(output_dir, f'attacks_w{window_size}_s{step_size}.png'))
    plt.close()


def prepare_sequence_datasets(window_dict, output_dir, window_size, step_size, test_size=0):
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

    # 获取所有窗口
    windows = window_dict['windows']

    # 提取特征和标签
    X = np.array([window['features'] for window in windows])
    y = np.array([window['is_malicious'] for window in windows])

    # 获取每个样本的原始标签
    original_labels = [window['labels'] for window in windows]

    # 获取每个样本的攻击类型（如果是异常窗口）
    attack_types = []
    for window in windows:
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


if __name__ == "__main__":
    # 文件路径配置
    dataset_path = "../cicids2017/clean/Tuesday-WorkingHours.pcap_ISCX.csv.parquet"  # 清洗后的数据集路径
    output_dir = "../cicids2017/flow_windows_w10000_s1000/Tuesday-WorkingHours"  # 输出目录

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 提取基于流量的窗口数据集
    extract_flow_based_windows(
        dataset_path=dataset_path,
        output_dir=output_dir,
        window_sizes=[1000, 5000, 10000],  # 窗口大小（流量数量）
        step_sizes=[500, 2000],  # 滑动步长（流量数量）
        apply_pca=False,  # 应用PCA降维
        pca_components=20  # 保留20个主成分
    )

    # 为特定窗口参数创建序列数据集
    window_dict_path = os.path.join(output_dir, "flow_windows_w5000_s500.pkl")
    with open(window_dict_path, 'rb') as f:
        window_dict = pickle.load(f)

    sequence_dir = os.path.join(output_dir, "sequence_data")
    prepare_sequence_datasets(window_dict, sequence_dir, window_size=5000, step_size=500)