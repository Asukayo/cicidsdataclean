"""
UNSW-NB15 数据预处理脚本
功能：清洗 → 滑动窗口 → 整合，对齐 CICIDS2017 的处理流程
用法：python process_unsw_nb15.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置区 —— 根据你的环境修改
# ============================================================
RAW_DIR = "/home/ubuntu/wyh/cicdis/UNSW_NB15/raw"
CLEAN_DIR = "/home/ubuntu/wyh/cicdis/UNSW_NB15/clean"
WINDOW_DIR = "/home/ubuntu/wyh/cicdis/UNSW_NB15/flow_windows"
INTEGRATED_DIR = "/home/ubuntu/wyh/cicdis/UNSW_NB15/integrated_windows"

WINDOW_SIZE = 100  # 与 CICIDS2017 保持一致
WINDOW_STEP = 20  # 与 CICIDS2017 保持一致

# UNSW-NB15 原始 CSV 无表头，列名来自官方 features 文件
# 参考: https://github.com/flerkenvn/unsw/blob/master/NUSW-NB15_features.csv
COLUMN_NAMES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
    'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
    'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'attack_cat', 'label'
]

# 需要删除的列（标识符 + 时间戳原始列）
DROP_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport',  # 网络标识符，不参与建模
    'Stime', 'Ltime',  # 时间戳仅用于排序，排序后删除
]

# 需要编码的分类列
CATEGORICAL_COLUMNS = ['proto', 'state', 'service']


# ============================================================
# Step 1: 数据清洗
# ============================================================
def step1_clean(raw_dir, clean_dir):
    """
    读取 4 个原始 CSV → 合并 → 清洗 → 保存
    对应 CICIDS2017 的 01_pre_data_cleaning.py
    """
    os.makedirs(clean_dir, exist_ok=True)

    csv_files = sorted([
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.startswith('UNSW-NB15_') and f.endswith('.csv') and 'features' not in f.lower()
    ])

    if not csv_files:
        raise FileNotFoundError(f"在 {raw_dir} 中未找到 UNSW-NB15_*.csv 文件")

    print(f"找到 {len(csv_files)} 个 CSV 文件")

    # ---- 读取并合并 ----
    dfs = []
    for f in csv_files:
        print(f"  读取: {os.path.basename(f)}")
        df = pd.read_csv(f, header=None, names=COLUMN_NAMES,
                         low_memory=False, encoding='latin')
        print(f"    形状: {df.shape}")
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总量: {all_data.shape}")
    del dfs
    gc.collect()

    # ---- 标签处理 ----
    # attack_cat: 攻击类别名称（含空格需清理），空值 → Normal
    # label: 二元标签 0=正常 1=攻击
    all_data['attack_cat'] = all_data['attack_cat'].astype(str).str.strip()
    all_data['attack_cat'] = all_data['attack_cat'].replace(
        {'': 'Normal', 'nan': 'Normal', 'None': 'Normal', ' ': 'Normal',
         'Backdoor': 'Backdoors', 'Backdoors': 'Backdoors'}
    )
    # 统一：label==0 的全部设为 Normal
    all_data.loc[all_data['label'] == 0, 'attack_cat'] = 'Normal'

    # 创建统一的 Label 列（与 CICIDS2017 对齐：Normal 对应 Benign）
    all_data['Label'] = all_data['attack_cat'].replace({'Normal': 'Benign'})
    all_data['Label'] = all_data['Label'].astype('category')

    print("\n标签分布:")
    print(all_data['Label'].value_counts())

    # ---- 时间戳处理 & 排序 ----
    # Stime 是 Unix 时间戳（秒），转为 datetime 用于排序
    all_data['Stime'] = pd.to_numeric(all_data['Stime'], errors='coerce')
    all_data['Ltime'] = pd.to_numeric(all_data['Ltime'], errors='coerce')
    all_data['Timestamp'] = pd.to_datetime(all_data['Stime'], unit='s', errors='coerce')
    all_data = all_data.sort_values(by='Stime').reset_index(drop=True)

    # ---- 分类特征编码 ----
    cat_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in all_data.columns:
            all_data[col] = all_data[col].astype(str).str.strip()
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
            cat_encoders[col] = le
            print(f"  编码 {col}: {len(le.classes_)} 个类别")

    # ---- ct_ftp_cmd 修复（该列应为数值但可能含空格）----
    if 'ct_ftp_cmd' in all_data.columns:
        all_data['ct_ftp_cmd'] = pd.to_numeric(all_data['ct_ftp_cmd'], errors='coerce').fillna(0)

    # ---- 删除标识符列 ----
    all_data.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')
    # 同时删除 attack_cat（已被 Label 替代）和原始 label 列
    all_data.drop(columns=['attack_cat', 'label'], inplace=True, errors='ignore')

    # ---- 数值类型转换 & 清洗 ----
    non_meta_cols = [c for c in all_data.columns if c not in ['Timestamp', 'Label']]
    all_data[non_meta_cols] = all_data[non_meta_cols].apply(
        pd.to_numeric, errors='coerce'
    )
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    critical_nan = all_data['Timestamp'].isna().sum()
    print(f"\n删除 {critical_nan} 行时间戳无效的数据")
    all_data.dropna(subset=['Timestamp'], inplace=True)
    all_data[non_meta_cols] = all_data[non_meta_cols].fillna(0)

    # ---- 去重 ----
    dup_cols = all_data.columns.difference(['Label', 'Timestamp'])
    before = len(all_data)
    all_data.drop_duplicates(subset=dup_cols, inplace=True)
    print(f"去重: {before} → {len(all_data)} (删除 {before - len(all_data)} 行)")

    all_data.reset_index(drop=True, inplace=True)

    # ---- 保存 ----
    feature_cols = [c for c in all_data.columns if c not in ['Timestamp', 'Label']]

    save_path_parquet = os.path.join(clean_dir, 'all_data.parquet')
    save_path_feather = os.path.join(clean_dir, 'all_data.feather')
    all_data.to_parquet(save_path_parquet, index=False)
    all_data.to_feather(save_path_feather)

    # 保存编码器
    with open(os.path.join(clean_dir, 'categorical_encoders.pkl'), 'wb') as f:
        pickle.dump(cat_encoders, f)

    print(f"\n清洗完成，最终形状: {all_data.shape}")
    print(f"特征数: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")
    print(f"保存至: {clean_dir}")

    return all_data, feature_cols


# ============================================================
# Step 2 & 3: 滑动窗口生成 + 整合
# ============================================================
def step2_create_windows(all_data, feature_cols, window_dir, integrated_dir,
                         window_size=100, step_size=20):
    """
    在已排序的全量数据上生成滑动窗口，直接输出整合后的结果。
    对应 CICIDS2017 的 02_by_flow_without_pca_scaler.py + 03_integrate_all_days.py

    UNSW-NB15 没有 CICIDS2017 的天然按天拆分，所以直接在合并数据上做窗口。
    """
    os.makedirs(window_dir, exist_ok=True)
    os.makedirs(integrated_dir, exist_ok=True)

    # ---- 标签编码 ----
    label_encoder = LabelEncoder()
    label_encoder.fit(all_data['Label'])
    all_data['label_encoded'] = label_encoder.transform(all_data['Label'])
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"标签映射: {label_mapping}")

    # ---- 提取特征矩阵 ----
    feature_matrix = all_data[feature_cols].values.astype(np.float32)
    label_array = all_data['label_encoded'].values
    label_str_array = all_data['Label'].values
    timestamp_array = all_data['Timestamp'].values

    total_records = len(all_data)
    num_windows = (total_records - window_size) // step_size + 1

    print(f"\n总记录数: {total_records}")
    print(f"窗口大小: {window_size}, 步长: {step_size}")
    print(f"预计窗口数: {num_windows}")

    if num_windows <= 0:
        raise ValueError("数据量不足以生成窗口")

    # ---- 生成窗口 ----
    X_windows = np.zeros((num_windows, window_size, len(feature_cols)), dtype=np.float32)
    y_windows = np.zeros((num_windows, window_size), dtype=np.int32)
    window_metadata = []

    for i in tqdm(range(num_windows), desc="生成滑动窗口"):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        X_windows[i] = feature_matrix[start_idx:end_idx]
        y_windows[i] = label_array[start_idx:end_idx]

        window_labels = label_str_array[start_idx:end_idx]
        has_malicious = np.any(window_labels != 'Benign')

        metadata = {
            'window_id': i,
            'global_window_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': str(timestamp_array[start_idx]),
            'end_time': str(timestamp_array[end_idx - 1]),
            'is_malicious': 1 if has_malicious else 0,
            'source_file': 'UNSW-NB15',
        }

        if has_malicious:
            attack_labels = window_labels[window_labels != 'Benign']
            unique, counts = np.unique(attack_labels, return_counts=True)
            attack_counts = dict(zip(unique, counts.astype(int)))
            metadata['attack_types'] = attack_counts
            metadata['primary_attack'] = max(attack_counts, key=attack_counts.get)

        window_metadata.append(metadata)

    # ---- 统计 ----
    malicious_count = sum(1 for w in window_metadata if w['is_malicious'] == 1)
    benign_count = len(window_metadata) - malicious_count
    print(f"\n窗口统计: 共 {len(window_metadata)} 个窗口")
    print(f"  正常: {benign_count} ({benign_count / len(window_metadata) * 100:.1f}%)")
    print(f"  异常: {malicious_count} ({malicious_count / len(window_metadata) * 100:.1f}%)")

    # ---- 保存到 window_dir（中间结果）----
    np.save(os.path.join(window_dir,
                         f'X_windows_w{window_size}_s{step_size}.npy'), X_windows)
    np.save(os.path.join(window_dir,
                         f'y_windows_w{window_size}_s{step_size}.npy'), y_windows)

    with open(os.path.join(window_dir,
                           f'metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        pickle.dump({
            'window_metadata': window_metadata,
            'feature_names': feature_cols,
            'label_mapping': label_mapping,
            'label_encoder': label_encoder,
            'config': {'window_size': window_size, 'step_size': step_size}
        }, f)

    # ---- 保存到 integrated_dir（最终结果，与 CICIDS2017 格式对齐）----
    np.save(os.path.join(integrated_dir,
                         f'integrated_X_w{window_size}_s{step_size}.npy'), X_windows)
    np.save(os.path.join(integrated_dir,
                         f'integrated_y_w{window_size}_s{step_size}.npy'), y_windows)

    integrated_metadata = {
        'window_metadata': window_metadata,
        'feature_names': feature_cols,
        'label_mapping': label_mapping,
        'file_window_indices': {
            'UNSW-NB15': {'start': 0, 'end': len(X_windows)}
        },
        'file_order': ['UNSW-NB15'],
        'config': {
            'window_size': window_size,
            'step_size': step_size,
            'total_windows': len(X_windows),
            'num_features': len(feature_cols),
        }
    }

    with open(os.path.join(integrated_dir,
                           f'integrated_metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        pickle.dump(integrated_metadata, f)

    # ---- 保存特征信息（与 CICIDS2017 的 features_info.pkl 对齐）----
    with open(os.path.join(window_dir, 'features_info.pkl'), 'wb') as f:
        pickle.dump({
            'features': feature_cols,
            'config': {'window_size': window_size, 'step_size': step_size}
        }, f)

    # ---- 生成摘要 ----
    create_summary(X_windows, y_windows, integrated_metadata,
                   integrated_dir, window_size, step_size)

    print(f"\n最终数据形状: X{X_windows.shape}, y{y_windows.shape}")
    return X_windows.shape, y_windows.shape


def create_summary(X, y, metadata, output_dir, window_size, step_size):
    """生成统计摘要文件"""
    summary_file = os.path.join(
        output_dir, f'integration_summary_w{window_size}_s{step_size}.txt'
    )

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("UNSW-NB15 Integration Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total windows: {X.shape[0]}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Features: {X.shape[2]}\n")
        f.write(f"Final shape: X{X.shape}, y{y.shape}\n\n")

        window_metadata = metadata['window_metadata']
        malicious_windows = sum(1 for w in window_metadata if w['is_malicious'] == 1)
        benign_windows = len(window_metadata) - malicious_windows

        f.write("Window-level distribution:\n")
        f.write(f"  Benign:    {benign_windows} "
                f"({benign_windows / len(window_metadata) * 100:.1f}%)\n")
        f.write(f"  Malicious: {malicious_windows} "
                f"({malicious_windows / len(window_metadata) * 100:.1f}%)\n\n")

        label_mapping = metadata['label_mapping']
        f.write("Flow-level distribution:\n")
        label_counts = {}
        for label_idx in label_mapping:
            count = int(np.sum(y == int(label_idx)))
            label_name = label_mapping[label_idx]
            if count > 0:
                label_counts[label_name] = count

        total_flows = sum(label_counts.values())
        for name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {name}: {count} ({count / total_flows * 100:.2f}%)\n")

        # 攻击类型分布
        f.write(f"\nAttack type breakdown (in malicious windows):\n")
        attack_type_counts = {}
        for w in window_metadata:
            if w['is_malicious'] == 1 and 'attack_types' in w:
                for atype, cnt in w['attack_types'].items():
                    attack_type_counts[atype] = attack_type_counts.get(atype, 0) + cnt
        for atype, cnt in sorted(attack_type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {atype}: {cnt}\n")

    print(f"摘要已保存: {summary_file}")


def verify_data(integrated_dir, window_size, step_size):
    """验证整合数据的一致性"""
    X = np.load(os.path.join(
        integrated_dir, f'integrated_X_w{window_size}_s{step_size}.npy'
    ), mmap_mode='r')
    y = np.load(os.path.join(
        integrated_dir, f'integrated_y_w{window_size}_s{step_size}.npy'
    ), mmap_mode='r')
    with open(os.path.join(
            integrated_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl'
    ), 'rb') as f:
        metadata = pickle.load(f)

    assert X.shape[0] == y.shape[0], "X 和 y 的窗口数不匹配"
    assert X.shape[0] == len(metadata['window_metadata']), "窗口数与元数据不匹配"
    assert X.shape[1] == window_size, f"窗口大小不匹配: {X.shape[1]} vs {window_size}"

    print(f"验证通过: X{X.shape}, y{y.shape}")
    return True


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UNSW-NB15 数据预处理流程")
    print("=" * 60)

    # Step 1: 清洗
    print("\n>>> Step 1: 数据清洗")
    print("-" * 40)
    all_data, feature_cols = step1_clean(RAW_DIR, CLEAN_DIR)

    # Step 2 & 3: 窗口生成 + 整合
    print("\n>>> Step 2 & 3: 滑动窗口生成 & 整合")
    print("-" * 40)
    X_shape, y_shape = step2_create_windows(
        all_data, feature_cols,
        WINDOW_DIR, INTEGRATED_DIR,
        window_size=WINDOW_SIZE,
        step_size=WINDOW_STEP
    )

    # 释放内存
    del all_data
    gc.collect()

    # 验证
    print("\n>>> 验证")
    print("-" * 40)
    verify_data(INTEGRATED_DIR, WINDOW_SIZE, WINDOW_STEP)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"  清洗数据: {CLEAN_DIR}")
    print(f"  窗口数据: {WINDOW_DIR}")
    print(f"  整合数据: {INTEGRATED_DIR}")
    print(f"  最终形状: X{X_shape}, y{y_shape}")
    print("=" * 60)