"""
CICIDS2018 数据预处理脚本（精简版）
====================================
单次运行：读取CSV → 清洗 → 滑动窗口 → 直接输出 integrated_windows
不生成任何中间文件，节省磁盘空间

用法：python process_cicids2018.py
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
# 配置
# ============================================================
RAW_DIR = "/home/ubuntu/wyh/cicdis/cicids2018/original"
OUTPUT_DIR = "/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"

WINDOW_SIZE = 100
WINDOW_STEP = 20

# 按日期排序的文件列表（时间顺序很重要）
FILE_ORDER = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",    # 原始文件名有拼写错误
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

# 与 CICIDS2017 完全相同的列名映射
MAPPER = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Fwd Packets Length Total',
    'Total Length of Fwd Packets': 'Fwd Packets Length Total',
    'TotLen Bwd Pkts': 'Bwd Packets Length Total',
    'Total Length of Bwd Packets': 'Bwd Packets Length Total',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Fwd Header Len': 'Fwd Header Length',
    'Bwd Header Len': 'Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Packet Length Min',
    'Min Packet Length': 'Packet Length Min',
    'Pkt Len Max': 'Packet Length Max',
    'Max Packet Length': 'Packet Length Max',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Pkt Size Avg': 'Avg Packet Size',
    'Average Packet Size': 'Avg Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets',
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init Fwd Win Bytes',
    'Init_Win_bytes_forward': 'Init Fwd Win Bytes',
    'Init Bwd Win Byts': 'Init Bwd Win Bytes',
    'Init_Win_bytes_backward': 'Init Bwd Win Bytes',
    'Fwd Act Data Pkts': 'Fwd Act Data Packets',
    'act_data_pkt_fwd': 'Fwd Act Data Packets',
    'Fwd Seg Size Min': 'Fwd Seg Size Min',
    'min_seg_size_forward': 'Fwd Seg Size Min',
}

DROP_COLUMNS = [
    "Flow ID",
    "Source IP", "Src IP",
    "Source Port", "Src Port",
    "Destination IP", "Dst IP",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk", "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate", "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk", "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk", "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate", "Bwd Blk Rate Avg",
    'Fwd Header Length.1',
]


def clean_single_file(file_path):
    """清洗单个CSV文件，返回清洗后的DataFrame"""
    file_name = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n--- {file_name} ({file_size_mb:.0f} MB) ---")

    # 大文件分块读取再拼接
    if file_size_mb > 1000:
        print("  大文件，分块读取...")
        chunks = []
        for chunk in pd.read_csv(file_path, skipinitialspace=True,
                                 encoding='latin', low_memory=False,
                                 chunksize=500_000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    else:
        df = pd.read_csv(file_path, skipinitialspace=True,
                         encoding='latin', low_memory=False)

    print(f"  原始形状: {df.shape}")
    print(f"  标签分布:\n{df['Label'].value_counts().to_string()}")

    # 列名统一
    df.rename(columns=MAPPER, inplace=True)

    # 删除无关列
    df.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')

    # 时间戳处理
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Timestamp'] = df['Timestamp'].apply(
        lambda x: x + pd.Timedelta(hours=12) if pd.notna(x) and x.hour < 8 else x
    )
    df = df.sort_values(by='Timestamp')

    # 标签统一
    df['Label'] = df['Label'].astype(str).str.strip()
    df['Label'] = df['Label'].replace({'BENIGN': 'Benign', 'Benign': 'Benign'})
    df['Label'] = df['Label'].astype('category')

    # 数值类型转换
    non_meta_cols = [c for c in df.columns if c not in ['Timestamp', 'Label']]
    obj_cols = df[non_meta_cols].select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')

    int_cols = df.select_dtypes(include='integer').columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce', downcast='integer')
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce', downcast='float')

    # 清洗异常值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ts_nan = df['Timestamp'].isna().sum()
    if ts_nan > 0:
        print(f"  删除 {ts_nan} 行时间戳无效数据")
        df.dropna(subset=['Timestamp'], inplace=True)
    df[non_meta_cols] = df[non_meta_cols].fillna(0)

    # 去重
    dup_cols = df.columns.difference(['Label', 'Timestamp'])
    before = len(df)
    df.drop_duplicates(subset=dup_cols, inplace=True)
    print(f"  去重: {before} → {len(df)} (删除 {before - len(df)})")

    df.reset_index(drop=True, inplace=True)
    print(f"  清洗后标签:\n{df['Label'].value_counts().to_string()}")

    return df


def get_common_features(raw_dir, file_order):
    """快速扫描所有文件获取共同特征列"""
    common = None
    for fname in file_order:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            continue
        sample = pd.read_csv(fpath, nrows=5, skipinitialspace=True,
                             encoding='latin', low_memory=False)
        sample.rename(columns=MAPPER, inplace=True)
        sample.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')
        cols = [c for c in sample.select_dtypes(include='number').columns
                if c not in ['Timestamp']]
        if common is None:
            common = set(cols)
        else:
            common = common.intersection(set(cols))
    return sorted(list(common))


def make_windows(df, feature_cols, label_encoder, window_size, step_size, source_file):
    """从单个文件的DataFrame生成滑动窗口"""
    feature_matrix = df[feature_cols].values.astype(np.float32)
    label_array = label_encoder.transform(df['Label'])
    label_str = df['Label'].values

    total = len(df)
    num_windows = (total - window_size) // step_size + 1

    if num_windows <= 0:
        print(f"  警告: {source_file} 数据不足，跳过")
        return None, None, []

    X = np.zeros((num_windows, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros((num_windows, window_size), dtype=np.int32)
    metadata = []

    for i in range(num_windows):
        s = i * step_size
        e = s + window_size
        X[i] = feature_matrix[s:e]
        y[i] = label_array[s:e]

        window_labels = label_str[s:e]
        has_mal = np.any(window_labels != 'Benign')

        meta = {
            'window_id': i,
            'start_idx': s,
            'end_idx': e,
            'is_malicious': 1 if has_mal else 0,
            'source_file': source_file,
        }
        if has_mal:
            atk = window_labels[window_labels != 'Benign']
            unique, counts = np.unique(atk, return_counts=True)
            meta['attack_types'] = dict(zip(unique, counts.astype(int)))
            meta['primary_attack'] = max(meta['attack_types'], key=meta['attack_types'].get)

        metadata.append(meta)

    return X, y, metadata


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CICIDS2018 预处理（直接输出 integrated_windows）")
    print("=" * 60)

    # 1. 获取共同特征
    print("\n>>> 扫描共同特征...")
    feature_cols = get_common_features(RAW_DIR, FILE_ORDER)
    print(f"共同特征数: {len(feature_cols)}")

    # 2. 先扫描所有标签以构建全局 LabelEncoder
    print("\n>>> 扫描所有标签...")
    all_labels = set()
    for fname in FILE_ORDER:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        sample = pd.read_csv(fpath, usecols=['Label'], skipinitialspace=True,
                             encoding='latin', low_memory=False)
        sample['Label'] = sample['Label'].astype(str).str.strip().replace(
            {'BENIGN': 'Benign'}
        )
        all_labels.update(sample['Label'].unique())
        del sample
        gc.collect()

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(all_labels))
    label_mapping = {i: l for i, l in enumerate(label_encoder.classes_)}
    print(f"标签类别: {label_mapping}")

    # 3. 逐文件处理：清洗 → 窗口 → 立即释放原始数据
    all_X = []
    all_y = []
    all_metadata = []
    file_window_indices = {}
    current_idx = 0

    for fname in FILE_ORDER:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n!!! 文件不存在: {fname}，跳过")
            continue

        # 清洗
        df = clean_single_file(fpath)

        # 确保使用共同特征（缺失的填0）
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # 生成窗口
        day_name = fname.replace('_TrafficForML_CICFlowMeter.csv', '')
        X, y, meta = make_windows(df, feature_cols, label_encoder,
                                  WINDOW_SIZE, WINDOW_STEP, day_name)

        # 立即释放原始数据
        del df
        gc.collect()

        if X is None:
            continue

        # 更新全局索引
        for m in meta:
            m['global_window_id'] = current_idx + m['window_id']

        file_window_indices[day_name] = {
            'start': current_idx,
            'end': current_idx + len(X)
        }
        current_idx += len(X)

        mal_count = sum(1 for m in meta if m['is_malicious'] == 1)
        print(f"  窗口: {len(X)} (正常: {len(X) - mal_count}, 异常: {mal_count})")

        all_X.append(X)
        all_y.append(y)
        all_metadata.extend(meta)

        del X, y, meta
        gc.collect()

    # 4. 合并并保存
    print("\n>>> 合并所有窗口...")
    integrated_X = np.vstack(all_X)
    integrated_y = np.vstack(all_y)
    del all_X, all_y
    gc.collect()

    print(f"最终形状: X{integrated_X.shape}, y{integrated_y.shape}")

    # 保存
    ws, ss = WINDOW_SIZE, WINDOW_STEP
    np.save(os.path.join(OUTPUT_DIR, f'integrated_X_w{ws}_s{ss}.npy'), integrated_X)
    np.save(os.path.join(OUTPUT_DIR, f'integrated_y_w{ws}_s{ss}.npy'), integrated_y)

    integrated_metadata = {
        'window_metadata': all_metadata,
        'feature_names': feature_cols,
        'label_mapping': label_mapping,
        'label_encoder': label_encoder,
        'file_window_indices': file_window_indices,
        'file_order': [f.replace('_TrafficForML_CICFlowMeter.csv', '') for f in FILE_ORDER],
        'config': {
            'window_size': WINDOW_SIZE,
            'step_size': WINDOW_STEP,
            'total_windows': len(integrated_X),
            'num_features': len(feature_cols),
        }
    }

    with open(os.path.join(OUTPUT_DIR, f'integrated_metadata_w{ws}_s{ss}.pkl'), 'wb') as f:
        pickle.dump(integrated_metadata, f)

    # 保存 features_info（兼容 provider）
    with open(os.path.join(OUTPUT_DIR, 'features_info.pkl'), 'wb') as f:
        pickle.dump({
            'features': feature_cols,
            'config': {'window_size': WINDOW_SIZE, 'step_size': WINDOW_STEP}
        }, f)

    # 5. 生成摘要
    mal_windows = sum(1 for m in all_metadata if m['is_malicious'] == 1)
    ben_windows = len(all_metadata) - mal_windows

    summary_file = os.path.join(OUTPUT_DIR, f'integration_summary_w{ws}_s{ss}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CICIDS2018 Integration Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total windows: {integrated_X.shape[0]}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Step size: {WINDOW_STEP}\n")
        f.write(f"Features: {integrated_X.shape[2]}\n")
        f.write(f"Shape: X{integrated_X.shape}, y{integrated_y.shape}\n\n")
        f.write(f"Window distribution:\n")
        f.write(f"  Benign:    {ben_windows} ({ben_windows / len(all_metadata) * 100:.1f}%)\n")
        f.write(f"  Malicious: {mal_windows} ({mal_windows / len(all_metadata) * 100:.1f}%)\n\n")
        f.write(f"Per-file breakdown:\n")
        for i, fname in enumerate(FILE_ORDER):
            day = fname.replace('_TrafficForML_CICFlowMeter.csv', '')
            if day in file_window_indices:
                idx = file_window_indices[day]
                n = idx['end'] - idx['start']
                f.write(f"  {i + 1}. {day}: {n} windows\n")

        f.write(f"\nFlow-level distribution:\n")
        label_counts = {}
        for li in label_mapping:
            count = int(np.sum(integrated_y == int(li)))
            if count > 0:
                label_counts[label_mapping[li]] = count
        total_flows = sum(label_counts.values())
        for name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {name}: {count} ({count / total_flows * 100:.2f}%)\n")

    print(f"\n摘要: {summary_file}")

    # 6. 验证
    X_check = np.load(os.path.join(OUTPUT_DIR, f'integrated_X_w{ws}_s{ss}.npy'), mmap_mode='r')
    y_check = np.load(os.path.join(OUTPUT_DIR, f'integrated_y_w{ws}_s{ss}.npy'), mmap_mode='r')
    assert X_check.shape[0] == y_check.shape[0]
    assert X_check.shape[1] == WINDOW_SIZE
    assert X_check.shape[0] == len(all_metadata)
    print(f"验证通过: X{X_check.shape}, y{y_check.shape}")

    print("\n" + "=" * 60)
    print("完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  X: {integrated_X.shape}")
    print(f"  y: {integrated_y.shape}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  窗口分布: 正常 {ben_windows} / 异常 {mal_windows}")
    print("=" * 60)


if __name__ == "__main__":
    main()