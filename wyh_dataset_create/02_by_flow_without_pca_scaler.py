import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
from config import CICIDS_WINDOW_SIZE,CICIDS_WINDOW_STEP

warnings.filterwarnings('ignore')


def get_common_features(data_files):
    """获取所有文件的共同特征"""
    common_features = None

    for file_path in data_files:
        try:
            if file_path.endswith('.feather'):
                df_sample = pd.read_feather(file_path).iloc[:100]
            elif file_path.endswith('.parquet'):
                df_sample = pd.read_parquet(file_path).iloc[:100]
            elif file_path.endswith('.csv'):
                df_sample = pd.read_csv(file_path, nrows=100)
            else:
                continue

            features = df_sample.select_dtypes(include=['number']).columns.tolist()
            features = [col for col in features if col not in ['Timestamp', 'Label', 'label']]

            if common_features is None:
                common_features = set(features)
            else:
                common_features = common_features.intersection(set(features))

            del df_sample
            gc.collect()

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    return list(common_features) if common_features else []


def process_single_file(file_path, output_dir, window_size=100, step_size=20, common_features=None):
    """处理单个文件生成滑动窗口"""
    file_name = os.path.basename(file_path)
    day_name = file_name.split('.')[0]
    day_output_dir = os.path.join(output_dir, day_name)
    os.makedirs(day_output_dir, exist_ok=True)

    print(f"Processing: {file_name}")

    # 加载数据
    if file_path.endswith('.feather'):
        df = pd.read_feather(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

    # 基本数据处理
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='S')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values('Timestamp')

    if 'label' in df.columns:
        df.rename(columns={'label': 'Label'}, inplace=True)

    df['Label'] = df['Label'].astype(str).replace({'BENIGN': 'Benign'})

    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Label'])
    df['label_encoded'] = label_encoder.transform(df['Label'])
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    # 处理特征
    feature_cols = common_features if common_features else [
        col for col in df.columns if col not in ['Timestamp', 'Label', 'label_encoded']
    ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)

    # 生成窗口
    total_records = len(df)
    num_windows = (total_records - window_size) // step_size + 1

    if num_windows <= 0:
        print(f"Warning: Not enough data for windows in {file_name}")
        return None

    X_windows = []
    y_windows = []
    window_metadata = []

    for i in tqdm(range(num_windows), desc="Creating windows"):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        window_data = df.iloc[start_idx:end_idx]
        has_malicious = (window_data['Label'] != 'Benign').any()

        X_windows.append(window_data[feature_cols].values)
        y_windows.append(window_data['label_encoded'].values)

        metadata = {
            'window_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': window_data['Timestamp'].iloc[0],
            'end_time': window_data['Timestamp'].iloc[-1],
            'is_malicious': 1 if has_malicious else 0,
            'original_labels': window_data['Label'].tolist()
        }

        if has_malicious:
            attack_counts = window_data[window_data['Label'] != 'Benign']['Label'].value_counts().to_dict()
            metadata['attack_types'] = attack_counts
            metadata['primary_attack'] = max(attack_counts, key=attack_counts.get)

        window_metadata.append(metadata)

    # 保存数据
    X_array = np.array(X_windows)
    y_array = np.array(y_windows)

    np.save(os.path.join(day_output_dir, f'X_windows_w{window_size}_s{step_size}.npy'), X_array)
    np.save(os.path.join(day_output_dir, f'y_windows_w{window_size}_s{step_size}.npy'), y_array)

    # 保存元数据
    metadata_dict = {
        'window_metadata': window_metadata,
        'feature_names': feature_cols,
        'label_mapping': label_mapping,
        'label_encoder': label_encoder,
        'config': {'window_size': window_size, 'step_size': step_size, 'file_name': file_name}
    }

    with open(os.path.join(day_output_dir, f'metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        pickle.dump(metadata_dict, f)

    malicious_count = sum(1 for w in window_metadata if w['is_malicious'] == 1)
    benign_count = len(window_metadata) - malicious_count

    print(f"Created {len(window_metadata)} windows: {benign_count} benign, {malicious_count} malicious")

    del df, X_array, y_array
    gc.collect()

    return metadata_dict


def process_all_files(data_dir, output_dir, window_size=100, step_size=20):
    """处理所有文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 查找数据文件
    data_files = []
    for ext in ['*.parquet', '*.feather', '*.csv']:
        data_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    print(f"Found {len(data_files)} files")
    data_files = sorted(data_files)

    # 获取共同特征
    print("Computing common features...")
    common_features = get_common_features(data_files)
    print(f"Found {len(common_features)} common features")

    # 保存特征信息
    with open(os.path.join(output_dir, 'features_info.pkl'), 'wb') as f:
        pickle.dump({
            'features': common_features,
            'config': {'window_size': window_size, 'step_size': step_size}
        }, f)

    # 处理每个文件
    results = {}
    for file_path in data_files:
        day_name = os.path.basename(file_path).split('.')[0]
        result = process_single_file(file_path, output_dir, window_size, step_size, common_features)
        if result:
            results[day_name] = result
        gc.collect()

    return results


if __name__ == "__main__":
    data_dir = "../cicids2017/clean"
    output_dir = "../cicids2017/flow_windows"

    results = process_all_files(
        data_dir=data_dir,
        output_dir=output_dir,
        window_size=CICIDS_WINDOW_SIZE,
        step_size=CICIDS_WINDOW_STEP
    )

    print("All files processed!")