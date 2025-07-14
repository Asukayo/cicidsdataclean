import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import gc
from config import CICIDS_WINDOW_SIZE,CICIDS_WINDOW_STEP

def integrate_all_windows(input_dir, output_dir, window_size=200, step_size=50):
    """整合所有日期的窗口数据"""
    os.makedirs(output_dir, exist_ok=True)

    # CICIDS2017文件顺序
    file_order = [
        "Monday-WorkingHours",
        "Tuesday-WorkingHours",
        "Wednesday-workingHours",
        "Thursday-WorkingHours-Morning-WebAttacks",
        "Thursday-WorkingHours-Afternoon-Infilteration",
        "Friday-WorkingHours-Morning",
        "Friday-WorkingHours-Afternoon-PortScan",
        "Friday-WorkingHours-Afternoon-DDos"
    ]

    all_X = []
    all_y = []
    all_metadata = []
    file_window_indices = {}
    current_window_idx = 0

    global_label_mapping = None
    global_feature_names = None

    print("Integrating data files...")

    for file_prefix in file_order:
        day_dir = os.path.join(input_dir, file_prefix)
        X_file = os.path.join(day_dir, f'X_windows_w{window_size}_s{step_size}.npy')
        y_file = os.path.join(day_dir, f'y_windows_w{window_size}_s{step_size}.npy')
        metadata_file = os.path.join(day_dir, f'metadata_w{window_size}_s{step_size}.pkl')

        if not all(os.path.exists(f) for f in [X_file, y_file, metadata_file]):
            print(f"Warning: Missing files for {file_prefix}, skipping")
            continue

        print(f"Processing: {file_prefix}")

        try:
            # 加载数据
            X = np.load(X_file)
            y = np.load(y_file)

            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 保存全局信息
            if global_label_mapping is None:
                global_label_mapping = metadata.get('label_mapping', {})
                global_feature_names = metadata.get('feature_names', [])

            # 记录窗口索引范围
            file_window_indices[file_prefix] = {
                'start': current_window_idx,
                'end': current_window_idx + len(X)
            }

            # 更新窗口元数据
            window_metadata = metadata.get('window_metadata', [])
            for i, window_meta in enumerate(window_metadata):
                window_meta['global_window_id'] = current_window_idx + i
                window_meta['source_file'] = file_prefix

            current_window_idx += len(X)

            all_X.append(X)
            all_y.append(y)
            all_metadata.extend(window_metadata)

            print(f"  Loaded {len(X)} windows, shape: {X.shape}")

            del X, y
            gc.collect()

        except Exception as e:
            print(f"Error processing {file_prefix}: {str(e)}")
            continue

    if not all_X:
        raise ValueError("No data was successfully loaded")

    # 合并数据
    print("Merging all data...")
    integrated_X = np.vstack(all_X)
    integrated_y = np.vstack(all_y)

    print(f"Final shapes: X{integrated_X.shape}, y{integrated_y.shape}")

    # 保存整合数据
    output_X_file = os.path.join(output_dir, f'integrated_X_w{window_size}_s{step_size}.npy')
    output_y_file = os.path.join(output_dir, f'integrated_y_w{window_size}_s{step_size}.npy')

    np.save(output_X_file, integrated_X)
    np.save(output_y_file, integrated_y)

    # 保存元数据
    integrated_metadata = {
        'window_metadata': all_metadata,
        'feature_names': global_feature_names,
        'label_mapping': global_label_mapping,
        'file_window_indices': file_window_indices,
        'file_order': file_order,
        'config': {
            'window_size': window_size,
            'step_size': step_size,
            'total_windows': len(integrated_X),
            'num_features': len(global_feature_names)
        }
    }

    metadata_output_file = os.path.join(output_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl')
    with open(metadata_output_file, 'wb') as f:
        pickle.dump(integrated_metadata, f)

    # 创建简单统计报告
    create_summary(integrated_X, integrated_y, integrated_metadata, output_dir, window_size, step_size)

    del all_X, all_y
    gc.collect()

    return {
        'X_shape': integrated_X.shape,
        'y_shape': integrated_y.shape,
        'metadata': integrated_metadata
    }


def create_summary(X, y, metadata, output_dir, window_size, step_size):
    """创建统计摘要"""
    summary_file = os.path.join(output_dir, f'integration_summary_w{window_size}_s{step_size}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CICIDS2017 Integration Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Total windows: {X.shape[0]}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Features: {X.shape[2]}\n")
        f.write(f"Final shape: X{X.shape}, y{y.shape}\n\n")

        # 窗口级别统计
        window_metadata = metadata['window_metadata']
        malicious_windows = sum(1 for w in window_metadata if w['is_malicious'] == 1)
        benign_windows = len(window_metadata) - malicious_windows

        f.write("Window-level distribution:\n")
        f.write(f"- Benign: {benign_windows} ({benign_windows / len(window_metadata) * 100:.1f}%)\n")
        f.write(f"- Malicious: {malicious_windows} ({malicious_windows / len(window_metadata) * 100:.1f}%)\n\n")

        # 流量级别统计
        label_mapping = metadata['label_mapping']
        f.write("Flow-level distribution:\n")

        label_counts = {}
        for label_idx in label_mapping.keys():
            count = np.sum(y == int(label_idx))
            label_name = label_mapping[label_idx]
            if count > 0:
                label_counts[label_name] = count

        total_flows = sum(label_counts.values())
        for label_name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {label_name}: {count} ({count / total_flows * 100:.2f}%)\n")

        # 文件分布
        f.write(f"\nFile distribution:\n")
        file_indices = metadata['file_window_indices']
        for i, file_name in enumerate(metadata['file_order']):
            if file_name in file_indices:
                indices = file_indices[file_name]
                num_windows = indices['end'] - indices['start']
                f.write(f"{i + 1}. {file_name}: {num_windows} windows\n")

    print(f"Summary saved to: {summary_file}")


def verify_data(integrated_dir, window_size=200, step_size=50):
    """验证整合数据"""
    X_file = os.path.join(integrated_dir, f'integrated_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(integrated_dir, f'integrated_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(integrated_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl')

    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    assert X.shape[0] == y.shape[0], "X and y shape mismatch"
    assert X.shape[0] == len(metadata['window_metadata']), "Window count mismatch"
    assert X.shape[1] == window_size, f"Window size mismatch: {X.shape[1]} vs {window_size}"

    print(f"Verification passed: X{X.shape}, y{y.shape}")
    return True


if __name__ == "__main__":
    input_dir = "../cicids2017/flow_windows"
    output_dir = "../cicids2017/integrated_windows"

    result = integrate_all_windows(
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=CICIDS_WINDOW_SIZE,
        step_size=CICIDS_WINDOW_STEP
    )

    if result:
        print(f"Integration completed: X{result['X_shape']}, y{result['y_shape']}")
        verify_data(output_dir, window_size=CICIDS_WINDOW_SIZE, step_size=CICIDS_WINDOW_STEP)
    else:
        print("Integration failed!")