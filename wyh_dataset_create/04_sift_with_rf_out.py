import numpy as np
import pickle
import os
from config import CICIDS_WINDOW_SIZE,CICIDS_WINDOW_STEP

def load_important_features(features_file):
    """加载重要特征列表"""
    with open(features_file, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(features)} important features")
    return features


def select_features(input_dir, output_dir, features_file, window_size=500, step_size=50):
    """基于重要特征列表进行特征选择"""
    os.makedirs(output_dir, exist_ok=True)

    # 文件路径
    X_file = os.path.join(input_dir, f'integrated_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(input_dir, f'integrated_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(input_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl')

    # 检查文件存在性
    if not all(os.path.exists(f) for f in [X_file, y_file, metadata_file]):
        raise FileNotFoundError("Integrated data files are incomplete")

    # 加载重要特征和元数据
    important_features = load_important_features(features_file)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    original_features = metadata['feature_names']
    print(f"Original features: {len(original_features)}")

    # 找出特征索引
    feature_indices = []
    found_features = []
    missing_features = []

    for feature in important_features:
        if feature in original_features:
            idx = original_features.index(feature)
            feature_indices.append(idx)
            found_features.append(feature)
        else:
            missing_features.append(feature)

    print(f"Matched features: {len(found_features)}")
    print(f"Missing features: {len(missing_features)}")

    if not feature_indices:
        raise ValueError("No matching important features found")

    # 加载和选择特征
    print("Loading and selecting features...")
    X = np.load(X_file, mmap_mode='r')
    X_selected = X[:, :, feature_indices]
    y = np.load(y_file)

    print(f"Feature selection: {X.shape} -> {X_selected.shape}")

    # 更新元数据
    updated_metadata = metadata.copy()
    updated_metadata['feature_names'] = found_features
    updated_metadata['original_feature_names'] = original_features
    updated_metadata['feature_indices'] = feature_indices
    updated_metadata['feature_selection_info'] = {
        'original_feature_count': len(original_features),
        'selected_feature_count': len(found_features),
        'found_features': found_features,
        'missing_features': missing_features,
        'selection_ratio': len(found_features) / len(original_features)
    }
    updated_metadata['config']['num_features'] = len(found_features)
    updated_metadata['config']['feature_selection_applied'] = True

    # 保存选择后的数据
    output_X_file = os.path.join(output_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    output_y_file = os.path.join(output_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    output_metadata_file = os.path.join(output_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    print("Saving selected data...")
    np.save(output_X_file, X_selected)
    np.save(output_y_file, y)

    with open(output_metadata_file, 'wb') as f:
        pickle.dump(updated_metadata, f)

    # 创建报告
    create_report(updated_metadata, output_dir, window_size, step_size)

    print(f"Feature selection completed!")
    print(f"Features: {len(original_features)} -> {len(found_features)}")
    print(f"Compression ratio: {len(found_features) / len(original_features):.3f}")
    print(f"Data saved to: {output_dir}")

    return {
        'original_shape': X.shape,
        'selected_shape': X_selected.shape,
        'selected_features': found_features,
        'missing_features': missing_features
    }


def create_report(metadata, output_dir, window_size, step_size):
    """创建特征选择报告"""
    report_file = os.path.join(output_dir, f'feature_selection_report_w{window_size}_s{step_size}.txt')

    selection_info = metadata['feature_selection_info']

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("CICIDS2017 Feature Selection Report\n")
        f.write("=" * 40 + "\n\n")

        f.write("Selection Summary:\n")
        f.write(f"- Original features: {selection_info['original_feature_count']}\n")
        f.write(f"- Selected features: {selection_info['selected_feature_count']}\n")
        f.write(f"- Retention ratio: {selection_info['selection_ratio']:.3f}\n")
        f.write(f"- Missing features: {len(selection_info['missing_features'])}\n\n")

        f.write("Selected features (by importance):\n")
        f.write("-" * 30 + "\n")
        for i, feature in enumerate(selection_info['found_features'], 1):
            f.write(f"{i:2d}. {feature}\n")

        if selection_info['missing_features']:
            f.write(f"\nMissing features ({len(selection_info['missing_features'])}):\n")
            f.write("-" * 30 + "\n")
            for feature in selection_info['missing_features']:
                f.write(f"- {feature}\n")

        f.write(f"\nDataset info:\n")
        f.write(f"- Total windows: {metadata['config']['total_windows']}\n")
        f.write(f"- Window size: {metadata['config']['window_size']}\n")
        f.write(f"- Final shape: ({metadata['config']['total_windows']}, "
                f"{metadata['config']['window_size']}, {metadata['config']['num_features']})\n")

    print(f"Report saved to: {report_file}")


def verify_selected_data(selected_dir, window_size=500, step_size=50):
    """验证特征选择后的数据"""
    X_file = os.path.join(selected_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(selected_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(selected_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    assert X.shape[0] == y.shape[0], "X and y window count mismatch"
    assert X.shape[2] == len(metadata['feature_names']), "Feature count mismatch"
    assert X.shape[1] == window_size, "Window size mismatch"

    print(f"Verification passed: X{X.shape}, y{y.shape}")
    print(f"Features: {len(metadata['feature_names'])}")

    print(f"\nTop 10 selected features:")
    for i, feature in enumerate(metadata['feature_names'][:10]):
        print(f"  {i + 1}. {feature}")

    return True


if __name__ == "__main__":
    input_dir = "../cicids2017/integrated_windows"
    output_dir = "../cicids2017/selected_features"
    features_file = "../resources/important_features_90.txt"

    result = select_features(
        input_dir=input_dir,
        output_dir=output_dir,
        features_file=features_file,
        window_size=CICIDS_WINDOW_SIZE,
        step_size=CICIDS_WINDOW_STEP
    )

    if result:
        print(f"\nFeature selection results:")
        print(f"Original shape: {result['original_shape']}")
        print(f"Selected shape: {result['selected_shape']}")
        print(f"Selected features: {len(result['selected_features'])}")

        if result['missing_features']:
            print(f"Missing features: {len(result['missing_features'])}")

        verify_selected_data(output_dir, window_size=CICIDS_WINDOW_SIZE, step_size=CICIDS_WINDOW_STEP)
    else:
        print("Feature selection failed!")