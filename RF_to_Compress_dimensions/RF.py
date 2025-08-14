import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd


def load_features_info(features_info_path):
    """加载特征信息"""
    with open(features_info_path, 'rb') as f:
        features_info = pickle.load(f)
    return features_info['features']


def prepare_data_for_rf(X_path, y_path, metadata_path):
    """
    为随机森林准备数据
    将窗口数据展平或聚合为每个窗口一个样本
    """
    # 加载数据
    X = np.load(X_path)  # shape: (num_windows, window_size, num_features)
    y = np.load(y_path)  # shape: (num_windows, window_size)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # 获取窗口级别的标签（恶意/正常）
    window_labels = np.array([w['is_malicious'] for w in metadata['window_metadata']])

    # 方法1: 使用窗口内的统计特征（均值、标准差、最大值、最小值等）
    print("准备随机森林数据...")
    print(f"原始X形状: {X.shape}")

    # 计算每个窗口的统计特征
    X_mean = np.mean(X, axis=1)  # (num_windows, num_features)
    X_std = np.std(X, axis=1)
    X_max = np.max(X, axis=1)
    X_min = np.min(X, axis=1)

    # 组合统计特征
    X_rf = np.hstack([X_mean, X_std, X_max, X_min])  # (num_windows, num_features * 4)

    print(f"随机森林输入形状: {X_rf.shape}")
    print(f"标签形状: {window_labels.shape}")

    return X_rf, window_labels, X_mean


def random_forest_feature_selection(X, y, feature_names, top_k=50):
    """
    使用随机森林进行特征选择

    参数:
    - X: 特征矩阵 (num_samples, num_features)
    - y: 标签 (num_samples,)
    - feature_names: 特征名称列表
    - top_k: 选择前k个重要特征
    """
    print("\n训练随机森林模型...")

    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # 处理类别不平衡
    )

    rf.fit(X, y)

    # 获取特征重要性
    feature_importances = rf.feature_importances_

    # 获取特征重要性排序
    indices = np.argsort(feature_importances)[::-1]

    print(f"\nTop {top_k} 最重要的特征:")
    print("-" * 50)

    selected_features = []
    selected_indices = []

    for i in range(min(top_k, len(feature_names))):
        idx = indices[i]
        selected_features.append(feature_names[idx])
        selected_indices.append(idx)
        print(f"{i + 1}. {feature_names[idx]}: {feature_importances[idx]:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))

    # 只显示前30个特征
    n_display = min(30, len(feature_names))
    plt.barh(range(n_display), feature_importances[indices[:n_display]])
    plt.yticks(range(n_display), [feature_names[i] for i in indices[:n_display]])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_display} Feature Importances from Random Forest')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 使用SelectFromModel进行特征选择
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_selected = selector.transform(X)

    print(f"\n使用中位数阈值选择了 {X_selected.shape[1]} 个特征")

    return {
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'feature_importances': feature_importances,
        'selector': selector,
        'rf_model': rf
    }


def save_selected_features(selection_result, output_path):
    """保存特征选择结果"""
    save_data = {
        'selected_features': selection_result['selected_features'],
        'selected_indices': selection_result['selected_indices'],
        'feature_importances': selection_result['feature_importances'],
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    # 同时保存为文本文件便于查看
    txt_path = output_path.replace('.pkl', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("随机森林特征选择结果\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"选择的特征数量: {len(selection_result['selected_features'])}\n\n")

        f.write("特征重要性排名:\n")
        for i, (feat, idx) in enumerate(zip(selection_result['selected_features'],
                                            selection_result['selected_indices'])):
            importance = selection_result['feature_importances'][idx]
            f.write(f"{i + 1}. {feat} (索引: {idx}): {importance:.6f}\n")


def apply_feature_selection_to_windows(X_windows_path, selected_indices, output_path):
    """
    将特征选择应用到窗口数据

    参数:
    - X_windows_path: 原始窗口数据路径
    - selected_indices: 选中的特征索引
    - output_path: 输出路径
    """
    # 加载原始数据
    X = np.load(X_windows_path)  # (num_windows, window_size, num_features)

    # 只选择重要的特征
    X_selected = X[:, :, selected_indices]  # (num_windows, window_size, num_selected_features)

    print(f"\n应用特征选择:")
    print(f"原始形状: {X.shape}")
    print(f"选择后形状: {X_selected.shape}")
    print(f"特征减少: {X.shape[2]} -> {X_selected.shape[2]} ({X_selected.shape[2] / X.shape[2] * 100:.1f}%)")

    # 保存选择后的数据
    np.save(output_path, X_selected)

    return X_selected


# 使用示例
if __name__ == "__main__":
    # 路径配置
    integrated_dir = "../cicids2017/integrated_windows"
    features_info_path = "../cicids2017/flow_windows/features_info.pkl"

    # 文件路径
    X_path = f"{integrated_dir}/integrated_X_w200_s50.npy"
    y_path = f"{integrated_dir}/integrated_y_w200_s50.npy"
    metadata_path = f"{integrated_dir}/integrated_metadata_w200_s50.pkl"

    # 1. 加载特征名称
    feature_names = load_features_info(features_info_path)
    print(f"总特征数: {len(feature_names)}")

    # 2. 准备数据
    X_rf, y_rf, X_mean = prepare_data_for_rf(X_path, y_path, metadata_path)

    # 3. 随机森林特征选择（使用均值特征）
    selection_result = random_forest_feature_selection(
        X_mean,  # 使用窗口均值特征
        y_rf,  # 窗口级别标签
        feature_names,
        top_k=50
    )

    # 4. 保存选择结果
    save_selected_features(selection_result, f"{integrated_dir}/rf_selected_features.pkl")

    # 5. 应用特征选择到原始窗口数据
    X_selected = apply_feature_selection_to_windows(
        X_path,
        selection_result['selected_indices'],
        f"{integrated_dir}/integrated_X_selected_w200_s50.npy"
    )

    print("\n特征选择完成!")