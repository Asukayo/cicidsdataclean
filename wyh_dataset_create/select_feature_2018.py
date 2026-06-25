"""
CICIDS2018 特征筛选脚本
========================
使用 Random Forest 特征重要性排序，选取 Top-K 特征
与 CICIDS2017 的特征筛选流程保持一致

输入：integrated_windows 目录下的完整数据（68特征）
输出：selected_features 目录下的筛选后数据（38特征）

用法：python select_features_cicids2018.py
"""

import numpy as np
import os
import pickle
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
INPUT_DIR = "/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"
OUTPUT_DIR = "/home/ubuntu/wyh/cicdis/cicids2018/selected_features"

WINDOW_SIZE = 100
WINDOW_STEP = 20
TOP_K = 38  # 与 CICIDS2017 保持一致

# RF 配置
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_RANDOM_STATE = 42
# 用于特征排序的采样量（全量60万窗口×100流=6000万太大，采样加速）
SAMPLE_SIZE = 500_000  # 采样50万条流用于RF训练


def load_data(input_dir, window_size, step_size):
    """加载整合数据"""
    X = np.load(os.path.join(input_dir, f'integrated_X_w{window_size}_s{step_size}.npy'))
    y = np.load(os.path.join(input_dir, f'integrated_y_w{window_size}_s{step_size}.npy'))

    with open(os.path.join(input_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    return X, y, metadata


def rf_feature_ranking(X, y, feature_names, n_estimators=200, max_depth=15,
                       sample_size=500_000, random_state=42):
    """
    用 Random Forest 对特征进行重要性排序

    为了节省内存，将窗口数据展平为 flow-level 后采样
    """
    print("  展平窗口数据为 flow-level...")
    X_flat = X.reshape(-1, X.shape[-1])  # [num_windows * window_size, num_features]
    y_flat = y.reshape(-1)               # [num_windows * window_size]

    # 二值化标签：0=正常，1=攻击
    y_binary = (y_flat > 0).astype(int)

    total_flows = len(X_flat)
    print(f"  总流量数: {total_flows:,}")
    print(f"  正常: {(y_binary == 0).sum():,}, 攻击: {(y_binary == 1).sum():,}")

    # 采样（全量太大，RF 不需要那么多数据）
    if total_flows > sample_size:
        print(f"  采样 {sample_size:,} 条用于 RF 训练...")
        # 分层采样保持类别比例
        np.random.seed(random_state)
        idx_normal = np.where(y_binary == 0)[0]
        idx_attack = np.where(y_binary == 1)[0]

        attack_ratio = len(idx_attack) / total_flows
        n_attack = int(sample_size * attack_ratio)
        n_normal = sample_size - n_attack

        sampled_normal = np.random.choice(idx_normal, size=min(n_normal, len(idx_normal)), replace=False)
        sampled_attack = np.random.choice(idx_attack, size=min(n_attack, len(idx_attack)), replace=False)

        sampled_idx = np.concatenate([sampled_normal, sampled_attack])
        np.random.shuffle(sampled_idx)

        X_sample = X_flat[sampled_idx]
        y_sample = y_binary[sampled_idx]
    else:
        X_sample = X_flat
        y_sample = y_binary

    del X_flat, y_flat, y_binary
    gc.collect()

    print(f"  训练样本: {len(X_sample):,} "
          f"(正常: {(y_sample == 0).sum():,}, 攻击: {(y_sample == 1).sum():,})")

    # 训练 Random Forest
    print(f"  训练 RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',
    )
    rf.fit(X_sample, y_sample)

    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 打印排序结果
    print(f"\n  特征重要性排序 (共 {len(feature_names)} 个):")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-'*58}")
    for rank, idx in enumerate(indices):
        marker = " ✓" if rank < TOP_K else ""
        print(f"  {rank+1:<6} {feature_names[idx]:<40} {importances[idx]:>12.6f}{marker}")

    del X_sample, y_sample, rf
    gc.collect()

    return importances, indices


def select_and_save(X, y, metadata, importances, indices, feature_names,
                    top_k, output_dir, window_size, step_size):
    """选取 Top-K 特征并保存"""
    os.makedirs(output_dir, exist_ok=True)

    # 选取 Top-K 特征索引
    selected_indices = indices[:top_k]
    selected_indices_sorted = np.sort(selected_indices)  # 保持原始列顺序
    selected_features = [feature_names[i] for i in selected_indices_sorted]
    selected_importances = importances[selected_indices_sorted]

    print(f"\n选取的 Top-{top_k} 特征:")
    for i, (fname, imp) in enumerate(zip(selected_features, selected_importances)):
        print(f"  {i+1:>3}. {fname:<40} {imp:.6f}")

    # 筛选特征
    print(f"\n筛选数据: {X.shape} → ", end="")
    X_selected = X[:, :, selected_indices_sorted]
    print(f"{X_selected.shape}")

    # 保存数据
    np.save(os.path.join(output_dir, f'integrated_X_w{window_size}_s{step_size}.npy'), X_selected)
    np.save(os.path.join(output_dir, f'integrated_y_w{window_size}_s{step_size}.npy'), y)

    # 更新 metadata
    selected_metadata = metadata.copy()
    selected_metadata['feature_names'] = selected_features
    selected_metadata['config'] = metadata['config'].copy()
    selected_metadata['config']['num_features'] = top_k
    selected_metadata['feature_selection'] = {
        'method': 'RandomForest',
        'n_estimators': RF_N_ESTIMATORS,
        'max_depth': RF_MAX_DEPTH,
        'original_features': len(feature_names),
        'selected_features': top_k,
        'selected_indices': selected_indices_sorted.tolist(),
        'importances': {feature_names[i]: float(importances[i]) for i in range(len(feature_names))},
    }

    with open(os.path.join(output_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl'), 'wb') as f:
        pickle.dump(selected_metadata, f)

    # 保存特征列表（文本文件，方便查看）
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        f.write(f"CICIDS2018 Top-{top_k} Features (Random Forest Ranking)\n")
        f.write(f"{'='*60}\n\n")
        for i, (fname, imp) in enumerate(zip(selected_features, selected_importances)):
            f.write(f"{i+1:>3}. {fname:<40} {imp:.6f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Total features: {len(feature_names)} → {top_k}\n")

    # 保存 features_info（兼容 provider）
    with open(os.path.join(output_dir, 'features_info.pkl'), 'wb') as f:
        pickle.dump({
            'features': selected_features,
            'config': {'window_size': window_size, 'step_size': step_size}
        }, f)

    print(f"\n保存完成: {output_dir}")
    print(f"  integrated_X: {X_selected.shape}")
    print(f"  integrated_y: {y.shape}")
    print(f"  特征数: {top_k}")

    return X_selected


def main():
    print("=" * 60)
    print("CICIDS2018 特征筛选 (Random Forest)")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    X, y, metadata = load_data(INPUT_DIR, WINDOW_SIZE, WINDOW_STEP)
    feature_names = metadata['feature_names']
    print(f"   X={X.shape}, y={y.shape}")
    print(f"   原始特征数: {len(feature_names)}")

    # 2. RF 特征排序
    print(f"\n2. Random Forest 特征排序...")
    importances, indices = rf_feature_ranking(
        X, y, feature_names,
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        sample_size=SAMPLE_SIZE,
        random_state=RF_RANDOM_STATE,
    )

    # 3. 选取 Top-K 并保存
    print(f"\n3. 选取 Top-{TOP_K} 特征并保存...")
    X_selected = select_and_save(
        X, y, metadata, importances, indices, feature_names,
        top_k=TOP_K,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        step_size=WINDOW_STEP,
    )

    # 4. 验证
    print(f"\n4. 验证...")
    X_check = np.load(
        os.path.join(OUTPUT_DIR, f'integrated_X_w{WINDOW_SIZE}_s{WINDOW_STEP}.npy'),
        mmap_mode='r'
    )
    assert X_check.shape[0] == X.shape[0], "窗口数不匹配"
    assert X_check.shape[1] == WINDOW_SIZE, "窗口大小不匹配"
    assert X_check.shape[2] == TOP_K, f"特征数不匹配: {X_check.shape[2]} vs {TOP_K}"
    print(f"   验证通过: {X_check.shape}")

    print("\n" + "=" * 60)
    print("完成！")
    print(f"  输入: {INPUT_DIR} ({len(feature_names)} 特征)")
    print(f"  输出: {OUTPUT_DIR} ({TOP_K} 特征)")
    print(f"  数据形状: {X_selected.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()