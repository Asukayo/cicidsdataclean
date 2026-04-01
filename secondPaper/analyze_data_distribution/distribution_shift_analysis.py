"""
CICIDS2017 窗口级数据集 训练集/测试集 分布差异量化分析
==========================================================
功能：
  1. 逐特征统计量对比（均值、标准差、偏度、峰度）
  2. 逐特征 KS 检验（Kolmogorov-Smirnov Test）
  3. 全局 MMD（Maximum Mean Discrepancy）估计
  4. 窗口级标签分布对比
  5. 可视化输出（特征分布对比图 + 热力图 + 汇总报告）

用法：
  修改下方 CONFIG 区域的 DATA_DIR 路径，然后运行：
  python distribution_shift_analysis.py

依赖：
  pip install numpy scipy matplotlib seaborn scikit-learn
"""

import os
import pickle
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 确保英文正常显示
import seaborn as sns

# ============================================================
# CONFIG — 请根据你的实际路径修改
# ============================================================
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/selected_features"  # <-- 修改为你的 data_dir 路径
WINDOW_SIZE = 100
STEP_SIZE = 20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
OUTPUT_DIR = "./shift_analysis_results"
# ============================================================

# 38个特征名称（基于 Random Forest ranking 选取的 CICIDS2017 特征）
# 如果你有实际的特征名列表，可以替换这里
FEATURE_NAMES = [f"Feature_{i}" for i in range(38)]


def load_and_split(data_dir, window_size, step_size, train_ratio, val_ratio):
    """加载数据并按时间顺序划分"""
    X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')

    print(f"Loading X from: {X_file}")
    print(f"Loading y from: {y_file}")

    X = np.load(X_file)  # [N, W, F]
    y = np.load(y_file)  # [N, W]

    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val':   (X[train_end:val_end], y[train_end:val_end]),
        'test':  (X[val_end:], y[val_end:]),
    }

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Train: {splits['train'][0].shape[0]} windows")
    print(f"Val:   {splits['val'][0].shape[0]} windows")
    print(f"Test:  {splits['test'][0].shape[0]} windows")

    return splits


def flatten_windows(X):
    """将窗口数据展平为 [N*W, F]，用于逐特征统计分析"""
    return X.reshape(-1, X.shape[-1])


def get_window_labels(y):
    """窗口级标签：只要包含恶意流量即为1"""
    return (np.any(y > 0, axis=1)).astype(int)


# ============================================================
# 1. 逐特征统计量对比
# ============================================================
def compute_feature_statistics(X_train_flat, X_test_flat, feature_names):
    """计算每个特征的均值、标准差、偏度、峰度"""
    results = []
    for i in range(X_train_flat.shape[1]):
        tr = X_train_flat[:, i]
        te = X_test_flat[:, i]
        results.append({
            'feature': feature_names[i],
            'train_mean': np.mean(tr),
            'test_mean': np.mean(te),
            'mean_diff': abs(np.mean(tr) - np.mean(te)),
            'train_std': np.std(tr),
            'test_std': np.std(te),
            'std_ratio': np.std(te) / (np.std(tr) + 1e-10),
            'train_skew': stats.skew(tr),
            'test_skew': stats.skew(te),
            'train_kurtosis': stats.kurtosis(tr),
            'test_kurtosis': stats.kurtosis(te),
        })
    return results


# ============================================================
# 2. 逐特征 KS 检验
# ============================================================
def compute_ks_tests(X_train_flat, X_test_flat, feature_names):
    """对每个特征执行 KS 检验，返回统计量和 p-value"""
    ks_results = []
    for i in range(X_train_flat.shape[1]):
        stat, pval = stats.ks_2samp(X_train_flat[:, i], X_test_flat[:, i])
        ks_results.append({
            'feature': feature_names[i],
            'ks_statistic': stat,
            'p_value': pval,
            'significant': pval < 0.05,  # 5% 显著性水平
        })
    return ks_results


# ============================================================
# 3. MMD 估计
# ============================================================
def compute_mmd(X_source, X_target, gamma=None, max_samples=5000):
    """
    计算 Maximum Mean Discrepancy (MMD) with RBF kernel.
    为了计算效率，随机采样 max_samples 个样本。
    """
    n_s = min(len(X_source), max_samples)
    n_t = min(len(X_target), max_samples)

    idx_s = np.random.choice(len(X_source), n_s, replace=False)
    idx_t = np.random.choice(len(X_target), n_t, replace=False)

    Xs = X_source[idx_s]
    Xt = X_target[idx_t]

    if gamma is None:
        # 使用 median heuristic 自动选择 gamma
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(Xs[:1000], Xt[:1000])
        median_dist = np.median(dists)
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)

    K_ss = rbf_kernel(Xs, Xs, gamma=gamma)
    K_tt = rbf_kernel(Xt, Xt, gamma=gamma)
    K_st = rbf_kernel(Xs, Xt, gamma=gamma)

    mmd2 = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
    return max(mmd2, 0.0), np.sqrt(max(mmd2, 0.0))


# ============================================================
# 4. 标签分布对比
# ============================================================
def compare_label_distributions(y_train, y_test):
    """对比训练集和测试集的窗口级标签分布"""
    train_labels = get_window_labels(y_train)
    test_labels = get_window_labels(y_test)

    train_pos_ratio = np.mean(train_labels)
    test_pos_ratio = np.mean(test_labels)

    return {
        'train_total': len(train_labels),
        'train_positive': int(np.sum(train_labels)),
        'train_negative': int(np.sum(train_labels == 0)),
        'train_pos_ratio': train_pos_ratio,
        'test_total': len(test_labels),
        'test_positive': int(np.sum(test_labels)),
        'test_negative': int(np.sum(test_labels == 0)),
        'test_pos_ratio': test_pos_ratio,
        'ratio_diff': abs(train_pos_ratio - test_pos_ratio),
    }


# ============================================================
# 5. 逐特征 MMD（可选，更细粒度）
# ============================================================
def compute_per_feature_mmd(X_train_flat, X_test_flat, feature_names, max_samples=5000):
    """对每个特征单独计算 MMD"""
    results = []
    for i in range(X_train_flat.shape[1]):
        tr = X_train_flat[:, i].reshape(-1, 1)
        te = X_test_flat[:, i].reshape(-1, 1)
        mmd2, mmd = compute_mmd(tr, te, max_samples=max_samples)
        results.append({
            'feature': feature_names[i],
            'mmd_squared': mmd2,
            'mmd': mmd,
        })
    return results


# ============================================================
# 可视化
# ============================================================
def plot_ks_heatmap(ks_results, output_dir):
    """绘制 KS 统计量热力图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    features = [r['feature'] for r in ks_results]
    ks_stats = [r['ks_statistic'] for r in ks_results]
    significant = [r['significant'] for r in ks_results]

    colors = ['#e74c3c' if s else '#3498db' for s in significant]
    bars = ax.bar(range(len(features)), ks_stats, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_ylabel('KS Statistic', fontsize=12)
    ax.set_title('Per-Feature KS Test: Train vs Test Distribution\n(Red = significant at p<0.05)', fontsize=13)
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Reference line (0.05)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ks_test_barplot.png'), dpi=150)
    plt.close()
    print(f"  Saved: ks_test_barplot.png")


def plot_feature_distribution_comparison(X_train_flat, X_test_flat, feature_names, output_dir, top_k=6):
    """绘制 KS 统计量最大的 top_k 个特征的分布对比图"""
    ks_stats = []
    for i in range(X_train_flat.shape[1]):
        stat, _ = stats.ks_2samp(X_train_flat[:, i], X_test_flat[:, i])
        ks_stats.append((i, stat))
    ks_stats.sort(key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (feat_idx, ks_val) in enumerate(ks_stats[:top_k]):
        ax = axes[idx]
        ax.hist(X_train_flat[:, feat_idx], bins=80, density=True, alpha=0.6,
                label='Train', color='#3498db')
        ax.hist(X_test_flat[:, feat_idx], bins=80, density=True, alpha=0.6,
                label='Test', color='#e74c3c')
        ax.set_title(f'{feature_names[feat_idx]}\nKS={ks_val:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel('Density')

    plt.suptitle('Top-6 Features with Largest Distribution Shift (Train vs Test)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top_features_distribution.png")


def plot_mean_std_comparison(feat_stats, output_dir):
    """绘制每个特征的均值和标准差对比"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    features = [r['feature'] for r in feat_stats]
    x = range(len(features))

    # 均值对比
    train_means = [r['train_mean'] for r in feat_stats]
    test_means = [r['test_mean'] for r in feat_stats]
    ax1.bar([i - 0.2 for i in x], train_means, width=0.4, label='Train', color='#3498db', alpha=0.8)
    ax1.bar([i + 0.2 for i in x], test_means, width=0.4, label='Test', color='#e74c3c', alpha=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(features, rotation=90, fontsize=7)
    ax1.set_ylabel('Mean')
    ax1.set_title('Per-Feature Mean Comparison: Train vs Test')
    ax1.legend()

    # 标准差对比
    train_stds = [r['train_std'] for r in feat_stats]
    test_stds = [r['test_std'] for r in feat_stats]
    ax2.bar([i - 0.2 for i in x], train_stds, width=0.4, label='Train', color='#3498db', alpha=0.8)
    ax2.bar([i + 0.2 for i in x], test_stds, width=0.4, label='Test', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(features, rotation=90, fontsize=7)
    ax2.set_ylabel('Std Dev')
    ax2.set_title('Per-Feature Standard Deviation Comparison: Train vs Test')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_std_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: mean_std_comparison.png")


def plot_label_distribution(label_info, output_dir):
    """绘制标签分布对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 训练集
    ax1.pie([label_info['train_negative'], label_info['train_positive']],
            labels=['Normal', 'Anomalous'],
            autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'])
    ax1.set_title(f"Train Set (n={label_info['train_total']})")

    # 测试集
    ax2.pie([label_info['test_negative'], label_info['test_positive']],
            labels=['Normal', 'Anomalous'],
            autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'])
    ax2.set_title(f"Test Set (n={label_info['test_total']})")

    plt.suptitle('Window-Level Label Distribution: Train vs Test', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: label_distribution.png")


def plot_per_feature_mmd(mmd_results, output_dir):
    """绘制逐特征 MMD"""
    fig, ax = plt.subplots(figsize=(14, 5))
    features = [r['feature'] for r in mmd_results]
    mmds = [r['mmd'] for r in mmd_results]

    colors = ['#e74c3c' if m > np.mean(mmds) + np.std(mmds) else '#3498db' for m in mmds]
    ax.bar(range(len(features)), mmds, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=np.mean(mmds), color='orange', linestyle='--', alpha=0.7, label=f'Mean MMD={np.mean(mmds):.4f}')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_ylabel('MMD')
    ax.set_title('Per-Feature MMD: Train vs Test\n(Red = above mean+1std)', fontsize=13)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_feature_mmd.png'), dpi=150)
    plt.close()
    print(f"  Saved: per_feature_mmd.png")


# ============================================================
# 汇总报告
# ============================================================
def generate_report(feat_stats, ks_results, global_mmd, global_mmd_sq,
                    label_info, mmd_per_feat, output_dir):
    """生成文本汇总报告"""
    report_path = os.path.join(output_dir, 'distribution_shift_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CICIDS2017 Train vs Test Distribution Shift Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        # 全局 MMD
        f.write("1. GLOBAL MMD (RBF Kernel, Median Heuristic)\n")
        f.write("-" * 50 + "\n")
        f.write(f"   MMD^2 = {global_mmd_sq:.6f}\n")
        f.write(f"   MMD   = {global_mmd:.6f}\n")
        if global_mmd > 0.1:
            f.write("   >> SIGNIFICANT distribution shift detected.\n")
        elif global_mmd > 0.01:
            f.write("   >> MODERATE distribution shift detected.\n")
        else:
            f.write("   >> MINOR distribution shift (distributions are similar).\n")
        f.write("\n")

        # 标签分布
        f.write("2. LABEL DISTRIBUTION\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Train: {label_info['train_positive']}/{label_info['train_total']} "
                f"anomalous ({label_info['train_pos_ratio']:.2%})\n")
        f.write(f"   Test:  {label_info['test_positive']}/{label_info['test_total']} "
                f"anomalous ({label_info['test_pos_ratio']:.2%})\n")
        f.write(f"   Ratio difference: {label_info['ratio_diff']:.2%}\n")
        if label_info['ratio_diff'] > 0.1:
            f.write("   >> WARNING: Significant label distribution shift!\n")
        f.write("\n")

        # KS 检验汇总
        f.write("3. KS TEST SUMMARY\n")
        f.write("-" * 50 + "\n")
        n_sig = sum(1 for r in ks_results if r['significant'])
        n_total = len(ks_results)
        f.write(f"   Features with significant shift (p<0.05): {n_sig}/{n_total}\n")
        f.write(f"   Percentage: {n_sig / n_total:.1%}\n\n")

        # Top-10 漂移最大的特征
        f.write("4. TOP-10 FEATURES WITH LARGEST SHIFT (by KS statistic)\n")
        f.write("-" * 50 + "\n")
        sorted_ks = sorted(ks_results, key=lambda x: x['ks_statistic'], reverse=True)
        f.write(f"   {'Feature':<20} {'KS Stat':>10} {'p-value':>12} {'Significant':>12}\n")
        for r in sorted_ks[:10]:
            f.write(f"   {r['feature']:<20} {r['ks_statistic']:>10.4f} "
                    f"{r['p_value']:>12.2e} {'YES' if r['significant'] else 'no':>12}\n")
        f.write("\n")

        # Top-10 MMD 最大特征
        f.write("5. TOP-10 FEATURES WITH LARGEST PER-FEATURE MMD\n")
        f.write("-" * 50 + "\n")
        sorted_mmd = sorted(mmd_per_feat, key=lambda x: x['mmd'], reverse=True)
        f.write(f"   {'Feature':<20} {'MMD':>10} {'MMD^2':>12}\n")
        for r in sorted_mmd[:10]:
            f.write(f"   {r['feature']:<20} {r['mmd']:>10.4f} {r['mmd_squared']:>12.6f}\n")
        f.write("\n")

        # 结论
        f.write("6. CONCLUSION\n")
        f.write("-" * 50 + "\n")
        if n_sig > n_total * 0.5 and global_mmd > 0.05:
            f.write("   The training and test sets exhibit SIGNIFICANT distribution\n")
            f.write("   shift. This provides strong motivation for designing drift-\n")
            f.write("   robust anomaly detection methods. The natural temporal split\n")
            f.write("   of CICIDS2017 already introduces meaningful covariate shift.\n")
            f.write("   >> RECOMMENDATION: Natural drift is sufficient for your study.\n")
            f.write("      Additional synthetic drift can be used for stress testing.\n")
        elif n_sig > n_total * 0.3:
            f.write("   MODERATE distribution shift is present. Some features show\n")
            f.write("   significant differences while others remain stable.\n")
            f.write("   >> RECOMMENDATION: Combine natural drift analysis with\n")
            f.write("      controlled synthetic drift experiments.\n")
        else:
            f.write("   Distribution shift between train and test is MINOR.\n")
            f.write("   >> RECOMMENDATION: You will need to construct synthetic\n")
            f.write("      covariate shift to simulate real-world drift scenarios.\n")

    print(f"\n  Report saved to: {report_path}")
    return report_path


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CICIDS2017 Distribution Shift Analysis")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print("\n[Step 1] Loading and splitting data...")
    splits = load_and_split(DATA_DIR, WINDOW_SIZE, STEP_SIZE, TRAIN_RATIO, VAL_RATIO)

    X_train, y_train = splits['train']
    X_test, y_test = splits['test']

    # 展平窗口用于特征级分析
    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)
    print(f"Flattened: Train={X_train_flat.shape}, Test={X_test_flat.shape}")

    # 1. 逐特征统计量
    print("\n[Step 2] Computing per-feature statistics...")
    feat_stats = compute_feature_statistics(X_train_flat, X_test_flat, FEATURE_NAMES)

    # 2. KS 检验
    print("[Step 3] Running KS tests...")
    ks_results = compute_ks_tests(X_train_flat, X_test_flat, FEATURE_NAMES)
    n_sig = sum(1 for r in ks_results if r['significant'])
    print(f"  Significant features: {n_sig}/{len(ks_results)}")

    # 3. 全局 MMD
    print("[Step 4] Computing global MMD (this may take a minute)...")
    np.random.seed(42)
    global_mmd_sq, global_mmd = compute_mmd(X_train_flat, X_test_flat, max_samples=5000)
    print(f"  Global MMD^2 = {global_mmd_sq:.6f}")
    print(f"  Global MMD   = {global_mmd:.6f}")

    # 4. 标签分布
    print("[Step 5] Comparing label distributions...")
    label_info = compare_label_distributions(y_train, y_test)
    print(f"  Train anomaly ratio: {label_info['train_pos_ratio']:.2%}")
    print(f"  Test anomaly ratio:  {label_info['test_pos_ratio']:.2%}")

    # 5. 逐特征 MMD
    print("[Step 6] Computing per-feature MMD...")
    mmd_per_feat = compute_per_feature_mmd(X_train_flat, X_test_flat, FEATURE_NAMES)

    # 可视化
    print("\n[Step 7] Generating visualizations...")
    plot_ks_heatmap(ks_results, OUTPUT_DIR)
    plot_feature_distribution_comparison(X_train_flat, X_test_flat, FEATURE_NAMES, OUTPUT_DIR)
    plot_mean_std_comparison(feat_stats, OUTPUT_DIR)
    plot_label_distribution(label_info, OUTPUT_DIR)
    plot_per_feature_mmd(mmd_per_feat, OUTPUT_DIR)

    # 生成报告
    print("\n[Step 8] Generating summary report...")
    generate_report(feat_stats, ks_results, global_mmd, global_mmd_sq,
                    label_info, mmd_per_feat, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis complete! All results saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()