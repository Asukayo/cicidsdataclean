"""
无监督范式下的分布漂移分析
==========================
对比组：
  A) 纯正常训练集 vs 测试集全部样本  → 模型实际面临的分布差距
  B) 纯正常训练集 vs 测试集正常样本  → 正常流量本身的协变量偏移
  C) 对比 A-B 的差异                → 分离漂移来源

修改 DATA_DIR 后运行即可。
"""

import os
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"  # <-- 修改为你的实际路径
WINDOW_SIZE = 100
STEP_SIZE = 20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
OUTPUT_DIR = "./unsupervised_shift_analysis"
RANDOM_SEED = 42
# ============================================================

FEATURE_NAMES = [f"Feature_{i}" for i in range(68)]


def load_and_prepare():
    """加载数据并按无监督范式准备"""
    X = np.load(os.path.join(DATA_DIR, f'integrated_X_w{WINDOW_SIZE}_s{STEP_SIZE}.npy'))
    y = np.load(os.path.join(DATA_DIR, f'integrated_y_w{WINDOW_SIZE}_s{STEP_SIZE}.npy'))

    total = len(X)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 窗口级标签
    train_labels = np.any(y_train > 0, axis=1).astype(int)
    test_labels = np.any(y_test > 0, axis=1).astype(int)

    # 过滤训练集：只保留正常窗口
    normal_mask = train_labels == 0
    X_train_normal = X_train[normal_mask]

    # 测试集：分出正常和异常
    test_normal_mask = test_labels == 0
    X_test_all = X_test
    X_test_normal = X_test[test_normal_mask]
    X_test_anomalous = X_test[~test_normal_mask]

    print(f"Train (normal only): {X_train_normal.shape[0]} windows")
    print(f"Test (all):          {X_test_all.shape[0]} windows")
    print(f"Test (normal only):  {X_test_normal.shape[0]} windows")
    print(f"Test (anomalous):    {X_test_anomalous.shape[0]} windows")

    return X_train_normal, X_test_all, X_test_normal, X_test_anomalous


def flatten(X):
    return X.reshape(-1, X.shape[-1])


def compute_mmd(X_s, X_t, gamma=None, max_samples=5000):
    """MMD with RBF kernel + median heuristic"""
    np.random.seed(RANDOM_SEED)
    n_s = min(len(X_s), max_samples)
    n_t = min(len(X_t), max_samples)

    Xs = X_s[np.random.choice(len(X_s), n_s, replace=False)]
    Xt = X_t[np.random.choice(len(X_t), n_t, replace=False)]

    if gamma is None:
        dists = euclidean_distances(Xs[:1000], Xt[:1000])
        median_dist = np.median(dists)
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)

    K_ss = rbf_kernel(Xs, Xs, gamma=gamma)
    K_tt = rbf_kernel(Xt, Xt, gamma=gamma)
    K_st = rbf_kernel(Xs, Xt, gamma=gamma)

    mmd2 = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
    return max(mmd2, 0.0), np.sqrt(max(mmd2, 0.0))


def compute_ks_tests(X_s_flat, X_t_flat):
    """逐特征 KS 检验"""
    results = []
    for i in range(X_s_flat.shape[1]):
        stat, pval = stats.ks_2samp(X_s_flat[:, i], X_t_flat[:, i])
        results.append({
            'feature': FEATURE_NAMES[i],
            'ks_statistic': stat,
            'p_value': pval,
            'significant': pval < 0.05,
        })
    return results


def compute_per_feature_mmd(X_s_flat, X_t_flat, max_samples=5000):
    """逐特征 MMD"""
    results = []
    for i in range(X_s_flat.shape[1]):
        s = X_s_flat[:, i].reshape(-1, 1)
        t = X_t_flat[:, i].reshape(-1, 1)
        mmd2, mmd = compute_mmd(s, t, max_samples=max_samples)
        results.append({
            'feature': FEATURE_NAMES[i],
            'mmd': mmd,
            'mmd_squared': mmd2,
        })
    return results


def plot_comparison_bar(ks_A, ks_B, output_dir):
    """对比两组 KS 统计量"""
    fig, ax = plt.subplots(figsize=(15, 6))

    features = [r['feature'] for r in ks_A]
    ks_A_vals = [r['ks_statistic'] for r in ks_A]
    ks_B_vals = [r['ks_statistic'] for r in ks_B]

    x = np.arange(len(features))
    w = 0.35

    ax.bar(x - w/2, ks_A_vals, w, label='Normal Train vs All Test', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, ks_B_vals, w, label='Normal Train vs Normal Test', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Per-Feature KS Test Comparison\n(Red = total gap model faces, Blue = pure covariate shift)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ks_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: ks_comparison.png")


def plot_mmd_comparison(mmd_A, mmd_B, output_dir):
    """对比两组逐特征 MMD"""
    fig, ax = plt.subplots(figsize=(15, 6))

    features = [r['feature'] for r in mmd_A]
    mmd_A_vals = [r['mmd'] for r in mmd_A]
    mmd_B_vals = [r['mmd'] for r in mmd_B]

    x = np.arange(len(features))
    w = 0.35

    ax.bar(x - w/2, mmd_A_vals, w, label='Normal Train vs All Test', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, mmd_B_vals, w, label='Normal Train vs Normal Test', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_ylabel('MMD')
    ax.set_title('Per-Feature MMD Comparison\n(Gap between red and blue = contribution of anomalous samples)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mmd_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: mmd_comparison.png")


def plot_top_features_distribution(X_train_flat, X_test_all_flat, X_test_normal_flat, ks_A, output_dir, top_k=6):
    """漂移最大的特征的三组分布对比"""
    sorted_ks = sorted(enumerate(ks_A), key=lambda x: x[1]['ks_statistic'], reverse=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (feat_idx, ks_info) in enumerate(sorted_ks[:top_k]):
        ax = axes[idx]
        ax.hist(X_train_flat[:, feat_idx], bins=80, density=True, alpha=0.5,
                label='Train (normal)', color='#2ecc71')
        ax.hist(X_test_normal_flat[:, feat_idx], bins=80, density=True, alpha=0.5,
                label='Test (normal)', color='#3498db')
        ax.hist(X_test_all_flat[:, feat_idx], bins=80, density=True, alpha=0.3,
                label='Test (all)', color='#e74c3c')
        ax.set_title(f'{FEATURE_NAMES[feat_idx]}\nKS={ks_info["ks_statistic"]:.4f}', fontsize=10)
        ax.legend(fontsize=7)

    plt.suptitle('Top-6 Drifted Features: Train(normal) vs Test(normal) vs Test(all)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_3way.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top_features_3way.png")


def generate_report(global_A, global_B, ks_A, ks_B, mmd_feat_A, mmd_feat_B, output_dir):
    """生成汇总报告"""
    path = os.path.join(output_dir, 'unsupervised_shift_report.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Unsupervised Setting: Distribution Shift Analysis Report\n")
        f.write("(Train = normal windows only)\n")
        f.write("=" * 70 + "\n\n")

        # ── Group A: Normal Train vs All Test ──
        f.write("1. GROUP A: Normal Train vs All Test (actual gap model faces)\n")
        f.write("-" * 55 + "\n")
        f.write(f"   Global MMD^2 = {global_A[0]:.6f}\n")
        f.write(f"   Global MMD   = {global_A[1]:.6f}\n")
        n_sig_A = sum(1 for r in ks_A if r['significant'])
        f.write(f"   KS significant features: {n_sig_A}/{len(ks_A)}\n\n")

        # ── Group B: Normal Train vs Normal Test ──
        f.write("2. GROUP B: Normal Train vs Normal Test (pure covariate shift)\n")
        f.write("-" * 55 + "\n")
        f.write(f"   Global MMD^2 = {global_B[0]:.6f}\n")
        f.write(f"   Global MMD   = {global_B[1]:.6f}\n")
        n_sig_B = sum(1 for r in ks_B if r['significant'])
        f.write(f"   KS significant features: {n_sig_B}/{len(ks_B)}\n\n")

        # ── 对比分析 ──
        f.write("3. DRIFT SOURCE DECOMPOSITION\n")
        f.write("-" * 55 + "\n")
        f.write(f"   Total gap (A):         MMD = {global_A[1]:.6f}\n")
        f.write(f"   Covariate shift (B):   MMD = {global_B[1]:.6f}\n")
        diff = global_A[1] - global_B[1]
        f.write(f"   Difference (A - B):    MMD = {diff:.6f}\n\n")

        if global_B[1] > 0.05:
            f.write("   >> Normal traffic ITSELF drifts significantly over time.\n")
            f.write("      This is genuine covariate shift that affects reconstruction.\n")
        else:
            f.write("   >> Normal traffic is relatively stable over time.\n")
            f.write("      The distribution gap is mainly caused by anomalous samples.\n")

        if diff > 0.03:
            f.write("   >> Anomalous samples contribute substantially to the total gap.\n")
        else:
            f.write("   >> Anomalous samples contribute minimally to the total gap.\n")

        f.write("\n")

        # ── Top-10 KS 两组对比 ──
        f.write("4. TOP-10 FEATURES: KS STATISTIC COMPARISON\n")
        f.write("-" * 55 + "\n")
        f.write(f"   {'Feature':<15} {'KS(A)':>8} {'KS(B)':>8} {'Diff':>8}  Source\n")

        # 按 Group A 排序
        paired = list(zip(ks_A, ks_B))
        paired.sort(key=lambda x: x[0]['ks_statistic'], reverse=True)

        for ra, rb in paired[:10]:
            diff_ks = ra['ks_statistic'] - rb['ks_statistic']
            # 判断漂移来源
            if rb['ks_statistic'] > 0.1 and diff_ks < 0.05:
                source = "Covariate"
            elif diff_ks > 0.1:
                source = "Anomaly"
            else:
                source = "Mixed"
            f.write(f"   {ra['feature']:<15} {ra['ks_statistic']:>8.4f} "
                    f"{rb['ks_statistic']:>8.4f} {diff_ks:>8.4f}  {source}\n")

        f.write("\n")

        # ── Top-10 per-feature MMD 两组对比 ──
        f.write("5. TOP-10 FEATURES: PER-FEATURE MMD COMPARISON\n")
        f.write("-" * 55 + "\n")
        f.write(f"   {'Feature':<15} {'MMD(A)':>8} {'MMD(B)':>8} {'Diff':>8}\n")

        paired_mmd = list(zip(mmd_feat_A, mmd_feat_B))
        paired_mmd.sort(key=lambda x: x[0]['mmd'], reverse=True)

        for ra, rb in paired_mmd[:10]:
            f.write(f"   {ra['feature']:<15} {ra['mmd']:>8.4f} "
                    f"{rb['mmd']:>8.4f} {ra['mmd'] - rb['mmd']:>8.4f}\n")

        f.write("\n")

        # ── 结论 ──
        f.write("6. CONCLUSION & RECOMMENDATION\n")
        f.write("-" * 55 + "\n")

        if global_B[1] > 0.05 and n_sig_B > len(ks_B) * 0.5:
            f.write("   SIGNIFICANT covariate shift exists in normal traffic itself.\n")
            f.write("   This will cause reconstruction errors to increase for normal\n")
            f.write("   samples in the test set, leading to elevated false positives.\n")
            f.write("\n")
            f.write("   >> Your anti-drift module should focus on ADAPTING the\n")
            f.write("      reconstruction baseline to account for shifting normal\n")
            f.write("      patterns. Techniques like adaptive normalization or\n")
            f.write("      test-time feature alignment are recommended.\n")
        elif global_A[1] > 0.05:
            f.write("   Distribution gap is mainly caused by anomalous samples\n")
            f.write("   in the test set, not by normal traffic drift.\n")
            f.write("\n")
            f.write("   >> Your anti-drift module should focus on ROBUST threshold\n")
            f.write("      adaptation rather than feature-level alignment.\n")
        else:
            f.write("   Distribution shift is minimal in the unsupervised setting.\n")
            f.write("   >> Consider adding synthetic drift for stress testing.\n")

    print(f"\n  Report saved: {path}")


def main():
    print("=" * 60)
    print("Unsupervised Setting: Distribution Shift Analysis")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print("\n[1/7] Loading data...")
    X_train_normal, X_test_all, X_test_normal, X_test_anomalous = load_and_prepare()

    # 展平
    train_flat = flatten(X_train_normal)
    test_all_flat = flatten(X_test_all)
    test_normal_flat = flatten(X_test_normal)

    print(f"\nFlattened shapes:")
    print(f"  Train (normal):  {train_flat.shape}")
    print(f"  Test (all):      {test_all_flat.shape}")
    print(f"  Test (normal):   {test_normal_flat.shape}")

    # Group A: Normal Train vs All Test
    print("\n[2/7] Global MMD: Normal Train vs All Test...")
    global_A = compute_mmd(train_flat, test_all_flat)
    print(f"  MMD = {global_A[1]:.6f}")

    # Group B: Normal Train vs Normal Test
    print("[3/7] Global MMD: Normal Train vs Normal Test...")
    global_B = compute_mmd(train_flat, test_normal_flat)
    print(f"  MMD = {global_B[1]:.6f}")

    # KS tests
    print("[4/7] KS tests...")
    ks_A = compute_ks_tests(train_flat, test_all_flat)
    ks_B = compute_ks_tests(train_flat, test_normal_flat)

    # Per-feature MMD
    print("[5/7] Per-feature MMD...")
    mmd_feat_A = compute_per_feature_mmd(train_flat, test_all_flat)
    mmd_feat_B = compute_per_feature_mmd(train_flat, test_normal_flat)

    # Plots
    print("[6/7] Generating plots...")
    plot_comparison_bar(ks_A, ks_B, OUTPUT_DIR)
    plot_mmd_comparison(mmd_feat_A, mmd_feat_B, OUTPUT_DIR)
    plot_top_features_distribution(train_flat, test_all_flat, test_normal_flat, ks_A, OUTPUT_DIR)

    # Report
    print("[7/7] Generating report...")
    generate_report(global_A, global_B, ks_A, ks_B, mmd_feat_A, mmd_feat_B, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done! Results saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()