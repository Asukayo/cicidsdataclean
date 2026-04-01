"""
训练集过滤统计：查看去掉异常窗口后还剩多少数据
================================================
修改 DATA_DIR 后直接运行即可。
"""

import os
import numpy as np

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"/home/ubuntu/wyh/cicdis/cicids2017/selected_features"  # <-- 修改为你的实际路径
WINDOW_SIZE = 100
STEP_SIZE = 20
TRAIN_RATIO = 0.5
VAL_RATIO = 0.2
# ============================================================


def main():
    # 加载数据
    X = np.load(os.path.join(DATA_DIR, f'selected_X_w{WINDOW_SIZE}_s{STEP_SIZE}.npy'))
    y = np.load(os.path.join(DATA_DIR, f'selected_y_w{WINDOW_SIZE}_s{STEP_SIZE}.npy'))

    total = len(X)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    # 划分
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 窗口级标签：只要窗口内存在异常流就标记为1
    train_labels = np.any(y_train > 0, axis=1).astype(int)
    val_labels = np.any(y_val > 0, axis=1).astype(int)
    test_labels = np.any(y_test > 0, axis=1).astype(int)

    # 过滤训练集
    normal_mask = train_labels == 0
    n_before = len(X_train)
    n_normal = int(normal_mask.sum())
    n_anomalous = n_before - n_normal

    # 打印结果
    print("=" * 55)
    print("  CICIDS2017 Window Filtering Statistics")
    print("=" * 55)

    print(f"\n  Total windows: {total}")
    print(f"  Features: {X.shape[2]}")
    print(f"  Window size: {WINDOW_SIZE}")

    print(f"\n{'─' * 55}")
    print(f"  TRAIN SET (first {TRAIN_RATIO:.0%} of data)")
    print(f"{'─' * 55}")
    print(f"  Before filtering:  {n_before} windows")
    print(f"    - Normal:        {n_normal} ({n_normal/n_before:.1%})")
    print(f"    - Anomalous:     {n_anomalous} ({n_anomalous/n_before:.1%})")
    print(f"  After filtering:   {n_normal} windows")
    print(f"  Removed:           {n_anomalous} windows ({n_anomalous/n_before:.1%})")

    print(f"\n{'─' * 55}")
    print(f"  VALIDATION SET ({VAL_RATIO:.0%} of data)")
    print(f"{'─' * 55}")
    n_val = len(X_val)
    n_val_normal = int((val_labels == 0).sum())
    n_val_anomalous = n_val - n_val_normal
    print(f"  Total:     {n_val} windows")
    print(f"    - Normal:    {n_val_normal} ({n_val_normal/n_val:.1%})")
    print(f"    - Anomalous: {n_val_anomalous} ({n_val_anomalous/n_val:.1%})")

    print(f"\n{'─' * 55}")
    print(f"  TEST SET (last {1-TRAIN_RATIO-VAL_RATIO:.0%} of data)")
    print(f"{'─' * 55}")
    n_test = len(X_test)
    n_test_normal = int((test_labels == 0).sum())
    n_test_anomalous = n_test - n_test_normal
    print(f"  Total:     {n_test} windows")
    print(f"    - Normal:    {n_test_normal} ({n_test_normal/n_test:.1%})")
    print(f"    - Anomalous: {n_test_anomalous} ({n_test_anomalous/n_test:.1%})")

    print(f"\n{'─' * 55}")
    print(f"  SUMMARY FOR UNSUPERVISED TRAINING")
    print(f"{'─' * 55}")
    print(f"  Training data:     {n_normal} normal windows")
    print(f"  Validation data:   {n_val} windows (for threshold tuning)")
    print(f"  Test data:         {n_test} windows (for evaluation)")

    # 判断训练数据是否充足
    print(f"\n{'─' * 55}")
    print(f"  ASSESSMENT")
    print(f"{'─' * 55}")
    if n_normal > 10000:
        print(f"  Training data is SUFFICIENT ({n_normal} windows).")
    elif n_normal > 3000:
        print(f"  Training data is ADEQUATE ({n_normal} windows).")
    else:
        print(f"  WARNING: Training data may be INSUFFICIENT ({n_normal} windows).")
        print(f"  Consider increasing train_ratio or reducing window step_size.")

    print("=" * 55)


if __name__ == '__main__':
    main()