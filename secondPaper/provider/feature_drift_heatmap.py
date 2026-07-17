"""
逐特征漂移热力图（基于 unsupervised_provider.py）
================================================

功能：
1. 复用 provider 中的 load_data() 加载滑动窗口数据；
2. 按 provider 相同的时间比例划分训练/验证/测试；
3. 以“训练阶段的正常窗口”建立每个特征的源域参考分布；
4. 对验证、测试或训练后完整时间流按时间块计算逐特征 KS 漂移；
5. 同时输出：
   - 完整在线流（正常 + 异常）的逐特征漂移热力图；
   - 仅正常流量的逐特征漂移热力图（只用于离线现象分析）；
   - 每个时间块的异常比例曲线；
   - 漂移矩阵、时间块统计和特征漂移摘要 CSV。

重要说明：
- 该脚本不训练模型，也不使用 StandardScaler。
  KS 统计量对同一特征上的线性单调缩放不敏感，因此直接使用原始特征即可。
- 数据由长度 WINDOW_SIZE、步长 STEP_SIZE 的重叠窗口构成。
  为避免同一条流量被重复统计，每个窗口只提取最后 STEP_SIZE 条新增流量。
- “仅正常流量”热力图使用测试标签筛选，仅可用于离线验证研究动机，
  不能作为真实在线方法的输入，也不能据此选择测试阶段参数。

运行前：
1. 将本文件与 unsupervised_provider.py 放在同一目录；
2. 修改下方 DATA_DIR；
3. 运行：python feature_drift_heatmap.py
"""

import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from unsupervised_provider import load_data, print_split_info


# ============================================================
# 配置区：通常只需要修改这里
# ============================================================
DATA_DIR = "/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"
OUTPUT_DIR = "./feature_drift_outputs"

WINDOW_SIZE = 100
STEP_SIZE = 20

# 与 provider 示例保持一致：前 40% 训练，中间 40% 验证，后 20% 测试
TRAIN_RATIO = 0.4
VAL_RATIO = 0.4

# 可选："val"、"test"、"post_train"
# post_train 表示从训练结束后开始，按时间顺序连续分析验证集 + 测试集。
ANALYSIS_SPLIT = "post_train"

# 一个漂移时间块包含多少个检测窗口。
# 每个窗口只取最后 STEP_SIZE 条新增流量，因此：
# 每个完整时间块约包含 BLOCK_WINDOWS * STEP_SIZE 条不重复流量。
BLOCK_WINDOWS = 1000

# 为控制 KS 计算耗时，对源域和当前时间块进行确定性随机抽样。
# 数据量较小时会自动使用全部样本。
REFERENCE_MAX_SAMPLES = 20000
CURRENT_MAX_SAMPLES = 10000
MIN_VALID_SAMPLES = 100
RANDOM_SEED = 42

# 是否生成两类热力图
MAKE_ALL_STREAM_HEATMAP = True
MAKE_NORMAL_ONLY_HEATMAP = True

# 特征名获取方式：
# 1. 优先使用 FEATURE_NAMES；
# 2. 其次读取 FEATURE_NAMES_FILE；
# 3. 再尝试从 metadata 中提取；
# 4. 均失败则使用 Feature_00、Feature_01 ...
FEATURE_NAMES = None
FEATURE_NAMES_FILE = None

# 颜色上限：None 表示根据全部漂移矩阵的 99% 分位数自动确定。
# 若要跨数据集严格比较，建议手动设为同一个值，例如 0.5 或 1.0。
HEATMAP_VMAX = None

# 图像设置
FIG_DPI = 800
MAX_X_TICKS = 12


# ============================================================
# 工具函数
# ============================================================
def validate_config():
    if not 0 < TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO 必须在 (0, 1) 内。")
    if not 0 <= VAL_RATIO < 1:
        raise ValueError("VAL_RATIO 必须在 [0, 1) 内。")
    if TRAIN_RATIO + VAL_RATIO >= 1:
        raise ValueError("TRAIN_RATIO + VAL_RATIO 必须小于 1。")
    if not 0 < STEP_SIZE <= WINDOW_SIZE:
        raise ValueError("STEP_SIZE 必须满足 0 < STEP_SIZE <= WINDOW_SIZE。")
    if BLOCK_WINDOWS <= 0:
        raise ValueError("BLOCK_WINDOWS 必须为正整数。")
    if ANALYSIS_SPLIT not in {"val", "test", "post_train"}:
        raise ValueError("ANALYSIS_SPLIT 只能是 'val'、'test' 或 'post_train'。")


def extract_feature_names(metadata, num_features):
    """尽可能从显式配置、文件或 metadata 中获取特征名。"""
    if FEATURE_NAMES is not None:
        names = list(FEATURE_NAMES)
        if len(names) != num_features:
            raise ValueError(
                "FEATURE_NAMES 数量为 {}，但数据特征数为 {}。".format(
                    len(names), num_features
                )
            )
        return [str(name) for name in names]

    if FEATURE_NAMES_FILE:
        names = load_feature_names_file(FEATURE_NAMES_FILE)
        if len(names) != num_features:
            raise ValueError(
                "特征名文件包含 {} 个名称，但数据特征数为 {}。".format(
                    len(names), num_features
                )
            )
        return names

    if isinstance(metadata, dict):
        candidate_keys = [
            "feature_names",
            "selected_features",
            "features",
            "columns",
            "column_names",
            "feature_columns",
        ]
        for key in candidate_keys:
            value = metadata.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                value = list(value)
                if len(value) == num_features:
                    return [str(name) for name in value]

    print(
        "[Warning] 未找到完整特征名，将使用 Feature_00 ... Feature_{:02d}。".format(
            num_features - 1
        )
    )
    return ["Feature_{:02d}".format(i) for i in range(num_features)]


def load_feature_names_file(file_path):
    """支持 txt/csv/json/npy 格式的特征名文件。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError("特征名文件不存在：{}".format(file_path))

    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".npy":
        values = np.load(file_path, allow_pickle=True).tolist()
        return [str(value).strip() for value in values if str(value).strip()]

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as file:
            values = json.load(file)
        if isinstance(values, dict):
            for key in ("feature_names", "features", "columns"):
                if key in values:
                    values = values[key]
                    break
        if not isinstance(values, list):
            raise ValueError("JSON 特征名文件应为列表，或包含 feature_names/features/columns。")
        return [str(value).strip() for value in values if str(value).strip()]

    if suffix == ".csv":
        names = []
        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                for value in row:
                    value = value.strip()
                    if value:
                        names.append(value)
        return names

    with open(file_path, "r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def build_split_info(X, y):
    """使用与 provider 一致的时间比例和窗口级标签统计。"""
    total = len(X)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_window_labels = np.any(y[:train_end] > 0, axis=1).astype(np.int8)
    val_window_labels = np.any(y[train_end:val_end] > 0, axis=1).astype(np.int8)
    test_window_labels = np.any(y[val_end:] > 0, axis=1).astype(np.int8)

    split_info = {
        "train_before_filter": train_end,
        "train_removed": int(train_window_labels.sum()),
        "train_normal": int((train_window_labels == 0).sum()),
        "val_total": len(val_window_labels),
        "val_normal": int((val_window_labels == 0).sum()),
        "val_anomalous": int((val_window_labels == 1).sum()),
        "test_total": len(test_window_labels),
        "test_normal": int((test_window_labels == 0).sum()),
        "test_anomalous": int((test_window_labels == 1).sum()),
    }
    return train_end, val_end, split_info


def get_analysis_range(total, train_end, val_end):
    """返回待分析时间流在完整 X/y 中的 [start, end) 索引。"""
    if ANALYSIS_SPLIT == "val":
        return train_end, val_end
    if ANALYSIS_SPLIT == "test":
        return val_end, total
    return train_end, total


def sample_rows(values, max_samples, rng):
    """按行无放回抽样，所有特征共享同一批行索引。"""
    if len(values) <= max_samples:
        return values
    indices = rng.choice(len(values), size=max_samples, replace=False)
    return values[indices]


def build_reference_samples(X, y, train_end, rng):
    """
    从训练阶段的完全正常窗口中构建源域参考样本。

    与 provider 保持一致：只有窗口内不存在恶意流时，该窗口才属于正常训练集。
    为去除滑动窗口重复，每个窗口只使用最后 STEP_SIZE 条新增流量。
    """
    train_window_normal = ~np.any(y[:train_end] > 0, axis=1)
    normal_window_indices = np.flatnonzero(train_window_normal)

    if len(normal_window_indices) == 0:
        raise RuntimeError("训练段中没有完全正常的窗口，无法建立源域参考分布。")

    needed_windows = int(math.ceil(float(REFERENCE_MAX_SAMPLES) / STEP_SIZE))
    if len(normal_window_indices) > needed_windows:
        selected_indices = rng.choice(
            normal_window_indices, size=needed_windows, replace=False
        )
        selected_indices.sort()
    else:
        selected_indices = normal_window_indices

    reference = X[selected_indices, -STEP_SIZE:, :].reshape(
        -1, X.shape[-1]
    )
    reference = sample_rows(reference, REFERENCE_MAX_SAMPLES, rng)

    print(
        "Reference: {} normal windows -> {} de-overlapped flows".format(
            len(selected_indices), len(reference)
        )
    )
    return reference


def prepare_reference_by_feature(reference):
    """提前清理每个参考特征中的 NaN/Inf，避免在每个时间块重复处理。"""
    result = []
    for feature_idx in range(reference.shape[1]):
        values = reference[:, feature_idx]
        values = values[np.isfinite(values)]
        result.append(values)
    return result


def compute_feature_ks(reference_by_feature, current, min_valid_samples):
    """计算一个时间块中每个特征相对源域参考分布的 KS 统计量。"""
    num_features = current.shape[1]
    ks_values = np.full(num_features, np.nan, dtype=np.float64)

    for feature_idx in range(num_features):
        source_values = reference_by_feature[feature_idx]
        current_values = current[:, feature_idx]
        current_values = current_values[np.isfinite(current_values)]

        if (
            len(source_values) < min_valid_samples
            or len(current_values) < min_valid_samples
        ):
            continue

        ks_values[feature_idx] = ks_2samp(
            source_values, current_values
        ).statistic

    return ks_values


def calculate_drift_matrices(X, y, start, end, reference, rng):
    """
    按时间块计算完整流和仅正常流的逐特征 KS 矩阵。

    Returns:
        matrices: dict，键为 all_stream / normal_only
        block_records: 每个时间块的范围、样本数和异常比例
    """
    reference_by_feature = prepare_reference_by_feature(reference)
    num_features = X.shape[-1]
    num_stream_windows = end - start
    num_blocks = int(math.ceil(float(num_stream_windows) / BLOCK_WINDOWS))

    matrices = {}
    if MAKE_ALL_STREAM_HEATMAP:
        matrices["all_stream"] = np.full(
            (num_features, num_blocks), np.nan, dtype=np.float64
        )
    if MAKE_NORMAL_ONLY_HEATMAP:
        matrices["normal_only"] = np.full(
            (num_features, num_blocks), np.nan, dtype=np.float64
        )

    block_records = []

    for block_idx in range(num_blocks):
        relative_start = block_idx * BLOCK_WINDOWS
        relative_end = min((block_idx + 1) * BLOCK_WINDOWS, num_stream_windows)
        absolute_start = start + relative_start
        absolute_end = start + relative_end

        # 只取每个滑动窗口最后 STEP_SIZE 条新增流量，避免重复计数。
        block_X = X[absolute_start:absolute_end, -STEP_SIZE:, :].reshape(
            -1, num_features
        )
        block_y = y[absolute_start:absolute_end, -STEP_SIZE:].reshape(-1)

        finite_label_mask = np.isfinite(block_y)
        block_X = block_X[finite_label_mask]
        block_y = block_y[finite_label_mask]

        normal_mask = block_y <= 0
        total_flows = len(block_y)
        normal_flows = int(normal_mask.sum())
        anomalous_flows = total_flows - normal_flows
        anomaly_ratio = (
            float(anomalous_flows) / total_flows if total_flows > 0 else np.nan
        )

        block_records.append(
            {
                "block": block_idx,
                "absolute_window_start": absolute_start,
                "absolute_window_end_exclusive": absolute_end,
                "windows": absolute_end - absolute_start,
                "total_flows": total_flows,
                "normal_flows": normal_flows,
                "anomalous_flows": anomalous_flows,
                "anomaly_ratio": anomaly_ratio,
            }
        )

        if "all_stream" in matrices and total_flows >= MIN_VALID_SAMPLES:
            current_all = sample_rows(block_X, CURRENT_MAX_SAMPLES, rng)
            matrices["all_stream"][:, block_idx] = compute_feature_ks(
                reference_by_feature, current_all, MIN_VALID_SAMPLES
            )

        if "normal_only" in matrices and normal_flows >= MIN_VALID_SAMPLES:
            current_normal = block_X[normal_mask]
            current_normal = sample_rows(
                current_normal, CURRENT_MAX_SAMPLES, rng
            )
            matrices["normal_only"][:, block_idx] = compute_feature_ks(
                reference_by_feature, current_normal, MIN_VALID_SAMPLES
            )

        if (
            block_idx == 0
            or (block_idx + 1) % 20 == 0
            or block_idx + 1 == num_blocks
        ):
            print(
                "Processed block {}/{} | windows={} | flows={} | anomaly_ratio={:.4f}".format(
                    block_idx + 1,
                    num_blocks,
                    absolute_end - absolute_start,
                    total_flows,
                    anomaly_ratio if np.isfinite(anomaly_ratio) else float("nan"),
                )
            )

    return matrices, block_records


def save_matrix_csv(matrix, feature_names, output_path):
    block_names = ["block_{:04d}".format(i) for i in range(matrix.shape[1])]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature"] + block_names)
        for name, row in zip(feature_names, matrix):
            writer.writerow([name] + row.tolist())


def save_block_records(block_records, output_path):
    if not block_records:
        return
    fieldnames = list(block_records[0].keys())
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(block_records)


def longest_run_above(values, threshold):
    """计算有效值中连续超过阈值的最长时间块数量。"""
    best = 0
    current = 0
    for value in values:
        if np.isfinite(value) and value >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def save_feature_summary(matrix, feature_names, output_path):
    """保存各特征跨时间块的漂移统计，便于寻找持续或剧烈漂移特征。"""
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "feature",
                "valid_blocks",
                "mean_ks",
                "median_ks",
                "p90_ks",
                "max_ks",
                "longest_run_ks_ge_0.2",
                "longest_run_ks_ge_0.3",
            ]
        )

        for feature_name, values in zip(feature_names, matrix):
            valid_values = values[np.isfinite(values)]
            if len(valid_values) == 0:
                writer.writerow(
                    [feature_name, 0, np.nan, np.nan, np.nan, np.nan, 0, 0]
                )
                continue

            writer.writerow(
                [
                    feature_name,
                    len(valid_values),
                    float(np.mean(valid_values)),
                    float(np.median(valid_values)),
                    float(np.percentile(valid_values, 90)),
                    float(np.max(valid_values)),
                    longest_run_above(values, 0.2),
                    longest_run_above(values, 0.3),
                ]
            )


def choose_shared_vmax(matrices):
    if HEATMAP_VMAX is not None:
        return float(HEATMAP_VMAX)

    finite_values = []
    for matrix in matrices.values():
        values = matrix[np.isfinite(matrix)]
        if len(values) > 0:
            finite_values.append(values)

    if not finite_values:
        return 1.0

    all_values = np.concatenate(finite_values)
    # 至少给出 0.1 的色阶范围，同时不超过 KS 理论上限 1。
    return min(1.0, max(0.1, float(np.percentile(all_values, 99))))


def plot_heatmap(matrix, feature_names, title, output_path, vmax):
    num_features, num_blocks = matrix.shape
    figure_height = max(8.0, num_features * 0.28)
    figure_width = min(24.0, max(12.0, num_blocks * 0.06))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("Chronological block index")
    ax.set_ylabel("Feature")

    ax.set_yticks(np.arange(num_features))
    ax.set_yticklabels(feature_names, fontsize=7)

    if num_blocks > 0:
        tick_count = min(MAX_X_TICKS, num_blocks)
        tick_positions = np.unique(
            np.linspace(0, num_blocks - 1, tick_count, dtype=int)
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(position) for position in tick_positions])

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Two-sample KS statistic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_ratio(block_records, output_path):
    block_indices = [record["block"] for record in block_records]
    anomaly_ratios = [record["anomaly_ratio"] for record in block_records]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(block_indices, anomaly_ratios, linewidth=1.2)
    ax.set_xlabel("Chronological block index")
    ax.set_ylabel("Anomalous-flow ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Anomalous-flow ratio in each chronological block")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_feature_names(feature_names, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for name in feature_names:
            file.write(str(name) + "\n")


# ============================================================
# 主程序
# ============================================================
def main():
    validate_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    print("Loading data ...")
    X, y, metadata = load_data(DATA_DIR, WINDOW_SIZE, STEP_SIZE)

    if X.ndim != 3:
        raise ValueError("X 应为 [num_windows, window_size, num_features]，实际为 {}".format(X.shape))
    if y.ndim != 2:
        raise ValueError("y 应为 [num_windows, window_size]，实际为 {}".format(y.shape))
    if X.shape[:2] != y.shape:
        raise ValueError("X 前两维 {} 与 y {} 不一致。".format(X.shape[:2], y.shape))
    if X.shape[1] != WINDOW_SIZE:
        raise ValueError(
            "配置 WINDOW_SIZE={}，但数据窗口长度为 {}。".format(
                WINDOW_SIZE, X.shape[1]
            )
        )

    print("Loaded: X={}, y={}".format(X.shape, y.shape))

    train_end, val_end, split_info = build_split_info(X, y)
    print_split_info(split_info)

    analysis_start, analysis_end = get_analysis_range(
        len(X), train_end, val_end
    )
    print(
        "Analysis split: {} | window range=[{}, {}) | {} windows".format(
            ANALYSIS_SPLIT,
            analysis_start,
            analysis_end,
            analysis_end - analysis_start,
        )
    )

    feature_names = extract_feature_names(metadata, X.shape[-1])
    save_feature_names(
        feature_names, os.path.join(OUTPUT_DIR, "feature_names_used.txt")
    )

    reference = build_reference_samples(X, y, train_end, rng)
    matrices, block_records = calculate_drift_matrices(
        X, y, analysis_start, analysis_end, reference, rng
    )

    shared_vmax = choose_shared_vmax(matrices)
    print("Shared heatmap vmax: {:.4f}".format(shared_vmax))

    for matrix_name, matrix in matrices.items():
        matrix_csv = os.path.join(
            OUTPUT_DIR, "ks_{}_matrix.csv".format(matrix_name)
        )
        summary_csv = os.path.join(
            OUTPUT_DIR, "ks_{}_feature_summary.csv".format(matrix_name)
        )
        heatmap_png = os.path.join(
            OUTPUT_DIR, "ks_{}_heatmap.png".format(matrix_name)
        )

        save_matrix_csv(matrix, feature_names, matrix_csv)
        save_feature_summary(matrix, feature_names, summary_csv)

        if matrix_name == "all_stream":
            title = (
                "Feature-wise distribution drift: complete chronological stream "
                "({})".format(ANALYSIS_SPLIT)
            )
        else:
            title = (
                "Feature-wise normal-behavior drift: label-filtered offline analysis "
                "({})".format(ANALYSIS_SPLIT)
            )

        plot_heatmap(matrix, feature_names, title, heatmap_png, shared_vmax)

    save_block_records(
        block_records, os.path.join(OUTPUT_DIR, "block_statistics.csv")
    )
    plot_anomaly_ratio(
        block_records, os.path.join(OUTPUT_DIR, "anomaly_ratio_by_block.png")
    )

    print("\nDone. Outputs saved to: {}".format(os.path.abspath(OUTPUT_DIR)))
    print("主要文件：")
    for file_name in sorted(os.listdir(OUTPUT_DIR)):
        print("  - {}".format(file_name))


if __name__ == "__main__":
    main()
