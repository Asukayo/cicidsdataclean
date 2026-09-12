"""
语义特征分组合理性验证
====================

输入：
    feature_drift_heatmap.py 输出的 ks_normal_only_matrix.csv

验证目标：
1. 同一语义组内特征的漂移轨迹是否比不同组更相似；
2. 语义分组是否优于保持相同组大小的随机分组；
3. 去除每个时间块的全局漂移成分后，上述结论是否仍成立。

输出：
- raw_feature_correlation.csv
- residual_feature_correlation.csv
- raw_pairwise_correlations.csv
- residual_pairwise_correlations.csv
- raw_summary.csv
- residual_summary.csv
- raw_correlation_heatmap.png
- residual_correlation_heatmap.png
- raw_within_between_boxplot.png
- residual_within_between_boxplot.png
- raw_permutation_test.png
- residual_permutation_test.png
- group_feature_mapping.csv
- unassigned_features.txt

运行：
    python evaluate_semantic_groups.py

依赖：
    pip install numpy pandas scipy matplotlib
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# 配置区
# ============================================================

# 改成你的 KS 矩阵 CSV 路径
KS_MATRIX_CSV = "./2018feature_drift_outputs/ks_normal_only_matrix.csv"

OUTPUT_DIR = "./2018semantic_group_evaluation"

# 置换次数。快速试跑可设为 200，正式结果建议至少 1000。
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# 两个特征至少需要多少个共同有效时间块才计算相关性
MIN_COMMON_BLOCKS = 12

# 是否使用相关系数绝对值。
# False 更合理：负相关不应被视为“同组轨迹一致”。
USE_ABSOLUTE_CORRELATION = False


# ============================================================
# 候选语义分组
#
# 说明：
# 1. 这是用于“验证分组是否合理”的候选方案，不代表最终方法必须使用；
# 2. 每个特征只能属于一个组；
# 3. 脚本按标准化后的特征名精确匹配；
# 4. 未匹配特征会写入 unassigned_features.txt。
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Temporal_IAT_Activity": [
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ],
    "Packet_Size": [
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Packet Length Min",
        "Packet Length Max",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Avg Packet Size",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
        "Fwd Seg Size Min",
    ],
    "Traffic_Volume_Rate": [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd Packets Length Total",
        "Bwd Packets Length Total",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
    ],
    "Direction_Header_Window": [
        "Fwd Header Length",
        "Bwd Header Length",
        "Init Fwd Win Bytes",
        "Init Bwd Win Bytes",
        "Fwd Act Data Packets",
        "Down/Up Ratio",
    ],
    "Protocol_Flag_Port": [
        "Destination Port",
        "Protocol",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "ECE Flag Count",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd FIN Flags",
        "Bwd FIN Flags",
        "Fwd SYN Flags",
        "Bwd SYN Flags",
        "Fwd RST Flags",
        "Bwd RST Flags",
    ],
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PairRecord:
    feature_i: str
    feature_j: str
    group_i: str
    group_j: str
    relation: str
    correlation: float
    common_blocks: int


# ============================================================
# 名称与输入处理
# ============================================================

def normalize_feature_name(name: str) -> str:
    """
    统一常见命名差异，例如：
    - 前后空格
    - 多余空格
    - Bwd Packet/s 与 Bwd Packets/s
    - Fwd Packet/s 与 Fwd Packets/s
    """
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    aliases = {
        "Fwd Packet/s": "Fwd Packets/s",
        "Bwd Packet/s": "Bwd Packets/s",
        "Fwd Packets Length Total": "Fwd Packets Length Total",
        "Bwd Packets Length Total": "Bwd Packets Length Total",
        "Total Length of Fwd Packets": "Fwd Packets Length Total",
        "Total Length of Bwd Packets": "Bwd Packets Length Total",
        "Init Fwd Win Byts": "Init Fwd Win Bytes",
        "Init Bwd Win Byts": "Init Bwd Win Bytes",
        "Fwd Act Data Pkts": "Fwd Act Data Packets",
        "Fwd Seg Size Min": "Fwd Seg Size Min",
        "Avg Fwd Segment Size": "Avg Fwd Segment Size",
        "Avg Bwd Segment Size": "Avg Bwd Segment Size",
    }
    return aliases.get(name, name)


def load_ks_matrix(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 KS 矩阵：{csv_path}")

    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        raise ValueError("CSV 必须包含 feature 列。")

    df["feature"] = df["feature"].map(normalize_feature_name)

    if df["feature"].duplicated().any():
        duplicated = df.loc[df["feature"].duplicated(), "feature"].tolist()
        raise ValueError(f"标准化后出现重复特征名：{duplicated}")

    matrix = df.set_index("feature").apply(pd.to_numeric, errors="coerce")
    if matrix.empty:
        raise ValueError("KS 矩阵为空。")

    return matrix


def build_group_mapping(
    feature_names: List[str],
    groups: Dict[str, List[str]],
) -> Tuple[Dict[str, str], List[str]]:
    available = set(feature_names)
    mapping: Dict[str, str] = {}
    duplicate_assignments = []

    for group_name, group_features in groups.items():
        for raw_name in group_features:
            feature_name = normalize_feature_name(raw_name)
            if feature_name not in available:
                continue
            if feature_name in mapping:
                duplicate_assignments.append(feature_name)
            mapping[feature_name] = group_name

    if duplicate_assignments:
        raise ValueError(
            "以下特征被重复分组：{}".format(sorted(set(duplicate_assignments)))
        )

    unassigned = [name for name in feature_names if name not in mapping]
    return mapping, unassigned


# ============================================================
# 去除全局时间块效应
# ============================================================

def remove_global_block_effect(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    对每个时间块减去所有特征的中位 KS。

    原始热力图中的竖向亮带表示多个特征同时发生全局变化。
    直接计算相关性时，这种全局成分可能让所有特征都显得高度相关。
    去除块级中位数后，更关注各特征相对全局漂移的偏离轨迹。
    """
    block_median = matrix.median(axis=0, skipna=True)
    return matrix.subtract(block_median, axis=1)


# ============================================================
# 相关性计算
# ============================================================

def pairwise_spearman(
    matrix: pd.DataFrame,
    group_mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, List[PairRecord]]:
    features = [f for f in matrix.index if f in group_mapping]
    corr_matrix = pd.DataFrame(
        np.nan,
        index=features,
        columns=features,
        dtype=float,
    )
    records: List[PairRecord] = []

    for i, feature_i in enumerate(features):
        values_i = matrix.loc[feature_i].to_numpy(dtype=float)
        corr_matrix.loc[feature_i, feature_i] = 1.0

        for j in range(i + 1, len(features)):
            feature_j = features[j]
            values_j = matrix.loc[feature_j].to_numpy(dtype=float)

            valid = np.isfinite(values_i) & np.isfinite(values_j)
            common_blocks = int(valid.sum())

            if common_blocks < MIN_COMMON_BLOCKS:
                continue

            x = values_i[valid]
            y = values_j[valid]

            # 常数序列无法计算 Spearman
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue

            result = spearmanr(x, y)

            if hasattr(result, "statistic"):
                correlation = float(result.statistic)
            elif hasattr(result, "correlation"):
                correlation = float(result.correlation)
            else:
                correlation = float(result[0])
            if not np.isfinite(correlation):
                continue

            if USE_ABSOLUTE_CORRELATION:
                correlation = abs(correlation)

            corr_matrix.loc[feature_i, feature_j] = correlation
            corr_matrix.loc[feature_j, feature_i] = correlation

            group_i = group_mapping[feature_i]
            group_j = group_mapping[feature_j]
            relation = "within" if group_i == group_j else "between"

            records.append(
                PairRecord(
                    feature_i=feature_i,
                    feature_j=feature_j,
                    group_i=group_i,
                    group_j=group_j,
                    relation=relation,
                    correlation=correlation,
                    common_blocks=common_blocks,
                )
            )

    return corr_matrix, records


def records_to_dataframe(records: List[PairRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_i": r.feature_i,
                "feature_j": r.feature_j,
                "group_i": r.group_i,
                "group_j": r.group_j,
                "relation": r.relation,
                "correlation": r.correlation,
                "common_blocks": r.common_blocks,
            }
            for r in records
        ]
    )


def calculate_gap_from_records(records: List[PairRecord]) -> Dict[str, float]:
    within = np.array(
        [r.correlation for r in records if r.relation == "within"],
        dtype=float,
    )
    between = np.array(
        [r.correlation for r in records if r.relation == "between"],
        dtype=float,
    )

    if len(within) == 0 or len(between) == 0:
        raise RuntimeError("有效的组内或组间特征对数量为 0。")

    return {
        "within_pairs": int(len(within)),
        "between_pairs": int(len(between)),
        "within_mean": float(np.mean(within)),
        "within_median": float(np.median(within)),
        "between_mean": float(np.mean(between)),
        "between_median": float(np.median(between)),
        "mean_gap": float(np.mean(within) - np.mean(between)),
        "median_gap": float(np.median(within) - np.median(between)),
    }


# ============================================================
# 随机分组置换检验
# ============================================================

def correlation_pairs_from_matrix(
    corr_matrix: pd.DataFrame,
) -> List[Tuple[str, str, float]]:
    pairs = []
    features = list(corr_matrix.index)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            value = corr_matrix.iloc[i, j]
            if np.isfinite(value):
                pairs.append((features[i], features[j], float(value)))

    return pairs


def grouping_gap(
    pairs: List[Tuple[str, str, float]],
    mapping: Dict[str, str],
) -> float:
    within = []
    between = []

    for feature_i, feature_j, correlation in pairs:
        if mapping[feature_i] == mapping[feature_j]:
            within.append(correlation)
        else:
            between.append(correlation)

    if not within or not between:
        return np.nan

    # 主检验使用均值差；摘要中同时提供中位数差。
    return float(np.mean(within) - np.mean(between))


def permutation_test(
    corr_matrix: pd.DataFrame,
    semantic_mapping: Dict[str, str],
    n_permutations: int,
    seed: int,
) -> Tuple[float, np.ndarray, float, float]:
    rng = np.random.RandomState(seed)
    features = list(corr_matrix.index)
    pairs = correlation_pairs_from_matrix(corr_matrix)

    semantic_mapping = {f: semantic_mapping[f] for f in features}
    observed_gap = grouping_gap(pairs, semantic_mapping)

    group_names = []
    group_sizes = []
    for group_name in dict.fromkeys(semantic_mapping.values()):
        members = [f for f in features if semantic_mapping[f] == group_name]
        if len(members) >= 2:
            group_names.append(group_name)
            group_sizes.append(len(members))

    if sum(group_sizes) != len(features):
        raise RuntimeError("随机分组时组大小之和与特征数不一致。")

    random_gaps = np.full(n_permutations, np.nan, dtype=float)

    for permutation_idx in range(n_permutations):
        shuffled = np.array(features, dtype=object)
        rng.shuffle(shuffled)

        random_mapping: Dict[str, str] = {}
        start = 0
        for group_name, group_size in zip(group_names, group_sizes):
            end = start + group_size
            for feature_name in shuffled[start:end]:
                random_mapping[str(feature_name)] = group_name
            start = end

        random_gaps[permutation_idx] = grouping_gap(pairs, random_mapping)

    valid_random = random_gaps[np.isfinite(random_gaps)]
    if len(valid_random) == 0:
        raise RuntimeError("随机分组未产生有效结果。")

    # 单侧检验：语义分组的 gap 是否显著高于随机分组
    p_value = (
        1.0 + float(np.sum(valid_random >= observed_gap))
    ) / (len(valid_random) + 1.0)

    percentile = 100.0 * float(np.mean(valid_random <= observed_gap))
    return observed_gap, valid_random, p_value, percentile


# ============================================================
# 输出
# ============================================================

def ordered_features_by_group(
    feature_names: List[str],
    mapping: Dict[str, str],
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    ordered = []
    boundaries = []
    cursor = 0

    for group_name in FEATURE_GROUPS:
        members = [
            feature_name
            for feature_name in feature_names
            if mapping.get(feature_name) == group_name
        ]
        if not members:
            continue

        start = cursor
        ordered.extend(members)
        cursor += len(members)
        boundaries.append((group_name, start, cursor))

    return ordered, boundaries


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    mapping: Dict[str, str],
    title: str,
    output_path: str,
):
    ordered, boundaries = ordered_features_by_group(
        list(corr_matrix.index), mapping
    )
    ordered_corr = corr_matrix.loc[ordered, ordered]

    size = max(11.0, len(ordered) * 0.18)
    fig, ax = plt.subplots(figsize=(size, size))

    image = ax.imshow(
        ordered_corr.to_numpy(dtype=float),
        aspect="equal",
        interpolation="nearest",
        vmin=-1.0 if not USE_ABSOLUTE_CORRELATION else 0.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered, rotation=90, fontsize=5)
    ax.set_yticklabels(ordered, fontsize=5)

    for _, start, end in boundaries[:-1]:
        ax.axhline(end - 0.5, linewidth=1.0)
        ax.axvline(end - 0.5, linewidth=1.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Spearman correlation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_within_between_boxplot(
    pair_df: pd.DataFrame,
    title: str,
    output_path: str,
):
    within = pair_df.loc[
        pair_df["relation"] == "within", "correlation"
    ].dropna()
    between = pair_df.loc[
        pair_df["relation"] == "between", "correlation"
    ].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [within.to_numpy(), between.to_numpy()],
        labels=["Within semantic groups", "Between groups"],
        showfliers=False,
    )
    ax.set_ylabel("Spearman correlation")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_distribution(
    random_gaps: np.ndarray,
    observed_gap: float,
    p_value: float,
    percentile: float,
    title: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_gaps, bins=35, alpha=0.85)
    ax.axvline(
        observed_gap,
        linewidth=2.0,
        linestyle="--",
        label=(
            f"Semantic grouping: {observed_gap:.4f}\n"
            f"p={p_value:.4f}, percentile={percentile:.1f}%"
        ),
    )
    ax.set_xlabel("Within-group mean correlation − between-group mean correlation")
    ax.set_ylabel("Random grouping count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    mode: str,
    stats: Dict[str, float],
    observed_gap: float,
    p_value: float,
    percentile: float,
    output_path: str,
):
    row = {
        "mode": mode,
        **stats,
        "permutation_observed_mean_gap": observed_gap,
        "permutation_p_value_one_sided": p_value,
        "semantic_grouping_percentile": percentile,
        "n_permutations": N_PERMUTATIONS,
        "min_common_blocks": MIN_COMMON_BLOCKS,
    }
    pd.DataFrame([row]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_group_mapping(
    mapping: Dict[str, str],
    output_path: str,
):
    rows = [
        {"feature": feature_name, "group": group_name}
        for feature_name, group_name in mapping.items()
    ]
    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 单种矩阵分析
# ============================================================

def analyze_mode(
    mode: str,
    matrix: pd.DataFrame,
    mapping: Dict[str, str],
):
    print(f"\n[{mode}] 计算特征相关性 ...")
    corr_matrix, records = pairwise_spearman(matrix, mapping)

    valid_features = [
        feature_name
        for feature_name in corr_matrix.index
        if corr_matrix.loc[feature_name].notna().sum() > 1
    ]
    corr_matrix = corr_matrix.loc[valid_features, valid_features]

    # 只保留仍在相关矩阵中的记录
    valid_set = set(valid_features)
    records = [
        r
        for r in records
        if r.feature_i in valid_set and r.feature_j in valid_set
    ]

    pair_df = records_to_dataframe(records)
    stats = calculate_gap_from_records(records)

    observed_gap, random_gaps, p_value, percentile = permutation_test(
        corr_matrix=corr_matrix,
        semantic_mapping=mapping,
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED,
    )

    prefix = os.path.join(OUTPUT_DIR, mode)

    corr_matrix.to_csv(
        f"{prefix}_feature_correlation.csv",
        encoding="utf-8-sig",
    )
    pair_df.to_csv(
        f"{prefix}_pairwise_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_summary(
        mode=mode,
        stats=stats,
        observed_gap=observed_gap,
        p_value=p_value,
        percentile=percentile,
        output_path=f"{prefix}_summary.csv",
    )

    plot_correlation_heatmap(
        corr_matrix=corr_matrix,
        mapping=mapping,
        title=f"{mode.capitalize()} feature-drift trajectory correlation",
        output_path=f"{prefix}_correlation_heatmap.png",
    )
    plot_within_between_boxplot(
        pair_df=pair_df,
        title=f"{mode.capitalize()}: within-group vs between-group correlation",
        output_path=f"{prefix}_within_between_boxplot.png",
    )
    plot_permutation_distribution(
        random_gaps=random_gaps,
        observed_gap=observed_gap,
        p_value=p_value,
        percentile=percentile,
        title=f"{mode.capitalize()} semantic grouping vs random grouping",
        output_path=f"{prefix}_permutation_test.png",
    )

    print(
        "  within mean={:.4f}, between mean={:.4f}, gap={:.4f}".format(
            stats["within_mean"],
            stats["between_mean"],
            stats["mean_gap"],
        )
    )
    print(
        "  permutation p={:.4f}, semantic percentile={:.1f}%".format(
            p_value,
            percentile,
        )
    )


# ============================================================
# 主程序
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ks_matrix = load_ks_matrix(KS_MATRIX_CSV)
    mapping, unassigned = build_group_mapping(
        feature_names=list(ks_matrix.index),
        groups=FEATURE_GROUPS,
    )

    grouped_counts = pd.Series(mapping).value_counts()
    print("已匹配特征数：", len(mapping))
    print("各组特征数：")
    print(grouped_counts.to_string())

    if unassigned:
        print("\n未分组特征：")
        for feature_name in unassigned:
            print("  -", feature_name)

    with open(
        os.path.join(OUTPUT_DIR, "unassigned_features.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        for feature_name in unassigned:
            file.write(feature_name + "\n")

    save_group_mapping(
        mapping,
        os.path.join(OUTPUT_DIR, "group_feature_mapping.csv"),
    )

    # 只分析已匹配的候选语义组特征。
    grouped_matrix = ks_matrix.loc[list(mapping.keys())]

    # 1. 原始 KS 轨迹相关性
    analyze_mode(
        mode="raw",
        matrix=grouped_matrix,
        mapping=mapping,
    )

    # 2. 去除每个时间块的全局漂移成分后再分析
    residual_matrix = remove_global_block_effect(grouped_matrix)
    analyze_mode(
        mode="residual",
        matrix=residual_matrix,
        mapping=mapping,
    )

    print("\n分析完成，结果目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
