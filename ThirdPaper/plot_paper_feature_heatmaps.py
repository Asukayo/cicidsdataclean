"""
论文版逐特征KS热力图绘图脚本
================================
用途：
1. 读取两个数据集的 ks_normal_only_matrix.csv
2. 按固定语义组对特征排序
3. 生成适合论文正文的双子图热力图（2017 / 2018）
4. 生成一个附加版（显示全部特征名），便于补充材料或自查

可直接在 PyCharm 中运行。

依赖：
    pip install numpy pandas matplotlib

默认输入路径：
    ./2017feature_drift_outputs/ks_normal_only_matrix.csv
    ./2018feature_drift_outputs/ks_normal_only_matrix.csv

输出：
    ./paper_heatmaps/
        figure_main_heatmap.png
        figure_main_heatmap.pdf
        figure_main_heatmap.svg
        figure_appendix_heatmap.png
        figure_appendix_heatmap.pdf
        figure_appendix_heatmap.svg
        feature_group_mapping_used.csv
        missing_features_report.txt
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 配置区：按需修改
# ============================================================

CSV_2017 = r"./2017feature_drift_outputs/ks_normal_only_matrix.csv"
CSV_2018 = r"./2018feature_drift_outputs/ks_normal_only_matrix.csv"
OUTPUT_DIR = r"./paper_heatmaps"

# 论文主图颜色范围：两个数据集必须共用同一色阶
VMIN = 0.0
VMAX = 0.8

# 图片分辨率
DPI = 600

# 是否在主图中只显示语义组名（推荐 True）
MAIN_SHOW_ONLY_GROUP_LABELS = True

# 是否额外导出一个“显示全部特征名”的附图（推荐 True）
EXPORT_APPENDIX_FIGURE = True

# 缺失值颜色（例如某些时间块正常样本不足）
NAN_COLOR = "#D9D9D9"

# 主图尺寸：适合论文双栏跨栏图
MAIN_FIGSIZE = (13.5, 8.5)

# 附图尺寸：显示全部特征名时更高一些
APPENDIX_FIGSIZE = (13.5, 14.0)

# 横轴主刻度数上限（防止2018时间块太多时过密）
MAX_X_TICKS = 8


# ============================================================
# 2. 固定语义组（与前面语义分组验证保持一致）
#    注意：这里只决定绘图顺序，不直接改变原始KS矩阵
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Temporal–IAT–Activity": [
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
    "Packet Size": [
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
    "Traffic Volume–Rate": [
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
    "Direction–Header–Window": [
        "Fwd Header Length",
        "Bwd Header Length",
        "Init Fwd Win Bytes",
        "Init Bwd Win Bytes",
        "Fwd Act Data Packets",
        "Down/Up Ratio",
    ],
    "Protocol–Flag–Port": [
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
# 3. 名称标准化
# ============================================================

def normalize_feature_name(name: str) -> str:
    """统一常见特征命名差异。"""
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    aliases = {
        "Fwd Packet/s": "Fwd Packets/s",
        "Bwd Packet/s": "Bwd Packets/s",
        "Total Length of Fwd Packets": "Fwd Packets Length Total",
        "Total Length of Bwd Packets": "Bwd Packets Length Total",
        "Init Fwd Win Byts": "Init Fwd Win Bytes",
        "Init Bwd Win Byts": "Init Bwd Win Bytes",
        "Fwd Act Data Pkts": "Fwd Act Data Packets",
        "Bwd Pkt Len Max": "Bwd Packet Length Max",
        "Bwd Pkt Len Min": "Bwd Packet Length Min",
        "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
        "Bwd Pkt Len Std": "Bwd Packet Length Std",
        "Fwd Pkt Len Max": "Fwd Packet Length Max",
        "Fwd Pkt Len Min": "Fwd Packet Length Min",
        "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
        "Fwd Pkt Len Std": "Fwd Packet Length Std",
    }
    return aliases.get(name, name)


# ============================================================
# 4. 读取与整理
# ============================================================

def load_ks_matrix(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件：{csv_path}")

    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        raise ValueError(f"{csv_path} 中必须包含列 'feature'。")

    df["feature"] = df["feature"].map(normalize_feature_name)

    if df["feature"].duplicated().any():
        duplicated = df.loc[df["feature"].duplicated(), "feature"].tolist()
        raise ValueError(f"{csv_path} 标准化后出现重复特征名：{duplicated}")

    matrix = df.set_index("feature").apply(pd.to_numeric, errors="coerce")
    return matrix


def build_feature_order(
    features_2017: List[str],
    features_2018: List[str],
) -> Tuple[List[str], List[Tuple[str, int, int]], List[str], List[str]]:
    """
    生成两个数据集共用的固定特征顺序，并返回：
    - ordered_features
    - boundaries: [(group_name, start, end), ...]
    - missing_2017
    - missing_2018
    """
    set17 = set(features_2017)
    set18 = set(features_2018)

    ordered_features = []
    boundaries = []

    missing_2017 = []
    missing_2018 = []

    cursor = 0

    for group_name, raw_features in FEATURE_GROUPS.items():
        current_group_features = []

        for raw_name in raw_features:
            feature = normalize_feature_name(raw_name)

            if feature not in set17:
                missing_2017.append(feature)
            if feature not in set18:
                missing_2018.append(feature)

            # 只要任一数据集存在，就保留在总顺序中；
            # 后续缺失的数据集用 NaN 行补齐
            if feature in set17 or feature in set18:
                current_group_features.append(feature)

        if current_group_features:
            start = cursor
            ordered_features.extend(current_group_features)
            cursor += len(current_group_features)
            boundaries.append((group_name, start, cursor))

    return ordered_features, boundaries, sorted(set(missing_2017)), sorted(set(missing_2018))


def reindex_matrix(matrix: pd.DataFrame, ordered_features: List[str]) -> pd.DataFrame:
    """按统一顺序重排，缺失特征自动补 NaN 行。"""
    return matrix.reindex(ordered_features)


def save_mapping(ordered_features: List[str], boundaries: List[Tuple[str, int, int]], output_dir: str):
    rows = []
    for group_name, start, end in boundaries:
        for idx in range(start, end):
            rows.append({
                "group": group_name,
                "order": idx,
                "feature": ordered_features[idx],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "feature_group_mapping_used.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_missing_report(missing_2017: List[str], missing_2018: List[str], output_dir: str):
    path = os.path.join(output_dir, "missing_features_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Features missing in 2017 CSV:\n")
        for item in missing_2017:
            f.write(f"  - {item}\n")

        f.write("\nFeatures missing in 2018 CSV:\n")
        for item in missing_2018:
            f.write(f"  - {item}\n")


# ============================================================
# 5. 绘图辅助
# ============================================================

def make_cmap():
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(NAN_COLOR)
    return cmap


def make_xticks(num_blocks: int, max_ticks: int = MAX_X_TICKS):
    if num_blocks <= 1:
        return [0], ["0"]
    n_ticks = min(max_ticks, num_blocks)
    positions = np.linspace(0, num_blocks - 1, n_ticks, dtype=int)
    positions = np.unique(positions)
    labels = [str(int(p)) for p in positions]
    return positions.tolist(), labels


def group_midpoints(boundaries: List[Tuple[str, int, int]]) -> Tuple[List[float], List[str]]:
    positions = []
    labels = []
    for group_name, start, end in boundaries:
        positions.append((start + end - 1) / 2.0)
        labels.append(group_name)
    return positions, labels


def plot_one_heatmap(
    ax,
    matrix: pd.DataFrame,
    boundaries: List[Tuple[str, int, int]],
    subplot_title: str,
    show_only_group_labels: bool,
    full_feature_labels: bool,
):
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=make_cmap(),
        vmin=VMIN,
        vmax=VMAX,
    )

    # 子图标题
    ax.set_title(subplot_title, fontsize=11, pad=6)

    # 横轴
    xticks, xlabels = make_xticks(matrix.shape[1], MAX_X_TICKS)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Chronological block index", fontsize=10)

    # 纵轴
    if full_feature_labels:
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.tolist(), fontsize=6)
    elif show_only_group_labels:
        mids, labels = group_midpoints(boundaries)
        ax.set_yticks(mids)
        ax.set_yticklabels(labels, fontsize=9)
    else:
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.tolist(), fontsize=7)

    ax.set_ylabel("Feature / semantic group", fontsize=10)

    # 组间分隔线
    for _, _, end in boundaries[:-1]:
        ax.axhline(end - 0.5, color="white", linewidth=1.1)

    # 去掉过于厚重的边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return image


def plot_main_figure(
    matrix_2017: pd.DataFrame,
    matrix_2018: pd.DataFrame,
    boundaries: List[Tuple[str, int, int]],
    output_dir: str,
):
    fig, axes = plt.subplots(
        2, 1,
        figsize=MAIN_FIGSIZE,
        sharey=True,
        constrained_layout=False,
    )

    im = plot_one_heatmap(
        ax=axes[0],
        matrix=matrix_2017,
        boundaries=boundaries,
        subplot_title="(a) CICIDS2017",
        show_only_group_labels=MAIN_SHOW_ONLY_GROUP_LABELS,
        full_feature_labels=False,
    )
    plot_one_heatmap(
        ax=axes[1],
        matrix=matrix_2018,
        boundaries=boundaries,
        subplot_title="(b) CICIDS2018",
        show_only_group_labels=MAIN_SHOW_ONLY_GROUP_LABELS,
        full_feature_labels=False,
    )

    # 共享颜色条
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Two-sample KS statistic", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # 总标题可省略；论文里通常用 caption 解释即可
    fig.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.09, hspace=0.16)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(output_dir, f"figure_main_heatmap.{ext}"),
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_appendix_figure(
    matrix_2017: pd.DataFrame,
    matrix_2018: pd.DataFrame,
    boundaries: List[Tuple[str, int, int]],
    output_dir: str,
):
    fig, axes = plt.subplots(
        1, 2,
        figsize=APPENDIX_FIGSIZE,
        sharey=True,
        constrained_layout=False,
    )

    im = plot_one_heatmap(
        ax=axes[0],
        matrix=matrix_2017,
        boundaries=boundaries,
        subplot_title="CICIDS2017 (all feature labels)",
        show_only_group_labels=False,
        full_feature_labels=True,
    )
    plot_one_heatmap(
        ax=axes[1],
        matrix=matrix_2018,
        boundaries=boundaries,
        subplot_title="CICIDS2018 (all feature labels)",
        show_only_group_labels=False,
        full_feature_labels=True,
    )

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Two-sample KS statistic", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.25, right=0.92, top=0.95, bottom=0.08, wspace=0.12)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(output_dir, f"figure_appendix_heatmap.{ext}"),
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading KS matrices ...")
    ks_2017 = load_ks_matrix(CSV_2017)
    ks_2018 = load_ks_matrix(CSV_2018)

    print(f"2017 shape: {ks_2017.shape}")
    print(f"2018 shape: {ks_2018.shape}")

    ordered_features, boundaries, missing_2017, missing_2018 = build_feature_order(
        features_2017=list(ks_2017.index),
        features_2018=list(ks_2018.index),
    )

    if not ordered_features:
        raise RuntimeError("没有构建出任何有效特征顺序，请检查 CSV 特征名。")

    print(f"Unified ordered feature count: {len(ordered_features)}")

    ks_2017_ord = reindex_matrix(ks_2017, ordered_features)
    ks_2018_ord = reindex_matrix(ks_2018, ordered_features)

    save_mapping(ordered_features, boundaries, OUTPUT_DIR)
    save_missing_report(missing_2017, missing_2018, OUTPUT_DIR)

    print("Plotting main paper figure ...")
    plot_main_figure(
        matrix_2017=ks_2017_ord,
        matrix_2018=ks_2018_ord,
        boundaries=boundaries,
        output_dir=OUTPUT_DIR,
    )

    if EXPORT_APPENDIX_FIGURE:
        print("Plotting appendix figure ...")
        plot_appendix_figure(
            matrix_2017=ks_2017_ord,
            matrix_2018=ks_2018_ord,
            boundaries=boundaries,
            output_dir=OUTPUT_DIR,
        )

    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
