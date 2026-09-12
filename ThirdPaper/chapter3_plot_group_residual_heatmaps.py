"""
论文版语义组残差漂移热力图
============================

核心计算：
1. 读取逐特征 KS 矩阵；
2. 对每个时间块减去该时间块所有有效特征的中位 KS：
       residual_j,t = KS_j,t - median_j(KS_j,t)
3. 在每个语义组内对 residual_j,t 取中位数：
       group_residual_g,t = median_{j in g}(residual_j,t)
4. 绘制 5 行语义组残差热力图。

颜色含义：
- 红色：该组在当前时间块的漂移高于全局特征中位水平；
- 蓝色：该组低于全局特征中位水平；
- 白色：接近全局水平；
- 灰色：缺失值或该时间块无足够有效数据。

可直接在 PyCharm 中运行。

依赖：
    pip install numpy pandas matplotlib

默认输入：
    ./2017feature_drift_outputs/ks_normal_only_matrix.csv
    ./2018feature_drift_outputs/ks_normal_only_matrix.csv

默认输出：
    ./paper_group_residual_heatmaps/
        figure_group_residual_heatmap.png
        figure_group_residual_heatmap.tiff
        figure_group_residual_heatmap.eps
        figure_group_residual_heatmap.pdf
        cicids2017_group_residual_matrix.csv
        cicids2018_group_residual_matrix.csv
        cicids2017_group_raw_median_matrix.csv
        cicids2018_group_raw_median_matrix.csv
        group_feature_mapping_used.csv
        missing_features_report.txt
"""

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


# ============================================================
# 论文插图统一样式
# ============================================================

# 图内所有文字、标题、坐标轴标签和刻度统一为
# 8 pt Times New Roman。
#
# 若操作系统中不存在 Times New Roman，
# Matplotlib 将按照顺序回退到其他衬线字体。
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,

    "axes.linewidth": 0.65,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    # 避免 PDF 和 EPS 使用 Type 3 字体。
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})


# ============================================================
# 1. 路径与绘图配置
# ============================================================

CSV_2017 = (
    r"./2017feature_drift_outputs/"
    r"ks_normal_only_matrix.csv"
)

CSV_2018 = (
    r"./2018feature_drift_outputs/"
    r"ks_normal_only_matrix.csv"
)

OUTPUT_DIR = r"./paper_group_residual_heatmaps"

# 截图要求建议约 300 dpi。
DPI = 600

# 7.16 inch 适合作为双栏通栏图片。
# 适当减小高度，以压缩上下两幅热力图之间的空白。
FIGSIZE = (7.16, 3.10)

MAX_X_TICKS = 8


# 色阶设置：
#
# None：
# 根据两个数据集残差绝对值的 98% 分位数
# 自动生成共享色阶。
#
# 固定数值，例如 0.20：
# 强制使用 [-0.20, 0.20]。
FIXED_ABS_VMAX = None

AUTO_VMAX_PERCENTILE = 98.0


# 避免自动色阶过窄或过宽。
MIN_AUTO_VMAX = 0.08
MAX_AUTO_VMAX = 0.35


# 缺失值使用灰色显示。
NAN_COLOR = "#D7D7D7"


# 是否裁去所有语义组均为 NaN 的头部或尾部时间块。
#
# 建议保持 True，可以避免 CICIDS2017 尾部大面积
# 无效灰色区域占用论文版面。
TRIM_ALL_NAN_EDGE_BLOCKS = True


# ============================================================
# 2. 固定语义分组
#    必须与语义分组验证脚本保持一致
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Temporal–Interarrival Time–Activity": [
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
# 3. 特征名标准化
# ============================================================

def normalize_feature_name(name: str) -> str:
    """
    统一不同 CICIDS 数据处理脚本中的常见特征命名差异。
    """
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    aliases = {
        "Fwd Packet/s": "Fwd Packets/s",
        "Bwd Packet/s": "Bwd Packets/s",

        "Total Length of Fwd Packets":
            "Fwd Packets Length Total",

        "Total Length of Bwd Packets":
            "Bwd Packets Length Total",

        "Init Fwd Win Byts":
            "Init Fwd Win Bytes",

        "Init Bwd Win Byts":
            "Init Bwd Win Bytes",

        "Fwd Act Data Pkts":
            "Fwd Act Data Packets",

        "Bwd Pkt Len Max":
            "Bwd Packet Length Max",

        "Bwd Pkt Len Min":
            "Bwd Packet Length Min",

        "Bwd Pkt Len Mean":
            "Bwd Packet Length Mean",

        "Bwd Pkt Len Std":
            "Bwd Packet Length Std",

        "Fwd Pkt Len Max":
            "Fwd Packet Length Max",

        "Fwd Pkt Len Min":
            "Fwd Packet Length Min",

        "Fwd Pkt Len Mean":
            "Fwd Packet Length Mean",

        "Fwd Pkt Len Std":
            "Fwd Packet Length Std",
    }

    return aliases.get(name, name)


# ============================================================
# 4. 数据读取与残差计算
# ============================================================

def load_ks_matrix(csv_path: str) -> pd.DataFrame:
    """
    读取逐特征 Kolmogorov–Smirnov 统计量矩阵。

    CSV 必须包含 feature 列，其余列表示按时间顺序排列的块。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 Kolmogorov–Smirnov 矩阵：{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "feature" not in df.columns:
        raise ValueError(
            f"{csv_path} 必须包含列：feature"
        )

    df["feature"] = df["feature"].map(
        normalize_feature_name
    )

    duplicated = df["feature"].duplicated()

    if duplicated.any():
        names = df.loc[
            duplicated,
            "feature",
        ].tolist()

        raise ValueError(
            f"特征名标准化后出现重复：{names}"
        )

    matrix = (
        df.set_index("feature")
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if matrix.empty:
        raise ValueError(
            f"{csv_path} 读取后为空。"
        )

    return matrix


def build_group_members(
    available_features: List[str],
) -> Tuple[
    Dict[str, List[str]],
    List[str],
]:
    """
    根据当前 CSV 中实际存在的特征构建语义组成员。

    返回：
    1. group_members：
       每个语义组实际匹配到的特征；

    2. unassigned_features：
       未被任何预定义语义组覆盖的特征。
    """
    available_set = set(
        available_features
    )

    assigned = set()

    group_members: Dict[
        str,
        List[str],
    ] = {}

    for group_name, raw_features in FEATURE_GROUPS.items():
        members = []

        for raw_name in raw_features:
            feature = normalize_feature_name(
                raw_name
            )

            if feature in available_set:
                members.append(feature)
                assigned.add(feature)

        group_members[group_name] = members

    unassigned = [
        feature
        for feature in available_features
        if feature not in assigned
    ]

    return group_members, unassigned


def compute_group_matrices(
    ks_matrix: pd.DataFrame,
    group_members: Dict[str, List[str]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    计算语义组级矩阵。

    返回：
    1. group_raw_median：
       每个语义组在每个时间块中的原始
       Kolmogorov–Smirnov 统计量中位数；

    2. group_residual：
       先减去当前时间块全部有效特征的中位数，
       再在语义组内取残差中位数。
    """
    # 每个时间块中，所有有效特征的统计量中位数。
    global_block_median = ks_matrix.median(
        axis=0,
        skipna=True,
    )

    # 每个特征相对于当前时间块全局水平的残差。
    feature_residual = ks_matrix.subtract(
        global_block_median,
        axis=1,
    )

    raw_rows = {}
    residual_rows = {}

    for group_name, members in group_members.items():
        if not members:
            raw_rows[group_name] = pd.Series(
                np.nan,
                index=ks_matrix.columns,
            )

            residual_rows[group_name] = pd.Series(
                np.nan,
                index=ks_matrix.columns,
            )

            continue

        raw_rows[group_name] = (
            ks_matrix
            .loc[members]
            .median(
                axis=0,
                skipna=True,
            )
        )

        residual_rows[group_name] = (
            feature_residual
            .loc[members]
            .median(
                axis=0,
                skipna=True,
            )
        )

    group_raw_median = pd.DataFrame(
        raw_rows
    ).T

    group_residual = pd.DataFrame(
        residual_rows
    ).T

    return (
        group_raw_median,
        group_residual,
    )


def trim_all_nan_edges(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    仅裁去矩阵左右边缘整列均为 NaN 的时间块。

    中间出现的全 NaN 时间块仍然保留，
    以真实反映时间序列中的数据缺失。
    """
    if matrix.empty:
        return matrix

    valid_columns = ~matrix.isna().all(
        axis=0
    )

    if not valid_columns.any():
        return matrix

    valid_positions = np.flatnonzero(
        valid_columns.to_numpy()
    )

    first = int(
        valid_positions[0]
    )

    last = int(
        valid_positions[-1]
    )

    return matrix.iloc[
        :,
        first:last + 1,
    ]


# ============================================================
# 5. 输出辅助信息
# ============================================================

def save_group_mapping(
    groups_2017: Dict[str, List[str]],
    groups_2018: Dict[str, List[str]],
):
    """
    保存两个数据集中实际使用的语义组与特征映射。
    """
    rows = []

    for group_name in FEATURE_GROUPS:
        union_members = sorted(
            set(
                groups_2017.get(
                    group_name,
                    [],
                )
            )
            |
            set(
                groups_2018.get(
                    group_name,
                    [],
                )
            )
        )

        for feature in union_members:
            rows.append({
                "group": group_name,
                "feature": feature,
                "in_2017": (
                    feature
                    in groups_2017.get(
                        group_name,
                        [],
                    )
                ),
                "in_2018": (
                    feature
                    in groups_2018.get(
                        group_name,
                        [],
                    )
                ),
            })

    pd.DataFrame(
        rows
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "group_feature_mapping_used.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )


def save_missing_report(
    unassigned_2017: List[str],
    unassigned_2018: List[str],
    groups_2017: Dict[str, List[str]],
    groups_2018: Dict[str, List[str]],
):
    """
    保存未分组特征和每个语义组的成员数量。
    """
    path = os.path.join(
        OUTPUT_DIR,
        "missing_features_report.txt",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Unassigned features in CICIDS2017:\n"
        )

        for feature in unassigned_2017:
            file.write(
                f"  - {feature}\n"
            )

        file.write(
            "\nUnassigned features in CICIDS2018:\n"
        )

        for feature in unassigned_2018:
            file.write(
                f"  - {feature}\n"
            )

        file.write(
            "\nGroup member counts:\n"
        )

        for group_name in FEATURE_GROUPS:
            file.write(
                f"  {group_name}: "
                f"2017="
                f"{len(groups_2017.get(group_name, []))}, "
                f"2018="
                f"{len(groups_2018.get(group_name, []))}\n"
            )


# ============================================================
# 6. 绘图
# ============================================================

def determine_shared_vmax(
    matrix_2017: pd.DataFrame,
    matrix_2018: pd.DataFrame,
) -> float:
    """
    为 CICIDS2017 和 CICIDS2018 计算共享的对称色阶范围。
    """
    if FIXED_ABS_VMAX is not None:
        if FIXED_ABS_VMAX <= 0:
            raise ValueError(
                "FIXED_ABS_VMAX 必须大于 0。"
            )

        return float(
            FIXED_ABS_VMAX
        )

    values = np.concatenate([
        matrix_2017
        .to_numpy(dtype=float)
        .ravel(),

        matrix_2018
        .to_numpy(dtype=float)
        .ravel(),
    ])

    values = np.abs(
        values[
            np.isfinite(values)
        ]
    )

    if len(values) == 0:
        return MIN_AUTO_VMAX

    vmax = float(
        np.percentile(
            values,
            AUTO_VMAX_PERCENTILE,
        )
    )

    vmax = max(
        vmax,
        MIN_AUTO_VMAX,
    )

    vmax = min(
        vmax,
        MAX_AUTO_VMAX,
    )

    return vmax


def make_xticks(
    num_blocks: int,
) -> Tuple[List[int], List[str]]:
    """
    根据时间块数量生成数量受控的横坐标刻度。
    """
    if num_blocks <= 1:
        return [0], ["0"]

    count = min(
        MAX_X_TICKS,
        num_blocks,
    )

    positions = np.unique(
        np.linspace(
            0,
            num_blocks - 1,
            count,
            dtype=int,
        )
    )

    labels = [
        str(int(position))
        for position in positions
    ]

    return (
        positions.tolist(),
        labels,
    )


def make_diverging_cmap():
    """
    创建以零为中心的红蓝发散色图。

    红色表示高于时间块全局中位水平，
    蓝色表示低于时间块全局中位水平。
    """
    cmap = plt.get_cmap(
        "RdBu_r"
    ).copy()

    cmap.set_bad(
        NAN_COLOR
    )

    return cmap


def draw_heatmap(
    ax,
    matrix: pd.DataFrame,
    title: str,
    vmax: float,
    show_xlabel: bool = True,
):
    """
    绘制一张语义组残差热力图。
    """
    masked = np.ma.masked_invalid(
        matrix.to_numpy(
            dtype=float
        )
    )

    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=make_diverging_cmap(),
        norm=TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0.0,
            vmax=vmax,
        ),
        resample=False,

        # 热力图本体栅格化，
        # 文字和坐标轴仍保留为矢量元素。
        rasterized=True,
    )

    # (a)、(b) 属于子图编号，
    # 不是整幅图片在论文中的图题。
    ax.set_title(
        title,
        loc="left",
        pad=1.2,
        fontweight="semibold",
        fontsize=8.0,
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index.tolist()
    )

    ax.set_ylabel(
        "Semantic Feature Group",
        labelpad=2.5,
    )

    xticks, xlabels = make_xticks(
        matrix.shape[1]
    )

    ax.set_xticks(
        xticks
    )

    ax.set_xticklabels(
        xlabels
    )

    # 上方的 CICIDS2017 热力图不重复显示横坐标名称，
    # 以减小两张热力图之间的空白。
    if show_xlabel:
        ax.set_xlabel(
            "Chronological Block Index",
            labelpad=2.0,
        )
    else:
        ax.set_xlabel("")

    # 使用浅色横向分隔线增强语义组行定位，
    # 同时避免遮挡热力图主体颜色。
    for y in np.arange(
        0.5,
        len(matrix.index),
        1.0,
    ):
        ax.axhline(
            y,
            linewidth=0.55,
            color="white",
            alpha=1.0,
            zorder=3,
        )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.8,
        width=0.6,
        pad=1.2,
        color="#4D4D4D",
        labelsize=8.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            0.65
        )

        spine.set_color(
            "#555555"
        )

    return image


def plot_combined_figure(
    residual_2017: pd.DataFrame,
    residual_2018: pd.DataFrame,
):
    """
    绘制 CICIDS2017 和 CICIDS2018 的组合热力图。
    """
    vmax = determine_shared_vmax(
        residual_2017,
        residual_2018,
    )

    print(
        "Shared residual color range: "
        f"[{-vmax:.4f}, {vmax:.4f}]"
    )

    # 不使用 constrained_layout，避免与手动颜色条布局冲突
    fig = plt.figure(
        figsize=(6.6, 2.75),
        facecolor="white",
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        hspace=0.28,
    )

    ax_2017 = fig.add_subplot(
        grid[0, 0]
    )

    ax_2018 = fig.add_subplot(
        grid[1, 0]
    )

    image = draw_heatmap(
        ax_2017,
        residual_2017,
        "(a) CICIDS2017",
        vmax,
        show_xlabel=False,
    )

    draw_heatmap(
        ax_2018,
        residual_2018,
        "(b) CICIDS2018",
        vmax,
        show_xlabel=True,
    )

    # 给左侧长语义组名称和右侧颜色条分别预留空间
    fig.subplots_adjust(
        left=0.27,
        right=0.88,
        top=0.95,
        bottom=0.15,
        hspace=0.28,
    )

    # 独立设置颜色条的位置：
    # [left, bottom, width, height]
    cax = fig.add_axes([
        0.905,   # 距离画布左侧的位置
        0.235,   # 距离画布底部的位置
        0.012,   # 颜色条宽度
        0.58,    # 颜色条高度
    ])

    colorbar = fig.colorbar(
        image,
        cax=cax,
        orientation="vertical",
    )

    colorbar.ax.tick_params(
        which="major",
        direction="out",
        length=2.4,
        width=0.55,
        pad=1.2,
        labelsize=8.0,
    )

    colorbar.outline.set_linewidth(
        0.6
    )

    colorbar.outline.set_edgecolor(
        "#555555"
    )

    # 不在颜色条旁放很长的竖排标题，
    # 颜色含义放在论文图注中说明
    base = os.path.join(
        OUTPUT_DIR,
        "figure_group_residual_heatmap",
    )

    fig.savefig(
        f"{base}.png",
        dpi=DPI,
    )

    fig.savefig(
        f"{base}.tiff",
        dpi=DPI,
        format="tiff",
        pil_kwargs={
            "compression": "tiff_lzw",
        },
    )

    fig.savefig(
        f"{base}.eps",
        format="eps",
    )

    fig.savefig(
        f"{base}.pdf",
        format="pdf",
    )

    plt.close(fig)

# ============================================================
# 7. 主程序
# ============================================================

def main():
    """
    程序入口。
    """
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        "Loading feature-wise "
        "Kolmogorov–Smirnov matrices ..."
    )

    ks_2017 = load_ks_matrix(
        CSV_2017
    )

    ks_2018 = load_ks_matrix(
        CSV_2018
    )

    print(
        f"CICIDS2017 matrix shape: "
        f"{ks_2017.shape}"
    )

    print(
        f"CICIDS2018 matrix shape: "
        f"{ks_2018.shape}"
    )

    groups_2017, unassigned_2017 = (
        build_group_members(
            list(ks_2017.index)
        )
    )

    groups_2018, unassigned_2018 = (
        build_group_members(
            list(ks_2018.index)
        )
    )

    raw_2017, residual_2017 = (
        compute_group_matrices(
            ks_2017,
            groups_2017,
        )
    )

    raw_2018, residual_2018 = (
        compute_group_matrices(
            ks_2018,
            groups_2018,
        )
    )

    if TRIM_ALL_NAN_EDGE_BLOCKS:
        raw_2017 = trim_all_nan_edges(
            raw_2017
        )

        residual_2017 = trim_all_nan_edges(
            residual_2017
        )

        raw_2018 = trim_all_nan_edges(
            raw_2018
        )

        residual_2018 = trim_all_nan_edges(
            residual_2018
        )

    # 保存语义组原始统计量中位数，
    # 便于论文结果复现和后续检查。
    raw_2017.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cicids2017_group_raw_median_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    raw_2018.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cicids2018_group_raw_median_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    # 保存语义组级残差矩阵。
    residual_2017.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cicids2017_group_residual_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    residual_2018.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cicids2018_group_residual_matrix.csv",
        ),
        encoding="utf-8-sig",
    )

    save_group_mapping(
        groups_2017,
        groups_2018,
    )

    save_missing_report(
        unassigned_2017,
        unassigned_2018,
        groups_2017,
        groups_2018,
    )

    print(
        "Plotting paper-ready "
        "group residual heatmap ..."
    )

    plot_combined_figure(
        residual_2017,
        residual_2018,
    )

    print(
        "Done. Outputs saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()