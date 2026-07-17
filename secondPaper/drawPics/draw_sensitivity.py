from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.9,
})

# =========================
# Data from your table
# =========================

groups = [
    {
        "title": r"Prototypes $K$",
        "labels": ["8", "16", "32", "64"],
        "auc": [0.809, 0.817, 0.816, 0.815],
        "f1":  [0.761, 0.772, 0.773, 0.768],
        "default_idx": 1,   # K = 16
    },
    {
        "title": r"Kernel size $K_{lp}$",
        "labels": ["10", "25", "40", "50"],
        "auc": [0.811, 0.817, 0.816, 0.814],
        "f1":  [0.763, 0.772, 0.770, 0.767],
        "default_idx": 1,   # K_lp = 25
    },
    {
        "title": r"Temperature $\tau$",
        "labels": ["0.05", "0.1", "0.2", "0.5"],
        "auc": [0.813, 0.817, 0.816, 0.808],
        "f1":  [0.764, 0.772, 0.771, 0.756],
        "default_idx": 1,   # tau = 0.1
    },
]

C_F1 = "#8b3a3a"
C_AUC = "#35688a"

# 默认配置对应的性能
DEFAULT_F1 = 0.772
DEFAULT_AUC = 0.817

# 构造横坐标：三组之间留空隙
x_groups = [
    np.array([0, 1, 2, 3]),
    np.array([5, 6, 7, 8]),
    np.array([10, 11, 12, 13]),
]

x_all = np.concatenate(x_groups)
x_labels = sum([g["labels"] for g in groups], [])

fig, (ax_f1, ax_auc) = plt.subplots(
    2,
    1,
    figsize=(7.0, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
)

fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.14)

# =========================
# Background shading
# =========================

# 中间 Kernel size 区域加灰色背景
for ax in [ax_f1, ax_auc]:
    ax.axvspan(4.45, 8.55, color="#efefef", zorder=0)

# =========================
# Plot function
# =========================

def plot_panel(ax, metric_key, color, default_y):
    for x, group in zip(x_groups, groups):
        y = np.array(group[metric_key])

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            zorder=3,
        )

        # 默认配置点使用实心圆
        default_x = x[group["default_idx"]]
        default_val = y[group["default_idx"]]

        ax.scatter(
            default_x,
            default_val,
            s=32,
            color=color,
            zorder=4,
        )

    # 默认性能水平线
    ax.axhline(
        default_y,
        color="#999999",
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        zorder=1,
    )

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# =========================
# Draw panels
# =========================

plot_panel(ax_f1, "f1", C_F1, DEFAULT_F1)
plot_panel(ax_auc, "auc", C_AUC, DEFAULT_AUC)

# =========================
# Axis settings
# =========================

ax_f1.set_ylabel(r"F1$^*$")
ax_auc.set_ylabel("AUC-ROC")

ax_f1.set_ylim(0.748, 0.781)
ax_f1.set_yticks([0.750, 0.760, 0.770, 0.780])

ax_auc.set_ylim(0.805, 0.822)
ax_auc.set_yticks([0.805, 0.810, 0.815, 0.820])

ax_auc.set_xticks(x_all)
ax_auc.set_xticklabels(x_labels)

# =========================
# Group titles
# =========================

for x, group in zip(x_groups, groups):
    center = x.mean()
    ax_f1.text(
        center,
        1.05,
        group["title"],
        transform=ax_f1.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=11,
    )

# =========================
# Default annotation
# =========================

ax_f1.text(
    12.55,
    DEFAULT_F1 + 0.001,
    "default",
    ha="left",
    va="bottom",
    fontsize=8.5,
    fontstyle="italic",
    color="#777777",
)

ax_auc.text(
    12.55,
    DEFAULT_AUC + 0.0003,
    "default",
    ha="left",
    va="bottom",
    fontsize=8.5,
    fontstyle="italic",
    color="#777777",
)

# Temperature tau = 0.5 的下降标注
ax_f1.text(
    12.45,
    0.751,
    r"$-0.016$",
    ha="center",
    va="center",
    fontsize=9,
    color=C_F1,
)

# =========================
# Save
# =========================

out_path = Path(__file__).resolve().parent / "hyperparameter_sensitivity.png"
fig.savefig(out_path, dpi=900, bbox_inches="tight")
print(f"saved to {out_path}")

plt.show()