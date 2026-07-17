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
# Data from your ablation table
# =========================

full_2017_f1 = 0.772
full_2018_f1 = 0.728

variants = [
    "A1\nw/o Freq\nBranch",
    "A2a\nManual\nWeights",
    "A2b\nUniform\nWeights",
    "A3\nw/o Spike\nBoost",
    "A4\nw/o Trend\nDecomp",
    "A5\nw/o Proto\nBank",
    "A6\nDot-Prod\nAttn",
    "A7\nw/o\n Direction–Norm\nDecoupling",
]

f1_2017 = np.array([
    0.732,  # A1
    0.765,  # A2a
    0.750,  # A2b
    0.762,  # A3
    0.744,  # A4
    0.738,  # A5
    0.748,  # A6
    0.743,  # A7
])

f1_2018 = np.array([
    0.697,  # A1
    0.714,  # A2a
    0.725,  # A2b
    0.720,  # A3
    0.717,  # A4
    0.704,  # A5
    0.711,  # A6
    0.716,  # A7
])

drop_2017 = full_2017_f1 - f1_2017
drop_2018 = full_2018_f1 - f1_2018

# 为了制造三组之间的视觉间隔
x = np.array([0, 2, 3, 4, 6, 7, 8, 9], dtype=float)
bar_w = 0.38

c_2017 = "#8b3a3a"
c_2018 = "#35688a"

fig, ax = plt.subplots(figsize=(9.0, 4.0))
fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.25)

# =========================
# Background region
# =========================

# Adaptive freq. weighting 区域，对应 A2a, A2b, A3
ax.axvspan(1.45, 4.45, color="#efefef", zorder=0)

# =========================
# Bars
# =========================

ax.bar(
    x - bar_w / 2,
    drop_2017,
    width=bar_w,
    color=c_2017,
    edgecolor="white",
    linewidth=0.5,
    label="CICIDS2017 (heterogeneous drift)",
    zorder=3,
)

ax.bar(
    x + bar_w / 2,
    drop_2018,
    width=bar_w,
    color=c_2018,
    edgecolor="white",
    linewidth=0.5,
    label="CICIDS2018 (near-uniform drift)",
    zorder=3,
)

# =========================
# Axis style
# =========================

ax.set_ylabel(r"F1$^*$ drop w.r.t. full model")
ax.set_xticks(x)
ax.set_xticklabels(variants)

ax.set_ylim(0, 0.046)
ax.set_yticks(np.arange(0, 0.041, 0.01))

ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# =========================
# Group titles
# =========================

ax.text(
    0.5,
    1.045,
    "Dual branch",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="bottom",
    fontsize=10,
    fontstyle="italic",
    color="#777777",
)

ax.text(
    3.0,
    1.045,
    "Adaptive freq. weighting",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="bottom",
    fontsize=10,
    fontstyle="italic",
    color="#777777",
)

ax.text(
    7.5,
    1.045,
    "Drift-robust reconstruction",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="bottom",
    fontsize=10,
    fontstyle="italic",
    color="#777777",
)

# =========================
# Legend and annotation
# =========================

ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.94, 0.96),
    frameon=False,
    fontsize=9,
    handlelength=1.4,
    borderaxespad=0.2,
)

ax.text(
    3.1,
    0.0305,
    r"$n \times$ = 2017/2018 drop ratio",
    ha="center",
    va="bottom",
    fontsize=9,
    fontstyle="italic",
    color="#777777",
)

# 只标注你第一张图中出现的三个 ratio
ratio_indices = {
    4: (r"2.5$\times$", 0.0292),  # A4: 0.028 / 0.011
    6: (r"1.4$\times$", 0.0292),  # A6: 0.024 / 0.017
    7: (r"2.4$\times$", 0.0337),  # A7: 0.029 / 0.012
}

for idx, (txt, y_pos) in ratio_indices.items():
    ax.text(
        x[idx],
        y_pos,
        txt,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold" if idx in [4, 7] else "normal",
        color=c_2017 if idx in [4, 7] else "#777777",
    )

# =========================
# Save
# =========================

out_path = Path(__file__).resolve().parent / "ablation_f1_drop.png"
fig.savefig(out_path, dpi=900, bbox_inches="tight")
print(f"saved to {out_path}")

plt.show()