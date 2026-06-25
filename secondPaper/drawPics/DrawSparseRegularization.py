import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch


# ============================================================
# 1. Output path
# ============================================================

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "Fig5_Sparsity_Weights_Before_After_clean.png"
pdf_path = out_dir / "Fig5_Sparsity_Weights_Before_After_clean.pdf"
svg_path = out_dir / "Fig5_Sparsity_Weights_Before_After_clean.svg"


# ============================================================
# 2. Simulate prototype attention weights
#    A = softmax(cos(z_norm, m_norm) / tau)
# ============================================================

K = 16
idx = np.arange(1, K + 1)

# 模拟某个时间步 z_{b,t} 与 16 个 prototype 的余弦相似度
# 2、8、13 号 prototype 与当前表示方向更接近
cos_sim = np.array([
    0.18, 0.64, 0.22, 0.10,
    0.27, 0.14, 0.31, 0.58,
    0.20, 0.16, 0.25, 0.12,
    0.49, 0.21, 0.15, 0.09
])


def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


# tau_before 大：注意力更分散
# tau_after 小：模拟稀疏正则后的尖锐分布
tau_before = 0.85
tau_after = 0.18

w_before = softmax(cos_sim / tau_before)
w_after = softmax(cos_sim / tau_after)

H_before = -(w_before * np.log(w_before + 1e-12)).sum()
H_after = -(w_after * np.log(w_after + 1e-12)).sum()


# ============================================================
# 3. Draw figure
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.0,
})

fig = plt.figure(figsize=(10.8, 5.2), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

fig.suptitle(
    "Sparsity Regularization on Prototype Attention Weights",
    fontsize=15,
    fontweight="bold",
    y=0.975
)

# 中间列稍微加宽，用来放箭头、文字和图例
gs = fig.add_gridspec(
    1, 3,
    left=0.075,
    right=0.965,
    bottom=0.14,
    top=0.78,
    width_ratios=[1.0, 0.30, 1.0]
)

ax1 = fig.add_subplot(gs[0, 0])
ax_mid = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)


# ============================================================
# 4. Style
# ============================================================

bar_color_before = "#B9A2D8"
bar_edge_before = "#8D6BBE"

bar_color_after = "#F2A6A6"
bar_edge_after = "#D55E5E"

highlight_color = "#E76F51"

ylim = max(w_after.max(), w_before.max()) * 1.18


# ============================================================
# 5. Left panel: without sparsity
# ============================================================

ax1.bar(
    idx,
    w_before,
    width=0.72,
    color=bar_color_before,
    edgecolor=bar_edge_before,
    linewidth=0.8
)

ax1.set_title(
    "Without sparsity\n(diffuse weights)",
    fontsize=12,
    fontweight="bold",
    pad=10
)

ax1.set_xlabel("Prototype index $k$", fontsize=11)
ax1.set_ylabel("Attention weight $A_{b,t,k}$", fontsize=11)
ax1.set_xticks(idx)
ax1.set_ylim(0, ylim)
ax1.grid(axis="y", alpha=0.25)

ax1.text(
    0.96,
    0.93,
    rf"$H(A)={H_before:.2f}$",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.30",
        fc="white",
        ec="#DDDDDD",
        alpha=0.96
    )
)


# ============================================================
# 6. Middle panel: arrow + legend
# ============================================================

ax_mid.axis("off")

ax_mid.text(
    0.50,
    0.64,
    "minimize\nentropy",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#333333"
)

ax_mid.annotate(
    "",
    xy=(0.82, 0.48),
    xytext=(0.18, 0.48),
    arrowprops=dict(
        arrowstyle="simple",
        color="#555555",
        lw=0,
        mutation_scale=26
    )
)

# 图例放在两图中间下方，不再占用顶部空间
handles = [
    Patch(
        facecolor=bar_color_before,
        edgecolor=bar_edge_before,
        label="Diffuse weights"
    ),
    Patch(
        facecolor=bar_color_after,
        edgecolor=bar_edge_after,
        label="Suppressed weights"
    ),
    Patch(
        facecolor=highlight_color,
        edgecolor=bar_edge_after,
        label="Dominant prototypes"
    ),
]

ax_mid.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.08),
    frameon=True,
    fontsize=8.8,
    edgecolor="#DDDDDD",
    handlelength=1.7,
    borderpad=0.6,
    labelspacing=0.6
)


# ============================================================
# 7. Right panel: with sparsity regularization
# ============================================================

top_ids = np.argsort(w_after)[-3:]

colors = [bar_color_after] * K
for i in top_ids:
    colors[i] = highlight_color

ax2.bar(
    idx,
    w_after,
    width=0.72,
    color=colors,
    edgecolor=bar_edge_after,
    linewidth=0.8
)

ax2.set_title(
    "With sparsity regularization\n(sharp weights)",
    fontsize=12,
    fontweight="bold",
    pad=10
)

ax2.set_xlabel("Prototype index $k$", fontsize=11)

# 右图不再重复 y 轴标签，避免和中间区域挤压
ax2.set_ylabel("")
ax2.tick_params(axis="y", labelleft=False)

ax2.set_xticks(idx)
ax2.set_ylim(0, ylim)
ax2.grid(axis="y", alpha=0.25)

ax2.text(
    0.96,
    0.93,
    rf"$H(A)={H_after:.2f}$",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.30",
        fc="white",
        ec="#DDDDDD",
        alpha=0.96
    )
)

# 标出 dominant prototypes
for i in top_ids:
    ax2.annotate(
        f"$m_{{{i + 1}}}$",
        xy=(i + 1, w_after[i]),
        xytext=(i + 1, w_after[i] + 0.035),
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="-",
            lw=0.8,
            color="#555555"
        )
    )


# ============================================================
# 8. Save
# ============================================================

fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)

plt.close(fig)

print("Saved:")
print(f"- {png_path}")
print(f"- {pdf_path}")
print(f"- {svg_path}")