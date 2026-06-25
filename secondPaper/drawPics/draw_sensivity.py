"""
Hyper-parameter sensitivity panel (FreqDAR, CICIDS2017).
Two stacked sub-plots sharing the x-axis:
  top    = F1*       (dark red)
  bottom = AUC-ROC   (steel blue)
Three parameter groups (K / kernel size / tau) laid out left-to-right with a
shaded middle band; the default setting in each group is drawn as a filled dot.

Data are taken directly from the sensitivity table; no source data required.
Outputs a vector PDF + a PNG preview.
"""

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],          # swap for "Times New Roman" if available
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.9,
})

C_F1  = "#8b3a3a"     # dark red  (F1*)
C_AUC = "#35688a"     # steel blue (AUC-ROC)
SHADE = "#efefef"     # alternating-band background
GRID  = dict(axis="y", ls=":", lw=0.6, alpha=0.55)

# ----------------------------------------------------------------------
# Data  (each group = 4 settings; index 1 is the default)
# ----------------------------------------------------------------------
groups = [
    dict(title=r"Prototypes $K$",
         x=[0, 1, 2, 3], labels=["8", "16", "32", "64"], default=1,
         f1=[0.761, 0.772, 0.773, 0.768], auc=[0.809, 0.817, 0.816, 0.815]),
    dict(title=r"Kernel size $K_{lp}$",
         x=[5, 6, 7, 8], labels=["10", "25", "40", "50"], default=1,
         f1=[0.763, 0.772, 0.770, 0.767], auc=[0.811, 0.817, 0.816, 0.814]),
    dict(title=r"Temperature $\tau$",
         x=[10, 11, 12, 13], labels=["0.05", "0.1", "0.2", "0.5"], default=1,
         f1=[0.764, 0.772, 0.771, 0.756], auc=[0.813, 0.817, 0.816, 0.808]),
]

F1_DEFAULT, AUC_DEFAULT = 0.772, 0.817
SHADE_SPAN = (4.3, 8.7)        # x-range of the shaded middle band

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
fig.subplots_adjust(left=0.11, right=0.965, top=0.90, bottom=0.085, hspace=0.12)


def draw(ax, key, color, y_default, ylim, yticks, ylabel):
    # shaded middle band + default reference line
    ax.axvspan(*SHADE_SPAN, color=SHADE, zorder=0)
    ax.axhline(y_default, color="#a9a9a9", ls=(0, (5, 4)), lw=0.9, zorder=1)

    for g in groups:
        xs, ys, d = np.array(g["x"]), np.array(g[key]), g["default"]
        ax.plot(xs, ys, color=color, lw=1.8, zorder=3)
        # non-default points: hollow markers
        mask = np.ones(len(xs), bool); mask[d] = False
        ax.scatter(xs[mask], ys[mask], s=46, marker="o",
                   facecolor="white", edgecolor=color, linewidth=1.5, zorder=4)
        # default point: filled marker
        ax.scatter(xs[d], ys[d], s=52, marker="o",
                   facecolor=color, edgecolor=color, linewidth=1.5, zorder=5)

    ax.text(13.9, y_default, "default", fontsize=8.5, style="italic",
            color="#888888", va="center", ha="left")

    ax.set_xlim(-0.8, 14.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


draw(ax1, "f1", C_F1, F1_DEFAULT, (0.748, 0.781),
     [0.75, 0.76, 0.77, 0.78], r"F1$^{*}$")
draw(ax2, "auc", C_AUC, AUC_DEFAULT, (0.804, 0.8215),
     [0.805, 0.810, 0.815, 0.820], "AUC-ROC")

# group titles above the top panel
for g in groups:
    xc = np.mean(g["x"])
    ax1.text(xc, 1.045, g["title"], transform=ax1.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=11)

# drop annotation on the most sensitive point (tau = 0.5), deviation from default
ax1.annotate(r"$-0.016$", xy=(13, 0.756), xytext=(13, 0.7525),
             ha="center", va="top", fontsize=8.5, color=C_F1)

# x tick labels = real hyper-parameter values
all_x = [xi for g in groups for xi in g["x"]]
all_lab = [l for g in groups for l in g["labels"]]
ax2.set_xticks(all_x)
ax2.set_xticklabels(all_lab, fontsize=9)
ax2.tick_params(axis="x", length=3)

fig.savefig("sensitivity_panel_repro.pdf", bbox_inches="tight")
fig.savefig("sensitivity_panel_repro.png", dpi=200, bbox_inches="tight")
print("saved")