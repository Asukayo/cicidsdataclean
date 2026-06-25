"""
Neutral accuracy-vs-cost view, series-consistent with the sensitivity panel:
two side-by-side scatters sharing the y-axis (F1, CICIDS2017).
  left  : F1 vs inference latency (ms, log)
  right : F1 vs GPU memory (MB, log)
Hollow steel-blue circles = baselines; filled dark-red = FreqDAR.
A light dashed line marks FreqDAR's F1; thin dotted connectors give the
exact FreqDAR-TransDe cost ratios. No rhetorical shading.
"""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.9,
})

C_MAIN, C_BASE = "#8b3a3a", "#35688a"

# name, latency(ms), mem(MB), F1, (dx,dy,ha) for panel A, (dx,dy,ha) for panel B
D = [
    ("USAD",        0.13,  21.7, 0.586, ( 7, -3, "left"),  ( 7, -3, "left")),
    ("OmniAnomaly", 0.44, 122.1, 0.664, (-7,  4, "right"), ( 8, -4, "left")),
    ("LSTM-AE",     0.52, 151.3, 0.554, ( 7, -3, "left"),  ( 7, -3, "left")),
    ("MemAE",       0.53,  25.1, 0.637, ( 7, -7, "left"),  ( 8, -5, "left")),
    ("STFT-TCAN",   0.72,  24.9, 0.669, (-2, 9, "center"), (-2,  9, "center")),
    ("DTAAD",       1.29,  81.3, 0.674, ( 8, -7, "left"),  (-2,  9, "center")),
    ("TranAD",      2.17, 108.3, 0.653, ( 0,-14, "center"),( 0,-14, "center")),
    ("FreqDAR",     1.21,  35.8, 0.721, (-4, 10, "center"),( 0, 10, "center")),
    ("TransDe",     8.92, 886.2, 0.697, ( 0, 10, "center"),(-9,  0, "right")),
]
F1_BEST = 0.721

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
fig.subplots_adjust(left=0.095, right=0.975, top=0.93, bottom=0.165, wspace=0.06)

def panel(ax, xi, label_i, xlim, xticks, xlab):
    ax.axhline(F1_BEST, color="#b3b8bd", ls=(0, (5, 4)), lw=0.8, zorder=1)
    for d in D:
        name, x, f1 = d[0], d[xi], d[3]
        dx, dy, ha = d[label_i]
        if name == "FreqDAR":
            ax.scatter(x, f1, s=62, facecolor=C_MAIN, edgecolor=C_MAIN,
                       linewidth=1.4, zorder=5)
            ax.annotate(name, (x, f1), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=9,
                        color=C_MAIN, fontweight="bold")
        else:
            ax.scatter(x, f1, s=56, facecolor="white", edgecolor=C_BASE,
                       linewidth=1.4, zorder=4)
            ax.annotate(name, (x, f1), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.8, color="#444444")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlabel(xlab)
    ax.grid(ls=":", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

panel(axA, 1, 4, (0.085, 16), [0.1, 0.3, 1, 3, 10],
      "Inference latency (ms, log)")
panel(axB, 2, 5, (14, 1500), [20, 50, 100, 300, 1000],
      "GPU memory (MB, log)")

axA.set_ylim(0.532, 0.748)
axA.set_yticks([0.55, 0.60, 0.65, 0.70])
axA.set_ylabel(r"F1 (CICIDS2017)")

# FreqDAR-TransDe ratio connectors (factual, italic gray)
axA.annotate("", xy=(1.5, 0.7185), xytext=(7.8, 0.7005),
             arrowprops=dict(arrowstyle="-", ls=":", lw=0.9, color="#9a9a9a"))
axA.text(3.4, 0.713, r"7.4$\times$", fontsize=8, style="italic",
         color="#777777", ha="center")
axB.annotate("", xy=(42, 0.7185), xytext=(740, 0.7005),
             arrowprops=dict(arrowstyle="-", ls=":", lw=0.9, color="#9a9a9a"))
axB.text(170, 0.7135, r"24.8$\times$", fontsize=8, style="italic",
         color="#777777", ha="center")

# "better" cue, panel A only
axA.annotate("", xy=(0.05, 0.965), xytext=(0.155, 0.875), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="#999999"))
axA.text(0.165, 0.882, "better", transform=axA.transAxes,
         fontsize=8.5, style="italic", color="#999999", va="center")

fig.savefig("/efficiency_2panel.pdf", bbox_inches="tight")
fig.savefig("/efficiency_2panel.png", dpi=200, bbox_inches="tight")
print("saved")