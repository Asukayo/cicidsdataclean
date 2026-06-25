"""
Ablation study as a grouped bar chart, styled as a series-mate of the
sensitivity panel: serif type, dark-red / steel-blue palette, three
innovation groups separated by gaps, shaded middle band, italic-gray
group titles, restrained ratio annotations, no heavy bar outlines.

x-axis : 8 ablation variants, grouped by the innovation they probe.
y-axis : F1* drop w.r.t. the full model (larger = the module matters more).
2017/2018 drop ratio annotated only for the reconstruction variants
being contrasted (A4/A7 drift-targeted, bold red; A6 general, gray).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],          # swap for "Times New Roman" if available
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.9,
})

C17, C18 = "#8b3a3a", "#35688a"              # dark red = 2017, steel blue = 2018
SHADE = "#f0f0f0"

# id, two-line label, drop17, drop18, x-position, ratio-style
V = [
    ("A1",  "w/o Freq\nBranch",     0.040, 0.031, 0.0, None),
    ("A2a", "Manual\nWeights",      0.007, 0.014, 2.0, None),
    ("A2b", "Uniform\nWeights",     0.022, 0.003, 3.0, None),
    ("A3",  "w/o Spike\nBoost",     0.010, 0.008, 4.0, None),
    ("A4",  "w/o Trend\nDecomp",    0.028, 0.011, 6.0, "hi"),
    ("A5",  "w/o Proto\nBank",      0.034, 0.024, 7.0, None),
    ("A6",  "Dot-Prod\nAttn",       0.024, 0.017, 8.0, "lo"),
    ("A7",  "w/o Norm\nRetention",  0.029, 0.012, 9.0, "hi"),
]
group_titles = [(0.0, "Dual branch"),
                (3.0, "Adaptive freq. weighting"),
                (7.5, "Drift-robust reconstruction")]

w = 0.38
fig, ax = plt.subplots(figsize=(7.8, 4.2))
fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.16)

# shaded middle group
ax.axvspan(1.3, 4.7, color=SHADE, zorder=0)

for vid, lab, d17, d18, x, rs in V:
    ax.bar(x - w/2, d17, w, color=C17, zorder=3)
    ax.bar(x + w/2, d18, w, color=C18, zorder=3)
    if rs is not None:
        top = max(d17, d18)
        ax.text(x, top + 0.0016, f"{d17/d18:.1f}" + r"$\times$",
                ha="center", va="bottom",
                fontsize=9, color=(C17 if rs == "hi" else "#777777"),
                fontweight=("bold" if rs == "hi" else "normal"))

# group titles (italic gray, panel voice)
for xc, name in group_titles:
    ax.text(xc, 1.055, name, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9.5, style="italic", color="#888888")

# x labels
xticks = [v[4] for v in V]
xlabs = [f"{v[0]}\n{v[1]}" for v in V]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs, fontsize=8)
ax.tick_params(axis="x", length=0)

ax.set_xlim(-0.85, 9.9)
ax.set_ylim(0, 0.046)
ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
ax.set_ylabel(r"F1$^{*}$ drop w.r.t. full model")
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.55)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# legend (no frame) over the empty space above the short middle group
leg = ax.legend(handles=[Patch(facecolor=C17, label="CICIDS2017 (strong drift)"),
                         Patch(facecolor=C18, label="CICIDS2018 (weak drift)")],
                loc="upper center", bbox_to_anchor=(0.40, 0.99),
                frameon=False, fontsize=8.6, handlelength=1.3, labelspacing=0.4)
ax.text(0.40, 0.70, r"$n\times$ = 2017/2018 drop ratio",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2, style="italic", color="#777777")

fig.savefig("ablation_grouped.pdf", bbox_inches="tight")
fig.savefig("ablation_grouped.png", dpi=200, bbox_inches="tight")
print("saved")