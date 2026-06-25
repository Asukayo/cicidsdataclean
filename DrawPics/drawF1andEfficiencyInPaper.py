import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Journal-style typography (serif, mathtext) ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,   # embed TrueType (editable text in PDF, required by many journals)
    "ps.fonttype": 42,
})

# Baselines: (name, inference latency [ms], F1 on CICIDS2017)
data = [("MLP", 0.04, 0.749), ("LSTM-AE", 1.91, 0.763), ("TCN", 2.22, 0.769),
        ("LSTM", 2.45, 0.755), ("Transformer", 3.98, 0.764),
        ("FEDformer", 4.43, 0.776), ("TreeMIL", 15.40, 0.804)]
ours = ("WSTD", 3.58, 0.864)

NAVY, ORANGE, GRAY, INK = "#1F4E79", "#C65911", "#7A8AA0", "#20303F"

fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)

# Baseline points (gray circles)
for n, x, f in data:
    ax.scatter(x, f, s=58, c=GRAY, marker='o',
               edgecolors='white', linewidths=0.9, zorder=3)

# Baseline labels
offsets = {"MLP": (7, -4), "LSTM-AE": (-2, 9), "TCN": (7, 5), "LSTM": (8, -12),
           "Transformer": (9, -3), "FEDformer": (8, 4), "TreeMIL": (-4, -15)}
for n, x, f in data:
    dx, dy = offsets.get(n, (6, 6))
    fs = 9.5 if n == "TreeMIL" else 8
    col = INK if n == "TreeMIL" else GRAY
    fw = 'bold' if n == "TreeMIL" else 'normal'
    ha = 'right' if n == "TreeMIL" else 'left'
    ax.annotate(n, (x, f), textcoords="offset points", xytext=(dx, dy),
                fontsize=fs, color=col, fontweight=fw, ha=ha)

# Our method: orange star (distinct marker -> survives grayscale printing)
ax.scatter(*ours[1:], s=260, marker='*', c=ORANGE,
           edgecolors='white', linewidths=1.1, zorder=5)
ax.annotate("WSTD (Ours)", (ours[1], ours[2]), textcoords="offset points",
            xytext=(11, 4), fontsize=11, color=ORANGE, fontweight='bold')

# Neutral "better" hint (acceptable in papers)
ax.annotate("Better", xy=(0.6, 0.93), xytext=(4.2, 0.965),
            fontsize=8.5, color=NAVY, alpha=0.8, style='italic',
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.1, alpha=0.6))

ax.set_xlabel("Inference Latency per Window (ms)", fontsize=10, color=INK)
ax.set_ylabel("F1-Score (CICIDS2017)", fontsize=10, color=INK)
ax.set_xlim(-0.8, 17.2)
ax.set_ylim(0.6, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, linestyle='--', linewidth=0.5, color='#E6EAF0', zorder=0)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color('#C5CDD8')
ax.tick_params(colors=INK, labelsize=8.5)

plt.tight_layout()
plt.savefig('perf_eff_scatter.pdf', bbox_inches='tight', facecolor='white')   # vector, preferred
plt.savefig('perf_eff_scatter.png', dpi=600, bbox_inches='tight', facecolor='white')  # raster backup