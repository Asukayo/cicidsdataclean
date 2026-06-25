import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm

# 中文字体（无此字体时改成系统里任一 CJK 字体路径，或删掉 fontproperties 参数用英文标签）
cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(cjk_path)
cjk = fm.FontProperties(fname=cjk_path)
mpl.rcParams['axes.unicode_minus'] = False

# 基线：(方法名, 推理延迟 ms, F1 on CICIDS2017)
data = [("MLP", 0.04, 0.749), ("LSTM-AE", 1.91, 0.763), ("TCN", 2.22, 0.769),
        ("LSTM", 2.45, 0.755), ("Transformer", 3.98, 0.764),
        ("FEDformer", 4.43, 0.776), ("TreeMIL", 15.40, 0.804)]
ours = ("WSTD", 3.58, 0.864)

NAVY, ORANGE, GRAY, INK = "#1F4E79", "#C65911", "#7A8AA0", "#20303F"

fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=200)

# 基线散点
for n, x, f in data:
    ax.scatter(x, f, s=66, c=GRAY, edgecolors='white', linewidths=1.0, zorder=3)

# 基线标签位置微调（offset points）
offsets = {"MLP": (7, -4), "LSTM-AE": (-2, 9), "TCN": (7, 5), "LSTM": (8, -12),
           "Transformer": (9, -3), "FEDformer": (8, 4), "TreeMIL": (-4, -15)}
for n, x, f in data:
    dx, dy = offsets.get(n, (6, 6))
    fs = 10.5 if n == "TreeMIL" else 8.5
    col = INK if n == "TreeMIL" else GRAY
    fw = 'bold' if n == "TreeMIL" else 'normal'
    ha = 'right' if n == "TreeMIL" else 'left'
    ax.annotate(n, (x, f), textcoords="offset points", xytext=(dx, dy),
                fontsize=fs, color=col, fontweight=fw, ha=ha, fontproperties=cjk)

# 本文方法：橙色星
ax.scatter(*ours[1:], s=300, marker='*', c=ORANGE,
           edgecolors='white', linewidths=1.3, zorder=5)
ax.annotate("WSTD（本文）", (ours[1], ours[2]), textcoords="offset points",
            xytext=(12, 4), fontsize=12, color=ORANGE, fontweight='bold',
            fontproperties=cjk)

# 最优区提示
ax.text(0.3, 0.97, "← 越靠左上越优（又快又准）", fontsize=9, color=NAVY,
        alpha=0.8, fontproperties=cjk, va='top')

ax.set_xlabel("单窗口推理延迟 (ms)", fontsize=10.5, color=INK, fontproperties=cjk)
ax.set_ylabel("F1 分数 (CICIDS2017)", fontsize=10.5, color=INK, fontproperties=cjk)
ax.set_xlim(-0.8, 17.2)
ax.set_ylim(0.6, 1.0)                      # 纵轴全程，诚实比例；想紧凑可改 (0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, linestyle='--', linewidth=0.5, color='#E6EAF0', zorder=0)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color('#C5CDD8')
ax.tick_params(colors=INK, labelsize=8.5)

plt.tight_layout()
plt.savefig('perf_eff_scatter.png', dpi=400, bbox_inches='tight', facecolor='white')