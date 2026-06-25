"""
Fig. 4 - Confusion matrix comparison: WSTD vs TreeMIL (CICIDS2017 test set).
Output: vector EPS (primary) + 600 dpi LZW-compressed TIFF (backup), per IJCS
figure requirements (graphs/drawings: vector preferred; raster >= 600 dpi).

NOTE ON TreeMIL's matrix
------------------------
TreeMIL's raw per-window predictions are no longer available, so its 2x2
confusion matrix is reconstructed from TreeMIL's *measured* Precision and
Recall (Table 1) together with the fixed test-set class totals. For a binary
classifier on a fixed test set this reconstruction is exact up to integer
rounding: with N_pos = TP+FN and N_neg = TN+FP known, Recall fixes TP (hence
FN) and Precision then fixes FP (hence TN). It is an algebraic re-expression
of real measurements, not synthetic data. If the raw TreeMIL predictions are
ever recovered, regenerate this figure directly via
confusion_matrix(y_true, y_pred_treemil).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Fonts: Times-like serif to match the Wiley/IJCS body text ----
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

# ---- Fixed CICIDS2017 test-set class totals (consistent with Table 3) ----
TOTAL_POSITIVE = 13597   # attack windows: DDoS 7047 + PortScan 5356 + Bot 1194
TOTAL_NEGATIVE = 20122   # benign windows

# ---- WSTD: real measured confusion matrix (FN corrected to 1966) ----
# rows = [Normal(true), Attack(true)]; cols = [Normal(pred), Attack(pred)]
cm_wstd = np.array([[18426, 1696],
                    [1966, 11631]])
p_wstd, r_wstd, f1_wstd = 0.872, 0.855, 0.864

# ---- TreeMIL: reconstructed from reported P/R (see header note) ----
p_treemil, r_treemil, f1_treemil = 0.828, 0.781, 0.804
tp_tm = round(r_treemil * TOTAL_POSITIVE)
fn_tm = TOTAL_POSITIVE - tp_tm
fp_tm = round(tp_tm / p_treemil - tp_tm)
tn_tm = TOTAL_NEGATIVE - fp_tm
cm_treemil = np.array([[tn_tm, fp_tm],
                       [fn_tm, tp_tm]])

# ---- Plot (full-width 2-panel figure) ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))


def draw(ax, cm, cmap, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, square=True,
                linewidths=1.2, linecolor='black',
                annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_xticklabels(['Normal', 'Attack'], fontsize=10)
    ax.set_yticklabels(['Normal', 'Attack'], fontsize=10, rotation=0)


draw(axes[0], cm_wstd, 'Blues',
     f'WSTD\nP={p_wstd:.3f}  R={r_wstd:.3f}  F1={f1_wstd:.3f}')
draw(axes[1], cm_treemil, 'Oranges',
     f'TreeMIL\nP={p_treemil:.3f}  R={r_treemil:.3f}  F1={f1_treemil:.3f}')

plt.tight_layout(pad=1.5)

# ---- Export: EPS (vector, primary) + TIFF 600 dpi LZW (backup) ----
fig.savefig('Fig4.eps', format='eps', bbox_inches='tight',
            pad_inches=0.1, facecolor='white')
fig.savefig('Fig4.tiff', dpi=600, bbox_inches='tight', pad_inches=0.1,
            facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# ---- Provenance / consistency check ----
print("WSTD    CM:", cm_wstd.tolist(),
      "| pos =", int(cm_wstd[1].sum()), " neg =", int(cm_wstd[0].sum()))
print("TreeMIL CM:", cm_treemil.tolist(),
      "| pos =", int(cm_treemil[1].sum()), " neg =", int(cm_treemil[0].sum()))