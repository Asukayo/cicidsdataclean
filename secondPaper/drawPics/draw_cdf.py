"""


关键设定（经审稿视角修正）：
  - val 的正常/异常分布【显著重叠】→ val 本身 F1<1（符合主表量级），
    而非完美可分。
  - 阈值【数据驱动】：在分数上真跑 grid-search 最优 F1 得到 τ_val / τ_test*，
    并据此算 F1（用 τ_val 评 test）与 F1*（test 上最优），Δ=F1*-F1。
  - 协议是 val(含异常) vs test(含异常)，两条 CDF 都带异常尾，不照搬
    参考文献 train(纯正常)的极陡形状。
  - 形状差异来自漂移对 test 正常段【均值+离散度】的双重抬升（非平移克隆）；
    异常段仅留数据固有的自然小缝（两模型相同）。
  - 横轴聚焦阈值区，高分异常尾段从略。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

C_VAL, C_TEST = "#2c6fbb", "#c0392b"      # val 蓝, test 红


def _cdf(x):
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


# ---------- 合成分数：正常 + 异常 两群，带标签 ----------
# 多模态混合：正常 = 多种正常流量模式；异常 = 多种攻击类型（末项宽尾 = 强/罕见异常）
# 每个成分 = (loc, scale, weight)，样本 ~ gamma(2.0, scale) + loc
_NORMAL_COMPS = [(0.14, 0.045, 0.50), (0.20, 0.055, 0.32), (0.27, 0.045, 0.18)]
_ANOM_COMPS   = [(0.36, 0.070, 0.45), (0.46, 0.100, 0.35), (0.58, 0.150, 0.20)]


def _mixture(rng, n, comps, drift, scale_mul):
    """从多成分混合中抽 n 个样本；drift 加到每个 loc，scale_mul 乘到每个 scale。"""
    ws = np.array([w for *_, w in comps], float); ws /= ws.sum()
    cnt = rng.multinomial(n, ws)
    return np.concatenate([rng.gamma(2.0, sc * scale_mul, k) + lo + drift
                           for k, (lo, sc, _) in zip(cnt, comps)])


def _synth_scores(rng, n_normal, n_anom, drift_n=0.0, spread_n=1.0, drift_a=0.0):
    """多模态分数：正常多模式 + 异常多攻击类型。漂移作用于正常成分(loc+drift_n,
    scale*spread_n)与异常成分(loc+drift_a)。返回 (scores, labels)。"""
    normal = _mixture(rng, n_normal, _NORMAL_COMPS, drift_n, spread_n)
    anom   = _mixture(rng, n_anom,   _ANOM_COMPS,   drift_a, 1.0)
    scores = np.concatenate([normal, anom])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    return scores, labels


def _best_f1_threshold(scores, labels, n_steps=600):
    ts = np.linspace(scores.min(), scores.max(), n_steps)
    best_f1, best_t = -1.0, float(ts[0])
    for t in ts:
        pred = scores > t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _f1_at(scores, labels, t):
    pred = scores > t
    tp = np.sum(pred & (labels == 1))
    fp = np.sum(pred & (labels == 0))
    fn = np.sum(~pred & (labels == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def _panel(ax, val, test, tau_val, tau_test, title, delta_label, xlim=None):
    xs, ys = _cdf(val);  ax.plot(xs, ys, color=C_VAL,  lw=2.2, label="Val")
    xs, ys = _cdf(test); ax.plot(xs, ys, color=C_TEST, lw=2.2, label="Test")
    ax.axvline(tau_val,  color=C_VAL,  ls="--", lw=1.4)
    ax.axvline(tau_test, color=C_TEST, ls="--", lw=1.4)
    ax.text(tau_val - 0.010,  1.04, r"$\tau_{val}$",      color=C_VAL,  ha="right", fontsize=10)
    ax.text(tau_test + 0.010, 1.04, r"$\tau^{*}_{test}$", color=C_TEST, ha="left",  fontsize=10)
    y_arr = 0.40
    arr = FancyArrowPatch((tau_val, y_arr), (tau_test, y_arr),
                          arrowstyle="<->", color="0.25", lw=1.6, mutation_scale=14)
    ax.add_patch(arr)
    ax.text((tau_val + tau_test) / 2, y_arr + 0.04,
            r"$\Delta\theta$" + f"\n({delta_label})", ha="center", va="bottom",
            fontsize=9.5, color="0.2")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.12)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)


def make_schematic(path="anomaly_score_cdf_schematic.png"):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # 验证集/测试集窗口数（贴近真实量级，留出经验 CDF 的局部阶梯感）
    N_NORMAL, N_ANOM = 900, 320

    def one(ax, drift_n, spread_n, drift_a, title, dl):
        rng = np.random.default_rng(7)   # 每子图 fresh：两图共享同一验证集抽样与 τ_val 起点
        v_scores, v_labels = _synth_scores(rng, N_NORMAL, N_ANOM)
        t_scores, t_labels = _synth_scores(rng, N_NORMAL, N_ANOM,
                                           drift_n=drift_n, spread_n=spread_n,
                                           drift_a=drift_a)
        tau_val,  _    = _best_f1_threshold(v_scores, v_labels)   # 数据驱动阈值
        f1_deploy      = _f1_at(t_scores, t_labels, tau_val)      # 用 τ_val 评 test
        tau_test, f1s  = _best_f1_threshold(t_scores, t_labels)   # test 最优
        print(f"{title.splitlines()[0]:<28} "
              f"tau_val={tau_val:.3f}  tau_test*={tau_test:.3f}  "
              f"F1={f1_deploy:.3f}  F1*={f1s:.3f}  Delta={f1s - f1_deploy:.3f}")
        _panel(ax, v_scores, t_scores, tau_val, tau_test, title, dl, xlim=(0.10, 0.60))

    # 左：LSTM-AE —— 正常段右移并适度摊平（仍可分→F1*不塌）；异常段自然小缝
    one(axL, 0.13, 1.18, 0.04,
        "(a) LSTM-AE  (large $\\Delta$)\nval/test distributions diverge",
        "wide → poor transfer")
    # 右：FreqDAR —— 正常段小右移、几乎不摊；异常段自然小缝（与 a 相同）
    one(axR, 0.04, 1.08, 0.04,
        "(b) FreqDAR  (small $\\Delta$)\nval/test distributions stay close",
        "narrow → robust transfer")

    fig.suptitle("Illustrative CDF of anomaly scores  ",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("saved:", path)


if __name__ == "__main__":
    make_schematic("anomaly_score_cdf_schematic.png")