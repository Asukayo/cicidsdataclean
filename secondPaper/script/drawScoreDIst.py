
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata, gaussian_kde, genpareto
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from secondPaper.provider.unsupervised_provider import (
    create_data_loaders, load_data, split_data_unsupervised,
)
from secondPaper.models.FullModel import FullModel

print('[SCRIPT] drawScoreDIst_val_test_FIXED.py loaded: will export val/test source data')

BRANCHES = ['recon_score', 'freq_score']   # 论文融合的两支


# ----------------------------- 推理 -----------------------------
@torch.no_grad()
def compute_scores(model, loader, device):
    """返回 {branch: (N,) 原始分数}, labels (N,)。与训练脚本一致：收集原始分支分数，
    全局拼接后再做秩融合（不在 batch 内融合）。"""
    model.eval()
    all_scores, all_labels = [], []
    for x, x_mark, labels in loader:
        x = x.to(device)
        s = model.compute_anomaly_score(x, x_mark)        # dict
        all_scores.append({k: s[k].cpu().numpy() for k in BRANCHES})
        all_labels.append(labels.numpy().squeeze(-1))
    labels_np = np.concatenate(all_labels)
    scores_np = {k: np.concatenate([d[k] for d in all_scores]) for k in BRANCHES}
    return scores_np, labels_np


def rnorm(s):
    """归一化秩，与训练脚本 rank_score 一致：(rank-1)/(N-1) ∈ [0,1]。"""
    return (rankdata(s) - 1) / (len(s) - 1)


def fuse(sdict):
    """S = R(S_recon) + R(S_freq) ∈ [0,2]。"""
    return sum(rnorm(sdict[k]) for k in BRANCHES)


# --------------------------- 阈值 ---------------------------
def best_f1_threshold(scores, labels, n_steps=1000):
    ts = np.linspace(scores.min(), scores.max(), n_steps)
    best_f1, best_t = -1.0, float(scores.min())
    for t in ts:
        f = f1_score(labels, (scores > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def pot_threshold(scores, q=0.95, risk=1e-5, max_xi=1.0):
    """与训练脚本同口径：仅在验证集纯正常分数上拟合 GPD。"""
    t = np.percentile(scores, q * 100)
    exc = scores[scores > t] - t
    if len(exc) < 10:
        return float(t)
    shape, _, scale = genpareto.fit(exc, floc=0)
    shape = min(shape, max_xi)
    N, Nt = len(scores), len(exc)
    if abs(shape) < 1e-8:
        thr = t + scale * np.log(Nt / (N * risk))
    else:
        thr = t + scale / shape * ((Nt / (N * risk)) ** shape - 1)
    return float(min(thr, np.max(scores) * 3.0))


# --------------------------- 模型重建 ---------------------------
def build_full_model(a, device):
    return FullModel(
        input_dim=a['n_features'],
        window_size=a['window_size'],
        kernel_size=a['tcn_kernel_size'],
        dropout=a['tcn_dropout'],
        smooth_weight=a['smooth_weight'],
        trend_kernel_size=a['trend_kernel_size'],
        num_prototypes=a['num_prototypes'],
        proto_tau=a['proto_tau'],
        sparse_weight=a['sparse_weight'],
        split_after=a['split_after'],
        freq_beta1=a['freq_beta1'],
        freq_beta2=a['freq_beta2'],
        freq_hidden_dim=a['freq_hidden_dim'],
        freq_num_layers=a['freq_num_layers'],
        freq_kernel_size=a['freq_kernel_size'],
        freq_loss_weight=a['freq_loss_weight'],
        freq_infer_segments=a['freq_infer_segments'],
    ).to(device)


# ----------------------------- 画图源数据导出 -----------------------------
def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_plot_source_data(s_norm, s_anom, bins, xs, kde_norm, kde_anom,
                            thr, metrics, tag, outbase, pot_thr=None):
    """导出绘图源数据：原始分数、直方图密度、KDE 曲线和阈值/指标元信息。"""
    # 1) 原始散点级分数：可以重新画直方图 / KDE / 阈值分割
    raw_rows = (
        [{'group': 'normal', 'label': 0, 'fused_score': float(s)} for s in s_norm] +
        [{'group': 'anomalous', 'label': 1, 'fused_score': float(s)} for s in s_anom]
    )
    _write_csv(
        outbase + '_source_scores.csv',
        ['group', 'label', 'fused_score'],
        raw_rows,
    )

    # 2) 直方图密度：对应图中的半透明柱状分布
    hist_norm, bin_edges = np.histogram(s_norm, bins=bins, density=True)
    hist_anom, _ = np.histogram(s_anom, bins=bins, density=True)
    hist_rows = []
    for group, label, hist in [('normal', 0, hist_norm), ('anomalous', 1, hist_anom)]:
        for left, right, density in zip(bin_edges[:-1], bin_edges[1:], hist):
            hist_rows.append({
                'group': group,
                'label': label,
                'bin_left': float(left),
                'bin_right': float(right),
                'density': float(density),
            })
    _write_csv(
        outbase + '_source_hist.csv',
        ['group', 'label', 'bin_left', 'bin_right', 'density'],
        hist_rows,
    )

    # 3) KDE 曲线：对应图中的两条平滑密度曲线
    kde_rows = [
        {
            'fused_score': float(x),
            'normal_density': float(n),
            'anomalous_density': float(a),
        }
        for x, n, a in zip(xs, kde_norm, kde_anom)
    ]
    _write_csv(
        outbase + '_source_kde.csv',
        ['fused_score', 'normal_density', 'anomalous_density'],
        kde_rows,
    )

    # 4) 元信息：阈值、指标、样本量，方便论文图表复现
    meta = {
        'dataset_tag': tag,
        'score_definition': 'S = R(S_recon) + R(S_freq)',
        'threshold': float(thr),
        'pot_threshold': None if pot_thr is None else float(pot_thr),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'n_normal': int(len(s_norm)),
        'n_anomalous': int(len(s_anom)),
        'hist_bins': int(len(bins) - 1),
        'kde_points': int(len(xs)),
    }
    with open(outbase + '_source_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'[Saved source data] {outbase}_source_scores.csv')
    print(f'[Saved source data] {outbase}_source_hist.csv')
    print(f'[Saved source data] {outbase}_source_kde.csv')
    print(f'[Saved source data] {outbase}_source_meta.json')


def _safe_kde(values, xs):
    """KDE 安全封装：样本过少或方差为 0 时返回全 0，避免 gaussian_kde 报错。"""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return np.zeros_like(xs, dtype=float)
    return gaussian_kde(values)(xs)


def _binary_metrics(labels, scores, thr):
    """按 scores > thr 计算二分类指标，与主图逻辑保持一致。"""
    preds = (scores > thr).astype(int)
    return {
        'P': precision_score(labels, preds, zero_division=0),
        'R': recall_score(labels, preds, zero_division=0),
        'F1': f1_score(labels, preds, zero_division=0),
    }


def export_val_test_source_data(val_raw, val_labels, val_fused,
                                test_raw, test_labels, test_fused,
                                thr, tag, outbase):
    """导出 val/test 对比图所需源数据。

    输出文件：
    1. *_val_test_source_scores.csv
       每一行是一个窗口级 fused score，包含 split/group/label/fused_score。

    2. *_val_test_source_branch_scores.csv
       同时导出 recon_score/freq_score/fused_score，便于之后检查或重新做统一 rank 融合。

    3. *_val_test_source_hist.csv
       按 split + group 统计直方图密度。

    4. *_val_test_source_kde.csv
       按 split 输出 normal/anomalous 两条 KDE 曲线。

    5. *_val_test_source_meta.json
       保存阈值、val/test 指标、样本量等元信息。
    """
    bins = np.linspace(0, 2, 46)
    xs = np.linspace(0, 2, 400)

    split_pack = {
        'val': {
            'raw': val_raw,
            'labels': np.asarray(val_labels).astype(int),
            'fused': np.asarray(val_fused, dtype=float),
        },
        'test': {
            'raw': test_raw,
            'labels': np.asarray(test_labels).astype(int),
            'fused': np.asarray(test_fused, dtype=float),
        },
    }

    # 1) fused score 源数据
    score_rows = []
    for split, item in split_pack.items():
        for y, s in zip(item['labels'], item['fused']):
            score_rows.append({
                'split': split,
                'group': 'anomalous' if int(y) == 1 else 'normal',
                'label': int(y),
                'fused_score': float(s),
            })
    _write_csv(
        outbase + '_val_test_source_scores.csv',
        ['split', 'group', 'label', 'fused_score'],
        score_rows,
    )

    # 2) 分支原始分数 + fused score，方便后续排查 rank 融合口径
    branch_rows = []
    for split, item in split_pack.items():
        raw = item['raw']
        labels = item['labels']
        fused = item['fused']
        n = len(labels)
        for i in range(n):
            row = {
                'split': split,
                'group': 'anomalous' if int(labels[i]) == 1 else 'normal',
                'label': int(labels[i]),
                'fused_score': float(fused[i]),
            }
            for b in BRANCHES:
                row[b] = float(raw[b][i])
            branch_rows.append(row)
    _write_csv(
        outbase + '_val_test_source_branch_scores.csv',
        ['split', 'group', 'label', 'recon_score', 'freq_score', 'fused_score'],
        branch_rows,
    )

    # 3) hist 源数据
    hist_rows = []
    for split, item in split_pack.items():
        labels = item['labels']
        fused = item['fused']
        for group, label in [('normal', 0), ('anomalous', 1)]:
            vals = fused[labels == label]
            if len(vals) == 0:
                hist = np.zeros(len(bins) - 1, dtype=float)
            else:
                hist, _ = np.histogram(vals, bins=bins, density=True)
            for left, right, density in zip(bins[:-1], bins[1:], hist):
                hist_rows.append({
                    'split': split,
                    'group': group,
                    'label': label,
                    'bin_left': float(left),
                    'bin_right': float(right),
                    'density': float(density),
                })
    _write_csv(
        outbase + '_val_test_source_hist.csv',
        ['split', 'group', 'label', 'bin_left', 'bin_right', 'density'],
        hist_rows,
    )

    # 4) KDE 源数据
    kde_rows = []
    for split, item in split_pack.items():
        labels = item['labels']
        fused = item['fused']
        normal_density = _safe_kde(fused[labels == 0], xs)
        anomalous_density = _safe_kde(fused[labels == 1], xs)
        for x, nd, ad in zip(xs, normal_density, anomalous_density):
            kde_rows.append({
                'split': split,
                'fused_score': float(x),
                'normal_density': float(nd),
                'anomalous_density': float(ad),
            })
    _write_csv(
        outbase + '_val_test_source_kde.csv',
        ['split', 'fused_score', 'normal_density', 'anomalous_density'],
        kde_rows,
    )

    # 5) meta
    val_metrics = _binary_metrics(val_labels, val_fused, thr)
    test_metrics = _binary_metrics(test_labels, test_fused, thr)
    meta = {
        'dataset_tag': tag,
        'score_definition': 'S = R(S_recon) + R(S_freq)',
        'rank_fusion_scope': 'val and test are fused independently, same as the original script',
        'threshold_source': 'best F1 threshold on validation split',
        'threshold': float(thr),
        'metrics': {
            'val_at_threshold': {k: float(v) for k, v in val_metrics.items()},
            'test_at_threshold': {k: float(v) for k, v in test_metrics.items()},
        },
        'counts': {
            'val': {
                'n_total': int(len(val_labels)),
                'n_normal': int(np.sum(np.asarray(val_labels) == 0)),
                'n_anomalous': int(np.sum(np.asarray(val_labels) == 1)),
            },
            'test': {
                'n_total': int(len(test_labels)),
                'n_normal': int(np.sum(np.asarray(test_labels) == 0)),
                'n_anomalous': int(np.sum(np.asarray(test_labels) == 1)),
            },
        },
        'hist_bins': int(len(bins) - 1),
        'kde_points': int(len(xs)),
    }
    with open(outbase + '_val_test_source_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'[Saved val/test source data] {outbase}_val_test_source_scores.csv')
    print(f'[Saved val/test source data] {outbase}_val_test_source_branch_scores.csv')
    print(f'[Saved val/test source data] {outbase}_val_test_source_hist.csv')
    print(f'[Saved val/test source data] {outbase}_val_test_source_kde.csv')
    print(f'[Saved val/test source data] {outbase}_val_test_source_meta.json')


def plot_val_test_distribution(val_fused, val_labels, test_fused, test_labels,
                               thr, tag, outbase):
    """直接输出 val/test 对比图：上下两个子图，每个子图显示 normal/anomalous。"""
    C_NORM, C_ANOM = '#2c6fbb', '#d1495b'
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 0.9,
                         'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})

    bins = np.linspace(0, 2, 46)
    xs = np.linspace(0, 2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150, sharey=True)

    for ax, split, scores, labels in [
        (axes[0], 'Validation', val_fused, val_labels),
        (axes[1], 'Test', test_fused, test_labels),
    ]:
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores, dtype=float)
        s_norm = scores[labels == 0]
        s_anom = scores[labels == 1]

        ax.hist(s_norm, bins=bins, density=True, color=C_NORM, alpha=0.30, edgecolor='none')
        ax.hist(s_anom, bins=bins, density=True, color=C_ANOM, alpha=0.30, edgecolor='none')
        ax.plot(xs, _safe_kde(s_norm, xs), color=C_NORM, lw=2.1, label='Normal windows')
        ax.plot(xs, _safe_kde(s_anom, xs), color=C_ANOM, lw=2.1, label='Anomalous windows')
        ax.axvline(thr, color='k', ls='--', lw=1.5)
        ax.set_xlim(0, 2)
        ax.set_xlabel(r'Fused anomaly score  $S=R(S_{rec})+R(S_{freq})$')
        ax.set_title(f'{split} split')
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper left', frameon=False, fontsize=10.0)
    fig.suptitle(f'FreqDAR Score Distribution — {tag}: Validation vs Test', fontsize=12.5)
    plt.tight_layout()

    fig.savefig(outbase + '_val_vs_test.pdf', bbox_inches='tight')
    fig.savefig(outbase + '_val_vs_test.png', dpi=150, bbox_inches='tight')
    print(f'[Saved] {outbase}_val_vs_test.pdf / .png')


# ----------------------------- 画图 -----------------------------
def plot_distribution(s_norm, s_anom, thr, metrics, tag, outbase,
                      pot_thr=None):
    C_NORM, C_ANOM = '#2c6fbb', '#d1495b'
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 0.9,
                         'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)

    bins = np.linspace(0, 2, 46)
    ax.hist(s_norm, bins=bins, density=True, color=C_NORM, alpha=0.30, edgecolor='none')
    ax.hist(s_anom, bins=bins, density=True, color=C_ANOM, alpha=0.30, edgecolor='none')

    xs = np.linspace(0, 2, 400)
    kde_norm = gaussian_kde(s_norm)(xs)
    kde_anom = gaussian_kde(s_anom)(xs)
    ax.plot(xs, kde_norm, color=C_NORM, lw=2.2, label='Normal windows')
    ax.plot(xs, kde_anom, color=C_ANOM, lw=2.2, label='Anomalous windows')

    export_plot_source_data(s_norm, s_anom, bins, xs, kde_norm, kde_anom,
                            thr, metrics, tag, outbase, pot_thr=pot_thr)

    ax.axvline(thr, color='k', ls='--', lw=1.6)
    ymax = ax.get_ylim()[1]
    ax.text(thr + 0.02, ymax * 0.95, f'Val-transferred\nthreshold = {thr:.2f}',
            fontsize=10.5, va='top', ha='left')
    if pot_thr is not None:
        ax.axvline(pot_thr, color='#666666', ls=':', lw=1.5)
        ax.text(pot_thr + 0.02, ymax * 0.62, f'POT = {pot_thr:.2f}',
                fontsize=9.5, color='#666666', va='top', ha='left')

    ax.set_xlabel(r'Fused anomaly score  $S=R(S_{rec})+R(S_{freq})$')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title(f'Anomaly Score Distribution — {tag} (FreqDAR)', fontsize=12.5, pad=10)
    ax.legend(loc='upper left', frameon=False, fontsize=10.5)
    ax.text(0.99, 0.02,
            f"@thr:  P={metrics['P']:.2f}  R={metrics['R']:.2f}  F1={metrics['F1']:.2f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f2f2f2', ec='#cccccc'))
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(outbase + '.pdf', bbox_inches='tight')   # 矢量，论文用
    fig.savefig(outbase + '.png', dpi=150, bbox_inches='tight')
    print(f'[Saved] {outbase}.pdf / .png')


# ----------------------------- Main -----------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 读回训练时的超参（保证模型结构与权重匹配）
    results_path = os.path.join(args.save_dir, f'{args.model}_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f'未找到 {results_path}；请确认 --save_dir 指向训练时的输出目录。')
    with open(results_path, 'r', encoding='utf-8') as f:
        a = json.load(f)['args']

    # 2) 数据（划分 / scaler 与训练完全一致）
    X, y, _ = load_data(args.data_dir, a['window_size'], a['step_size'])
    train_data, val_data, test_data, _ = split_data_unsupervised(
        X, y, train_ratio=a['train_ratio'], val_ratio=a['val_ratio'])
    _, val_loader, test_loader, _ = create_data_loaders(
        train_data, val_data, test_data, batch_size=a['batch_size'])

    # 3) 重建模型并载入最优权重
    model = build_full_model(a, device)
    state = torch.load(os.path.join(args.save_dir, f'{args.model}_best.pt'),
                       map_location=device)
    model.load_state_dict(state)

    # 4) 推理 -> 原始分支分数；各 split 独立秩融合
    val_raw, val_labels = compute_scores(model, val_loader, device)
    test_raw, test_labels = compute_scores(model, test_loader, device)
    val_fused = fuse(val_raw)
    test_fused = fuse(test_raw)

    # 5) 阈值：验证集 grid-search 最优 F1（论文部署阈值），迁移到测试集
    thr = best_f1_threshold(val_fused, val_labels)
    preds = (test_fused > thr).astype(int)
    metrics = {
        'P': precision_score(test_labels, preds, zero_division=0),
        'R': recall_score(test_labels, preds, zero_division=0),
        'F1': f1_score(test_labels, preds, zero_division=0),
    }
    print(f"[Test @ val-threshold]  thr={thr:.4f}  "
          f"P={metrics['P']:.4f}  R={metrics['R']:.4f}  F1={metrics['F1']:.4f}")

    pot_thr = None
    if args.show_pot:
        vn = val_fused[val_labels == 0]
        if len(vn) > 0:
            pot_thr = pot_threshold(vn, q=args.pot_q, risk=args.pot_risk)
            print(f"[POT threshold] {pot_thr:.4f}")

    # 6) 导出 val/test 对比所需源数据，并画 val vs test 对比图
    outbase = os.path.join(args.save_dir, f'fig_score_dist_{args.dataset_tag}')
    export_val_test_source_data(
        val_raw, val_labels, val_fused,
        test_raw, test_labels, test_fused,
        thr, args.dataset_tag, outbase,
    )
    plot_val_test_distribution(
        val_fused, val_labels, test_fused, test_labels,
        thr, args.dataset_tag, outbase,
    )

    # 7) 保留原逻辑：画测试集 normal/anomalous 分布
    plot_distribution(test_fused[test_labels == 0], test_fused[test_labels == 1],
                      thr, metrics, args.dataset_tag, outbase, pot_thr=pot_thr)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Anomaly score distribution plot')
    p.add_argument('--data_dir', type=str, default=r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows")
    p.add_argument('--save_dir', type=str, default='./results')
    p.add_argument('--model', type=str, default='FullModel')
    p.add_argument('--dataset_tag', type=str, default='CICIDS2017')
    p.add_argument('--show_pot', action='store_true', help='叠加 POT 阈值竖线对照')
    p.add_argument('--pot_q', type=float, default=0.95)
    p.add_argument('--pot_risk', type=float, default=1e-5)
    main(p.parse_args())