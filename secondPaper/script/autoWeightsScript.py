"""
无监督异常检测 · 自适应频率加权模型训练脚本
=============================================
基于 TCNaddMaskScript.py 修改，专为 withAutoFreqWeights 模型设计。
新增功能：在训练结束后，自动提取并保存模型学习到的频率不确定性参数 (log_var)，供论文画图使用。

用法：
  python train_auto_freq.py --data_dir /path/to/data --model TCNAE_AutoFreq
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from scipy.stats import rankdata
from secondPaper.provider.unsupervised_provider import (
    create_data_loaders,
    load_data,
    print_split_info,
    split_data_unsupervised,
)

from secondPaper.models.OmniAnomaly import OmniAnomaly
from secondPaper.models.TCN_Autoencoder import TCNAE
# 【修改 1】：导入新的自适应权重模型
from secondPaper.models.withAutoFreqWeights import TCNAEWithFreq as TCNAE_AutoFreq

MODEL_REGISTRY = {
    'OmniAnomaly': OmniAnomaly,
    'TCNAE': TCNAE,
    'TCNAE_AutoFreq': TCNAE_AutoFreq,  # 【修改 2】：更新注册表
}


# ====================================================================
#  POT (Peak Over Threshold) — 基于极值理论的自适应阈值
# ====================================================================

def pot_threshold(scores, q=0.98, risk=1e-4, max_xi=1.0):
    t = np.percentile(scores, q * 100)
    exceedances = scores[scores > t] - t
    if len(exceedances) < 10:
        print(f"  [POT] 超额样本不足 ({len(exceedances)})，回退到 {q * 100:.0f}th 百分位数")
        return float(t)

    shape, _, scale = genpareto.fit(exceedances, floc=0)

    if shape > max_xi:
        print(f"  [POT] ξ={shape:.4f} > {max_xi}，截断为 {max_xi}（原始拟合不可靠）")
        shape = max_xi

    N, Nt = len(scores), len(exceedances)
    if abs(shape) < 1e-8:
        threshold = t + scale * np.log(Nt / (N * risk))
    else:
        threshold = t + scale / shape * ((Nt / (N * risk)) ** shape - 1)

    score_max = np.max(scores)
    upper_bound = score_max * 3.0
    if threshold > upper_bound:
        print(f"  [POT] threshold={threshold:.6f} 超出合理范围，截断为 {upper_bound:.6f} (3x max)")
        threshold = upper_bound

    print(f"  [POT] init_t={t:.6f}, exceedances={Nt}, GPD(ξ={shape:.4f}, σ={scale:.4f})")
    print(f"  [POT] threshold={threshold:.6f}, score_range=[{np.min(scores):.6f}, {score_max:.6f}]")
    return float(threshold)


# ====================================================================
#  在测试集上网格搜索最优F1
# ====================================================================

def best_f1_search(scores, labels, n_steps=1000):
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    best_f1, best_t, best_p, best_r = 0, 0, 0, 0
    for t in thresholds:
        preds = (scores > t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
            best_p = precision_score(labels, preds, zero_division=0)
            best_r = recall_score(labels, preds, zero_division=0)

    return {
        'f1_star': float(best_f1),
        'threshold': float(best_t),
        'precision': float(best_p),
        'recall': float(best_r),
    }


# ====================================================================
#  训练 / 推理
# ====================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    components = {}

    loop = tqdm(loader, desc='  Train', leave=False)
    for x, x_mark, _ in loop:
        x, x_mark = x.to(device), x_mark.to(device)

        result = model.compute_loss(x, x_mark)
        loss = result['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in result.items():
            if k != 'loss':
                components[k] = components.get(k, 0.0) + v

        loop.set_postfix(loss=f"{loss.item():.5f}")

    n = len(loader)
    avg = {k: v / n for k, v in components.items()}
    avg['loss'] = total_loss / n
    return avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for x, x_mark, _ in tqdm(loader, desc='  Val  ', leave=False):
        x, x_mark = x.to(device), x_mark.to(device)
        result = model.compute_loss(x, x_mark)
        total_loss += result['loss'].item()

    return total_loss / len(loader)


@torch.no_grad()
def compute_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    is_dict = None

    for x, x_mark, labels in loader:
        x, x_mark = x.to(device), x_mark.to(device)
        s = model.compute_anomaly_score(x, x_mark)
        if isinstance(s, dict):
            is_dict = True
            all_scores.append({k: v.cpu().numpy() for k, v in s.items()})
        else:
            is_dict = False
            all_scores.append(s.cpu().numpy())
        all_labels.append(labels.numpy().squeeze(-1))

    labels_np = np.concatenate(all_labels)
    if is_dict:
        keys = all_scores[0].keys()
        scores_np = {k: np.concatenate([d[k] for d in all_scores]) for k in keys}
    else:
        scores_np = np.concatenate(all_scores)
    return scores_np, labels_np


def fuse_multi_branch_scores(val_scores, test_scores, val_labels):
    print("\n--- Multi-branch Score Fusion (Rank Fusion) ---")

    def rank_score(s):
        return (rankdata(s) - 1) / (len(s) - 1)

    def fuse(sdict):
        return sum(rank_score(sdict[k]) for k in sdict)

    return fuse(val_scores), fuse(test_scores), {}


# ====================================================================
#  评估
# ====================================================================

def evaluate(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    has_both = len(np.unique(labels)) > 1
    return {
        'threshold': float(threshold),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(labels, scores)) if has_both else 0.0,
        'auc_pr': float(average_precision_score(labels, scores)) if has_both else 0.0,
    }


def print_results(results, title='Evaluation'):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  Threshold  : {results['threshold']:.6f}")
    print(f"  F1         : {results['f1']:.4f}")
    print(f"  Precision  : {results['precision']:.4f}")
    print(f"  Recall     : {results['recall']:.4f}")
    print(f"  AUC-ROC    : {results['auc_roc']:.4f}")
    print(f"  AUC-PR     : {results['auc_pr']:.4f}")
    print(f"{'=' * 55}")


# ====================================================================
#  Early Stopping
# ====================================================================

class EarlyStopping:
    def __init__(self, patience=7, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ====================================================================
#  模型构建
# ====================================================================

def build_model(args, device):
    cls = MODEL_REGISTRY[args.model]

    if args.model == 'OmniAnomaly':
        model = cls(
            feats=args.n_features,
            device=device,
            n_hidden=args.n_hidden,
            n_latent=args.n_latent,
            beta=args.beta,
        )
    elif args.model == 'TCNAE':
        model = cls(
            input_dim=args.n_features,
            window_size=args.window_size,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
        )
    elif args.model == 'TCNAE_AutoFreq':  # 【修改 3】：增加对应条件
        model = cls(
            input_dim=args.n_features,
            window_size=args.window_size,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
            freq_beta1=args.freq_beta1,
            freq_beta2=args.freq_beta2,
            freq_hidden_dim=args.freq_hidden_dim,
            freq_num_layers=args.freq_num_layers,
            freq_kernel_size=args.freq_kernel_size,
            freq_loss_weight=args.freq_loss_weight,
            freq_infer_segments=args.freq_infer_segments,
        )
    else:
        raise ValueError(f"未注册的模型配置: {args.model}")

    return model.to(device)


# ====================================================================
#  Main
# ====================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 数据 ──
    X, y, metadata = load_data(args.data_dir, args.window_size, args.step_size)
    print(f"Loaded: X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(
        X, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    print_split_info(split_info)

    args.n_features = X.shape[-1]

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, batch_size=args.batch_size,
    )

    # ── 模型 ──
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    early_stopping = EarlyStopping(patience=args.patience)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}  |  Parameters: {n_params:,}\n")

    # ── 训练 ──
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        log = f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.5f} | val_loss={val_loss:.5f}"
        for k, v in train_metrics.items():
            if k != 'loss':
                log += f" | {k}={v:.5f}"
        print(log)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n>>> Early stopping at epoch {epoch}\n")
            break

    early_stopping.load_best(model)

    # ── 推理：得到 val / test 分数 ──
    val_scores_raw, val_labels = compute_scores(model, val_loader, device)
    test_scores_raw, test_labels = compute_scores(model, test_loader, device)

    # ── 多分支模型：rank 融合 ──
    fusion_stats = None
    if isinstance(val_scores_raw, dict):
        val_scores, test_scores, fusion_stats = fuse_multi_branch_scores(
            val_scores_raw, test_scores_raw, val_labels
        )
    else:
        val_scores, test_scores = val_scores_raw, test_scores_raw

    # ── POT 阈值（仅在验证集纯正常数据上拟合） ──
    print("\n--- POT Threshold ---")
    val_normal_scores = val_scores[val_labels == 0]
    if len(val_normal_scores) == 0:
        raise ValueError("验证集中没有正常的窗口数据，无法拟合 POT 阈值！")
    threshold = pot_threshold(val_normal_scores, q=args.pot_q, risk=args.pot_risk)

    # ── 评估 ──
    val_results = evaluate(val_scores, val_labels, threshold)
    print_results(val_results, title='Validation Results')

    test_results = evaluate(test_scores, test_labels, threshold)
    print_results(test_results, title='Test Results')

    val_f1_result = best_f1_search(val_scores, val_labels)
    val_threshold = val_f1_result['threshold']
    test_results_val_th = evaluate(test_scores, test_labels, val_threshold)
    print_results(test_results_val_th, title='Test Results (Val Grid-Search Threshold)')

    # ── F1*（测试集 oracle threshold）──
    f1_star = best_f1_search(test_scores, test_labels)
    print(f"\n{'=' * 55}")
    print(f"  Test F1* (Oracle Threshold)")
    print(f"{'=' * 55}")
    print(f"  Threshold  : {f1_star['threshold']:.6f}")
    print(f"  F1* : {f1_star['f1_star']:.4f}")
    print(f"  Precision  : {f1_star['precision']:.4f}")
    print(f"  Recall     : {f1_star['recall']:.4f}")
    print(f"{'=' * 55}")

    if isinstance(val_scores_raw, dict):
        print(f"\n{'=' * 55}")
        print(f"  Per-branch Metrics (Test, fixed threshold per branch via val grid)")
        print(f"{'=' * 55}")
        per_branch = {}
        for k in val_scores_raw:
            vb = val_scores_raw[k]
            tb = test_scores_raw[k]
            bt = best_f1_search(vb, val_labels)['threshold']
            per_branch[k] = evaluate(tb, test_labels, bt)
            print(f"  [{k}]  F1={per_branch[k]['f1']:.4f}  "
                  f"AUC-ROC={per_branch[k]['auc_roc']:.4f}  "
                  f"AUC-PR={per_branch[k]['auc_pr']:.4f}")

    # ── 保存 ──
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model}_best.pt'))
        np.save(os.path.join(args.save_dir, f'{args.model}_test_scores.npy'), test_scores)

        # 【修改 4】：新增导出学习到的频率权重 log_var，供论文分析画图
        if hasattr(model, 'freq_log_var'):
            learned_log_var = model.freq_log_var.detach().cpu().numpy()
            np.save(os.path.join(args.save_dir, f'{args.model}_learned_freq_log_var.npy'), learned_log_var)
            print(f"\n  [Feature] 自适应频率不确定性参数 (log_var) 已保存至: {args.model}_learned_freq_log_var.npy")
            print(f"            在论文中可直接绘制 exp(-log_var) 曲线，证明低频分量被自发降权！")

        output = {
            'args': {k: v for k, v in vars(args).items()},
            'val': val_results,
            'test': test_results,
            'test_val_th': test_results_val_th,
            'f1_star': f1_star,
        }
        if fusion_stats is not None:
            output['fusion_stats'] = fusion_stats
            output['per_branch_test'] = per_branch

        with open(os.path.join(args.save_dir, f'{args.model}_results.json'), 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {args.save_dir}/")


# ====================================================================
#  CLI
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='无监督异常检测训练脚本')

    g = parser.add_argument_group('Data')
    g.add_argument('--data_dir', type=str,
                   default=r"/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows")
    g.add_argument('--window_size', type=int, default=100)
    g.add_argument('--step_size', type=int, default=20)
    g.add_argument('--train_ratio', type=float, default=0.6)
    g.add_argument('--val_ratio', type=float, default=0.2)

    g = parser.add_argument_group('Training')
    # 【修改 5】：默认参数调整为新的模型名称
    g.add_argument('--model', type=str, default='TCNAE_AutoFreq',
                   choices=list(MODEL_REGISTRY.keys()))
    g.add_argument('--batch_size', type=int, default=128)
    g.add_argument('--epochs', type=int, default=50)
    g.add_argument('--lr', type=float, default=1e-3)
    g.add_argument('--patience', type=int, default=5)
    g.add_argument('--seed', type=int, default=42)

    g = parser.add_argument_group('OmniAnomaly')
    g.add_argument('--n_hidden', type=int, default=256)
    g.add_argument('--n_latent', type=int, default=64)
    g.add_argument('--beta', type=float, default=0.0001)

    g = parser.add_argument_group('TCNAE')
    g.add_argument('--tcn_kernel_size', type=int, default=7)
    g.add_argument('--tcn_dropout', type=float, default=0.3)

    g = parser.add_argument_group('TCNAE_AutoFreq')  # 【修改 6】：组名同步更新
    g.add_argument('--freq_beta1', type=float, default=0.2,
                   help='每样本 mask 比例下界；>0 避免某些样本一个 bin 都不被 mask')
    g.add_argument('--freq_beta2', type=float, default=0.7)
    g.add_argument('--freq_hidden_dim', type=int, default=64)
    g.add_argument('--freq_num_layers', type=int, default=2)
    g.add_argument('--freq_kernel_size', type=int, default=7)
    g.add_argument('--freq_loss_weight', type=float, default=0.7)
    g.add_argument('--freq_infer_segments', type=int, default=25,
                   help='推理时 rolling mask 的段数；每个 bin 恰被 mask 一次')

    g = parser.add_argument_group('POT')
    g.add_argument('--pot_q', type=float, default=0.95, help='初始阈值百分位数')
    g.add_argument('--pot_risk', type=float, default=1e-5, help='期望误报概率')

    g = parser.add_argument_group('Output')
    g.add_argument('--save_dir', type=str, default='./results')

    main(parser.parse_args())