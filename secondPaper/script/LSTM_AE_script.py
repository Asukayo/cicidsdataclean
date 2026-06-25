"""
LSTM-AE 无监督异常检测训练脚本（PyCharm 直接运行版）
=================================================

使用方式：
1. 把本文件放到项目根目录，保证能正常 import:
   secondPaper.provider.unsupervised_provider
2. 修改下面 Config 中的 DATA_DIR / SAVE_DIR / 超参数
3. 在 PyCharm 中右键本文件，选择 Run 即可

说明：
- 不使用 argparse，不需要命令行参数
- 模型为 LSTM AutoEncoder
- 保留原训练脚本中的 POT 阈值、F1 / Precision / Recall / AUC-ROC / AUC-PR 评估逻辑
"""

import json
import os
import random
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# 如果本文件放在项目根目录，下面这段可以确保 PyCharm 直接运行时能找到 secondPaper 包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from secondPaper.provider.unsupervised_provider import (  # noqa: E402
    create_data_loaders,
    load_data,
    print_split_info,
    split_data_unsupervised,
)


# ====================================================================
#  配置区：PyCharm 直接修改这里，然后点击运行
# ====================================================================

@dataclass
class Config:
    # 数据路径
    DATA_DIR: str = r"/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"

    # 输出路径
    SAVE_DIR: str = "./results_lstm_ae"

    # 数据窗口
    WINDOW_SIZE: int = 100
    STEP_SIZE: int = 20
    TRAIN_RATIO: float = 0.6
    VAL_RATIO: float = 0.2

    # 训练参数
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 5
    SEED: int = 42
    GRAD_CLIP: float = 10.0

    # LSTM-AE 参数
    HIDDEN_DIM: int = 128
    LATENT_DIM: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2

    # POT 阈值参数
    POT_Q: float = 0.95
    POT_RISK: float = 1e-4


# ====================================================================
#  LSTM AutoEncoder
# ====================================================================

class LSTMAutoEncoder(nn.Module):
    """
    输入:
        x: [batch_size, window_size, n_features]

    输出:
        recon: [batch_size, window_size, n_features]

    异常分数:
        每个窗口的平均重构误差 MSE
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        # decoder 每个时间步都输入同一个 latent 表示
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]

        Returns:
            recon: [B, T, F]
        """
        batch_size, seq_len, _ = x.shape

        _, (h_n, _) = self.encoder(x)

        # 取最后一层最后时刻的隐状态作为窗口级表示
        h_last = h_n[-1]                  # [B, H]
        z = self.to_latent(h_last)        # [B, Z]

        # 把窗口级 latent 复制到每个时间步，作为 decoder 输入
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, Z]
        decoder_output, _ = self.decoder(decoder_input)       # [B, T, H]

        recon = self.output_layer(decoder_output)             # [B, T, F]
        return recon

    def compute_loss(self, x: torch.Tensor, x_mark=None) -> dict:
        """
        与原训练脚本保持一致的接口。
        x_mark 对 LSTM-AE 不使用，但保留参数以兼容 DataLoader 输出。
        """
        x = x.float()
        recon = self.forward(x)
        loss = self.criterion(recon, x)

        return {
            "loss": loss,
            "recon_loss": loss.item(),
        }

    @torch.no_grad()
    def compute_anomaly_score(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        """
        返回每个窗口的异常分数，shape: [B]
        分数越大，表示重构误差越大，越可能异常。
        """
        x = x.float()
        recon = self.forward(x)
        scores = torch.mean((recon - x) ** 2, dim=(1, 2))
        return scores


# ====================================================================
#  工具函数
# ====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 追求可复现时开启；如果更看重速度，可以注释掉下面两行
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pot_threshold(scores, q=0.98, risk=1e-4, max_xi=1.0):
    """
    使用广义 Pareto 分布拟合尾部超额分布，推导异常阈值。
    """
    scores = np.asarray(scores, dtype=np.float64)

    t = np.percentile(scores, q * 100)
    exceedances = scores[scores > t] - t

    if len(exceedances) < 10:
        print(f"  [POT] 超额样本不足 ({len(exceedances)})，回退到 {q * 100:.0f}th 百分位数")
        return float(t)

    shape, _, scale = genpareto.fit(exceedances, floc=0)

    if shape > max_xi:
        print(f"  [POT] ξ={shape:.4f} > {max_xi}，截断为 {max_xi}")
        shape = max_xi

    n, nt = len(scores), len(exceedances)

    if abs(shape) < 1e-8:
        threshold = t + scale * np.log(nt / (n * risk))
    else:
        threshold = t + scale / shape * ((nt / (n * risk)) ** shape - 1)

    score_max = np.max(scores)
    upper_bound = score_max * 3.0

    if threshold > upper_bound:
        print(f"  [POT] threshold={threshold:.6f} 超出合理范围，截断为 {upper_bound:.6f}")
        threshold = upper_bound

    print(f"  [POT] init_t={t:.6f}, exceedances={nt}, GPD(ξ={shape:.4f}, σ={scale:.4f})")
    print(f"  [POT] threshold={threshold:.6f}, score_range=[{np.min(scores):.6f}, {score_max:.6f}]")

    return float(threshold)


def best_f1_search(scores, labels, n_steps=1000):
    """
    在给定 scores 和 labels 上网格搜索最优 F1。
    注意：测试集上的 F1* 是 oracle 指标，只用于论文对比，不应用作真实部署阈值。
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    thresholds = np.linspace(scores.min(), scores.max(), n_steps)

    best_f1, best_t, best_p, best_r = 0.0, 0.0, 0.0, 0.0

    for t in thresholds:
        preds = (scores > t).astype(int)
        f = f1_score(labels, preds, zero_division=0)

        if f > best_f1:
            best_f1 = f
            best_t = t
            best_p = precision_score(labels, preds, zero_division=0)
            best_r = recall_score(labels, preds, zero_division=0)

    return {
        "f1_star": float(best_f1),
        "threshold": float(best_t),
        "precision": float(best_p),
        "recall": float(best_r),
    }


# ====================================================================
#  训练 / 验证 / 推理
# ====================================================================

def train_one_epoch(model, loader, optimizer, device, grad_clip: float):
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0

    loop = tqdm(loader, desc="  Train", leave=False)

    for x, x_mark, _ in loop:
        x = x.to(device)
        x_mark = x_mark.to(device) if x_mark is not None else None

        result = model.compute_loss(x, x_mark)
        loss = result["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += float(result.get("recon_loss", loss.item()))

        loop.set_postfix(loss=f"{loss.item():.5f}")

    n = max(len(loader), 1)

    return {
        "loss": total_loss / n,
        "recon_loss": total_recon_loss / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss = 0.0

    for x, x_mark, _ in tqdm(loader, desc="  Val  ", leave=False):
        x = x.to(device)
        x_mark = x_mark.to(device) if x_mark is not None else None

        result = model.compute_loss(x, x_mark)
        total_loss += result["loss"].item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def compute_scores(model, loader, device):
    """
    返回:
        scores: [num_windows]
        labels: [num_windows]
    """
    model.eval()

    all_scores = []
    all_labels = []

    for x, x_mark, labels in tqdm(loader, desc="  Score", leave=False):
        x = x.to(device)
        x_mark = x_mark.to(device) if x_mark is not None else None

        scores = model.compute_anomaly_score(x, x_mark)

        all_scores.append(scores.detach().cpu().numpy())
        all_labels.append(labels.numpy().squeeze(-1))

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ====================================================================
#  评估
# ====================================================================

def evaluate(scores, labels, threshold):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)

    preds = (scores > threshold).astype(int)
    has_both_classes = len(np.unique(labels)) > 1

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, scores)) if has_both_classes else 0.0,
        "auc_pr": float(average_precision_score(labels, scores)) if has_both_classes else 0.0,
    }


def print_results(results, title="Evaluation"):
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
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.counter = 0
            return

        self.counter += 1
        print(f"  EarlyStopping: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ====================================================================
#  主流程
# ====================================================================

def main(cfg: Config):
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 数据 ──
    X, y, metadata = load_data(cfg.DATA_DIR, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    print(f"Loaded: X={X.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(
        X,
        y,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
    )
    print_split_info(split_info)

    n_features = X.shape[-1]

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=cfg.BATCH_SIZE,
    )

    # ── 模型 ──
    model = LSTMAutoEncoder(
        input_dim=n_features,
        hidden_dim=cfg.HIDDEN_DIM,
        latent_dim=cfg.LATENT_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.9,
    )
    early_stopping = EarlyStopping(patience=cfg.PATIENCE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: LSTM-AE | Parameters: {n_params:,}\n")

    # ── 训练 ──
    for epoch in range(1, cfg.EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.GRAD_CLIP,
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"recon_loss={train_metrics['recon_loss']:.5f} | "
            f"val_loss={val_loss:.5f}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"\n>>> Early stopping at epoch {epoch}\n")
            break

    early_stopping.load_best(model)

    # ── 验证集分数 ──
    val_scores, val_labels = compute_scores(model, val_loader, device)

    # ── POT 阈值：只用验证集正常窗口拟合 ──
    print("\n--- POT Threshold ---")

    val_normal_scores = val_scores[val_labels == 0]
    if len(val_normal_scores) == 0:
        raise ValueError("验证集中没有正常窗口，无法拟合 POT 阈值。请检查数据划分或标签。")

    threshold = pot_threshold(
        val_normal_scores,
        q=cfg.POT_Q,
        risk=cfg.POT_RISK,
    )

    # ── 验证集 / 测试集评估 ──
    val_results = evaluate(val_scores, val_labels, threshold)
    print_results(val_results, title="Validation Results")

    test_scores, test_labels = compute_scores(model, test_loader, device)
    test_results = evaluate(test_scores, test_labels, threshold)
    print_results(test_results, title="Test Results")

    # 用验证集网格搜索得到阈值，再评估测试集
    val_f1_result = best_f1_search(val_scores, val_labels)
    val_threshold = val_f1_result["threshold"]
    test_results_val_th = evaluate(test_scores, test_labels, val_threshold)
    print_results(test_results_val_th, title="Test Results (Val Grid-Search Threshold)")

    # 测试集 F1*：只作为 oracle 对比指标
    f1_star = best_f1_search(test_scores, test_labels)

    print(f"\n{'=' * 55}")
    print("  Test F1* (Oracle Threshold)")
    print(f"{'=' * 55}")
    print(f"  Threshold  : {f1_star['threshold']:.6f}")
    print(f"  F1*        : {f1_star['f1_star']:.4f}")
    print(f"  Precision  : {f1_star['precision']:.4f}")
    print(f"  Recall     : {f1_star['recall']:.4f}")
    print(f"{'=' * 55}")

    # ── 保存 ──
    if cfg.SAVE_DIR:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(cfg.SAVE_DIR, "LSTM_AE_best.pt"),
        )
        np.save(
            os.path.join(cfg.SAVE_DIR, "LSTM_AE_test_scores.npy"),
            test_scores,
        )

        output = {
            "config": asdict(cfg),
            "n_features": int(n_features),
            "val": val_results,
            "test": test_results,
            "test_val_th": test_results_val_th,
            "f1_star": f1_star,
        }

        with open(
            os.path.join(cfg.SAVE_DIR, "LSTM_AE_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {cfg.SAVE_DIR}/")


if __name__ == "__main__":
    config = Config()
    main(config)
