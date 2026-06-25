"""
Isolation Forest 无监督异常检测训练脚本（PyCharm 直接运行版）
=======================================================

特点：
  - 不使用命令行参数，直接在 Config 中修改配置
  - 复用 secondPaper.provider.unsupervised_provider 中的数据加载与无监督划分逻辑
  - 使用 IsolationForest 对窗口级特征进行异常检测
  - 支持 POT 阈值、验证集阈值搜索、测试集 F1* 评估
  - 保存模型、标准化器、分数与结果 JSON

使用：
  1. 将本文件放到项目根目录，确保可以 import secondPaper
  2. 修改 Config.DATA_DIR
  3. 在 PyCharm 中右键运行本文件
"""

import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# 让 PyCharm 直接运行时更容易找到项目包
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secondPaper.provider.unsupervised_provider import (  # noqa: E402
    load_data,
    print_split_info,
    split_data_unsupervised,
)


# ====================================================================
#  配置区：PyCharm 直接改这里即可
# ====================================================================

@dataclass
class Config:
    # 数据路径
    DATA_DIR: str = r"/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"

    # 窗口参数：需要与你生成 integrated_windows 时保持一致
    WINDOW_SIZE: int = 100
    STEP_SIZE: int = 20

    # 无监督划分比例
    TRAIN_RATIO: float = 0.6
    VAL_RATIO: float = 0.2

    # 特征模式：
    #   "flatten": 直接把 [window_size, n_features] 拉平成一维，保留时序位置，但维度较高
    #   "stat":    使用 mean/std/min/max/median/last 统计特征，维度更低，通常更稳
    #   "last":    只取窗口最后一个时间步，最简单但丢失窗口信息
    FEATURE_MODE: str = "stat"

    # 训练时是否只使用正常样本。若训练集没有标签 0，则自动回退为使用全部训练样本
    FIT_NORMAL_ONLY: bool = True

    # Isolation Forest 参数
    N_ESTIMATORS: int = 300
    MAX_SAMPLES: Any = "auto"      # 可设为 "auto"、整数或 0~1 浮点数
    CONTAMINATION: Any = "auto"    # 无监督时建议先用 "auto"，最终阈值由 POT/验证集决定
    MAX_FEATURES: float = 1.0
    BOOTSTRAP: bool = False
    N_JOBS: int = -1
    RANDOM_STATE: int = 42

    # POT 阈值
    POT_Q: float = 0.95
    POT_RISK: float = 1e-4

    # 网格搜索阈值粒度
    THRESHOLD_SEARCH_STEPS: int = 1000

    # 输出
    SAVE_DIR: str = "./results_isolation_forest"


# ====================================================================
#  工具函数
# ====================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def unpack_split(split_data) -> Tuple[np.ndarray, np.ndarray]:
    """
    兼容不同 split_data_unsupervised 返回格式。

    预期常见格式：
      1. (X, y)
      2. {"X": X, "y": y}
      3. {"data": X, "label": y}
    """
    if isinstance(split_data, dict):
        x = (
            split_data.get("X")
            if "X" in split_data else
            split_data.get("x")
            if "x" in split_data else
            split_data.get("data")
        )
        y = (
            split_data.get("y")
            if "y" in split_data else
            split_data.get("label")
            if "label" in split_data else
            split_data.get("labels")
        )
    elif isinstance(split_data, (tuple, list)) and len(split_data) >= 2:
        x, y = split_data[0], split_data[1]
    else:
        raise TypeError(
            "无法解析 split_data_unsupervised 的返回结果。"
            "请确认 train_data/val_data/test_data 是 (X, y) 或包含 X/y 的 dict。"
        )

    if x is None or y is None:
        raise ValueError("split_data 中没有找到 X/y，请检查 unsupervised_provider 的返回格式。")

    return np.asarray(x), np.asarray(y)


def to_window_labels(y: np.ndarray) -> np.ndarray:
    """
    将标签压成窗口级 1-D 标签：
      - 若 y 已经是 [N] 或 [N, 1]，直接 squeeze
      - 若 y 是 [N, T] 或 [N, T, 1]，只要窗口内出现异常，就认为该窗口异常
    """
    y = np.asarray(y)
    y = np.squeeze(y)

    if y.ndim == 1:
        return y.astype(int)

    axes = tuple(range(1, y.ndim))
    return (np.max(y, axis=axes) > 0).astype(int)


def build_window_features(x: np.ndarray, mode: str) -> np.ndarray:
    """
    将窗口数据 [N, T, C] 转成 IsolationForest 可用的二维特征 [N, D]。
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        # 已经是 [N, D]
        return x

    if x.ndim != 3:
        raise ValueError(f"期望 X 形状为 [N, T, C] 或 [N, D]，但得到 {x.shape}")

    mode = mode.lower()

    if mode == "flatten":
        return x.reshape(x.shape[0], -1)

    if mode == "last":
        return x[:, -1, :]

    if mode == "stat":
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        min_ = np.min(x, axis=1)
        max_ = np.max(x, axis=1)
        median = np.median(x, axis=1)
        last = x[:, -1, :]
        return np.concatenate([mean, std, min_, max_, median, last], axis=1)

    raise ValueError(f"未知 FEATURE_MODE={mode}，可选：flatten/stat/last")


# ====================================================================
#  POT 阈值
# ====================================================================

def pot_threshold(scores: np.ndarray, q: float = 0.98, risk: float = 1e-4, max_xi: float = 1.0) -> float:
    """
    使用广义 Pareto 分布拟合尾部超额分布，推导异常阈值。
    注意：scores 越大表示越异常。
    """
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        scores = scores.reshape(-1)

    if len(scores) == 0:
        raise ValueError("POT 输入 scores 为空，无法计算阈值。")

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


# ====================================================================
#  评估
# ====================================================================

def best_f1_search(scores: np.ndarray, labels: np.ndarray, n_steps: int = 1000) -> dict:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)

    if np.min(scores) == np.max(scores):
        threshold = float(scores[0])
        preds = (scores > threshold).astype(int)
        return {
            "f1_star": float(f1_score(labels, preds, zero_division=0)),
            "threshold": threshold,
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        }

    thresholds = np.linspace(scores.min(), scores.max(), n_steps)

    best_f1, best_t, best_p, best_r = 0.0, float(thresholds[0]), 0.0, 0.0
    for t in thresholds:
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_p = precision_score(labels, preds, zero_division=0)
            best_r = recall_score(labels, preds, zero_division=0)

    return {
        "f1_star": float(best_f1),
        "threshold": float(best_t),
        "precision": float(best_p),
        "recall": float(best_r),
    }


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    preds = (scores > threshold).astype(int)

    has_both = len(np.unique(labels)) > 1

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, scores)) if has_both else 0.0,
        "auc_pr": float(average_precision_score(labels, scores)) if has_both else 0.0,
    }


def print_results(results: dict, title: str = "Evaluation") -> None:
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
#  Isolation Forest 训练与打分
# ====================================================================

def fit_isolation_forest(x_train_feat: np.ndarray, cfg: Config) -> IsolationForest:
    model = IsolationForest(
        n_estimators=cfg.N_ESTIMATORS,
        max_samples=cfg.MAX_SAMPLES,
        contamination=cfg.CONTAMINATION,
        max_features=cfg.MAX_FEATURES,
        bootstrap=cfg.BOOTSTRAP,
        n_jobs=cfg.N_JOBS,
        random_state=cfg.RANDOM_STATE,
        verbose=0,
    )

    print("\n--- Fit Isolation Forest ---")
    print(f"  train_features={x_train_feat.shape}")
    print(f"  n_estimators={cfg.N_ESTIMATORS}, contamination={cfg.CONTAMINATION}")

    model.fit(x_train_feat)
    return model


def compute_anomaly_scores(model: IsolationForest, x_feat: np.ndarray) -> np.ndarray:
    """
    sklearn 的 score_samples 越大越正常。
    这里取负号，使得 scores 越大越异常，与原训练脚本 evaluate/POT 保持一致。
    """
    return -model.score_samples(x_feat)


# ====================================================================
#  Main
# ====================================================================

def main() -> None:
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print("\nIsolation Forest Unsupervised Anomaly Detection")
    print("=" * 55)
    print(f"DATA_DIR     : {cfg.DATA_DIR}")
    print(f"FEATURE_MODE : {cfg.FEATURE_MODE}")
    print(f"SAVE_DIR     : {cfg.SAVE_DIR}")

    # ── 数据加载与划分 ──
    x, y, metadata = load_data(cfg.DATA_DIR, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    print(f"\nLoaded: X={x.shape}, y={y.shape}")

    train_data, val_data, test_data, split_info = split_data_unsupervised(
        x,
        y,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
    )
    print_split_info(split_info)

    x_train, y_train = unpack_split(train_data)
    x_val, y_val = unpack_split(val_data)
    x_test, y_test = unpack_split(test_data)

    y_train = to_window_labels(y_train)
    y_val = to_window_labels(y_val)
    y_test = to_window_labels(y_test)

    # ── 窗口特征化 ──
    x_train_feat = build_window_features(x_train, cfg.FEATURE_MODE)
    x_val_feat = build_window_features(x_val, cfg.FEATURE_MODE)
    x_test_feat = build_window_features(x_test, cfg.FEATURE_MODE)

    print("\n--- Feature Shape ---")
    print(f"  train: {x_train_feat.shape}")
    print(f"  val  : {x_val_feat.shape}")
    print(f"  test : {x_test_feat.shape}")

    # ── 只用正常训练样本拟合，避免异常污染 ──
    if cfg.FIT_NORMAL_ONLY:
        normal_mask = y_train == 0
        if np.sum(normal_mask) > 0:
            x_fit_feat = x_train_feat[normal_mask]
            print(f"\nUse normal train windows only: {x_fit_feat.shape[0]} / {x_train_feat.shape[0]}")
        else:
            x_fit_feat = x_train_feat
            print("\n[Warning] 训练集中没有标签 0，回退为使用全部训练样本。")
    else:
        x_fit_feat = x_train_feat
        print("\nUse all train windows.")

    # ── 标准化：只 fit 训练数据，避免数据泄露 ──
    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit_feat)
    x_train_scaled = scaler.transform(x_train_feat)
    x_val_scaled = scaler.transform(x_val_feat)
    x_test_scaled = scaler.transform(x_test_feat)

    # ── 训练 Isolation Forest ──
    model = fit_isolation_forest(x_fit_scaled, cfg)

    # 训练集分数只用于观察，不参与测试评估
    train_scores = compute_anomaly_scores(model, x_train_scaled)
    val_scores = compute_anomaly_scores(model, x_val_scaled)
    test_scores = compute_anomaly_scores(model, x_test_scaled)

    print("\n--- Score Range ---")
    print(f"  train: [{train_scores.min():.6f}, {train_scores.max():.6f}]")
    print(f"  val  : [{val_scores.min():.6f}, {val_scores.max():.6f}]")
    print(f"  test : [{test_scores.min():.6f}, {test_scores.max():.6f}]")

    # ── POT 阈值：只在验证集正常样本分数上拟合 ──
    print("\n--- POT Threshold ---")
    val_normal_scores = val_scores[y_val == 0]
    if len(val_normal_scores) == 0:
        raise ValueError("验证集中没有正常窗口，无法拟合 POT 阈值。")

    pot_th = pot_threshold(val_normal_scores, q=cfg.POT_Q, risk=cfg.POT_RISK)

    val_results = evaluate(val_scores, y_val, pot_th)
    print_results(val_results, title="Validation Results (POT Threshold)")

    test_results = evaluate(test_scores, y_test, pot_th)
    print_results(test_results, title="Test Results (POT Threshold)")

    # ── 验证集网格搜索阈值，然后迁移到测试集 ──
    val_f1_result = best_f1_search(
        val_scores,
        y_val,
        n_steps=cfg.THRESHOLD_SEARCH_STEPS,
    )
    val_grid_th = val_f1_result["threshold"]
    test_results_val_th = evaluate(test_scores, y_test, val_grid_th)
    print_results(test_results_val_th, title="Test Results (Val Grid-Search Threshold)")

    # ── 测试集 Oracle F1*：仅用于论文/实验对照，不应作为真实部署阈值 ──
    f1_star = best_f1_search(
        test_scores,
        y_test,
        n_steps=cfg.THRESHOLD_SEARCH_STEPS,
    )

    print(f"\n{'=' * 55}")
    print("  Test F1* (Oracle Threshold)")
    print(f"{'=' * 55}")
    print(f"  Threshold  : {f1_star['threshold']:.6f}")
    print(f"  F1*        : {f1_star['f1_star']:.4f}")
    print(f"  Precision  : {f1_star['precision']:.4f}")
    print(f"  Recall     : {f1_star['recall']:.4f}")
    print(f"{'=' * 55}")

    # ── 保存结果 ──
    if cfg.SAVE_DIR:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

        model_path = os.path.join(cfg.SAVE_DIR, "IsolationForest_model.pkl")
        scaler_path = os.path.join(cfg.SAVE_DIR, "IsolationForest_scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        np.save(os.path.join(cfg.SAVE_DIR, "IsolationForest_train_scores.npy"), train_scores)
        np.save(os.path.join(cfg.SAVE_DIR, "IsolationForest_val_scores.npy"), val_scores)
        np.save(os.path.join(cfg.SAVE_DIR, "IsolationForest_test_scores.npy"), test_scores)
        np.save(os.path.join(cfg.SAVE_DIR, "IsolationForest_test_labels.npy"), y_test)

        output = {
            "config": asdict(cfg),
            "feature_shape": {
                "train": list(x_train_feat.shape),
                "val": list(x_val_feat.shape),
                "test": list(x_test_feat.shape),
            },
            "val": val_results,
            "test": test_results,
            "test_val_th": test_results_val_th,
            "val_f1_search": val_f1_result,
            "f1_star": f1_star,
        }

        with open(os.path.join(cfg.SAVE_DIR, "IsolationForest_results.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()
