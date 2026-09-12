# -*- coding: utf-8 -*-
"""
从零训练并评估 TranAD / DTAAD 静态基准。

使用方式
--------
1. 在 PyCharm 中打开本文件；
2. 修改下方“用户配置区”中的路径和实验列表；
3. 直接点击 Run。

脚本行为
--------
- 不要求已有 checkpoint；
- 若 checkpoint 不存在，则从零训练并保存最佳模型；
- 若 checkpoint 已存在，则直接加载并评估；
- 数据按时间顺序划分为 40% / 20% / 20% / 20%：
    0%-40%   Source Train
    40%-60%  Drift Baseline（本脚本只保留划分，不参与静态模型训练）
    60%-80%  Validation
    80%-100% Online Test
- StandardScaler 只在 Source Train 的全正常窗口上拟合；
- 模型只在 Source Train 的全正常窗口上训练；
- Early stopping 与检测阈值只使用 Validation 的全正常窗口；
- 默认删除后续集合开头的4个重叠窗口，避免滑窗边界泄漏；
- 输出 TP、FP、TN、FN、Precision、Recall、FPR、F1、AUROC、AUPRC；
- 输出验证正常分数 Q0.99 / Q0.995 / Q0.999 下的测试结果；
- 输出 JSON、CSV、checkpoint，可选保存原始分数。

注意
----
TranAD.py 和 DTAAD.py 必须与本脚本放在可访问路径中。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 用户配置区：只需要修改这里，然后直接在 PyCharm 中运行
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

# 模型文件路径
TRANAD_MODEL_FILE = PROJECT_DIR / "TranAD.py"
DTAAD_MODEL_FILE = PROJECT_DIR / "DTAAD.py"

# 结果输出目录
OUTPUT_DIR = PROJECT_DIR / "baseline_outputs"

# 要运行的实验。
# enabled=False 可暂时跳过某项实验。
EXPERIMENTS = [
    {
        "enabled": False,
        "dataset_name": "CICIDS2017",
        "data_dir": Path("/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"),
        "model_name": "tranad",
    },
    {
        "enabled": False,
        "dataset_name": "CICIDS2017",
        "data_dir": Path("/home/ubuntu/wyh/cicdis/cicids2017/integrated_windows"),
        "model_name": "dtaad",
    },
    {
        "enabled": False,
        "dataset_name": "CICIDS2018",
        "data_dir": Path("/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"),
        "model_name": "tranad",
    },
    {
        "enabled": True,
        "dataset_name": "CICIDS2018",
        "data_dir": Path("/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows"),
        "model_name": "dtaad",
    },
]

# 数据参数
WINDOW_SIZE = 100
STEP_SIZE = 20
INPUT_DIM = 68
NORMAL_LABEL = 0

# 时间划分：40 / 20 / 20 / 20
TRAIN_RATIO = 0.40
DRIFT_BASELINE_RATIO = 0.20
VALIDATION_RATIO = 0.20
PURGE_OVERLAP_AT_BOUNDARIES = True

# 模型参数
# 68和136都能被4整除，因此两个模型都可使用4个注意力头。
N_HEADS = 4
DIM_FEEDFORWARD = 16
OUTPUT_ACTIVATION = "identity"
TRANAD_SCORE_ALPHA = 0.20

# 训练参数
SEED = 2026
EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0  # Windows/PyCharm下建议保持0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 调试或显存不足时可限制窗口数；正式实验必须设为None。
MAX_TRAIN_WINDOWS = None
MAX_VAL_NORMAL_WINDOWS = None

# 阈值
NORMAL_SCORE_QUANTILES = (0.99, 0.995, 0.999)
PRIMARY_QUANTILE = 0.995

# 输出
SAVE_RAW_SCORES = True
FORCE_RETRAIN = False  # True：忽略已有checkpoint并重新训练


# =============================================================================
# 工具函数
# =============================================================================

SPLIT_NAMES = ("source_train", "drift_baseline", "validation", "online_test")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽量保证可复现；若某些CUDA算子不支持确定性，仍可能有极小差异。
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def choose_device() -> torch.device:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，自动切换到CPU。")
        return torch.device("cpu")
    return torch.device(DEVICE)


def load_arrays(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    x_file = data_dir / f"integrated_X_w{WINDOW_SIZE}_s{STEP_SIZE}.npy"
    y_file = data_dir / f"integrated_y_w{WINDOW_SIZE}_s{STEP_SIZE}.npy"
    metadata_file = (
        data_dir / f"integrated_metadata_w{WINDOW_SIZE}_s{STEP_SIZE}.pkl"
    )

    if not x_file.exists():
        raise FileNotFoundError(f"未找到X文件：{x_file}")
    if not y_file.exists():
        raise FileNotFoundError(f"未找到y文件：{y_file}")

    # CICIDS2018体积较大，使用mmap避免一次性占满内存。
    X = np.load(str(x_file), mmap_mode="r")
    y = np.load(str(y_file), mmap_mode="r")

    metadata = None
    if metadata_file.exists():
        with metadata_file.open("rb") as f:
            metadata = pickle.load(f)

    if X.ndim != 3 or y.ndim != 2:
        raise ValueError(f"期望X=[N,W,F], y=[N,W]，实际X={X.shape}, y={y.shape}")
    if X.shape[:2] != y.shape:
        raise ValueError(f"X和y窗口维度不一致：X={X.shape}, y={y.shape}")
    if X.shape[1] != WINDOW_SIZE or X.shape[2] != INPUT_DIM:
        raise ValueError(
            f"数据维度与配置不一致：X={X.shape}, "
            f"配置W={WINDOW_SIZE}, F={INPUT_DIM}"
        )
    return X, y, metadata


def build_split_ranges(n_windows: int) -> Dict[str, Tuple[int, int]]:
    train_end = int(n_windows * TRAIN_RATIO)
    baseline_end = int(n_windows * (TRAIN_RATIO + DRIFT_BASELINE_RATIO))
    val_end = int(
        n_windows * (TRAIN_RATIO + DRIFT_BASELINE_RATIO + VALIDATION_RATIO)
    )

    ranges = {
        "source_train": (0, train_end),
        "drift_baseline": (train_end, baseline_end),
        "validation": (baseline_end, val_end),
        "online_test": (val_end, n_windows),
    }

    if PURGE_OVERLAP_AT_BOUNDARIES:
        purge_windows = int(
            math.ceil((WINDOW_SIZE - STEP_SIZE) / float(STEP_SIZE))
        )
        purged = {}
        for idx, name in enumerate(SPLIT_NAMES):
            start, end = ranges[name]
            if idx > 0:
                start = min(start + purge_windows, end)
            purged[name] = (start, end)
        return purged

    return ranges


def window_binary_labels(y_part: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(y_part) != NORMAL_LABEL, axis=1).astype(np.int64)


def all_normal_mask(y_part: np.ndarray) -> np.ndarray:
    return np.all(np.asarray(y_part) == NORMAL_LABEL, axis=1)


def collect_all_normal_indices(
    y: np.ndarray,
    split_range: Tuple[int, int],
    max_windows: Optional[int],
    seed: int,
) -> np.ndarray:
    start, end = split_range
    mask = all_normal_mask(np.asarray(y[start:end]))
    indices = np.flatnonzero(mask).astype(np.int64) + start

    if indices.size == 0:
        raise RuntimeError(f"区间{split_range}中没有全正常窗口")

    if max_windows is not None and indices.size > max_windows:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(indices, size=max_windows, replace=False)
        )
    return indices


def fit_scaler(
    X: np.ndarray,
    normal_indices: np.ndarray,
    batch_windows: int = 2048,
) -> StandardScaler:
    scaler = StandardScaler()

    for offset in range(0, len(normal_indices), batch_windows):
        batch_indices = normal_indices[offset:offset + batch_windows]
        x_batch = np.asarray(X[batch_indices], dtype=np.float64)
        scaler.partial_fit(x_batch.reshape(-1, x_batch.shape[-1]))

    return scaler


class WindowDataset(Dataset):
    """按索引从mmap数据读取窗口并使用固定Scaler标准化。"""

    def __init__(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        scaler: StandardScaler,
    ) -> None:
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)
        self.scale = np.where(self.scale > 0.0, self.scale, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> torch.Tensor:
        idx = int(self.indices[item])
        x = np.asarray(self.X[idx], dtype=np.float32)
        x = (x - self.mean) / self.scale
        return torch.from_numpy(x.astype(np.float32, copy=False))


def make_loader(
    X: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    shuffle: bool,
) -> DataLoader:
    dataset = WindowDataset(X, indices, scaler)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def dynamic_import_model(model_file: Path, model_name: str):
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_file}")

    module_name = f"_baseline_{model_name}_{abs(hash(str(model_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模型文件：{model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = "TranAD" if model_name == "tranad" else "DTAAD"
    if not hasattr(module, class_name):
        raise AttributeError(f"{model_file}中没有找到类{class_name}")
    return getattr(module, class_name)


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "tranad":
        model_file = TRANAD_MODEL_FILE
    elif model_name == "dtaad":
        model_file = DTAAD_MODEL_FILE
    else:
        raise ValueError(f"未知模型：{model_name}")

    model_cls = dynamic_import_model(model_file, model_name)
    return model_cls(
        input_dim=INPUT_DIM,
        window_size=WINDOW_SIZE,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        output_activation=OUTPUT_ACTIVATION,
    )


def compute_training_loss(
    model: torch.nn.Module,
    model_name: str,
    batch_x: torch.Tensor,
    epoch: int,
) -> torch.Tensor:
    if model_name == "tranad":
        result = model.compute_loss(batch_x, epoch=epoch)
    else:
        result = model.compute_loss(batch_x)
    return result["loss"]


def compute_scores(
    model: torch.nn.Module,
    model_name: str,
    batch_x: torch.Tensor,
) -> torch.Tensor:
    if model_name == "tranad":
        return model.compute_anomaly_score(
            batch_x,
            alpha=TRANAD_SCORE_ALPHA,
        )
    return model.compute_anomaly_score(batch_x)


def evaluate_normal_loss(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_windows = 0

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            loss = compute_training_loss(model, model_name, batch_x, epoch)
            batch_count = int(batch_x.shape[0])
            total_loss += float(loss.item()) * batch_count
            total_windows += batch_count

    if total_windows == 0:
        raise RuntimeError("验证正常DataLoader为空")
    return total_loss / total_windows


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    scaler: StandardScaler,
    epoch: int,
    best_val_loss: float,
    dataset_name: str,
    model_name: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "dataset_name": dataset_name,
            "model_name": model_name,
            "model_config": {
                "input_dim": INPUT_DIM,
                "window_size": WINDOW_SIZE,
                "n_heads": N_HEADS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "output_activation": OUTPUT_ACTIVATION,
                "tranad_score_alpha": (
                    TRANAD_SCORE_ALPHA if model_name == "tranad" else None
                ),
            },
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seed": SEED,
        },
        str(checkpoint_path),
    )


def torch_load_compat(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def train_or_load_model(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    train_loader: DataLoader,
    val_normal_loader: DataLoader,
    scaler: StandardScaler,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if checkpoint_path.exists() and not FORCE_RETRAIN:
        checkpoint = torch_load_compat(checkpoint_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        print(f"[LOAD] 已加载checkpoint：{checkpoint_path}")
        return model, {
            "loaded_existing_checkpoint": True,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        }

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    print(f"[TRAIN] {dataset_name} / {model_name.upper()} 从零开始训练")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        total_windows = 0
        start_time = time.time()

        for batch_x in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = compute_training_loss(model, model_name, batch_x, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"训练出现非有限loss：epoch={epoch}, loss={loss.item()}"
                )

            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )
            optimizer.step()

            batch_count = int(batch_x.shape[0])
            epoch_loss += float(loss.item()) * batch_count
            total_windows += batch_count

        train_loss = epoch_loss / max(total_windows, 1)
        val_loss = evaluate_normal_loss(
            model=model,
            model_name=model_name,
            loader=val_normal_loader,
            device=device,
            epoch=epoch,
        )
        elapsed = time.time() - start_time

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_normal_loss": float(val_loss),
                "seconds": float(elapsed),
            }
        )
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.8f} | "
            f"val_normal={val_loss:.8f} | "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            print(f"    [SAVE] 保存最佳模型：{checkpoint_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"    [EARLY STOP] 连续{PATIENCE}轮未改善，停止训练。"
                )
                break

    checkpoint = torch_load_compat(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, {
        "loaded_existing_checkpoint": False,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }


def score_indices(
    model: torch.nn.Module,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        X=X,
        indices=indices,
        scaler=scaler,
        shuffle=False,
    )

    scores_list: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            scores = compute_scores(model, model_name, batch_x)
            scores_list.append(
                scores.detach().cpu().numpy().reshape(-1).astype(np.float64)
            )

    scores = np.concatenate(scores_list)
    labels = window_binary_labels(np.asarray(y[indices]))
    return scores, labels


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        pred,
        labels=[0, 1],
    ).ravel()

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "FPR": fpr,
        "specificity": specificity,
        "F1": f1,
        "accuracy": accuracy,
        "predicted_attack_ratio": float(np.mean(pred)),
    }


def ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Optional[float]]:
    if np.unique(labels).size < 2:
        return {"AUROC": None, "AUPRC": None}
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
    }


def best_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    if np.unique(labels).size < 2:
        threshold = float(np.median(scores))
        return threshold, classification_metrics(labels, scores, threshold)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        threshold = float(np.median(scores))
        return threshold, classification_metrics(labels, scores, threshold)

    denom = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_values))
    threshold = float(thresholds[best_idx])
    return threshold, classification_metrics(labels, scores, threshold)


def score_summary(scores: np.ndarray) -> Dict[str, float]:
    quantiles = (0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
    return {
        f"q{q:g}": float(np.quantile(scores, q))
        for q in quantiles
    }


def normalize_label_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def extract_label_map(metadata: Any) -> Dict[str, str]:
    if not isinstance(metadata, Mapping):
        return {}

    for key in (
        "label_mapping",
        "label_map",
        "id_to_label",
        "class_names",
        "attack_label_map",
    ):
        raw = metadata.get(key)
        if not isinstance(raw, Mapping):
            continue

        result: Dict[str, str] = {}
        for key_value, name_value in raw.items():
            if isinstance(name_value, (int, np.integer)):
                result[normalize_label_key(name_value)] = str(key_value)
            else:
                result[normalize_label_key(key_value)] = str(name_value)
        if result:
            return result
    return {}


def per_attack_window_recall(
    y: np.ndarray,
    test_indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metadata: Any,
) -> Dict[str, Dict[str, Any]]:
    y_test = np.asarray(y[test_indices])
    pred = scores >= threshold
    label_map = extract_label_map(metadata)
    result: Dict[str, Dict[str, Any]] = {}

    for attack_id in np.unique(y_test):
        if normalize_label_key(attack_id) == normalize_label_key(NORMAL_LABEL):
            continue

        mask = np.any(y_test == attack_id, axis=1)
        count = int(np.sum(mask))
        if count == 0:
            continue

        key = normalize_label_key(attack_id)
        name = label_map.get(key, f"label_{key}")
        result[name] = {
            "window_count": count,
            "detected_windows": int(np.sum(pred[mask])),
            "window_recall": float(np.mean(pred[mask])),
        }
    return result


def split_summary(
    y: np.ndarray,
    ranges: Mapping[str, Tuple[int, int]],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for name, (start, end) in ranges.items():
        labels = window_binary_labels(np.asarray(y[start:end]))
        result[name] = {
            "window_index_range": [int(start), int(end)],
            "total_windows": int(labels.size),
            "normal_windows": int(np.sum(labels == 0)),
            "attack_windows": int(np.sum(labels == 1)),
            "attack_window_ratio": float(np.mean(labels)),
        }
    return result


def write_metrics_csv(
    output_path: Path,
    dataset_name: str,
    model_name: str,
    threshold_results: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = [
        "dataset",
        "model",
        "threshold_name",
        "threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "FPR",
        "specificity",
        "F1",
        "accuracy",
        "predicted_attack_ratio",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for threshold_name, metrics in threshold_results.items():
            row = {
                "dataset": dataset_name,
                "model": model_name.upper(),
                "threshold_name": threshold_name,
            }
            for field in fields:
                if field in metrics:
                    row[field] = metrics[field]
            writer.writerow(row)


def print_metric_line(name: str, metrics: Mapping[str, Any]) -> None:
    print(
        f"{name:<30}"
        f"TP={metrics['TP']:>6d} "
        f"FP={metrics['FP']:>6d} "
        f"TN={metrics['TN']:>6d} "
        f"FN={metrics['FN']:>6d} | "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"FPR={metrics['FPR']:.4f} "
        f"F1={metrics['F1']:.4f}"
    )


def run_experiment(experiment: Mapping[str, Any]) -> Path:
    dataset_name = str(experiment["dataset_name"])
    data_dir = Path(experiment["data_dir"]).expanduser().resolve()
    model_name = str(experiment["model_name"]).lower()

    if model_name not in {"tranad", "dtaad"}:
        raise ValueError(f"model_name必须是tranad或dtaad，当前为{model_name}")

    set_global_seed(SEED)
    device = choose_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        OUTPUT_DIR / f"{dataset_name}_{model_name}_best_checkpoint.pth"
    )
    json_path = (
        OUTPUT_DIR / f"{dataset_name}_{model_name}_static_baseline_metrics.json"
    )
    csv_path = json_path.with_suffix(".csv")
    score_path = json_path.with_suffix(".npz")

    print("\n" + "=" * 120)
    print(f"Experiment: {dataset_name} / {model_name.upper()}")
    print(f"Data:       {data_dir}")
    print(f"Device:     {device}")
    print("=" * 120)

    X, y, metadata = load_arrays(data_dir)
    ranges = build_split_ranges(len(X))
    summaries = split_summary(y, ranges)

    for split_name in SPLIT_NAMES:
        info = summaries[split_name]
        print(
            f"{split_name:<16}: "
            f"total={info['total_windows']:,}, "
            f"normal={info['normal_windows']:,}, "
            f"attack={info['attack_windows']:,}, "
            f"attack_ratio={info['attack_window_ratio']:.4f}"
        )

    print("\n[1/5] 收集Source Train与Validation正常窗口...")
    train_normal_indices = collect_all_normal_indices(
        y=y,
        split_range=ranges["source_train"],
        max_windows=MAX_TRAIN_WINDOWS,
        seed=SEED,
    )
    val_normal_indices = collect_all_normal_indices(
        y=y,
        split_range=ranges["validation"],
        max_windows=MAX_VAL_NORMAL_WINDOWS,
        seed=SEED + 1,
    )
    print(f"  Train normal windows: {len(train_normal_indices):,}")
    print(f"  Val normal windows:   {len(val_normal_indices):,}")

    print("\n[2/5] 仅在Source Train正常窗口上拟合StandardScaler...")
    scaler = fit_scaler(X, train_normal_indices)
    train_loader = make_loader(
        X=X,
        indices=train_normal_indices,
        scaler=scaler,
        shuffle=True,
    )
    val_normal_loader = make_loader(
        X=X,
        indices=val_normal_indices,
        scaler=scaler,
        shuffle=False,
    )

    print("\n[3/5] 构建、训练或加载模型...")
    model = build_model(model_name)
    model, training_info = train_or_load_model(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        train_loader=train_loader,
        val_normal_loader=val_normal_loader,
        scaler=scaler,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print("\n[4/5] 计算Validation和Online Test异常分数...")
    val_start, val_end = ranges["validation"]
    test_start, test_end = ranges["online_test"]
    val_indices = np.arange(val_start, val_end, dtype=np.int64)
    test_indices = np.arange(test_start, test_end, dtype=np.int64)

    val_scores, val_labels = score_indices(
        model=model,
        model_name=model_name,
        X=X,
        y=y,
        indices=val_indices,
        scaler=scaler,
        device=device,
    )
    test_scores, test_labels = score_indices(
        model=model,
        model_name=model_name,
        X=X,
        y=y,
        indices=test_indices,
        scaler=scaler,
        device=device,
    )

    val_normal_scores = val_scores[val_labels == 0]
    if val_normal_scores.size < 2:
        raise RuntimeError("Validation正常窗口不足，无法生成分位数阈值")

    threshold_results: Dict[str, Dict[str, Any]] = {}
    thresholds: Dict[str, float] = {}

    for quantile in sorted(set(NORMAL_SCORE_QUANTILES + (PRIMARY_QUANTILE,))):
        threshold_name = f"val_normal_q{quantile:g}"
        threshold = float(np.quantile(val_normal_scores, quantile))
        thresholds[threshold_name] = threshold
        threshold_results[threshold_name] = classification_metrics(
            test_labels,
            test_scores,
            threshold,
        )

    # 可部署的替代方案：使用整个Validation的标签选择阈值。
    # 该结果可以用于诊断，但若论文强调纯正常验证阈值，则不要作为主结果。
    best_val_threshold, _ = best_f1_threshold(val_labels, val_scores)
    threshold_results["best_validation_F1_diagnostic"] = classification_metrics(
        test_labels,
        test_scores,
        best_val_threshold,
    )

    # 仅用于观察模型上限，绝不能作为论文主结果。
    _, oracle_test_metrics = best_f1_threshold(test_labels, test_scores)
    threshold_results[
        "oracle_best_test_F1_DIAGNOSTIC_ONLY"
    ] = oracle_test_metrics

    primary_name = f"val_normal_q{PRIMARY_QUANTILE:g}"
    primary_metrics = threshold_results[primary_name]

    print("\n[5/5] 测试结果")
    print("-" * 120)
    for threshold_name, metrics in threshold_results.items():
        print_metric_line(threshold_name, metrics)

    val_ranking = ranking_metrics(val_labels, val_scores)
    test_ranking = ranking_metrics(test_labels, test_scores)
    print(
        f"\nValidation AUROC={val_ranking['AUROC']}, "
        f"AUPRC={val_ranking['AUPRC']}"
    )
    print(
        f"Test       AUROC={test_ranking['AUROC']}, "
        f"AUPRC={test_ranking['AUPRC']}"
    )

    report = {
        "dataset": dataset_name,
        "model": model_name.upper(),
        "protocol": {
            "ratios": {
                "source_train": TRAIN_RATIO,
                "drift_baseline": DRIFT_BASELINE_RATIO,
                "validation": VALIDATION_RATIO,
                "online_test": (
                    1.0
                    - TRAIN_RATIO
                    - DRIFT_BASELINE_RATIO
                    - VALIDATION_RATIO
                ),
            },
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "purge_overlap_at_boundaries": PURGE_OVERLAP_AT_BOUNDARIES,
            "evaluation_level": "window",
            "window_label_rule": (
                "attack if any flow label in the window is non-zero"
            ),
            "scaler_rule": (
                "StandardScaler fitted only on source-train all-normal windows"
            ),
            "threshold_rule": (
                "primary threshold is a validation-normal score quantile"
            ),
        },
        "model_config": {
            "input_dim": INPUT_DIM,
            "n_heads": N_HEADS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "output_activation": OUTPUT_ACTIVATION,
            "tranad_score_alpha": (
                TRANAD_SCORE_ALPHA if model_name == "tranad" else None
            ),
            "seed": SEED,
        },
        "training_config": {
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "max_train_windows": MAX_TRAIN_WINDOWS,
            "max_val_normal_windows": MAX_VAL_NORMAL_WINDOWS,
        },
        "training_info": training_info,
        "checkpoint_path": str(checkpoint_path),
        "split_summary": summaries,
        "train_normal_window_count": int(len(train_normal_indices)),
        "val_normal_window_count": int(len(val_normal_indices)),
        "score_summary": {
            "validation_all": score_summary(val_scores),
            "validation_normal": score_summary(val_normal_scores),
            "validation_attack": (
                score_summary(val_scores[val_labels == 1])
                if np.any(val_labels == 1)
                else None
            ),
            "test_all": score_summary(test_scores),
            "test_normal": score_summary(test_scores[test_labels == 0]),
            "test_attack": score_summary(test_scores[test_labels == 1]),
        },
        "ranking_metrics": {
            "validation": val_ranking,
            "test": test_ranking,
        },
        "threshold_results_on_test": threshold_results,
        "primary_threshold_name": primary_name,
        "primary_threshold": thresholds[primary_name],
        "primary_test_metrics": primary_metrics,
        "per_attack_window_recall_at_primary_threshold": (
            per_attack_window_recall(
                y=y,
                test_indices=test_indices,
                scores=test_scores,
                threshold=thresholds[primary_name],
                metadata=metadata,
            )
        ),
        "warning": (
            "oracle_best_test_F1_DIAGNOSTIC_ONLY uses test labels and must "
            "not be used as the main paper result."
        ),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_metrics_csv(
        output_path=csv_path,
        dataset_name=dataset_name,
        model_name=model_name,
        threshold_results=threshold_results,
    )

    if SAVE_RAW_SCORES:
        np.savez_compressed(
            score_path,
            validation_scores=val_scores,
            validation_labels=val_labels,
            test_scores=test_scores,
            test_labels=test_labels,
        )

    print("\n输出文件：")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  JSON:       {json_path}")
    print(f"  CSV:        {csv_path}")
    if SAVE_RAW_SCORES:
        print(f"  Scores:     {score_path}")

    return json_path


def validate_global_config() -> None:
    ratio_sum = TRAIN_RATIO + DRIFT_BASELINE_RATIO + VALIDATION_RATIO
    if not 0.0 < ratio_sum < 1.0:
        raise ValueError("前三个划分比例之和必须位于(0,1)")
    if WINDOW_SIZE <= 0 or STEP_SIZE <= 0 or INPUT_DIM <= 0:
        raise ValueError("WINDOW_SIZE、STEP_SIZE和INPUT_DIM必须为正")
    if STEP_SIZE > WINDOW_SIZE:
        raise ValueError("STEP_SIZE不能大于WINDOW_SIZE")
    if INPUT_DIM % N_HEADS != 0:
        raise ValueError(
            f"DTAAD要求INPUT_DIM={INPUT_DIM}能被N_HEADS={N_HEADS}整除"
        )
    if (2 * INPUT_DIM) % N_HEADS != 0:
        raise ValueError(
            f"TranAD要求2*INPUT_DIM={2 * INPUT_DIM}能被N_HEADS={N_HEADS}整除"
        )
    if OUTPUT_ACTIVATION not in {"identity", "sigmoid"}:
        raise ValueError("OUTPUT_ACTIVATION必须是identity或sigmoid")
    if not 0.0 <= TRANAD_SCORE_ALPHA <= 1.0:
        raise ValueError("TRANAD_SCORE_ALPHA必须位于[0,1]")


def main() -> None:
    validate_global_config()
    enabled_experiments = [
        item for item in EXPERIMENTS if bool(item.get("enabled", True))
    ]
    if not enabled_experiments:
        raise RuntimeError("EXPERIMENTS中没有启用的实验")

    completed: List[Path] = []
    failed: List[Tuple[str, str, str]] = []

    for experiment in enabled_experiments:
        dataset_name = str(experiment.get("dataset_name"))
        model_name = str(experiment.get("model_name"))

        try:
            completed.append(run_experiment(experiment))
        except Exception as exc:
            failed.append((dataset_name, model_name, repr(exc)))
            print(
                f"\n[ERROR] {dataset_name}/{model_name} 运行失败：{exc}",
                file=sys.stderr,
            )

    print("\n" + "=" * 120)
    print("全部实验执行完毕")
    print("=" * 120)

    if completed:
        print("成功输出：")
        for path in completed:
            print(f"  - {path}")

    if failed:
        print("\n失败实验：")
        for dataset_name, model_name, message in failed:
            print(f"  - {dataset_name}/{model_name}: {message}")
        raise RuntimeError(
            "存在实验失败，请根据上方错误信息检查路径、显存或模型配置。"
        )


if __name__ == "__main__":
    main()
