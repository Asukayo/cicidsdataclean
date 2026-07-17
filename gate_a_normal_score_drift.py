#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate A: frozen LSTM-AE normal-score drift diagnostic.

This script reuses:

* secondPaper.provider.unsupervised_provider
* secondPaper.models.LSTMAE.LSTMAE

Protocol:

1. Keep the provider's chronological train/validation/test split.
2. Train LSTM-AE on normal training windows only.
3. Select one fixed threshold from normal validation-window scores.
4. Freeze the model and threshold for the complete test stream.
5. In consecutive test blocks, use labels only offline to summarize the
   reconstruction scores and false-positive rate of normal windows.

Run from the cicidsdataclean repository root, for example:

    python gate_a_normal_score_drift.py \
        --data-dir /home/ubuntu/wyh/cicdis/cicids2017/integrated_windows \
        --output-dir ./ThirdPaper/outputs/gate_a/cicids2017_seed42

The script does not update the model on validation or test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _add_project_root_to_path() -> Path:
    """Find a directory containing ``secondPaper`` and add it to sys.path."""
    candidates: List[Path] = [Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        provider = candidate / "secondPaper" / "provider" / "unsupervised_provider.py"
        if provider.is_file():
            sys.path.insert(0, str(candidate))
            return candidate

    raise RuntimeError(
        "Cannot find the cicidsdataclean project root. Run this script from the "
        "repository root or place it inside that repository."
    )


PROJECT_ROOT = _add_project_root_to_path()

from secondPaper.models.LSTMAE import LSTMAE  # noqa: E402
from secondPaper.provider.unsupervised_provider import (  # noqa: E402
    create_data_loaders,
    load_data,
    print_split_info,
    split_data_unsupervised,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frozen LSTM-AE normal-window anomaly scores over time."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing integrated_*.npy files.")
    parser.add_argument("--output-dir", default="./gate_a_outputs", help="Directory for figures and metrics.")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--block-size", type=int, default=512, help="Number of chronological test windows per block.")
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Fixed threshold quantile of normal validation scores; 0.95 targets 5%% validation FPR.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use CUDA automatically when available unless explicitly overridden.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1).")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be smaller than 1.")
    if not (0.0 < args.threshold_quantile < 1.0):
        raise ValueError("--threshold-quantile must be in (0, 1).")
    if args.block_size <= 0 or args.batch_size <= 0:
        raise ValueError("--block-size and --batch-size must be positive.")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_loader_workers(loaders: Iterable[torch.utils.data.DataLoader], num_workers: int) -> None:
    # DataLoader.num_workers is read-only after construction. The provider uses
    # zero workers, which is safe and deterministic; keep the argument only to
    # make this behavior explicit instead of silently pretending to change it.
    if num_workers != 0:
        print("[warning] The supplied provider constructs DataLoaders with num_workers=0; --num-workers is ignored.")


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_windows = 0

    for batch_x, batch_mark, _ in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_mark = batch_mark.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch_x, batch_mark)["loss"]
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_n = batch_x.size(0)
        total_loss += float(loss.detach().item()) * batch_n
        total_windows += batch_n

    if total_windows == 0:
        raise RuntimeError("The normal training loader is empty.")
    return total_loss / total_windows


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return scores and labels in DataLoader order."""
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch_x, batch_mark, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_mark = batch_mark.to(device, non_blocking=True)
        score = model.compute_anomaly_score(batch_x, batch_mark)

        all_scores.append(score.detach().cpu().numpy().reshape(-1))
        all_labels.append(batch_y.detach().cpu().numpy().reshape(-1).astype(np.int64))

    if not all_scores:
        raise RuntimeError("Cannot score an empty loader.")
    return np.concatenate(all_scores), np.concatenate(all_labels)


def fit_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, List[Dict[str, float]], int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_scores, val_labels = collect_scores(model, val_loader, device)
        val_normal = val_scores[val_labels == 0]
        if val_normal.size == 0:
            raise RuntimeError("Validation split contains no normal window.")
        val_loss = float(np.mean(val_normal))

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_normal_mse": val_loss,
                "seconds": float(time.time() - started),
            }
        )
        print(
            "Epoch {0:03d} | train={1:.6f} | val_normal={2:.6f} | {3:.1f}s".format(
                epoch, train_loss, val_loss, history[-1]["seconds"]
            )
        )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping at epoch {0}; best epoch={1}.".format(epoch, best_epoch))
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, history, best_epoch, best_val


def most_common_text(values: Sequence[Any]) -> str:
    text = [str(value) for value in values if value not in (None, "")]
    if not text:
        return ""
    return Counter(text).most_common(1)[0][0]


def metadata_slice(metadata: Any, start: int, end: int) -> List[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    windows = metadata.get("window_metadata")
    if not isinstance(windows, list) or len(windows) < end:
        return []
    return windows[start:end]


def optional_distribution_distances(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return KS statistic, KS p-value, and Wasserstein distance if SciPy exists."""
    if reference.size == 0 or current.size == 0:
        return None, None, None
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        try:
            ks_result = ks_2samp(
                reference,
                current,
                alternative="two-sided",
                method="auto",
            )
        except TypeError:
            # Compatibility with the older SciPy version used by the original
            # second-paper conda environment.
            ks_result = ks_2samp(reference, current)
        return (
            float(getattr(ks_result, "statistic", ks_result[0])),
            float(getattr(ks_result, "pvalue", ks_result[1])),
            float(wasserstein_distance(reference, current)),
        )
    except ImportError:
        return None, None, None


def summarize_blocks(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    block_size: int,
    val_normal_scores: np.ndarray,
    test_metadata: Sequence[Dict[str, Any]],
    test_global_start: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for block_id, start in enumerate(range(0, len(scores), block_size)):
        end = min(start + block_size, len(scores))
        block_scores = scores[start:end]
        block_labels = labels[start:end]
        normal_scores = block_scores[block_labels == 0]
        block_meta = test_metadata[start:end] if test_metadata else []

        ks_stat, ks_pvalue, wasserstein = optional_distribution_distances(
            val_normal_scores, normal_scores
        )

        if normal_scores.size:
            mean = float(np.mean(normal_scores))
            median = float(np.median(normal_scores))
            q25, q75, q90, q95 = [
                float(value)
                for value in np.quantile(normal_scores, [0.25, 0.75, 0.90, 0.95])
            ]
            fpr = float(np.mean(normal_scores > threshold))
        else:
            mean = median = q25 = q75 = q90 = q95 = fpr = None

        sources = [item.get("source_file") for item in block_meta if isinstance(item, dict)]
        start_times = [item.get("start_time") for item in block_meta if isinstance(item, dict)]
        end_times = [item.get("end_time") for item in block_meta if isinstance(item, dict)]

        rows.append(
            {
                "block_id": block_id,
                "test_start_index": start,
                "test_end_index_exclusive": end,
                "global_start_index": test_global_start + start,
                "global_end_index_exclusive": test_global_start + end,
                "n_windows": int(end - start),
                "n_normal": int(np.sum(block_labels == 0)),
                "n_anomalous": int(np.sum(block_labels == 1)),
                "attack_window_rate": float(np.mean(block_labels == 1)),
                "normal_score_mean": mean,
                "normal_score_median": median,
                "normal_score_q25": q25,
                "normal_score_q75": q75,
                "normal_score_q90": q90,
                "normal_score_q95": q95,
                "normal_fpr": fpr,
                "ks_vs_val_normal": ks_stat,
                "ks_pvalue_vs_val_normal": ks_pvalue,
                "wasserstein_vs_val_normal": wasserstein,
                "dominant_source_file": most_common_text(sources),
                "block_start_time": str(start_times[0]) if start_times else "",
                "block_end_time": str(end_times[-1]) if end_times else "",
            }
        )

    return rows


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def period_stats(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Any]:
    normal = scores[labels == 0]
    if normal.size == 0:
        return {
            "n_windows": int(scores.size),
            "n_normal": 0,
            "normal_score_mean": None,
            "normal_score_median": None,
            "normal_score_q90": None,
            "normal_fpr": None,
        }
    return {
        "n_windows": int(scores.size),
        "n_normal": int(normal.size),
        "normal_score_mean": float(np.mean(normal)),
        "normal_score_median": float(np.median(normal)),
        "normal_score_q90": float(np.quantile(normal, 0.90)),
        "normal_fpr": float(np.mean(normal > threshold)),
    }


def build_shift_summary(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    period_n = max(1, int(math.ceil(len(test_scores) * 0.20)))
    early_scores = test_scores[:period_n]
    early_labels = test_labels[:period_n]
    late_scores = test_scores[-period_n:]
    late_labels = test_labels[-period_n:]

    early = period_stats(early_scores, early_labels, threshold)
    late = period_stats(late_scores, late_labels, threshold)
    early_normal = early_scores[early_labels == 0]
    late_normal = late_scores[late_labels == 0]
    ks_stat, ks_pvalue, wasserstein = optional_distribution_distances(
        early_normal, late_normal
    )

    median_change = None
    median_ratio = None
    fpr_change = None
    if early["normal_score_median"] is not None and late["normal_score_median"] is not None:
        median_change = late["normal_score_median"] - early["normal_score_median"]
        if abs(early["normal_score_median"]) > 1e-12:
            median_ratio = late["normal_score_median"] / early["normal_score_median"]
    if early["normal_fpr"] is not None and late["normal_fpr"] is not None:
        fpr_change = late["normal_fpr"] - early["normal_fpr"]

    valid_rows = [row for row in rows if row["normal_score_median"] is not None]
    spearman_rho = None
    spearman_pvalue = None
    if len(valid_rows) >= 3:
        try:
            from scipy.stats import spearmanr

            result = spearmanr(
                [row["block_id"] for row in valid_rows],
                [row["normal_score_median"] for row in valid_rows],
            )
            spearman_rho = finite_or_none(result.statistic)
            spearman_pvalue = finite_or_none(result.pvalue)
        except ImportError:
            pass

    return {
        "early_test_20_percent": early,
        "late_test_20_percent": late,
        "late_minus_early_normal_median": median_change,
        "late_over_early_normal_median": median_ratio,
        "late_minus_early_normal_fpr": fpr_change,
        "early_vs_late_ks_statistic": ks_stat,
        "early_vs_late_ks_pvalue": ks_pvalue,
        "early_vs_late_wasserstein": wasserstein,
        "spearman_block_vs_normal_median": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No block metric was produced.")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_block_metrics(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    val_normal_scores: np.ndarray,
    threshold: float,
    target_fpr: float,
    title: str,
) -> None:
    valid = [row for row in rows if row["normal_score_median"] is not None]
    if not valid:
        raise RuntimeError("No test block contains normal windows, so Gate A cannot be plotted.")

    x = np.asarray([row["block_id"] for row in valid], dtype=float)
    median = np.asarray([row["normal_score_median"] for row in valid], dtype=float)
    q25 = np.asarray([row["normal_score_q25"] for row in valid], dtype=float)
    q75 = np.asarray([row["normal_score_q75"] for row in valid], dtype=float)
    q90 = np.asarray([row["normal_score_q90"] for row in valid], dtype=float)
    fpr = np.asarray([row["normal_fpr"] for row in valid], dtype=float)
    n_normal = np.asarray([row["n_normal"] for row in valid], dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35]},
    )
    ax_score, ax_fpr = axes

    ax_score.fill_between(x, q25, q75, color="#4C78A8", alpha=0.20, label="Normal-score IQR")
    ax_score.plot(x, median, color="#1F4E79", marker="o", markersize=3.2, linewidth=1.7, label="Normal-score median")
    ax_score.plot(x, q90, color="#F58518", linewidth=1.25, label="Normal-score 90th percentile")
    ax_score.axhline(
        float(np.median(val_normal_scores)),
        color="#54A24B",
        linestyle="--",
        linewidth=1.2,
        label="Validation-normal median",
    )
    ax_score.axhline(
        threshold,
        color="#E45756",
        linestyle=":",
        linewidth=1.7,
        label="Fixed validation threshold",
    )
    ax_score.set_ylabel("Reconstruction anomaly score (MSE)")
    ax_score.grid(axis="y", alpha=0.25)
    ax_score.legend(loc="best", ncol=2, fontsize=9)
    ax_score.set_title(title)

    ax_fpr.plot(x, fpr, color="#B22222", marker="o", markersize=3.2, linewidth=1.6, label="Normal-window FPR")
    ax_fpr.axhline(
        target_fpr,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="Validation target FPR",
    )
    ax_fpr.set_ylim(bottom=0.0)
    ax_fpr.set_ylabel("FPR at fixed threshold")
    ax_fpr.set_xlabel("Chronological deployment block in frozen test stream")
    ax_fpr.grid(axis="y", alpha=0.25)

    ax_count = ax_fpr.twinx()
    width = 0.75 if len(x) > 1 else 0.4
    ax_count.bar(x, n_normal, width=width, color="#BAB0AC", alpha=0.22, label="Normal windows")
    ax_count.set_ylabel("Normal windows per block", color="#666666")

    handles1, labels1 = ax_fpr.get_legend_handles_labels()
    handles2, labels2 = ax_count.get_legend_handles_labels()
    ax_fpr.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=9)

    # Mark source/day changes when metadata is available. These annotations are
    # offline-only diagnostics and are never inputs to the model.
    previous = None
    y_top = ax_score.get_ylim()[1]
    for row in rows:
        source = row.get("dominant_source_file", "")
        if source and source != previous:
            block_id = row["block_id"]
            ax_score.axvline(block_id - 0.5, color="#999999", linewidth=0.7, alpha=0.55)
            short_source = Path(str(source)).stem
            if len(short_source) > 28:
                short_source = short_source[:25] + "..."
            ax_score.text(
                block_id,
                y_top,
                short_source,
                rotation=90,
                va="top",
                ha="left",
                fontsize=6.5,
                color="#666666",
                clip_on=True,
            )
            previous = source

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Project root: {0}".format(PROJECT_ROOT))
    print("Device: {0}".format(device))
    print("Loading data from: {0}".format(args.data_dir))
    X, y, metadata = load_data(args.data_dir, args.window_size, args.step_size)
    print("Loaded X={0}, y={1}".format(X.shape, y.shape))
    if X.ndim != 3 or y.ndim != 2 or len(X) != len(y):
        raise ValueError("Expected X=[N,W,F], y=[N,W] with matching N.")

    train_data, val_data, test_data, split_info = split_data_unsupervised(
        X,
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print_split_info(split_info)

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
    )
    configure_loader_workers((train_loader, val_loader, test_loader), args.num_workers)

    input_dim = int(X.shape[-1])
    model = LSTMAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Model: LSTM-AE | parameters={0:,}".format(parameter_count))

    model, history, best_epoch, best_val = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        args,
    )

    # From here onward, the model is frozen. No optimizer/update touches test data.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    val_scores, val_labels = collect_scores(model, val_loader, device)
    test_scores, test_labels = collect_scores(model, test_loader, device)
    val_normal_scores = val_scores[val_labels == 0]
    if val_normal_scores.size == 0:
        raise RuntimeError("Cannot calibrate threshold: validation has no normal window.")
    threshold = float(np.quantile(val_normal_scores, args.threshold_quantile))
    observed_val_fpr = float(np.mean(val_normal_scores > threshold))

    total = len(X)
    train_end = int(total * args.train_ratio)
    val_end = int(total * (args.train_ratio + args.val_ratio))
    test_meta = metadata_slice(metadata, val_end, total)

    rows = summarize_blocks(
        scores=test_scores,
        labels=test_labels,
        threshold=threshold,
        block_size=args.block_size,
        val_normal_scores=val_normal_scores,
        test_metadata=test_meta,
        test_global_start=val_end,
    )
    shift_summary = build_shift_summary(test_scores, test_labels, threshold, rows)

    csv_path = output_dir / "gate_a_block_metrics.csv"
    figure_path = output_dir / "gate_a_normal_score_over_time.png"
    summary_path = output_dir / "gate_a_summary.json"
    score_path = output_dir / "gate_a_window_scores.npz"
    checkpoint_path = output_dir / "gate_a_lstm_ae.pt"
    scaler_path = output_dir / "gate_a_scaler.pkl"
    history_path = output_dir / "gate_a_training_history.csv"

    write_csv(csv_path, rows)
    write_csv(history_path, history)
    np.savez_compressed(
        score_path,
        val_scores=val_scores,
        val_labels=val_labels,
        test_scores=test_scores,
        test_labels=test_labels,
        test_local_indices=np.arange(len(test_scores), dtype=np.int64),
        test_global_indices=np.arange(val_end, total, dtype=np.int64),
        threshold=np.asarray([threshold], dtype=np.float64),
    )
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    torch.save(
        {
            "model_name": "LSTM-AE",
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "best_epoch": best_epoch,
            "best_val_normal_mse": best_val,
            "threshold": threshold,
            "threshold_quantile": args.threshold_quantile,
        },
        checkpoint_path,
    )

    dataset_name = Path(os.path.normpath(args.data_dir)).name
    plot_block_metrics(
        path=figure_path,
        rows=rows,
        val_normal_scores=val_normal_scores,
        threshold=threshold,
        target_fpr=1.0 - args.threshold_quantile,
        title="Gate A: frozen LSTM-AE normal-score drift ({0})".format(dataset_name),
    )

    summary = {
        "protocol": {
            "model": "LSTM-AE",
            "model_frozen_during_test": True,
            "test_labels_used_by_model": False,
            "test_labels_used_offline_only_to_select_normal_windows": True,
            "threshold_source": "normal validation-window reconstruction scores",
        },
        "arguments": vars(args),
        "data": {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "train_end_global_index": train_end,
            "validation_end_global_index": val_end,
            "split_info": split_info,
            "metadata_available_for_test": bool(test_meta),
        },
        "training": {
            "device": str(device),
            "parameter_count": int(parameter_count),
            "best_epoch": int(best_epoch),
            "best_val_normal_mse": float(best_val),
        },
        "calibration": {
            "normal_validation_windows": int(val_normal_scores.size),
            "normal_validation_score_mean": float(np.mean(val_normal_scores)),
            "normal_validation_score_median": float(np.median(val_normal_scores)),
            "threshold_quantile": float(args.threshold_quantile),
            "fixed_threshold": threshold,
            "observed_normal_validation_fpr": observed_val_fpr,
        },
        "test": {
            "test_windows": int(test_scores.size),
            "normal_test_windows": int(np.sum(test_labels == 0)),
            "anomalous_test_windows": int(np.sum(test_labels == 1)),
            "overall_normal_test_fpr": float(np.mean(test_scores[test_labels == 0] > threshold))
            if np.any(test_labels == 0)
            else None,
            "number_of_blocks": len(rows),
            "shift_summary": shift_summary,
        },
        "outputs": {
            "figure": str(figure_path),
            "block_metrics": str(csv_path),
            "window_scores": str(score_path),
            "checkpoint": str(checkpoint_path),
            "scaler": str(scaler_path),
            "training_history": str(history_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2)

    print("\nFixed threshold: {0:.8f}".format(threshold))
    print("Observed validation-normal FPR: {0:.4f}".format(observed_val_fpr))
    print("Overall test-normal FPR: {0}".format(summary["test"]["overall_normal_test_fpr"]))
    print("Late minus early normal-score median: {0}".format(shift_summary["late_minus_early_normal_median"]))
    print("Late minus early normal FPR: {0}".format(shift_summary["late_minus_early_normal_fpr"]))
    print("Outputs written to: {0}".format(output_dir))


if __name__ == "__main__":
    main()
