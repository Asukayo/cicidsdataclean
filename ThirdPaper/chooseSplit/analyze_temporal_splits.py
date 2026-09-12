#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 CICIDS 预构建滑动窗口在不同时间划分下的数据分布。

主要输出：
1. 每个区间的窗口数、正常/异常窗口数；
2. 去重后的近似流级正常/攻击数量与攻击比例；
3. 每种攻击标签的流级数量、窗口覆盖数；
4. 连续正常片段的数量与长度；
5. 可用于训练、漂移基线和阈值拟合的全正常窗口数；
6. 预构建重叠窗口直接切分造成的边界重叠；
7. 60/10/10/20 与 50/10/10/30 两种候选方案的原始版和 purge 后结果。

注意：
- 脚本只分析数据，不训练模型，也不会修改原始文件。
- 默认正常标签为 0。
- 对预构建窗口按索引直接切分时，相邻集合会共享 W-S 条流。
  脚本同时给出 purge 版本：从后一个集合开头删除
  ceil((W-S)/S) 个窗口，以避免跨集合样本重叠。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CANDIDATES = (
    ("60-10-10-20", (0.60, 0.10, 0.10, 0.20)),
    ("50-10-10-30", (0.50, 0.10, 0.10, 0.30)),
)
SPLIT_NAMES = ("source_train", "drift_baseline", "validation", "online_test")


def parse_ratio(text: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "划分比例必须包含4个数，例如 0.6,0.1,0.1,0.2"
        )
    if any(x <= 0 for x in parts):
        raise argparse.ArgumentTypeError("每个比例都必须大于0")
    if not np.isclose(sum(parts), 1.0, atol=1e-8):
        raise argparse.ArgumentTypeError(
            f"比例之和必须为1，当前为 {sum(parts):.8f}"
        )
    return tuple(parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 CICIDS 时间顺序划分、攻击比例与窗口重叠。"
    )
    parser.add_argument("--data-dir", required=True, help="integrated_*.npy 所在目录")
    parser.add_argument("--dataset", default="CICIDS", help="输出中显示的数据集名称")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_ratio,
        help=(
            "自定义候选比例，可重复传入。"
            "例如 --candidate 0.6,0.1,0.1,0.2"
        ),
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help='可选JSON文件，例如 {"0":"BENIGN","1":"DoS","2":"PortScan"}',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出JSON路径；默认写入数据目录下的 temporal_split_analysis.json",
    )
    parser.add_argument(
        "--consistency-samples",
        type=int,
        default=200,
        help="抽查相邻窗口标签重叠一致性的窗口对数量",
    )
    parser.add_argument(
        "--top-normal-segments",
        type=int,
        default=10,
        help="输出最长连续正常片段的数量",
    )
    return parser.parse_args()


def load_arrays(
    data_dir: Path,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, Any, Path, Path]:
    x_file = data_dir / f"integrated_X_w{window_size}_s{step_size}.npy"
    y_file = data_dir / f"integrated_y_w{window_size}_s{step_size}.npy"
    metadata_file = data_dir / f"integrated_metadata_w{window_size}_s{step_size}.pkl"

    if not x_file.exists():
        raise FileNotFoundError(f"未找到X文件：{x_file}")
    if not y_file.exists():
        raise FileNotFoundError(f"未找到y文件：{y_file}")

    # mmap 避免仅为统计形状而把整个 X 全部载入内存。
    X = np.load(x_file, mmap_mode="r")
    y = np.load(y_file, mmap_mode="r")

    metadata = None
    if metadata_file.exists():
        with metadata_file.open("rb") as f:
            metadata = pickle.load(f)

    return X, y, metadata, x_file, y_file


def normalize_label_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def load_label_map(path: Optional[str], metadata: Any) -> Dict[str, str]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, Mapping):
            raise ValueError("--label-map 必须是JSON对象")
        return {normalize_label_key(k): str(v) for k, v in raw.items()}

    # 尝试从 metadata 中读取常见映射字段。
    if isinstance(metadata, Mapping):
        for key in (
            "label_map",
            "label_mapping",
            "id_to_label",
            "class_names",
            "attack_label_map",
        ):
            raw = metadata.get(key)
            if isinstance(raw, Mapping):
                # 兼容 name -> id 和 id -> name 两种形式。
                result: Dict[str, str] = {}
                for k, v in raw.items():
                    if isinstance(v, (int, np.integer)):
                        result[normalize_label_key(v)] = str(k)
                    else:
                        result[normalize_label_key(k)] = str(v)
                if result:
                    return result
    return {}


def label_name(label: Any, label_map: Mapping[str, str]) -> str:
    key = normalize_label_key(label)
    return label_map.get(key, f"label_{key}")


def metadata_summary(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {"available": False}
    result: Dict[str, Any] = {
        "available": True,
        "type": type(metadata).__name__,
    }
    if isinstance(metadata, Mapping):
        result["keys"] = [str(k) for k in list(metadata.keys())[:50]]
    elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        result["length"] = len(metadata)
        if len(metadata) > 0:
            result["first_item_type"] = type(metadata[0]).__name__
            if isinstance(metadata[0], Mapping):
                result["first_item_keys"] = [
                    str(k) for k in list(metadata[0].keys())[:50]
                ]
    return result


def validate_shapes(X: np.ndarray, y: np.ndarray, window_size: int) -> None:
    if X.ndim != 3:
        raise ValueError(f"X应为[N,W,F]，当前形状为 {X.shape}")
    if y.ndim != 2:
        raise ValueError(f"y应为[N,W]，当前形状为 {y.shape}")
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ValueError(f"X与y窗口维度不一致：X={X.shape}, y={y.shape}")
    if X.shape[1] != window_size:
        raise ValueError(
            f"参数 window_size={window_size}，但文件中的窗口长度为 {X.shape[1]}"
        )


def overlap_consistency(
    y: np.ndarray,
    step_size: int,
    sample_count: int,
) -> Dict[str, Any]:
    n, w = y.shape
    overlap = w - step_size
    if n < 2 or overlap <= 0:
        return {
            "checked_pairs": 0,
            "overlap_flows_per_boundary": max(0, overlap),
            "mismatched_pairs": 0,
            "mismatch_rate": 0.0,
        }

    pair_count = min(sample_count, n - 1)
    indices = np.unique(np.linspace(1, n - 1, pair_count, dtype=int))
    mismatched = 0
    mismatched_elements = 0
    checked_elements = 0

    for idx in indices:
        left = np.asarray(y[idx - 1, step_size:])
        right = np.asarray(y[idx, :overlap])
        unequal = left != right
        if np.any(unequal):
            mismatched += 1
            mismatched_elements += int(np.sum(unequal))
        checked_elements += int(unequal.size)

    return {
        "checked_pairs": int(len(indices)),
        "overlap_flows_per_boundary": int(overlap),
        "mismatched_pairs": int(mismatched),
        "mismatched_pair_rate": (
            float(mismatched / len(indices)) if len(indices) else 0.0
        ),
        "mismatched_elements": int(mismatched_elements),
        "checked_elements": int(checked_elements),
        "mismatched_element_rate": (
            float(mismatched_elements / checked_elements)
            if checked_elements
            else 0.0
        ),
    }


def ratio_boundaries(
    n_windows: int,
    ratios: Sequence[float],
) -> List[Tuple[int, int]]:
    ends = []
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += ratio
        ends.append(int(n_windows * cumulative))
    bounds = [0, *ends, n_windows]
    return [(bounds[i], bounds[i + 1]) for i in range(4)]


def purge_ranges(
    ranges: Sequence[Tuple[int, int]],
    purge_windows: int,
) -> List[Tuple[int, int]]:
    purged: List[Tuple[int, int]] = []
    for idx, (start, end) in enumerate(ranges):
        if idx > 0:
            start = min(end, start + purge_windows)
        purged.append((start, end))
    return purged


def extract_flow_labels(
    y: np.ndarray,
    start_window: int,
    end_window: int,
    step_size: int,
) -> np.ndarray:
    """
    将连续窗口区间近似还原为不重复的流标签序列：
    第一个窗口保留全部W个标签，后续窗口只追加最后S个新标签。
    """
    if end_window <= start_window:
        return np.empty(0, dtype=y.dtype)

    first = np.asarray(y[start_window]).reshape(-1)
    if end_window - start_window == 1:
        return first.copy()

    tails = np.asarray(y[start_window + 1:end_window, -step_size:]).reshape(-1)
    return np.concatenate((first, tails))


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """返回布尔序列中所有True连续段长度。"""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return (changes[1::2] - changes[::2]).astype(np.int64)


def counter_to_named_dict(
    counter: Counter,
    label_map: Mapping[str, str],
    normal_label: int,
) -> Dict[str, int]:
    items = []
    for key, count in counter.items():
        if normalize_label_key(key) == normalize_label_key(normal_label):
            continue
        items.append((label_name(key, label_map), int(count)))
    return dict(sorted(items, key=lambda item: (-item[1], item[0])))


def analyze_range(
    y: np.ndarray,
    start: int,
    end: int,
    step_size: int,
    normal_label: int,
    label_map: Mapping[str, str],
    top_segments: int,
) -> Dict[str, Any]:
    y_part = np.asarray(y[start:end])
    total_windows = int(end - start)

    if total_windows == 0:
        return {
            "window_index_range": [int(start), int(end)],
            "total_windows": 0,
            "normal_windows": 0,
            "anomalous_windows": 0,
            "flow_count_approx": 0,
            "normal_flows_approx": 0,
            "attack_flows_approx": 0,
            "attack_ratio_flow_approx": None,
            "attack_types_flow": {},
            "attack_types_window_coverage": {},
            "normal_segments": {},
        }

    window_attack_mask = np.any(y_part != normal_label, axis=1)
    normal_windows = int(np.sum(~window_attack_mask))
    anomalous_windows = int(np.sum(window_attack_mask))

    # 一个窗口可包含多个攻击类型，因此这里统计“包含该类型的窗口数”。
    window_type_counter: Counter = Counter()
    unique_labels = np.unique(y_part)
    for value in unique_labels:
        if normalize_label_key(value) == normalize_label_key(normal_label):
            continue
        window_type_counter[value.item() if isinstance(value, np.generic) else value] = int(
            np.sum(np.any(y_part == value, axis=1))
        )

    flow_labels = extract_flow_labels(y, start, end, step_size)
    flow_attack_mask = flow_labels != normal_label
    attack_count = int(np.sum(flow_attack_mask))
    normal_count = int(flow_labels.size - attack_count)

    flow_counter = Counter(
        value.item() if isinstance(value, np.generic) else value
        for value in flow_labels[flow_attack_mask]
    )

    normal_lengths = run_lengths(~flow_attack_mask)
    sorted_lengths = np.sort(normal_lengths)[::-1]
    normal_segments = {
        "count": int(normal_lengths.size),
        "mean_length": (
            float(np.mean(normal_lengths)) if normal_lengths.size else 0.0
        ),
        "median_length": (
            float(np.median(normal_lengths)) if normal_lengths.size else 0.0
        ),
        "max_length": (
            int(np.max(normal_lengths)) if normal_lengths.size else 0
        ),
        "segments_at_least_one_window": int(
            np.sum(normal_lengths >= y.shape[1])
        ),
        "top_lengths": [int(v) for v in sorted_lengths[:top_segments]],
    }

    return {
        "window_index_range": [int(start), int(end)],
        "total_windows": total_windows,
        "normal_windows": normal_windows,
        "anomalous_windows": anomalous_windows,
        "normal_window_ratio": float(normal_windows / total_windows),
        "flow_count_approx": int(flow_labels.size),
        "normal_flows_approx": normal_count,
        "attack_flows_approx": attack_count,
        "attack_ratio_flow_approx": float(attack_count / flow_labels.size),
        "attack_type_count": len(flow_counter),
        "attack_types_flow": counter_to_named_dict(
            flow_counter, label_map, normal_label
        ),
        "attack_types_window_coverage": counter_to_named_dict(
            window_type_counter, label_map, normal_label
        ),
        "normal_segments": normal_segments,
    }


def analyze_candidate(
    y: np.ndarray,
    name: str,
    ratios: Sequence[float],
    step_size: int,
    normal_label: int,
    label_map: Mapping[str, str],
    top_segments: int,
    purge_windows: int,
) -> Dict[str, Any]:
    raw_ranges = ratio_boundaries(len(y), ratios)
    purged_ranges = purge_ranges(raw_ranges, purge_windows)

    def analyze_ranges(ranges: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
        return {
            split_name: analyze_range(
                y=y,
                start=start,
                end=end,
                step_size=step_size,
                normal_label=normal_label,
                label_map=label_map,
                top_segments=top_segments,
            )
            for split_name, (start, end) in zip(SPLIT_NAMES, ranges)
        }

    dropped = {
        split_name: int(raw[1] - raw[0] - (purged[1] - purged[0]))
        for split_name, raw, purged in zip(
            SPLIT_NAMES, raw_ranges, purged_ranges
        )
    }

    return {
        "name": name,
        "ratios": [float(x) for x in ratios],
        "raw_window_split": analyze_ranges(raw_ranges),
        "purged_window_split": analyze_ranges(purged_ranges),
        "purge": {
            "purge_windows_from_start_of_each_later_split": int(purge_windows),
            "dropped_windows_by_split": dropped,
            "raw_ranges": [[int(a), int(b)] for a, b in raw_ranges],
            "purged_ranges": [[int(a), int(b)] for a, b in purged_ranges],
        },
    }


def print_split_block(title: str, data: Mapping[str, Any]) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 112)
    print(
        f"  {'Split':<16}"
        f"{'Windows':>11}"
        f"{'Normal-W':>12}"
        f"{'Attack-W':>12}"
        f"{'Flows~':>12}"
        f"{'Attack-F~':>12}"
        f"{'Attack%~':>11}"
        f"{'Types':>8}"
        f"{'Max normal seg':>16}"
    )
    print("  " + "-" * 112)

    for split_name in SPLIT_NAMES:
        item = data[split_name]
        ratio = item["attack_ratio_flow_approx"]
        ratio_text = "N/A" if ratio is None else f"{100.0 * ratio:.2f}%"
        max_seg = item.get("normal_segments", {}).get("max_length", 0)
        print(
            f"  {split_name:<16}"
            f"{item['total_windows']:>11,d}"
            f"{item['normal_windows']:>12,d}"
            f"{item['anomalous_windows']:>12,d}"
            f"{item['flow_count_approx']:>12,d}"
            f"{item['attack_flows_approx']:>12,d}"
            f"{ratio_text:>11}"
            f"{item.get('attack_type_count', 0):>8,d}"
            f"{max_seg:>16,d}"
        )

    test = data["online_test"]
    print("\n  Online-test attack flow counts:")
    if test["attack_types_flow"]:
        for attack_name, count in test["attack_types_flow"].items():
            print(f"    - {attack_name}: {count:,}")
    else:
        print("    - 未发现攻击标签，或当前y仅包含正常标签")


def print_report(report: Mapping[str, Any]) -> None:
    print("=" * 120)
    print(f"Temporal Split Analysis: {report['dataset']}")
    print("=" * 120)
    print(
        f"X shape={tuple(report['data']['x_shape'])}, "
        f"y shape={tuple(report['data']['y_shape'])}, "
        f"window={report['data']['window_size']}, "
        f"step={report['data']['step_size']}"
    )
    consistency = report["overlap_consistency"]
    print(
        "Overlap check: "
        f"{consistency['overlap_flows_per_boundary']} flows shared by "
        "adjacent prebuilt windows; "
        f"sampled mismatch pair rate="
        f"{100.0 * consistency.get('mismatched_pair_rate', 0.0):.2f}%"
    )
    print(
        "Leak-free purge: remove "
        f"{report['purge_windows']} window(s) from the beginning of each "
        "later split."
    )

    for candidate in report["candidates"]:
        ratios = "/".join(f"{int(round(x * 100))}" for x in candidate["ratios"])
        print("\n" + "#" * 120)
        print(f"Candidate: {candidate['name']} ({ratios})")
        print("#" * 120)
        print_split_block(
            "A. Direct split of prebuilt windows (contains boundary overlap)",
            candidate["raw_window_split"],
        )
        print_split_block(
            "B. Purged split (recommended when raw sequences cannot be re-windowed)",
            candidate["purged_window_split"],
        )


def main() -> None:
    args = parse_args()
    if args.window_size <= 0 or args.step_size <= 0:
        raise ValueError("window_size 和 step_size 必须大于0")
    if args.step_size > args.window_size:
        raise ValueError("step_size 不能大于 window_size")

    data_dir = Path(args.data_dir).expanduser().resolve()
    X, y, metadata, x_file, y_file = load_arrays(
        data_dir, args.window_size, args.step_size
    )
    validate_shapes(X, y, args.window_size)

    label_map = load_label_map(args.label_map, metadata)
    consistency = overlap_consistency(
        y, args.step_size, args.consistency_samples
    )

    overlap_flows = max(0, args.window_size - args.step_size)
    purge_windows = int(math.ceil(overlap_flows / args.step_size))

    if args.candidate:
        candidates = [
            (
                "-".join(str(int(round(v * 100))) for v in ratios),
                ratios,
            )
            for ratios in args.candidate
        ]
    else:
        candidates = list(DEFAULT_CANDIDATES)

    candidate_reports = [
        analyze_candidate(
            y=y,
            name=name,
            ratios=ratios,
            step_size=args.step_size,
            normal_label=args.normal_label,
            label_map=label_map,
            top_segments=args.top_normal_segments,
            purge_windows=purge_windows,
        )
        for name, ratios in candidates
    ]

    unique_labels = np.unique(np.asarray(y))
    report: Dict[str, Any] = {
        "dataset": args.dataset,
        "files": {
            "x_file": str(x_file),
            "y_file": str(y_file),
        },
        "data": {
            "x_shape": [int(v) for v in X.shape],
            "y_shape": [int(v) for v in y.shape],
            "x_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "normal_label": int(args.normal_label),
            "unique_label_ids": [
                normalize_label_key(v) for v in unique_labels.tolist()
            ],
            "label_map": dict(label_map),
        },
        "metadata": metadata_summary(metadata),
        "overlap_consistency": consistency,
        "purge_windows": purge_windows,
        "candidates": candidate_reports,
    }

    print_report(report)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_dir / f"{args.dataset}_temporal_split_analysis.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 120)
    print(f"JSON report saved to: {output_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
