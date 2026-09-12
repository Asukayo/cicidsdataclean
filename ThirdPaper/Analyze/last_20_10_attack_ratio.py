#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCharm 直接运行：统计在线测试集 10 个时间段的真实攻击窗口比例。

与 unsupervised_provider.py 及当前实验协议保持一致：
1. 从 integrated_y_w100_s20.npy 读取标签；
2. 按原始时间顺序采用 40% / 20% / 20% / 20% 划分；
3. 最后 20% 为 online test；
4. 为避免滑动窗口跨边界共享原始流量，从 online test 开头移除 4 个窗口；
5. 一个窗口内只要存在 y > 0，就视为攻击窗口；
6. 将 purged online test 按时间划分为 10 段；
7. 输出每段真实的正常窗口数、攻击窗口数和攻击窗口比例。

无需命令行参数。只修改“配置区”后，在 PyCharm 中直接运行即可。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 配置区：只需要修改这里
# =============================================================================

DATASET_NAME = "CICIDS2018"

# 与你上传的 unsupervised_provider.py 中路径一致
DATA_DIR = Path("/home/ubuntu/wyh/cicdis/cicids2018/integrated_windows")

WINDOW_SIZE = 100
STEP_SIZE = 20

# 当前论文实验协议：40 / 20 / 20 / 20
SPLIT_RATIOS = (0.40, 0.20, 0.20, 0.20)

NUM_TIME_SEGMENTS = 10
NORMAL_LABEL = 0

# 为了与之前 time_segment_predictions_v2 中的边界一致：
# 当测试窗口数不能被10整除时，将多出的窗口放到最后几个时间段。
# CICIDS2018将得到：前8段12106个窗口，后2段12107个窗口。
ASSIGN_REMAINDER_TO_LAST_SEGMENTS = True

# 输出到脚本所在目录，方便在 PyCharm 中直接找到
OUTPUT_DIR = Path(__file__).resolve().parent / "real_attack_ratio_results"

# 可选：核对测试集汇总数量。
# 当前 CICIDS2018 在 40/20/20/20 + purge=4 下应为：
EXPECTED_NORMAL_WINDOWS: Optional[int] = 80850
EXPECTED_ATTACK_WINDOWS: Optional[int] = 40212

# CICIDS2017可改成：
# DATASET_NAME = "CICIDS2017"
# DATA_DIR = Path("/home/ubuntu/wyh/cicdis/cicids2017/selected_features")
# EXPECTED_NORMAL_WINDOWS = 12243
# EXPECTED_ATTACK_WINDOWS = 12103


# =============================================================================
# 数据加载与划分
# =============================================================================

def locate_y_file(
    data_dir: Path,
    window_size: int,
    step_size: int,
) -> Path:
    """定位 integrated_y 文件，兼容大小写 Y。"""
    candidates = [
        data_dir / f"integrated_y_w{window_size}_s{step_size}.npy",
        data_dir / f"integrated_Y_w{window_size}_s{step_size}.npy",
    ]

    for path in candidates:
        if path.exists():
            return path

    existing = sorted(p.name for p in data_dir.glob("*.npy"))
    existing_text = "\n".join(f"  - {name}" for name in existing[:30])

    raise FileNotFoundError(
        "没有找到标签文件。\n"
        "尝试过：\n"
        + "\n".join(f"  - {path}" for path in candidates)
        + "\n\n当前数据目录中的 .npy 文件：\n"
        + (existing_text if existing_text else "  （没有找到 .npy 文件）")
    )


def load_y(
    data_dir: Path,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, Path]:
    """
    只加载标签，不读取巨大的 integrated_X。
    mmap_mode='r' 可避免把整个标签数组一次性复制到内存。
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    y_file = locate_y_file(data_dir, window_size, step_size)
    y = np.load(y_file, mmap_mode="r")

    if y.ndim != 2:
        raise ValueError(
            f"标签应为二维数组 [N,W]，当前形状为 {y.shape}"
        )

    if y.shape[1] != window_size:
        raise ValueError(
            f"配置 WINDOW_SIZE={window_size}，"
            f"但标签窗口长度为 {y.shape[1]}"
        )

    return y, y_file


def ratio_boundaries(
    n_windows: int,
    ratios: Sequence[float],
) -> List[Tuple[int, int]]:
    """根据比例生成四个连续区间。"""
    if len(ratios) != 4:
        raise ValueError("SPLIT_RATIOS 必须包含4个比例")

    if not np.isclose(sum(ratios), 1.0, atol=1e-8):
        raise ValueError(
            f"SPLIT_RATIOS之和必须为1，当前为 {sum(ratios):.8f}"
        )

    cumulative = 0.0
    ends: List[int] = []

    for ratio in ratios[:-1]:
        cumulative += ratio
        ends.append(int(n_windows * cumulative))

    bounds = [0, *ends, n_windows]
    return [
        (bounds[i], bounds[i + 1])
        for i in range(4)
    ]


def calculate_purge_windows(
    window_size: int,
    step_size: int,
) -> int:
    """
    相邻预构建窗口共享 W-S 条流。
    删除 ceil((W-S)/S) 个窗口可避免跨集合边界重叠。
    """
    overlap = max(0, window_size - step_size)
    return int(math.ceil(overlap / step_size))


def purge_later_splits(
    ranges: Sequence[Tuple[int, int]],
    purge_windows: int,
) -> List[Tuple[int, int]]:
    """从第二、第三、第四个区间开头删除 purge_windows 个窗口。"""
    purged: List[Tuple[int, int]] = []

    for index, (start, end) in enumerate(ranges):
        if index > 0:
            start = min(start + purge_windows, end)
        purged.append((start, end))

    return purged


def split_test_into_segments(
    start: int,
    end: int,
    num_segments: int,
    remainder_to_last: bool,
) -> List[Tuple[int, int, int]]:
    """
    将 [start,end) 按时间划分为 num_segments 段。

    remainder_to_last=True 时，多出的窗口分配给最后几个区段，
    用于复现当前 time_segment_predictions_v2 的边界。
    """
    total = end - start

    if total <= 0:
        raise ValueError("online test 为空")

    if num_segments <= 0:
        raise ValueError("NUM_TIME_SEGMENTS 必须大于0")

    base, remainder = divmod(total, num_segments)

    lengths = [base] * num_segments

    if remainder_to_last:
        for i in range(num_segments - remainder, num_segments):
            if remainder > 0:
                lengths[i] += 1
    else:
        for i in range(remainder):
            lengths[i] += 1

    result: List[Tuple[int, int, int]] = []
    cursor = start

    for segment_id, length in enumerate(lengths, start=1):
        next_cursor = cursor + length
        result.append((segment_id, cursor, next_cursor))
        cursor = next_cursor

    if cursor != end:
        raise RuntimeError(
            f"时间段划分错误：最终位置={cursor}，应为={end}"
        )

    return result


# =============================================================================
# 分段统计
# =============================================================================

def analyze_segment(
    y: np.ndarray,
    segment_id: int,
    start: int,
    end: int,
    normal_label: int,
) -> Dict[str, Any]:
    """
    窗口级规则与 UnsupervisedTrafficDataset 一致：
        window_attack = any(y > normal_label)
    当前正常标签为0，因此等价于 any(y > 0)。
    """
    y_part = np.asarray(y[start:end])

    total_windows = int(end - start)
    attack_mask = np.any(y_part > normal_label, axis=1)

    attack_windows = int(np.sum(attack_mask))
    normal_windows = total_windows - attack_windows
    attack_ratio = attack_windows / total_windows

    return {
        "segment": int(segment_id),
        "window_start_index": int(start),
        "window_end_index_exclusive": int(end),
        "window_range": f"{start}–{end}",
        "total_windows": total_windows,
        "normal_windows": normal_windows,
        "attack_windows": attack_windows,
        "attack_ratio": float(attack_ratio),
    }


def check_expected_totals(
    normal_windows: int,
    attack_windows: int,
) -> None:
    """核对当前结果是否与已知的测试集汇总一致。"""
    if EXPECTED_NORMAL_WINDOWS is not None:
        if normal_windows != EXPECTED_NORMAL_WINDOWS:
            raise ValueError(
                "正常窗口总数与预期不一致："
                f"实际={normal_windows:,}，"
                f"预期={EXPECTED_NORMAL_WINDOWS:,}"
            )
        print(
            f"正常窗口总数核对通过：{normal_windows:,}"
        )

    if EXPECTED_ATTACK_WINDOWS is not None:
        if attack_windows != EXPECTED_ATTACK_WINDOWS:
            raise ValueError(
                "攻击窗口总数与预期不一致："
                f"实际={attack_windows:,}，"
                f"预期={EXPECTED_ATTACK_WINDOWS:,}"
            )
        print(
            f"攻击窗口总数核对通过：{attack_windows:,}"
        )


# =============================================================================
# 输出
# =============================================================================

def print_report(
    y_file: Path,
    y_shape: Tuple[int, ...],
    purge_windows: int,
    online_test_range: Tuple[int, int],
    records: Sequence[Dict[str, Any]],
) -> None:
    print("=" * 108)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Label file: {y_file}")
    print(f"Label shape: {y_shape}")
    print(
        "Split protocol: "
        + "/".join(
            str(int(round(value * 100)))
            for value in SPLIT_RATIOS
        )
    )
    print(
        f"Boundary purge: {purge_windows} windows "
        f"(W={WINDOW_SIZE}, S={STEP_SIZE})"
    )
    print(
        "Online-test range after purge: "
        f"[{online_test_range[0]},{online_test_range[1]})"
    )
    print("=" * 108)

    header = (
        f"{'Seg':>3} "
        f"{'Window range':>22} "
        f"{'Total':>10} "
        f"{'Normal':>10} "
        f"{'Attack':>10} "
        f"{'Attack ratio':>14}"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in records:
        range_text = (
            f"[{row['window_start_index']},"
            f"{row['window_end_index_exclusive']})"
        )

        print(
            f"{row['segment']:>3} "
            f"{range_text:>22} "
            f"{row['total_windows']:>10,d} "
            f"{row['normal_windows']:>10,d} "
            f"{row['attack_windows']:>10,d} "
            f"{row['attack_ratio']:>14.6f}"
        )


def save_results(
    records: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_dataset = DATASET_NAME.lower().replace(" ", "_")

    csv_path = (
        OUTPUT_DIR
        / f"{safe_dataset}_real_time_segment_attack_ratios.csv"
    )
    json_path = (
        OUTPUT_DIR
        / f"{safe_dataset}_real_time_segment_attack_ratios.json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "metadata": metadata,
                "segments": list(records),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    y, y_file = load_y(
        data_dir=DATA_DIR,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    n_windows = int(y.shape[0])

    raw_ranges = ratio_boundaries(
        n_windows=n_windows,
        ratios=SPLIT_RATIOS,
    )

    purge_windows = calculate_purge_windows(
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    purged_ranges = purge_later_splits(
        ranges=raw_ranges,
        purge_windows=purge_windows,
    )

    online_test_start, online_test_end = purged_ranges[3]

    segment_ranges = split_test_into_segments(
        start=online_test_start,
        end=online_test_end,
        num_segments=NUM_TIME_SEGMENTS,
        remainder_to_last=ASSIGN_REMAINDER_TO_LAST_SEGMENTS,
    )

    records = [
        analyze_segment(
            y=y,
            segment_id=segment_id,
            start=start,
            end=end,
            normal_label=NORMAL_LABEL,
        )
        for segment_id, start, end in segment_ranges
    ]

    total_windows = sum(
        row["total_windows"] for row in records
    )
    normal_windows = sum(
        row["normal_windows"] for row in records
    )
    attack_windows = sum(
        row["attack_windows"] for row in records
    )
    overall_attack_ratio = attack_windows / total_windows

    print_report(
        y_file=y_file,
        y_shape=tuple(int(v) for v in y.shape),
        purge_windows=purge_windows,
        online_test_range=(online_test_start, online_test_end),
        records=records,
    )

    print("\nOnline-test summary")
    print("-" * 52)
    print(f"Total windows : {total_windows:,}")
    print(f"Normal windows: {normal_windows:,}")
    print(f"Attack windows: {attack_windows:,}")
    print(f"Attack ratio  : {overall_attack_ratio:.6f}")

    check_expected_totals(
        normal_windows=normal_windows,
        attack_windows=attack_windows,
    )

    metadata = {
        "dataset": DATASET_NAME,
        "data_dir": str(DATA_DIR),
        "label_file": str(y_file),
        "label_shape": [
            int(value) for value in y.shape
        ],
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "normal_label": NORMAL_LABEL,
        "window_label_rule": (
            "attack if any flow label in the window is greater than 0"
        ),
        "split_ratios": [
            float(value) for value in SPLIT_RATIOS
        ],
        "purge_windows": purge_windows,
        "online_test_range": [
            online_test_start,
            online_test_end,
        ],
        "num_time_segments": NUM_TIME_SEGMENTS,
        "remainder_assignment": (
            "last_segments"
            if ASSIGN_REMAINDER_TO_LAST_SEGMENTS
            else "first_segments"
        ),
        "total_windows": total_windows,
        "normal_windows": normal_windows,
        "attack_windows": attack_windows,
        "overall_attack_ratio": float(overall_attack_ratio),
    }

    csv_path, json_path = save_results(
        records=records,
        metadata=metadata,
    )

    print("\nSaved files")
    print("-" * 52)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()