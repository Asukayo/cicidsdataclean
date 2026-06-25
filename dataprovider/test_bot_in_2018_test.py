"""
验证各攻击类型在 train/val/test 中的分布
用法：python verify_attack_split.py
"""

import pickle
import numpy as np
from collections import defaultdict


def check_split(data_dir, dataset_name, window_size=100, step_size=20,
                train_ratio=0.6, val_ratio=0.2):
    metadata_file = f'{data_dir}/integrated_metadata_w{window_size}_s{step_size}.pkl'

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    all_meta = metadata['window_metadata']
    total = len(all_meta)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        'Train': all_meta[:train_end],
        'Val': all_meta[train_end:val_end],
        'Test': all_meta[val_end:]
    }

    print(f"\n{'=' * 70}")
    print(f"  {dataset_name} Attack Distribution (total={total})")
    print(f"  Train: 0-{train_end}  Val: {train_end}-{val_end}  Test: {val_end}-{total}")
    print(f"{'=' * 70}")

    # 统计每个split中各攻击类型的窗口数
    all_attacks = defaultdict(lambda: {'Train': 0, 'Val': 0, 'Test': 0, 'Total': 0})

    for split_name, split_meta in splits.items():
        for m in split_meta:
            if m['is_malicious'] == 1:
                attack = m.get('primary_attack', 'Unknown')
                all_attacks[attack][split_name] += 1
                all_attacks[attack]['Total'] += 1

    # 打印
    print(f"\n{'Attack Type':<25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}  "
          f"{'Train%':>7} {'Test%':>7}")
    print("-" * 70)

    for attack in sorted(all_attacks.keys(), key=lambda x: all_attacks[x]['Total'], reverse=True):
        d = all_attacks[attack]
        t_pct = d['Train'] / d['Total'] * 100 if d['Total'] > 0 else 0
        te_pct = d['Test'] / d['Total'] * 100 if d['Total'] > 0 else 0
        print(f"{attack:<25} {d['Train']:>7} {d['Val']:>7} {d['Test']:>7} {d['Total']:>7}  "
              f"{t_pct:>6.1f}% {te_pct:>6.1f}%")

    # Benign统计
    benign = {'Train': 0, 'Val': 0, 'Test': 0}
    for split_name, split_meta in splits.items():
        benign[split_name] = sum(1 for m in split_meta if m['is_malicious'] == 0)

    print("-" * 70)
    b_total = sum(benign.values())
    print(f"{'Benign':<25} {benign['Train']:>7} {benign['Val']:>7} {benign['Test']:>7} {b_total:>7}")


if __name__ == '__main__':
    # ---- 修改为你的实际路径 ----
    # check_split("../cicids2017/selected_features", "CICIDS2017")
    check_split("../cicids2018/selected_features", "CICIDS2018")