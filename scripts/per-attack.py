"""
Window-level Attack-Type-wise Analysis
=====================================
加载已训练模型 -> 在测试集上推理 -> 统计“包含某攻击类型的窗口”被判为 malicious 的比例。

重要说明：
1. 模型是二分类窗口检测器，不是多分类攻击分类器。
2. 一个窗口可能包含多个攻击类型。
3. 因此这里的指标不是 per-attack classification performance，
   而是 attack-type-wise malicious-window recall：
   DetectionRate(a) =
       # windows containing attack type a and predicted as malicious
       / # windows containing attack type a

用法：
python per_attack_window_multilabel_fixed.py
"""

import os
import pickle
import json
from collections import defaultdict, Counter

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models.mymodel.amd.Model_with_ddi import Model
from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


class Config:
    """与训练时相同的配置"""
    def __init__(self):
        self.seq_len = CICIDS_WINDOW_SIZE
        self.enc_in = 38
        self.pred_len = 38
        self.num_classes = 2
        self.individual = False
        self.batch_size = 128
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = "checkpoints"


def get_test_metadata(data_dir, train_ratio=0.6, val_ratio=0.2,
                      window_size=100, step_size=20):
    """获取测试集对应的 window_metadata。"""
    metadata_file = os.path.join(
        data_dir, f"integrated_metadata_w{window_size}_s{step_size}.pkl"
    )
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    all_window_metadata = metadata["window_metadata"]
    total = len(all_window_metadata)
    val_end = int(total * (train_ratio + val_ratio))

    return all_window_metadata[val_end:]


def run_inference(model, test_loader, device):
    """在测试集上跑推理，返回逐窗口预测、标签和恶意概率。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in test_loader:
            batch_X = batch_X.to(device)

            output = model(batch_X)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

            # batch_y shape 通常是 [batch, 1]
            labels = batch_y.squeeze(-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def normalize_attack_name(name):
    """统一清理攻击名，避免空字符串、BENIGN、None 被计为攻击。"""
    if name is None:
        return None

    name = str(name).strip()
    if not name:
        return None

    benign_names = {"BENIGN", "Benign", "benign", "Normal", "NORMAL", "normal"}
    if name in benign_names:
        return None

    # 统一常见拼写
    if name == "Infilteration":
        name = "Infiltration"

    return name


def extract_attack_types(meta):
    """
    从单个窗口 metadata 中提取“该窗口包含的所有攻击类型”。

    优先使用多标签字段；如果没有多标签字段，才回退到 primary_attack。
    你需要根据自己的 metadata 实际结构确认字段名。
    """
    candidate_list_keys = [
        "attack_types",
        "attacks",
        "attack_type_list",
        "unique_attack_types",
        "contained_attacks",
    ]

    candidate_counter_keys = [
        "attack_counts",
        "label_counts",
        "type_counts",
        "attack_type_counts",
    ]

    attacks = set()

    # 情况 1：metadata 里已经有攻击类型列表
    for key in candidate_list_keys:
        value = meta.get(key)
        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            for item in value:
                attack = normalize_attack_name(item)
                if attack:
                    attacks.add(attack)

    # 情况 2：metadata 里有 {attack_type: count}
    for key in candidate_counter_keys:
        value = meta.get(key)
        if value is None:
            continue

        if isinstance(value, dict):
            for attack_name, count in value.items():
                try:
                    count_value = int(count)
                except Exception:
                    count_value = 0

                if count_value <= 0:
                    continue

                attack = normalize_attack_name(attack_name)
                if attack:
                    attacks.add(attack)

    # 情况 3：没有多标签字段时，回退到 primary_attack
    # 注意：这只能得到主攻击类型，不能还原多攻击窗口。
    if not attacks:
        attack = normalize_attack_name(meta.get("primary_attack"))
        if attack:
            attacks.add(attack)

    return attacks


def is_malicious_meta(meta, label):
    """
    判断窗口是否为恶意。
    优先用 metadata['is_malicious']，否则用二分类 label。
    """
    if "is_malicious" in meta:
        return int(meta["is_malicious"]) == 1
    return int(label) == 1


def attack_type_wise_window_recall(preds, labels, probs, test_metadata):
    """
    统计：
    - overall binary metrics
    - benign accuracy
    - 每种攻击类型：包含该攻击类型的窗口中，有多少被判为 malicious
    - mixed_windows：含多种攻击类型的窗口数
    - pure_attack：只含一种攻击类型的窗口统计
    """
    if not (len(preds) == len(labels) == len(test_metadata) == len(probs)):
        raise ValueError(
            f"长度不一致: preds={len(preds)}, labels={len(labels)}, "
            f"probs={len(probs)}, metadata={len(test_metadata)}"
        )

    benign_items = []
    attack_groups = defaultdict(list)
    pure_attack_groups = defaultdict(list)

    mixed_windows = 0
    malicious_windows = 0
    unknown_malicious_windows = 0
    attack_set_size_counter = Counter()

    for i, meta in enumerate(test_metadata):
        pred = int(preds[i])
        label = int(labels[i])
        prob = float(probs[i])

        is_mal = is_malicious_meta(meta, label)

        if not is_mal:
            benign_items.append((pred, label, prob))
            continue

        malicious_windows += 1
        attacks = extract_attack_types(meta)
        attack_set_size_counter[len(attacks)] += 1

        if len(attacks) == 0:
            unknown_malicious_windows += 1
            attack_groups["Unknown"].append((pred, label, prob))
            continue

        if len(attacks) > 1:
            mixed_windows += 1

        # 多标签统计：一个窗口包含多个攻击类型，就进入多个攻击类型的分母
        for attack in attacks:
            attack_groups[attack].append((pred, label, prob))

        # 纯攻击窗口统计：只含一种攻击类型时，额外记录
        if len(attacks) == 1:
            only_attack = next(iter(attacks))
            pure_attack_groups[only_attack].append((pred, label, prob))

    # benign 统计
    if benign_items:
        benign_preds = np.array([x[0] for x in benign_items])
        benign_count = len(benign_items)
        tn = int((benign_preds == 0).sum())
        fp = int((benign_preds == 1).sum())
    else:
        benign_count = tn = fp = 0

    benign_result = {
        "count": benign_count,
        "correct": tn,
        "incorrect": fp,
        "accuracy": tn / benign_count if benign_count > 0 else 0.0,
    }

    def build_attack_entries(groups):
        entries = []
        for attack_type, items in groups.items():
            group_preds = np.array([x[0] for x in items])
            group_probs = np.array([x[2] for x in items])

            count = len(items)
            tp = int((group_preds == 1).sum())
            fn = int((group_preds == 0).sum())

            entries.append({
                "attack_type": attack_type,
                "count": count,
                "tp": tp,
                "fn": fn,
                "detection_rate": tp / count if count > 0 else 0.0,
                "mean_malicious_prob": float(group_probs.mean()) if count > 0 else 0.0,
                "median_malicious_prob": float(np.median(group_probs)) if count > 0 else 0.0,
            })

        entries.sort(key=lambda x: x["detection_rate"], reverse=True)
        return entries

    per_attack = build_attack_entries(attack_groups)
    pure_per_attack = build_attack_entries(pure_attack_groups)

    overall_precision = precision_score(labels, preds, average="binary", zero_division=0)
    overall_recall = recall_score(labels, preds, average="binary", zero_division=0)
    overall_f1 = f1_score(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    results = {
        "metric_definition": (
            "Attack-type-wise malicious-window recall. "
            "For each attack type, count means the number of test windows containing that attack type. "
            "A window containing multiple attack types is counted once for each contained type. "
            "This is not multi-class attack classification performance."
        ),
        "window_statistics": {
            "total_windows": int(len(labels)),
            "benign_windows": int(benign_count),
            "malicious_windows": int(malicious_windows),
            "mixed_attack_windows": int(mixed_windows),
            "unknown_malicious_windows": int(unknown_malicious_windows),
            "attack_set_size_distribution": {
                str(k): int(v) for k, v in sorted(attack_set_size_counter.items())
            },
        },
        "per_attack": per_attack,
        "pure_per_attack": pure_per_attack,
        "benign": benign_result,
        "overall": {
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "confusion_matrix": cm.tolist(),
        },
    }

    return results


def print_results(results):
    print("\n" + "=" * 100)
    print("Attack-Type-wise Malicious Window Recall")
    print("=" * 100)
    print(results["metric_definition"])

    stats = results["window_statistics"]
    print("\nWindow statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nPer attack, multi-label counting:")
    print(f"{'Attack Type':<25} {'Count':>8} {'TP':>8} {'FN':>8} {'Recall':>8} {'MeanProb':>10}")
    print("-" * 80)
    for e in results["per_attack"]:
        print(
            f"{e['attack_type']:<25} {e['count']:>8} {e['tp']:>8} {e['fn']:>8} "
            f"{e['detection_rate']:>8.3f} {e['mean_malicious_prob']:>10.3f}"
        )

    print("\nPure attack windows only:")
    print(f"{'Attack Type':<25} {'Count':>8} {'TP':>8} {'FN':>8} {'Recall':>8} {'MeanProb':>10}")
    print("-" * 80)
    for e in results["pure_per_attack"]:
        print(
            f"{e['attack_type']:<25} {e['count']:>8} {e['tp']:>8} {e['fn']:>8} "
            f"{e['detection_rate']:>8.3f} {e['mean_malicious_prob']:>10.3f}"
        )

    print("\nBenign:")
    print(results["benign"])

    print("\nOverall:")
    print(results["overall"])


def main():
    configs = Config()

    print("1. Loading data...")
    data_dir = "../cicids2018/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)
    configs.enc_in = len(metadata["feature_names"])
    print(f"   X={X.shape}, features={configs.enc_in}")

    print("2. Splitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )
    _, _, test_loader, _ = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    print("3. Loading test metadata...")
    test_metadata = get_test_metadata(
        data_dir,
        train_ratio=configs.train_ratio,
        val_ratio=configs.val_ratio,
        window_size=CICIDS_WINDOW_SIZE,
        step_size=CICIDS_WINDOW_STEP,
    )
    print(f"   Test windows: {len(test_metadata)}")

    print("4. Loading best model...")
    model = Model(configs).to(configs.device)
    checkpoint = torch.load(
        os.path.join(configs.save_dir, "best_model.pth"),
        map_location=configs.device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"   Loaded from epoch {checkpoint.get('epoch')}, "
        f"val_f1={checkpoint.get('best_val_f1')}"
    )

    print("5. Running inference on test set...")
    preds, labels, probs = run_inference(model, test_loader, configs.device)
    print(f"   Predictions: {len(preds)}, Positive rate: {preds.mean():.3f}")

    assert len(preds) == len(test_metadata), (
        f"Mismatch: preds={len(preds)}, metadata={len(test_metadata)}"
    )

    results = attack_type_wise_window_recall(preds, labels, probs, test_metadata)
    print_results(results)

    save_path = os.path.join(configs.save_dir, "attack_type_window_recall_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
