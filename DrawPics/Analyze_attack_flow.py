import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


def analyze_attack_duration(metadata_file):
    """分析攻击持续时间（以流数量计量）"""

    # 加载元数据
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    window_metadata = metadata['window_metadata']

    # 统计每种攻击的持续流数量
    attack_durations = defaultdict(list)
    current_attack = None
    current_duration = 0

    for window in window_metadata:
        if window['is_malicious'] == 1:
            # 获取主要攻击类型
            primary_attack = window.get('primary_attack', 'Unknown')

            if current_attack == primary_attack:
                # 继续当前攻击
                current_duration += CICIDS_WINDOW_SIZE
            else:
                # 保存前一个攻击的持续时间
                if current_attack is not None:
                    attack_durations[current_attack].append(current_duration)
                # 开始新攻击
                current_attack = primary_attack
                current_duration = CICIDS_WINDOW_SIZE
        else:
            # 遇到正常流量，结束当前攻击
            if current_attack is not None:
                attack_durations[current_attack].append(current_duration)
                current_attack = None
                current_duration = 0

    # 保存最后一个攻击
    if current_attack is not None:
        attack_durations[current_attack].append(current_duration)

    # 生成统计报告
    print("=" * 60)
    print("攻击持续时间分析 (以流数量计量)")
    print("=" * 60)

    for attack_type in sorted(attack_durations.keys()):
        durations = attack_durations[attack_type]
        print(f"\n{attack_type}:")
        print(f"  攻击事件数: {len(durations)}")
        print(f"  平均持续流数: {np.mean(durations):.2f}")
        print(f"  中位数持续流数: {np.median(durations):.2f}")
        print(f"  最短持续流数: {np.min(durations)}")
        print(f"  最长持续流数: {np.max(durations)}")
        print(f"  标准差: {np.std(durations):.2f}")

    # 创建DataFrame以便进一步分析
    results = []
    for attack_type, durations in attack_durations.items():
        for duration in durations:
            results.append({
                'attack_type': attack_type,
                'duration_flows': duration
            })

    df = pd.DataFrame(results)

    # 保存结果
    df.to_csv('attack_duration_analysis.csv', index=False)
    print("\n结果已保存至 attack_duration_analysis.csv")

    return df, attack_durations


if __name__ == "__main__":
    metadata_file = "../cicids2017/selected_features/selected_metadata_w100_s20.pkl"

    df, durations = analyze_attack_duration(metadata_file)

    # 可选：绘制分布图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for i, (attack_type, duration_list) in enumerate(durations.items()):
        plt.subplot(2, 4, i + 1)
        plt.hist(duration_list, bins=20, edgecolor='black')
        plt.title(f'{attack_type}', fontsize=10)
        plt.xlabel('持续流数')
        plt.ylabel('频次')

    plt.tight_layout()
    plt.savefig('attack_duration_distribution.png', dpi=300)
    print("分布图已保存至 attack_duration_distribution.png")