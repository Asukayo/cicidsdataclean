import pickle
import os
from collections import defaultdict
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


def analyze_window_distribution(selected_dir, output_file="window_distribution_analysis.txt",
                                window_size=100, step_size=20):
    """分析窗口分布并生成统计报告"""

    # 加载元数据
    metadata_file = os.path.join(selected_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    # 提取窗口元数据
    window_metadata = metadata['window_metadata']
    file_order = metadata.get('file_order', [])

    # 按日期统计窗口数量
    daily_stats = defaultdict(lambda: {'normal': 0, 'abnormal': 0})

    for window in window_metadata:
        source_file = window.get('source_file', 'Unknown')
        is_malicious = window.get('is_malicious', 0)

        if is_malicious == 0:
            daily_stats[source_file]['normal'] += 1
        else:
            daily_stats[source_file]['abnormal'] += 1

    # 计算总计
    total_normal = sum(stats['normal'] for stats in daily_stats.values())
    total_abnormal = sum(stats['abnormal'] for stats in daily_stats.values())
    total_windows = total_normal + total_abnormal

    # 生成报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CICIDS2017 Window Distribution Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Window Configuration:\n")
        f.write(f"- Window Size: {window_size}\n")
        f.write(f"- Step Size: {step_size}\n")
        f.write(f"- Selected Features: {len(metadata['feature_names'])}\n\n")

        f.write("Daily Window Distribution:\n")
        f.write("-" * 30 + "\n")

        # 按照文件顺序输出每日统计
        for i, day_file in enumerate(file_order, 1):
            if day_file in daily_stats:
                stats = daily_stats[day_file]
                normal = stats['normal']
                abnormal = stats['abnormal']
                total_day = normal + abnormal
                abnormal_ratio = abnormal / total_day * 100 if total_day > 0 else 0

                f.write(f"{i:2d}. {day_file}:\n")
                f.write(f"    Normal Windows: {normal:,}\n")
                f.write(f"    Abnormal Windows: {abnormal:,}\n")
                f.write(f"    Total: {total_day:,}\n")
                f.write(f"    Abnormal Ratio: {abnormal_ratio:.2f}%\n\n")

        # 总计统计
        f.write("Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Normal Windows: {total_normal:,}\n")
        f.write(f"Total Abnormal Windows: {total_abnormal:,}\n")
        f.write(f"Total Windows: {total_windows:,}\n\n")

        # 计算比例
        if total_abnormal > 0:
            normal_abnormal_ratio = total_normal / total_abnormal
            f.write(f"Normal : Abnormal Ratio = {normal_abnormal_ratio:.3f} : 1\n")
        else:
            f.write("Normal : Abnormal Ratio = ∞ : 1 (No abnormal windows)\n")

        f.write(f"Normal Window Percentage: {total_normal / total_windows * 100:.2f}%\n")
        f.write(f"Abnormal Window Percentage: {total_abnormal / total_windows * 100:.2f}%\n")

    # 控制台输出简要统计
    print(f"Window Distribution Analysis Completed!")
    print(f"Results saved to: {output_file}")
    print(f"\nQuick Summary:")
    print(f"- Total Windows: {total_windows:,}")
    print(f"- Normal: {total_normal:,} ({total_normal / total_windows * 100:.1f}%)")
    print(f"- Abnormal: {total_abnormal:,} ({total_abnormal / total_windows * 100:.1f}%)")
    if total_abnormal > 0:
        print(f"- Normal:Abnormal Ratio = {total_normal / total_abnormal:.3f}:1")

    return {
        'total_normal': total_normal,
        'total_abnormal': total_abnormal,
        'daily_stats': dict(daily_stats),
        'ratio': total_normal / total_abnormal if total_abnormal > 0 else float('inf')
    }


if __name__ == "__main__":
    selected_dir = "../cicids2017/selected_features"
    output_file = "window_distribution_analysis.txt"

    result = analyze_window_distribution(
        selected_dir=selected_dir,
        output_file=output_file,
        window_size=CICIDS_WINDOW_SIZE,
        step_size=CICIDS_WINDOW_STEP
    )