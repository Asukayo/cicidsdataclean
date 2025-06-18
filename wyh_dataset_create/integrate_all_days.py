import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import gc


def integrate_windows_by_weekday(input_dir, output_dir, window_size=200, step_size=50):
    """
    按照星期顺序整合CICIDS2017数据集的窗口文件

    参数:
    - input_dir: 包含各天窗口数据的输入目录
    - output_dir: 输出目录
    - window_size: 窗口大小
    - step_size: 步长
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义文件顺序（按照您指定的顺序）
    file_order = [
        "Monday-WorkingHours",
        "Tuesday-WorkingHours",
        "Wednesday-workingHours",
        "Thursday-WorkingHours-Morning-WebAttacks",
        "Thursday-WorkingHours-Afternoon-Infilteration",
        "Friday-WorkingHours-Morning",
        "Friday-WorkingHours-Afternoon-PortScan",
        "Friday-WorkingHours-Afternoon-DDos"
    ]

    # 收集所有数据
    all_X = []
    all_y = []
    all_metadata = []

    # 记录每个文件的窗口起始索引
    file_window_indices = {}
    current_window_idx = 0

    # 记录全局信息
    global_label_mapping = None
    global_feature_names = None

    print("开始整合数据文件...")

    for file_prefix in file_order:
        # 构建文件路径
        day_dir = os.path.join(input_dir, file_prefix)
        X_file = os.path.join(day_dir, f'X_windows_w{window_size}_s{step_size}.npy')
        y_file = os.path.join(day_dir, f'y_windows_w{window_size}_s{step_size}.npy')
        metadata_file = os.path.join(day_dir, f'metadata_w{window_size}_s{step_size}.pkl')

        # 检查文件是否存在
        if not os.path.exists(X_file) or not os.path.exists(y_file) or not os.path.exists(metadata_file):
            print(f"警告: {file_prefix} 的文件不完整，跳过")
            continue

        print(f"\n处理: {file_prefix}")

        try:
            # 加载数据
            X = np.load(X_file)
            y = np.load(y_file)

            # 加载元数据
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 保存第一个文件的全局信息
            if global_label_mapping is None:
                global_label_mapping = metadata.get('label_mapping', {})
                global_feature_names = metadata.get('feature_names', [])

            # 记录该文件的窗口索引范围
            file_window_indices[file_prefix] = {
                'start': current_window_idx,
                'end': current_window_idx + len(X)
            }

            # 更新窗口元数据中的全局索引
            window_metadata = metadata.get('window_metadata', [])
            for i, window_meta in enumerate(window_metadata):
                window_meta['global_window_id'] = current_window_idx + i
                window_meta['source_file'] = file_prefix

            current_window_idx += len(X)

            # 添加到总数据中
            all_X.append(X)
            all_y.append(y)
            all_metadata.extend(window_metadata)

            print(f"  - 加载了 {len(X)} 个窗口")
            print(f"  - X形状: {X.shape}, y形状: {y.shape}")

            # 清理内存
            del X, y
            gc.collect()

        except Exception as e:
            print(f"处理 {file_prefix} 时出错: {str(e)}")
            continue

    # 合并所有数据
    print("\n合并所有数据...")

    if not all_X:
        print("错误: 没有成功加载任何数据")
        return None

    # 使用vstack合并，这样更节省内存
    integrated_X = np.vstack(all_X)
    integrated_y = np.vstack(all_y)

    print(f"\n整合完成!")
    print(f"最终X形状: {integrated_X.shape}")
    print(f"最终y形状: {integrated_y.shape}")

    # 验证形状
    expected_shape = (integrated_X.shape[0], window_size, len(global_feature_names))
    if integrated_X.shape != expected_shape:
        print(f"警告: X的形状 {integrated_X.shape} 与预期 {expected_shape} 不符")

    # 保存整合后的数据
    output_X_file = os.path.join(output_dir, f'integrated_X_w{window_size}_s{step_size}.npy')
    output_y_file = os.path.join(output_dir, f'integrated_y_w{window_size}_s{step_size}.npy')

    print(f"\n保存整合数据到: {output_dir}")
    np.save(output_X_file, integrated_X)
    np.save(output_y_file, integrated_y)

    # 保存整合后的元数据
    integrated_metadata = {
        'window_metadata': all_metadata,
        'feature_names': global_feature_names,
        'label_mapping': global_label_mapping,
        'file_window_indices': file_window_indices,
        'file_order': file_order,
        'config': {
            'window_size': window_size,
            'step_size': step_size,
            'total_windows': len(integrated_X),
            'num_features': len(global_feature_names)
        }
    }

    metadata_output_file = os.path.join(output_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl')
    with open(metadata_output_file, 'wb') as f:
        pickle.dump(integrated_metadata, f)

    # 创建数据集统计报告
    create_summary_report(integrated_X, integrated_y, integrated_metadata, output_dir, window_size, step_size)

    # 清理内存
    del all_X, all_y
    gc.collect()

    return {
        'X_shape': integrated_X.shape,
        'y_shape': integrated_y.shape,
        'metadata': integrated_metadata
    }


def create_summary_report(X, y, metadata, output_dir, window_size, step_size):
    """
    创建整合数据集的统计报告
    """
    summary_file = os.path.join(output_dir, f'integrated_summary_w{window_size}_s{step_size}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CICIDS2017 整合数据集统计报告\n")
        f.write("=" * 50 + "\n\n")

        # 基本信息
        f.write("基本信息:\n")
        f.write(f"- 总窗口数: {X.shape[0]}\n")
        f.write(f"- 窗口大小: {window_size} 条流量记录\n")
        f.write(f"- 窗口步长: {step_size} 条流量记录\n")
        f.write(f"- 特征数量: {X.shape[2]}\n")
        f.write(f"- X形状: {X.shape} (窗口数, 窗口大小, 特征数)\n")
        f.write(f"- y形状: {y.shape}\n\n")

        # 文件顺序和窗口分布
        f.write("文件顺序和窗口分布:\n")
        f.write("-" * 30 + "\n")

        file_indices = metadata['file_window_indices']
        for i, file_name in enumerate(metadata['file_order']):
            if file_name in file_indices:
                indices = file_indices[file_name]
                num_windows = indices['end'] - indices['start']
                f.write(
                    f"{i + 1}. {file_name}: {num_windows} 个窗口 (索引 {indices['start']} - {indices['end'] - 1})\n")

        # 标签分布统计
        f.write("\n标签分布统计:\n")
        f.write("-" * 30 + "\n")

        # 统计窗口级别的标签（恶意/正常）
        window_metadata = metadata['window_metadata']
        malicious_windows = sum(1 for w in window_metadata if w['is_malicious'] == 1)
        benign_windows = len(window_metadata) - malicious_windows

        f.write(f"窗口级别:\n")
        f.write(f"- 正常窗口: {benign_windows} ({benign_windows / len(window_metadata) * 100:.2f}%)\n")
        f.write(f"- 异常窗口: {malicious_windows} ({malicious_windows / len(window_metadata) * 100:.2f}%)\n\n")

        # 统计流量级别的标签分布
        label_mapping = metadata['label_mapping']
        f.write("流量级别标签分布:\n")

        # 计算每个标签的出现次数
        label_counts = {}
        for label_idx in label_mapping.keys():
            count = np.sum(y == int(label_idx))
            label_name = label_mapping[label_idx]
            label_counts[label_name] = count

        total_flows = sum(label_counts.values())
        for label_name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                f.write(f"- {label_name}: {count} 条记录 ({count / total_flows * 100:.2f}%)\n")

        # 攻击类型分布
        f.write("\n攻击类型分布（按文件）:\n")
        f.write("-" * 30 + "\n")

        for file_name in metadata['file_order']:
            if file_name in file_indices:
                indices = file_indices[file_name]
                file_windows = [w for w in window_metadata if w.get('source_file') == file_name]

                if file_windows:
                    f.write(f"\n{file_name}:\n")

                    # 统计该文件中的攻击类型
                    attack_stats = {}
                    for window in file_windows:
                        if window['is_malicious'] == 1 and 'attack_types' in window:
                            for attack_type, count in window['attack_types'].items():
                                if attack_type in attack_stats:
                                    attack_stats[attack_type] += count
                                else:
                                    attack_stats[attack_type] = count

                    if attack_stats:
                        for attack_type, count in sorted(attack_stats.items(), key=lambda x: x[1], reverse=True):
                            f.write(f"  - {attack_type}: {count} 条记录\n")
                    else:
                        f.write("  - 仅包含正常流量\n")

        # 特征信息
        f.write("\n特征信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"特征数量: {len(metadata['feature_names'])}\n")
        f.write("前10个特征:\n")
        for i, feature in enumerate(metadata['feature_names'][:10]):
            f.write(f"  {i + 1}. {feature}\n")

        if len(metadata['feature_names']) > 10:
            f.write(f"  ... 以及其他 {len(metadata['feature_names']) - 10} 个特征\n")

    print(f"统计报告已保存到: {summary_file}")


def verify_integrated_data(integrated_dir, window_size=200, step_size=50):
    """
    验证整合后的数据
    """
    print("\n验证整合数据...")

    # 加载数据
    X_file = os.path.join(integrated_dir, f'integrated_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(integrated_dir, f'integrated_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(integrated_dir, f'integrated_metadata_w{window_size}_s{step_size}.pkl')

    # 使用mmap_mode='r'来避免完全加载到内存
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    print(f"窗口元数据数量: {len(metadata['window_metadata'])}")

    # 验证形状一致性
    assert X.shape[0] == y.shape[0], "X和y的第一维度不匹配"
    assert X.shape[0] == len(metadata['window_metadata']), "窗口数量与元数据不匹配"
    assert X.shape[1] == window_size, f"窗口大小不正确: {X.shape[1]} vs {window_size}"

    print("数据验证通过!")

    return True


if __name__ == "__main__":
    # 配置参数
    input_dir = "../cicids2017/flow_windows"  # 包含各天窗口数据的目录
    output_dir = "../cicids2017/integrated_windows"  # 输出目录

    # 执行整合
    result = integrate_windows_by_weekday(
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=570,
        step_size=50
    )

    if result:
        print(f"\n整合完成!")
        print(f"最终数据形状: X{result['X_shape']}, y{result['y_shape']}")

        # 验证数据
        verify_integrated_data(output_dir, window_size=570, step_size=50)
    else:
        print("整合失败!")