"""
Simplified Autocorrelation-based Window Size Analysis for CICIDS2017 Dataset
Purpose: Analyze autocorrelation structure to provide theoretical basis for window size selection

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class AutocorrelationWindowAnalyzer:
    """
    简化的自相关窗口大小分析器

    核心理论：最优窗口大小应覆盖时间序列的记忆长度
    """

    def __init__(self, current_window_size=200, max_lag=None, significance_level=0.05):
        """
        初始化分析器

        参数:
        - current_window_size: 当前使用的窗口大小（可修改参数）
        - max_lag: 最大滞后长度，默认为窗口大小的2.5倍
        - significance_level: 显著性水平
        """
        self.current_window_size = current_window_size
        self.max_lag = max_lag if max_lag else int(current_window_size * 2.5)
        self.significance_level = significance_level

        # 保持关键特征不变（基于随机森林重要性排序）
        self.key_features = [
            'Bwd Packet Length Std',  # 重要性: 0.0808
            'Packet Length Variance',  # 重要性: 0.0734
            'Packet Length Std',  # 重要性: 0.0727
            'Avg Bwd Segment Size',  # 重要性: 0.0565
            'Bwd Packet Length Mean',  # 重要性: 0.0541
            'Avg Packet Size',  # 重要性: 0.0446
            'Bwd Packet Length Max',  # 重要性: 0.0417
            'Packet Length Max',  # 重要性: 0.0373
            'Packet Length Mean',  # 重要性: 0.0345
            'Subflow Bwd Bytes',  # 重要性: 0.0267
            'Bwd Packets Length Total',  # 重要性: 0.0266
            'Destination Port',  # 重要性: 0.0261
            'Fwd Packets Length Total',  # 重要性: 0.0206
            'Subflow Fwd Packets',  # 重要性: 0.0200
            'Subflow Fwd Bytes',  # 重要性: 0.0197
            'Total Fwd Packets',  # 重要性: 0.0196
            'Fwd Header Length',  # 重要性: 0.0155
            'Fwd Seg Size Min',  # 重要性: 0.0151
            'Fwd Packet Length Max',  # 重要性: 0.0148
            'Fwd IAT Std',  # 重要性: 0.0148
            'Fwd IAT Max',  # 重要性: 0.0141
            'PSH Flag Count',  # 重要性: 0.0130
            'Bwd Header Length',  # 重要性: 0.0129
            'Fwd Act Data Packets',  # 重要性: 0.0128
            'Flow IAT Max',  # 重要性: 0.0125
            'Avg Fwd Segment Size',  # 重要性: 0.0122
            'Fwd Packet Length Mean',  # 重要性: 0.0115
            'Init Bwd Win Bytes',  # 重要性: 0.0111
            'Flow IAT Std',  # 重要性: 0.0100
            'ACK Flag Count'  # 重要性: 0.0100
        ]

        # 特征权重
        self.feature_weights = {
            'Bwd Packet Length Std': 0.0808, 'Packet Length Variance': 0.0734,
            'Packet Length Std': 0.0727, 'Avg Bwd Segment Size': 0.0565,
            'Bwd Packet Length Mean': 0.0541, 'Avg Packet Size': 0.0446,
            'Bwd Packet Length Max': 0.0417, 'Packet Length Max': 0.0373,
            'Packet Length Mean': 0.0345, 'Subflow Bwd Bytes': 0.0267,
            'Bwd Packets Length Total': 0.0266, 'Destination Port': 0.0261,
            'Fwd Packets Length Total': 0.0206, 'Subflow Fwd Packets': 0.0200,
            'Subflow Fwd Bytes': 0.0197, 'Total Fwd Packets': 0.0196,
            'Fwd Header Length': 0.0155, 'Fwd Seg Size Min': 0.0151,
            'Fwd Packet Length Max': 0.0148, 'Fwd IAT Std': 0.0148,
            'Fwd IAT Max': 0.0141, 'PSH Flag Count': 0.0130,
            'Bwd Header Length': 0.0129, 'Fwd Act Data Packets': 0.0128,
            'Flow IAT Max': 0.0125, 'Avg Fwd Segment Size': 0.0122,
            'Fwd Packet Length Mean': 0.0115, 'Init Bwd Win Bytes': 0.0111,
            'Flow IAT Std': 0.0100, 'ACK Flag Count': 0.0100
        }

    def compute_autocorrelation(self, series):
        """
        计算时间序列的自相关函数

        参数:
        - series: 输入时间序列

        返回:
        - autocorr: 自相关系数数组
        - lags: 对应的滞后数组
        """
        # 数据预处理：标准化
        series = np.array(series)
        series = (series - np.mean(series)) / (np.std(series) + 1e-8)

        n = len(series)
        autocorr = np.correlate(series, series, mode='full')

        # 取右半部分（正滞后）
        autocorr = autocorr[n - 1:]

        # 归一化
        autocorr = autocorr / autocorr[0]

        # 截断到指定最大滞后
        max_lag = min(self.max_lag, len(autocorr) - 1)
        autocorr = autocorr[:max_lag + 1]
        lags = np.arange(max_lag + 1)

        return autocorr, lags

    def find_memory_length(self, autocorr, n_samples):
        """
        基于Bartlett公式找到记忆长度

        参数:
        - autocorr: 自相关系数数组
        - n_samples: 原始样本数量

        返回:
        - memory_length: 记忆长度
        - critical_value: 临界值
        """
        # Bartlett公式：对于大样本，自相关的标准误差 ≈ 1/√n
        z_critical = stats.norm.ppf(1 - self.significance_level / 2)  # 双尾检验
        critical_value = z_critical / np.sqrt(n_samples)

        # 确定哪些滞后是显著的
        significant_mask = np.abs(autocorr) > critical_value
        significant_lags = np.where(significant_mask)[0]

        if len(significant_lags) > 1:
            # 寻找连续性断点
            diff = np.diff(significant_lags)
            breaks = np.where(diff > 1)[0]

            if len(breaks) > 0:
                memory_length = significant_lags[breaks[0]]
            else:
                memory_length = significant_lags[-1]
        else:
            memory_length = significant_lags[0] if len(significant_lags) > 0 else 0

        return memory_length, critical_value

    def analyze_feature(self, data, feature_name):
        """
        分析单个特征的自相关结构

        参数:
        - data: 数据DataFrame
        - feature_name: 特征名称

        返回:
        - 分析结果字典
        """
        if feature_name not in data.columns:
            return None

        # 提取特征序列并处理缺失值
        series = data[feature_name].fillna(method='ffill').fillna(0)

        # 计算自相关
        autocorr, lags = self.compute_autocorrelation(series)

        # 找到记忆长度
        memory_length, critical_value = self.find_memory_length(autocorr, len(series))

        return {
            'feature_name': feature_name,
            'autocorr': autocorr,
            'lags': lags,
            'memory_length': memory_length,
            'critical_value': critical_value
        }

    def analyze_by_class(self, data, label_column='Label', max_features=10):
        """
        按类别分析自相关结构

        参数:
        - data: 数据DataFrame
        - label_column: 标签列名
        - max_features: 最大分析特征数量

        返回:
        - 分类分析结果
        """
        results = {}

        # 获取可用特征
        available_features = [f for f in self.key_features if f in data.columns][:max_features]
        print(f"分析 {len(available_features)} 个关键特征")

        # 分析正常流量
        benign_data = data[data[label_column] == 'Benign']
        if len(benign_data) > 100:
            print(f"分析正常流量 (样本数: {len(benign_data)})")
            results['benign'] = {}

            for feature in available_features:
                result = self.analyze_feature(benign_data, feature)
                if result:
                    results['benign'][feature] = result

        # 分析攻击流量
        attack_data = data[data[label_column] != 'Benign']
        if len(attack_data) > 100:
            print(f"分析攻击流量 (样本数: {len(attack_data)})")
            results['attack'] = {}

            for feature in available_features:
                result = self.analyze_feature(attack_data, feature)
                if result:
                    results['attack'][feature] = result

        return results

    def compute_weighted_memory_length(self, data, max_features=15):
        """
        计算加权记忆长度

        参数:
        - data: 数据DataFrame
        - max_features: 最大特征数量

        返回:
        - 加权记忆长度
        """
        available_features = [f for f in self.key_features if f in data.columns][:max_features]

        weighted_sum = 0
        total_weight = 0

        for feature in available_features:
            if feature in self.feature_weights:
                result = self.analyze_feature(data, feature)
                if result:
                    weight = self.feature_weights[feature]
                    weighted_sum += weight * result['memory_length']
                    total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def recommend_window_size(self, data, window_sizes=None):
        """
        推荐窗口大小

        参数:
        - data: 数据DataFrame
        - window_sizes: 候选窗口大小列表

        返回:
        - 推荐结果
        """
        if window_sizes is None:
            window_sizes = [100, 200, 300, 400, 500, 570, 600, 800, 1000]

        print("\n=== 窗口大小推荐分析 ===")

        # 计算加权记忆长度
        weighted_memory = self.compute_weighted_memory_length(data)
        print(f"加权平均记忆长度: {weighted_memory:.1f}")

        # 分类分析
        class_results = self.analyze_by_class(data)

        # 计算各类别的平均记忆长度
        class_memories = {}
        for class_name, class_data in class_results.items():
            if isinstance(class_data, dict):
                memories = [r['memory_length'] for r in class_data.values()
                            if isinstance(r, dict)]
                if memories:
                    class_memories[class_name] = np.mean(memories)

        # 综合推荐
        all_memories = [weighted_memory] + list(class_memories.values())
        overall_memory = np.mean(all_memories)

        recommendations = {
            'theoretical_optimal': int(overall_memory * 1.2),
            'conservative': int(overall_memory * 0.8),
            'aggressive': int(overall_memory * 2.0),
            'current_choice': self.current_window_size,
            'weighted_memory': weighted_memory,
            'class_memories': class_memories,
            'overall_memory': overall_memory
        }

        # 评估候选窗口大小
        print(f"\n理论最优窗口大小: {recommendations['theoretical_optimal']}")
        print(f"保守估计: {recommendations['conservative']}")
        print(f"激进估计: {recommendations['aggressive']}")
        print(f"当前选择: {recommendations['current_choice']}")

        # 评估当前选择
        ratio = self.current_window_size / recommendations['theoretical_optimal']
        if 0.8 <= ratio <= 1.2:
            evaluation = "✓ 在理论最优范围内"
        elif 0.5 <= ratio < 0.8:
            evaluation = "⚠ 略小，可能错失长期依赖"
        elif 1.2 < ratio <= 2.0:
            evaluation = "⚠ 略大，但可接受"
        else:
            evaluation = "✗ 显著偏离理论最优"

        print(f"评估结果: {evaluation} (比值: {ratio:.2f})")

        # 评估所有候选窗口大小
        print(f"\n候选窗口大小评估:")
        for size in window_sizes:
            ratio = size / recommendations['theoretical_optimal']
            if 0.8 <= ratio <= 1.2:
                status = "推荐 ✓"
            elif 0.5 <= ratio <= 2.0:
                status = "可接受"
            else:
                status = "不推荐"
            print(f"  窗口大小 {size}: 比值 {ratio:.2f} - {status}")

        return recommendations

    def visualize_results(self, data, save_path=None):
        """
        简化的可视化结果

        参数:
        - data: 数据DataFrame
        - save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'自相关窗口分析 (当前窗口: {self.current_window_size})', fontsize=16, fontweight='bold')

        # 1. 加权综合自相关
        ax1 = axes[0, 0]
        weighted_memory = self.compute_weighted_memory_length(data)

        # 选择一个代表性特征进行展示
        representative_feature = None
        for feature in self.key_features[:5]:
            if feature in data.columns:
                representative_feature = feature
                break

        if representative_feature:
            result = self.analyze_feature(data, representative_feature)
            if result:
                autocorr = result['autocorr']
                lags = result['lags']
                critical_value = result['critical_value']
                memory_length = result['memory_length']

                ax1.plot(lags, autocorr, 'b-', linewidth=2, label=f'{representative_feature}')
                ax1.axhline(y=critical_value, color='r', linestyle='--',
                            label=f'显著性阈值 (±{critical_value:.3f})')
                ax1.axhline(y=-critical_value, color='r', linestyle='--')
                ax1.axvline(x=memory_length, color='g', linestyle=':', linewidth=2,
                            label=f'记忆长度: {memory_length}')
                ax1.axvline(x=self.current_window_size, color='orange', linestyle='-',
                            linewidth=2, label=f'当前窗口: {self.current_window_size}')

        ax1.set_xlabel('滞后')
        ax1.set_ylabel('自相关系数')
        ax1.set_title('代表性特征自相关函数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 记忆长度分布
        ax2 = axes[0, 1]
        memory_lengths = []

        for feature in self.key_features[:10]:
            if feature in data.columns:
                result = self.analyze_feature(data, feature)
                if result:
                    memory_lengths.append(result['memory_length'])

        if memory_lengths:
            ax2.hist(memory_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=self.current_window_size, color='red', linestyle='--',
                        linewidth=2, label=f'当前窗口: {self.current_window_size}')
            ax2.axvline(x=np.mean(memory_lengths), color='green', linestyle='-',
                        linewidth=2, label=f'平均记忆长度: {np.mean(memory_lengths):.1f}')

        ax2.set_xlabel('记忆长度')
        ax2.set_ylabel('频次')
        ax2.set_title('记忆长度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 类别对比
        ax3 = axes[1, 0]
        class_results = self.analyze_by_class(data, max_features=5)

        colors = ['blue', 'red']
        class_names = ['benign', 'attack']

        for idx, class_name in enumerate(class_names):
            if class_name in class_results:
                class_memories = [r['memory_length'] for r in class_results[class_name].values()
                                  if isinstance(r, dict)]
                if class_memories:
                    ax3.scatter([idx] * len(class_memories), class_memories,
                                color=colors[idx], alpha=0.6, s=50, label=f'{class_name} 流量')
                    ax3.scatter(idx, np.mean(class_memories), color=colors[idx],
                                s=200, marker='x', label=f'{class_name} 平均值')

        ax3.axhline(y=self.current_window_size, color='orange', linestyle='--',
                    label=f'当前窗口: {self.current_window_size}')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['正常流量', '攻击流量'])
        ax3.set_ylabel('记忆长度')
        ax3.set_title('不同类别流量记忆长度对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 窗口推荐总结
        ax4 = axes[1, 1]
        ax4.axis('off')

        # 计算推荐
        recommendations = self.recommend_window_size(data)

        summary_text = f"""
窗口大小理论分析总结

当前选择: {self.current_window_size} 条流量

理论分析结果:
• 加权记忆长度: {recommendations['weighted_memory']:.1f}
• 理论最优窗口: {recommendations['theoretical_optimal']}
• 保守估计: {recommendations['conservative']}
• 激进估计: {recommendations['aggressive']}

评估结果:
"""

        ratio = self.current_window_size / recommendations['theoretical_optimal']
        if 0.8 <= ratio <= 1.2:
            evaluation = "✓ 在理论最优范围内"
        elif 0.5 <= ratio < 0.8:
            evaluation = "⚠ 略小，可能错失长期依赖"
        elif 1.2 < ratio <= 2.0:
            evaluation = "⚠ 略大，但可接受"
        else:
            evaluation = "✗ 显著偏离理论最优"

        summary_text += f"• {evaluation}\n• 理论比值: {ratio:.2f}"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()


def analyze_window_size(data_path, current_window_size=200, max_features=15):
    """
    主分析函数 - 窗口大小可修改

    参数:
    - data_path: 数据文件路径
    - current_window_size: 当前窗口大小（可修改参数）
    - max_features: 最大分析特征数
    """
    print(f"CICIDS2017 自相关窗口分析 (当前窗口: {current_window_size})")
    print("=" * 60)

    # 初始化分析器
    analyzer = AutocorrelationWindowAnalyzer(
        current_window_size=current_window_size,
        max_lag=int(current_window_size * 2.5)
    )

    # 加载数据
    print(f"\n加载数据: {data_path}")
    if data_path.endswith('.feather'):
        data = pd.read_feather(data_path)
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else:
        raise ValueError("支持的文件格式: .feather 或 .csv")

    print(f"数据形状: {data.shape}")
    print(f"标签分布:\n{data['Label'].value_counts()}")

    # 执行分析
    recommendations = analyzer.recommend_window_size(data)

    # 可视化
    analyzer.visualize_results(data, save_path=f'window_analysis_{current_window_size}.png')

    print(f"\n分析完成! 当前窗口大小 {current_window_size} 的理论评估已生成。")

    return analyzer, recommendations


# 使用示例
if __name__ == "__main__":
    # 可以轻松修改窗口大小进行分析
    CURRENT_WINDOW_SIZE = 570  # 修改这里来测试不同的窗口大小

    # 分析不同窗口大小
    window_sizes_to_test = [200, 400, 570, 600, 800]

    for window_size in window_sizes_to_test:
        print(f"\n{'=' * 50}")
        print(f"测试窗口大小: {window_size}")
        print(f"{'=' * 50}")

        try:
            analyzer, recommendations = analyze_window_size(
                data_path='../cicids2017/clean/all_data.feather',
                current_window_size=window_size,
                max_features=10
            )

            print(f"窗口 {window_size} 分析完成!")

        except Exception as e:
            print(f"分析窗口 {window_size} 时出错: {e}")
            continue