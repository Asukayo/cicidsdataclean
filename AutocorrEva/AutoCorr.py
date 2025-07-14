"""
修复版本：解决硬编码不一致和记忆长度计算问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class FixedAutocorrelationWindowAnalyzer:
    """
    修复版本的自相关窗口大小分析器
    解决硬编码不一致和记忆长度计算问题
    """

    def __init__(self, current_window_size=570, max_lag=None, significance_level=0.05):
        """
        初始化分析器

        参数:
        - current_window_size: 当前使用的窗口大小（一致性参数）
        - max_lag: 最大滞后长度
        - significance_level: 显著性水平
        """
        self.current_window_size = current_window_size  # 统一的当前窗口大小
        self.max_lag = max_lag if max_lag else min(int(current_window_size * 2.5), 1000)
        self.significance_level = significance_level

        # 保持关键特征不变
        self.key_features = [
            'Bwd Packet Length Std', 'Packet Length Variance', 'Packet Length Std',
            'Avg Bwd Segment Size', 'Bwd Packet Length Mean', 'Avg Packet Size',
            'Bwd Packet Length Max', 'Packet Length Max', 'Packet Length Mean',
            'Subflow Bwd Bytes', 'Bwd Packets Length Total', 'Destination Port',
            'Fwd Packets Length Total', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Total Fwd Packets', 'Fwd Header Length', 'Fwd Seg Size Min',
            'Fwd Packet Length Max', 'Fwd IAT Std', 'Fwd IAT Max',
            'PSH Flag Count', 'Bwd Header Length', 'Fwd Act Data Packets',
            'Flow IAT Max', 'Avg Fwd Segment Size', 'Fwd Packet Length Mean',
            'Init Bwd Win Bytes', 'Flow IAT Std', 'ACK Flag Count'
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
        """计算自相关函数"""
        series = np.array(series)
        series = (series - np.mean(series)) / (np.std(series) + 1e-8)

        n = len(series)
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]

        max_lag = min(self.max_lag, len(autocorr) - 1)
        autocorr = autocorr[:max_lag + 1]
        lags = np.arange(max_lag + 1)

        return autocorr, lags

    def significance_test(self, autocorr, n_samples, alpha=None):
        """
        修复的显著性检验：改进记忆长度计算逻辑
        """
        if alpha is None:
            alpha = self.significance_level

        z_critical = stats.norm.ppf(1 - alpha/2)
        critical_value = z_critical / np.sqrt(n_samples)

        # 找到显著的滞后
        significant_mask = np.abs(autocorr) > critical_value
        significant_lags = np.where(significant_mask)[0]

        if len(significant_lags) == 0:
            return 0, critical_value

        # 修复的记忆长度计算逻辑
        # 方法1：连续显著性方法
        if len(significant_lags) > 1:
            # 寻找连续性断点
            diff = np.diff(significant_lags)
            gaps = np.where(diff > 5)[0]  # 允许小的间隙

            if len(gaps) > 0:
                # 第一个大间隙之前的最后一个显著滞后
                memory_length = significant_lags[gaps[0]]
            else:
                # 所有显著滞后都相对连续
                memory_length = significant_lags[-1]
        else:
            memory_length = significant_lags[0]

        # 方法2：阈值衰减方法（备选）
        # 找到自相关首次持续低于阈值的位置
        threshold = critical_value * 2  # 更严格的阈值

        consecutive_below = 0
        threshold_memory = len(autocorr) - 1

        for i in range(1, len(autocorr)):
            if np.abs(autocorr[i]) < threshold:
                consecutive_below += 1
                if consecutive_below >= 5:  # 连续5个点都低于阈值
                    threshold_memory = i - 5
                    break
            else:
                consecutive_below = 0

        # 取两种方法的较小值作为保守估计
        memory_length = min(memory_length, threshold_memory)

        # 确保记忆长度不超过合理范围
        memory_length = min(memory_length, self.max_lag // 2)

        return memory_length, critical_value

    def analyze_feature(self, data, feature_name):
        """分析单个特征"""
        if feature_name not in data.columns:
            return None

        series = data[feature_name].fillna(method='ffill').fillna(0)
        autocorr, lags = self.compute_autocorrelation(series)
        memory_length, critical_value = self.significance_test(autocorr, len(series))

        return {
            'feature_name': feature_name,
            'autocorr': autocorr,
            'lags': lags,
            'memory_length': memory_length,
            'critical_value': critical_value,
            'series_length': len(series)
        }

    def analyze_by_class(self, data, label_column='Label', max_features=10):
        """按类别分析"""
        results = {}
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

    def recommend_window_size(self, data):
        """
        修复的窗口大小推荐：统一使用self.current_window_size
        """
        print(f"\n=== 窗口大小推荐分析 (当前选择: {self.current_window_size}) ===")

        # 计算加权记忆长度
        available_features = [f for f in self.key_features if f in data.columns][:15]

        weighted_sum = 0
        total_weight = 0
        feature_memories = []

        for feature in available_features:
            if feature in self.feature_weights:
                result = self.analyze_feature(data, feature)
                if result and result['memory_length'] > 0:
                    weight = self.feature_weights[feature]
                    weighted_sum += weight * result['memory_length']
                    total_weight += weight
                    feature_memories.append(result['memory_length'])

        weighted_memory = weighted_sum / total_weight if total_weight > 0 else 0

        # 分类分析
        class_results = self.analyze_by_class(data)
        class_memories = {}

        for class_name, class_data in class_results.items():
            if isinstance(class_data, dict):
                memories = [r['memory_length'] for r in class_data.values()
                           if isinstance(r, dict) and r['memory_length'] > 0]
                if memories:
                    class_memories[class_name] = np.mean(memories)

        # 综合记忆长度（使用多种方法的中位数）
        all_memories = []
        if weighted_memory > 0:
            all_memories.append(weighted_memory)
        all_memories.extend(list(class_memories.values()))
        all_memories.extend(feature_memories)

        if not all_memories:
            print("警告：无法计算有效的记忆长度")
            return None

        # 使用中位数而不是平均值，更鲁棒
        overall_memory = np.median(all_memories)

        print(f"加权记忆长度: {weighted_memory:.1f}")
        print(f"综合记忆长度 (中位数): {overall_memory:.1f}")
        print(f"记忆长度范围: {np.min(all_memories):.1f} - {np.max(all_memories):.1f}")

        # 推荐计算
        recommendations = {
            'memory_length': overall_memory,
            'conservative': int(overall_memory * 0.8),
            'recommended': int(overall_memory * 1.2),
            'aggressive': int(overall_memory * 2.0),
            'current_choice': self.current_window_size,  # 使用统一的当前选择
            'class_memories': class_memories,
            'all_memories': all_memories
        }

        print(f"\n理论推荐:")
        print(f"  保守估计: {recommendations['conservative']}")
        print(f"  理论最优: {recommendations['recommended']}")
        print(f"  激进估计: {recommendations['aggressive']}")
        print(f"  当前选择: {recommendations['current_choice']}")

        # 评估当前选择
        if recommendations['recommended'] > 0:
            ratio = self.current_window_size / recommendations['recommended']
            print(f"  理论比值: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                evaluation = "✓ 在理论最优范围内"
            elif 0.5 <= ratio < 0.8:
                evaluation = "⚠ 略小，可能错失长期依赖"
            elif 1.2 < ratio <= 2.0:
                evaluation = "⚠ 略大，但可接受"
            else:
                evaluation = "✗ 显著偏离理论最优"

            print(f"  评估结果: {evaluation}")
            recommendations['evaluation'] = evaluation
            recommendations['ratio'] = ratio

        return recommendations

    def visualize_results(self, data, save_path=None):
        """修复的可视化：统一使用self.current_window_size"""
        recommendations = self.recommend_window_size(data)

        if not recommendations:
            print("无法生成推荐，跳过可视化")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'自相关窗口分析 (当前窗口: {self.current_window_size})',
                     fontsize=16, fontweight='bold')

        # 1. 代表性特征自相关
        ax1 = axes[0, 0]
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

                ax1.plot(lags, autocorr, 'b-', linewidth=2,
                        label=f'{representative_feature}')
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
        all_memories = recommendations['all_memories']

        if all_memories:
            ax2.hist(all_memories, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=self.current_window_size, color='red', linestyle='--',
                       linewidth=2, label=f'当前窗口: {self.current_window_size}')
            ax2.axvline(x=recommendations['memory_length'], color='green', linestyle='-',
                       linewidth=2, label=f'中位记忆长度: {recommendations["memory_length"]:.1f}')
            ax2.axvline(x=recommendations['recommended'], color='purple', linestyle=':',
                       linewidth=2, label=f'理论推荐: {recommendations["recommended"]}')

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
                                if isinstance(r, dict) and r['memory_length'] > 0]
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

        # 4. 推荐总结
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
窗口大小理论分析总结

当前选择: {self.current_window_size} 条流量

理论分析结果:
• 综合记忆长度: {recommendations['memory_length']:.1f}
• 记忆长度范围: {np.min(all_memories):.0f} - {np.max(all_memories):.0f}
• 理论最优窗口: {recommendations['recommended']}
• 保守估计: {recommendations['conservative']}
• 激进估计: {recommendations['aggressive']}

评估结果:
• {recommendations.get('evaluation', '无法评估')}
• 理论比值: {recommendations.get('ratio', 0):.2f}

说明:
• 记忆长度: 自相关显著性的最大滞后
• 窗口大小: 用于异常检测的时间窗口
• 推荐关系: 窗口 = 1.2 × 记忆长度
"""

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()
        return recommendations


def analyze_window_size_fixed(data_path, current_window_size=570):
    """
    修复版本的主分析函数
    """
    print(f"CICIDS2017 修复版自相关窗口分析 (当前窗口: {current_window_size})")
    print("=" * 70)

    # 初始化修复版分析器
    analyzer = FixedAutocorrelationWindowAnalyzer(
        current_window_size=current_window_size
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

    # 执行分析和可视化
    recommendations = analyzer.visualize_results(
        data,
        save_path=f'fixed_window_analysis_{current_window_size}.png'
    )

    print(f"\n修复版分析完成! 所有计算使用统一的当前窗口大小: {current_window_size}")

    return analyzer, recommendations


# 使用示例
if __name__ == "__main__":
    # 测试修复版本
    try:
        analyzer, recommendations = analyze_window_size_fixed(
            data_path='../cicids2017/clean/all_data.feather',
            current_window_size=570
        )
        print("\n修复版分析成功完成!")
    except Exception as e:
        print(f"分析过程中出错: {e}")