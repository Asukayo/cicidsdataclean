import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 方案1: 使用matplotlib内置字体
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号



warnings.filterwarnings('ignore')




def load_and_preprocess_data(file_path):
    """
    加载并预处理数据

    参数:
    - file_path: parquet文件路径

    返回:
    - df: 预处理后的DataFrame
    - feature_cols: 特征列名列表
    """
    print("正在加载数据...")
    df = pd.read_parquet(file_path)
    print(f"数据形状: {df.shape}")

    # 打印标签分布
    print("\n标签分布:")
    print(df['Label'].value_counts())

    # 将标签进行二分类处理 (正常 vs 异常)
    # 将所有非'Benign'的标签都视为异常
    df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)

    # 获取特征列（排除时间戳和标签列）
    exclude_cols = ['Timestamp', 'Label', 'binary_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 处理无穷值和缺失值
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 统计缺失值
    missing_counts = df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n发现缺失值，共 {missing_counts.sum()} 个")
        # 使用0填充缺失值（也可以使用其他策略）
        df[feature_cols] = df[feature_cols].fillna(0)

    print(f"\n特征数量: {len(feature_cols)}")

    return df, feature_cols


def calculate_feature_importance(X, y, n_estimators=100, max_samples=50000):
    """
    使用随机森林计算特征重要性

    参数:
    - X: 特征矩阵
    - y: 标签向量
    - n_estimators: 随机森林中树的数量
    - max_samples: 最大样本数（用于加速计算）

    返回:
    - importance_df: 包含特征重要性的DataFrame
    """
    print(f"\n使用随机森林计算特征重要性...")
    print(f"参数: n_estimators={n_estimators}")

    # 如果数据量太大，进行采样
    if len(X) > max_samples:
        print(f"数据量较大({len(X)}条)，随机采样{max_samples}条进行分析...")
        # 分层采样，保持类别比例
        _, X_sample, _, y_sample = train_test_split(
            X, y, test_size=max_samples / len(X),
            random_state=42, stratify=y
        )
    else:
        X_sample = X
        y_sample = y

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1  # 显示进度
    )

    # 训练模型
    print("正在训练随机森林模型...")
    rf_classifier.fit(X_sample, y_sample)

    # 获取特征重要性
    feature_importances = rf_classifier.feature_importances_

    # 计算准确率（可选）
    print(f"\n模型OOB得分: {rf_classifier.oob_score if hasattr(rf_classifier, 'oob_score') else '未计算'}")

    return feature_importances


def analyze_and_visualize_importance(feature_cols, feature_importances, top_n=30):
    """
    分析并可视化特征重要性

    参数:
    - feature_cols: 特征名称列表
    - feature_importances: 特征重要性数组
    - top_n: 显示前N个最重要的特征

    返回:
    - importance_df: 排序后的特征重要性DataFrame
    """
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importances
    })

    # 按重要性降序排序
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 添加累积重要性
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    # 打印最重要的特征
    print(f"\n前{top_n}个最重要的特征:")
    print("-" * 60)
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{idx + 1:3d}. {row['feature']:40s} {row['importance']:.6f}")

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 图1: 前N个特征的重要性条形图
    top_features = importance_df.head(top_n)
    ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('特征重要性')
    ax1.set_title(f'前{top_n}个最重要的特征')
    ax1.invert_yaxis()

    # 图2: 累积重要性曲线
    ax2.plot(range(1, len(importance_df) + 1), importance_df['cumulative_importance'])
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%累积重要性')
    ax2.set_xlabel('特征数量')
    ax2.set_ylabel('累积重要性')
    ax2.set_title('特征累积重要性曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 找出达到95%累积重要性所需的特征数量
    n_features_95 = (importance_df['cumulative_importance'] >= 0.95).idxmax() + 1
    ax2.axvline(x=n_features_95, color='g', linestyle='--',
                label=f'{n_features_95}个特征达到95%')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"\n达到95%累积重要性需要前{n_features_95}个特征")

    return importance_df


def save_results(importance_df, output_path='feature_importance_results.csv'):
    """
    保存特征重要性结果

    参数:
    - importance_df: 特征重要性DataFrame
    - output_path: 输出文件路径
    """
    importance_df.to_csv(output_path, index=False)
    print(f"\n特征重要性结果已保存到: {output_path}")

    # 同时保存一个简化版本，只包含达到95%累积重要性的特征
    n_features_95 = (importance_df['cumulative_importance'] >= 0.95).idxmax() + 1
    important_features = importance_df.head(n_features_95)['feature'].tolist()

    with open('important_features_95.txt', 'w') as f:
        for feature in important_features:
            f.write(f"{feature}\n")

    print(f"前{n_features_95}个重要特征列表已保存到: important_features_95.txt")


def main():
    """
    主函数
    """
    # 配置参数
    data_path = "../cicids2017/clean/all_data.parquet"  # 数据文件路径
    n_estimators = 10000  # 随机森林中树的数量
    max_samples = 100000  # 用于训练的最大样本数
    top_n_display = 30  # 显示前N个最重要的特征

    try:
        # 1. 加载和预处理数据
        df, feature_cols = load_and_preprocess_data(data_path)

        # 2. 准备特征矩阵和标签
        X = df[feature_cols].values
        y = df['binary_label'].values

        print(f"\n特征矩阵形状: {X.shape}")
        print(f"正常样本: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"异常样本: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

        # 3. 计算特征重要性
        feature_importances = calculate_feature_importance(
            X, y, n_estimators=n_estimators, max_samples=max_samples
        )

        # 4. 分析和可视化结果
        importance_df = analyze_and_visualize_importance(
            feature_cols, feature_importances, top_n=top_n_display
        )

        # 5. 保存结果
        save_results(importance_df)

        # 6. 额外分析：特征重要性分布
        plt.figure(figsize=(10, 6))
        plt.hist(importance_df['importance'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('特征重要性')
        plt.ylabel('特征数量')
        plt.title('特征重要性分布直方图')
        plt.axvline(x=importance_df['importance'].mean(), color='r',
                    linestyle='--', label=f'平均值: {importance_df["importance"].mean():.6f}')
        plt.axvline(x=importance_df['importance'].median(), color='g',
                    linestyle='--', label=f'中位数: {importance_df["importance"].median():.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # 打印统计信息
        print("\n特征重要性统计:")
        print(f"- 平均重要性: {importance_df['importance'].mean():.6f}")
        print(f"- 中位数: {importance_df['importance'].median():.6f}")
        print(f"- 标准差: {importance_df['importance'].std():.6f}")
        print(f"- 最大值: {importance_df['importance'].max():.6f}")
        print(f"- 最小值: {importance_df['importance'].min():.6f}")

        # 低重要性特征分析
        low_importance_threshold = 0.0001
        low_importance_features = importance_df[importance_df['importance'] < low_importance_threshold]
        print(f"\n重要性低于{low_importance_threshold}的特征数量: {len(low_importance_features)}")

        if len(low_importance_features) > 0:
            print("这些特征可以考虑在降维时移除:")
            for _, row in low_importance_features.head(10).iterrows():
                print(f"  - {row['feature']}: {row['importance']:.8f}")
            if len(low_importance_features) > 10:
                print(f"  ... 以及其他 {len(low_importance_features) - 10} 个特征")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()