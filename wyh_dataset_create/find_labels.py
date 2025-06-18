import pandas as pd
import pickle


# 方法1：从清洗后的数据文件中读取
def get_feature_info_from_data(file_path):
    """从数据文件中获取特征信息"""
    if file_path.endswith('.feather'):
        df = pd.read_feather(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)

    # 排除Timestamp和Label
    feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Label']]

    print(f"特征总数: {len(feature_cols)}")
    print("\n特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:3d}. {col}")

    return feature_cols


# 方法2：从您保存的features_info.pkl文件中读取
def get_feature_info_from_pkl(pkl_path):
    """从保存的pkl文件中获取特征信息"""
    with open(pkl_path, 'rb') as f:
        features_info = pickle.load(f)

    features = features_info['features']
    print(f"特征总数: {len(features)}")
    print("\n特征列表:")
    for i, col in enumerate(features, 1):
        print(f"{i:3d}. {col}")

    return features

# 使用示例
feature_cols = (
    get_feature_info_from_data('../cicids2017/clean/all_data.parquet'))
# 或
# feature_cols = get_feature_info_from_pkl('../cicids2017/flow_windows/features_info.pkl')