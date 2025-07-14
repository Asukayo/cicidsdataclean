import pandas as pd
import numpy as np
import random 
import os
import glob
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"  # 或者选择其他支持该字符的字体
random.seed(0)


# 与数据集特定信息相关的列：比如流量标识符（Flow ID）、源/目标 IP 和端口，这些通常对机器学习特征无帮助。
# 无方差或冗余的特征：
# 例如部分 TCP 标志（如 URG、PSH）和 Bulk 相关特征，由于在数据中可能始终为 0 或缺乏变化，对建模无用。
# 同时删除一个重复的列 'Fwd Header Length.1'。
drop_columns = [
    # Dataset Specific Information
    "Flow ID", 
    "Source IP", "Src IP", 
    "Source Port", "Src Port", 
    "Destination IP", "Dst IP",
    # Features Without Observed Variance
    "Bwd PSH Flags", 
    "Fwd URG Flags", 
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Byts/b Avg", 
    "Fwd Avg Packets/Bulk", "Fwd Pkts/b Avg", 
    "Fwd Avg Bulk Rate", "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk", "Bwd Byts/b Avg", 
    "Bwd Avg Packets/Bulk", "Bwd Pkts/b Avg", 
    "Bwd Avg Bulk Rate", "Bwd Blk Rate Avg",
    # Duplicate Column
    'Fwd Header Length.1'
]

# 目的：统一列名。
# 数据集中由于不同文件或工具提取，列名可能不统一。
# 这个字典将一些常见的列名别名映射为统一的名称，
# 比如将 'Dst Port' 重命名为 'Destination Port'，
# 将多个表示“总包长”的列统一为 'Fwd Packets Length Total' 等，便于后续分析和模型训练时的一致性。
mapper = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Fwd Packets Length Total', 
    'Total Length of Fwd Packets': 'Fwd Packets Length Total',
    'TotLen Bwd Pkts': 'Bwd Packets Length Total',
    'Total Length of Bwd Packets': 'Bwd Packets Length Total', 
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min', 
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean', 
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max', 
    'Bwd Pkt Len Min': 'Bwd Packet Length Min', 
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std', 
    'Flow Byts/s': 'Flow Bytes/s', 
    'Flow Pkts/s': 'Flow Packets/s', 
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total', 
    'Fwd Header Len': 'Fwd Header Length', 
    'Bwd Header Len': 'Bwd Header Length', 
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s', 
    'Pkt Len Min': 'Packet Length Min', 
    'Min Packet Length': 'Packet Length Min',
    'Pkt Len Max': 'Packet Length Max', 
    'Max Packet Length': 'Packet Length Max',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std', 
    'Pkt Len Var': 'Packet Length Variance', 
    'FIN Flag Cnt': 'FIN Flag Count', 
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count', 
    'PSH Flag Cnt': 'PSH Flag Count', 
    'ACK Flag Cnt': 'ACK Flag Count', 
    'URG Flag Cnt': 'URG Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count', 
    'Pkt Size Avg': 'Avg Packet Size',
    'Average Packet Size': 'Avg Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size', 
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk', 
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate', 
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk', 
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate', 
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes', 
    'Subflow Bwd Pkts': 'Subflow Bwd Packets', 
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init Fwd Win Bytes', 
    'Init_Win_bytes_forward': 'Init Fwd Win Bytes',
    'Init Bwd Win Byts': 'Init Bwd Win Bytes', 
    'Init_Win_bytes_backward': 'Init Bwd Win Bytes',
    'Fwd Act Data Pkts': 'Fwd Act Data Packets',
    'act_data_pkt_fwd': 'Fwd Act Data Packets',
    'Fwd Seg Size Min': 'Fwd Seg Size Min',
    'min_seg_size_forward': 'Fwd Seg Size Min'
}
# 功能：绘制一个文件中每条记录的 Timestamp 分布。
# 对于标签为 “Benign” 的流量，用浅绿色点显示；其他攻击类别则以不同颜色显示。
# 通过散点图可以直观观察数据的时间分布，检查数据的连续性和异常值。
def plot_day(df,file_name):
    # 如果提供了文件名，则设置为图表标题
    if file_name:
        plt.title(file_name)

    df.loc[df["Label"] == "Benign", 'Timestamp'].plot(style='.', color="lightgreen", label='Benign')
    for label in df.Label.unique():
        if label != 'Benign':
            df.loc[df["Label"] == label, 'Timestamp'].plot(style='.', label=label)
    plt.legend()
    plt.show()

# 作用：对 dataset（此处为 'cicids2017'）目录下 original 子目录中的所有 CSV 文件进行预处理。
# 读取文件：通过 pd.read_csv 读取 CSV 文件，设置 skipinitialspace=True（跳过字段前空格）和 encoding='latin'（编码设置）。
# 打印当前文件的 标签分布 和 数据形状，方便查看初始数据情况。
def clean_dataset(dataset, filetypes=['feather']):
    # Will search for all files in the dataset subdirectory 'orignal'
    for file in os.listdir(f'{dataset}/original'):
        print(f"------- {file} -------")
        # 使用 pd.read_csv 读取 CSV 文件，参数 skipinitialspace=True 去除分隔符后多余的空格，encoding='latin' 指定编码格式。
        # 添加low_memory = False 目的是避免DtypeWarning 警告
        df = pd.read_csv(f"{dataset}/original/{file}", skipinitialspace=True, encoding='latin',low_memory=False)
        # 输出 Label 列中每个类别的计数，便于了解数据集初始状态。
        print(df["Label"].value_counts())
        # 输出数据形状（行数和列数）。
        print(f"Shape: {df.shape}")

        # Rename column names for uniform column names across files
        df.rename(columns=mapper, inplace=True)

        # Drop unrelevant columns
        # 删除 drop_columns 列表中列名对应的列。
        # errors="ignore" 表示如果某个列不存在，则忽略该错误。
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

        # Parse Timestamp column to pandas datetime
        # 将 Timestamp 列转换为 pandas 的 datetime 类型，errors='coerce' 将无法解析的值置为 NaT。
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # 应用 lambda 函数：如果小时数小于 8，则加 12 小时（可能是处理时区或录入错误的情况）。
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x + pd.Timedelta(hours=12) if x.hour < 8 else x)
        # 根据时间戳对数据排序。
        df = df.sort_values(by=['Timestamp'])

        # Make Label column Categorical
        # 将 Label 中的 'BENIGN' 替换为 'Benign'，统一大小写。
        df['Label'].replace({'BENIGN': 'Benign'}, inplace=True)
        # 将 Label 列转换为 categorical 类型，提高内存使用效率和后续分析速度。
        df['Label'] = df.Label.astype('category')

        # Parse Columns to correct dtype
        # 找到所有整数类型的列，并使用 pd.to_numeric 转换，同时用 downcast 参数降低内存占用。
        int_col = df.select_dtypes(include='integer').columns
        df[int_col] = df[int_col].apply(pd.to_numeric, errors='coerce', downcast='integer')
        # 浮点列：同上，针对浮点数进行转换和降维。
        float_col = df.select_dtypes(include='float').columns
        df[float_col] = df[float_col].apply(pd.to_numeric, errors='coerce', downcast='float')
        # 对象列：找出所有 dtype 为 object 的列，打印出这些列名，然后尝试将它们转换为数值型。
        # 如果转换失败，则该值会被置为 NaN（errors='coerce'）。
        obj_col = df.select_dtypes(include='object').columns
        print(f'Columns with dtype == object: {obj_col}')
        df[obj_col] = df[obj_col].apply(pd.to_numeric, errors='coerce')

        # Drop rows with invalid data
        # 将所有正无穷（inf）和负无穷（-inf）替换为 NaN。
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 打印出存在 NaN 的行数。
        print(f"{df.isna().any(axis=1).sum()} invalid rows dropped")
        # 删除所有包含 NaN 的行，确保数据有效性。
        df.dropna(inplace=True)

        # Drop duplicate rows
        # 使用 drop_duplicates 删除重复行，但保留 Label 和 Timestamp 列的重复值（因为这两个字段可能不用于判断数据重复）。
        df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label', 'Timestamp']))
        # 再次打印 Label 的计数和数据形状，以确认去重效果。
        print(df["Label"].value_counts())
        print(f"shape: {df.shape}\n")

        # Reset index
        # 重置 DataFrame 的索引，丢弃原来的索引，确保行索引连续。
        df.reset_index(inplace=True, drop=True)

        # Plot resulting file
        # 调用之前定义的 plot_day 函数，对当前文件处理后的数据按照时间戳绘图，帮助直观检查数据分布。
        plot_day(df,file_name=file);

        # Save to file
        # 根据参数 filetypes，将清洗后的 DataFrame 保存为 feather 格式和/或 parquet 格式文件，
        # 存储在 {dataset}/clean/ 目录下，文件名基于原文件名加上相应后缀。
        if 'feather' in filetypes:
            df.to_feather(f'{dataset}/clean/{file}.feather')
        if 'parquet' in filetypes:
            df.to_parquet(f'{dataset}/clean/{file}.parquet', index=False)

#     # 将 {dataset}/clean/ 目录下的所有清洗后的文件聚合成一个 DataFrame，并进一步拆分为恶意和正常数据集。
def aggregate_data(dataset, save=True, filetype='feather'):
    # Will search for all files in the 'clean' directory of the correct filetype and aggregate them
    # 初始化空的 DataFrame all_data。
    all_data = pd.DataFrame()
    # 使用 glob.glob 查找 {dataset}/clean/ 目录下所有指定文件格式（feather 或 parquet）的文件。
    for file in glob.glob(f'{dataset}/clean/*.{filetype}'):
        # 打印文件路径
        print(file)
        # 根据文件格式读取数据。
        df = pd.DataFrame()
        if filetype == 'feather':
            df = pd.read_feather(file)
        if filetype == 'parquet':
            df = pd.read_parquet(file)

        print(df.shape)
        # 打印该文件的形状和各标签的计数。
        print(f'{df["Label"].value_counts()}\n')
        # 将当前文件的数据追加到 all_data 中（使用 append，设置 ignore_index=True 保证索引连续）。
        all_data = all_data.append(df, ignore_index=True)

    # 打印 “ALL DATA” 以提示已聚合所有数据。
    print('ALL DATA')
    # 利用 duplicated 检测重复行，这里同样排除 Label 和 Timestamp 列（这些字段不用于判断是否重复）。
    duplicates = all_data[all_data.duplicated(subset=all_data.columns.difference(['Label', 'Timestamp']))]
    # 打印重复行中各标签的计数，便于了解重复数据的情况。
    print('Removed duplicates after aggregating:')
    print(duplicates.Label.value_counts())
    print('Resulting Dataset')
    # 删除这些重复行，并重置索引。
    all_data.drop(duplicates.index, axis=0, inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    # 再次打印数据集形状及各标签分布，确认聚合后的数据质量。
    print(all_data.shape)
    print(f'{all_data["Label"].value_counts()}\n')

    if save:
        # 根据 Label 划分出恶意（非 "Benign"）数据和正常数据，重置索引。
        malicious = all_data[all_data.Label != 'Benign'].reset_index(drop = True)
        benign = all_data[all_data.Label == 'Benign'].reset_index(drop = True)
        # 分别保存整个聚合数据、恶意数据和正常数据到 {dataset}/clean/ 目录下，文件格式根据 filetype 参数决定（feather 或 parquet）。
        if filetype == 'feather':
            all_data.to_feather(f'{dataset}/clean/all_data.feather')
            malicious.to_feather(f'{dataset}/clean/all_malicious.feather')
            benign.to_feather(f'{dataset}/clean/all_benign.feather')
        if filetype == 'parquet':
            all_data.to_parquet(f'{dataset}/clean/all_data.parquet', index=False)
            malicious.to_parquet(f'{dataset}/clean/all_malicious.parquet', index=False)
            benign.to_parquet(f'{dataset}/clean/all_benign.parquet', index=False)
            
if __name__ == "__main__":
    # Adjust for cleaning the correct dataset into the desired format
    
    # Needs directory with dataset name containing empty dir 'clean' and dir 'original' containing de csv's
    clean_dataset('../cicids2017', filetypes=['feather', 'parquet'])
    aggregate_data('../cicids2017', save=True, filetype='feather')
    aggregate_data('../cicids2017', save=True, filetype='parquet')