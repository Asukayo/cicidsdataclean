# IDS Dataset Cleaning

## Fetch Datasets

### CIC-IDS-2017
1. Download GeneratedLabelled flows from http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/

### CSE-CIC-IDS-2018
1. Install aws cli: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
2. Fetch dataset files from S3 bucket:

## wyh_dataset_create
create       
01 pre_cleaning  : output in cicids2017/clean folder
02 by_flow_without_pca_scaler  
03

## RF_to_Compress_dimensions
use RF_01.ipynb to calculate feature importance,data is all_parquet.py

## dataprovider 
622Analyze  spilt according to 60:20:20,shows severe data imbalance in validation set.

analysis_total is a script show everyday's normal and abnormal window nums
