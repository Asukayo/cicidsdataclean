# IDS Dataset Cleaning

## Fetch Datasets

### CIC-IDS-2017
1. Download GeneratedLabelled flows from http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/

### CSE-CIC-IDS-2018
1. Install aws cli: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
2. Fetch dataset files from S3 bucket:
```bash
aws s3 cp --no-sign-request --region eu-west-3 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" Documents\onderzoek\experiment\real-time\data --recursive
```

## Execute Cleaning
* Run `data_cleaning.ipynb` notebook
* Alternatively, run the `data_cleaning.py` script