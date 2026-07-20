import argparse
import os
import zipfile
import json
import pandas as pd
from datasets import Dataset

zip_path = 'dataset/longbench/data.zip'  # 请替换为您本地真正的 data.zip 路径
save_dir = './dataset/longbench'

all_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"not find：{zip_path}")

with zipfile.ZipFile(zip_path, 'r') as z:
    file_list = z.namelist()
    for dataset_name in all_datasets:
        target_file = None
        for f in file_list:
            if f.endswith(f"{dataset_name}.jsonl"):
                target_file = f
                break
        
        if not target_file:
            print(f"{target_file} not find")
            continue
            
        try:
           with z.open(target_file) as f:
                lines = f.read().decode('utf-8').splitlines()
                data_list = [json.loads(line) for line in lines if line.strip()]
            
           valid_data = [item for item in data_list if item.get('split', 'test') == 'test']
            
           if not valid_data:
               print("test not find")
               continue
               
           #df = pd.DataFrame(valid_data)
           #hf_dataset = Dataset.from_pandas(df) 
           hf_dataset = Dataset.from_list(valid_data)
           hf_dataset.save_to_disk(os.path.join(save_dir, dataset_name))
        except Exception as e:
            print(f"❌ convert {dataset_name} fail: {e}")

