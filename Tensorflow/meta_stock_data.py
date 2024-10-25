import kagglehub
import tensorflow as tf
import pandas as pd
import numpy as np

path = kagglehub.dataset_download(r"zongaobian/meta-stock-data-and-key-affiliated-companies")
print("Path to dataset files:", path)

def convert_path(pa):
    for i in pa:
        if i == "\\":
            pa[pa.index(i)] = "/"
        else:
            continue
    return pa
new_path = convert_path(path) + "META_daily_data.csv"
data = pd.read_csv(new_path)

print(data.head)