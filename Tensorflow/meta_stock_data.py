import kagglehub
import tensorflow as tf
import pandas as pd
import numpy as np


# Download latest version
path = kagglehub.dataset_download("zongaobian/meta-stock-data-and-key-affiliated-companies")
print("Path to dataset files:", path)