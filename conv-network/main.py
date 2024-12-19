import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import model
import cv2

#instantate the convolution model
conv_model = model.ConvModel()
print(conv_model)

# list of parameters
params = list(conv_model.parameters())
print("length parameters: ", len(params))
total_params = sum(
    param.numel() for param in conv_model.parameters()
)
print("total parameters: ", total_params)
print("output_size: ", params[0].size())

x = torch.randn(size=(2, 13, 13))
print("x: ", conv_model(x))


