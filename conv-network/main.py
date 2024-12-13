import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from main import ConvModel

# instantate the convolution model
model = ConvModel()
print(model)

# list of parameters
params = list(model.parameters())
print("length parameters: ", len(params))
print("output_size: ", params[0].size())

