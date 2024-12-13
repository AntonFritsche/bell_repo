import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import model

# instantate the convolution model
conv_model = model.ConvModel()
print(model)

# list of parameters
params = list(conv_model.parameters())
print("length parameters: ", len(params))
print("output_size: ", params[0].size())

