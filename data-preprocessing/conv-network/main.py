# Conv-Network mit PyTorch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class ConvModel(nn.Module):
    def __init__(self) -> None:
        super(ConvModel, self).__init__()
        self.input_shape = 5
        self.conv1 = nn.Conv2d(input=3, ouput=3, kernel_size=self.input_shape, axtivation="sigmoid")
        self.conv2 = nn.Conv2d(input=3, ouput=6, axtivation="sigmoid")
        