# Conv-Network mit PyTorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class ConvModel(nn.Module):
    def __init__(self) -> None:
        super(ConvModel, self).__init__()
        self.input_shape = 3 # -> 6 conv layers 
        
        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.input_shape)

        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=self.input_shape)

        # convolution 3
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=self.input_shape)

        # convolution 4
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.input_shape)

        # convolution 5
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.input_shape)

        # convolution 6
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=self.input_shape)
        
        # fully connected layer
        self.fc1 = nn.Linear(24, 2)
    
    def forward(self, x):
        # convolution 1
        out = self.conv1(x)
        # sigmoid activation 1
        out = F.sigmoid(out)

        # convolution 2
        out = self.conv2(out)
        # sigmoid activation 2
        out = F.sigmoid(out)

        # convolution 3
        out = self.conv3(out)
        # sigmoid activation 3
        out = F.sigmoid(out)

        # convolution 4
        out = self.conv3(out)
        # sigmoid activation 4
        out = F.sigmoid(out)

        # convolution 5
        out = self.conv3(out)
        # sigmoid activation 5
        out = F.sigmoid(out)

        # convolution 6
        out = self.conv3(out)
        # sigmoid activation 6
        out = F.sigmoid(out)

        # fully connected layer
        out = self.fc1(out)

        return out
    
