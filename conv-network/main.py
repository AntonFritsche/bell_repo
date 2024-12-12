# Conv-Network mit PyTorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class ConvModel(nn.Module):
    def __init__(self) -> None:
        super(ConvModel, self).__init__()
        self.input_shape = 13
        
        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=self.input_shape)
        # max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.input_shape)

        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=self.input_shape)
        # max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.input_shape)

        # convolution 3
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=self.input_shape)
        # max pool 3
        self.maxpool3 = nn.MaxPool2d(self.input_shape)

        # fully connected layer
        self.fc1 = nn.Linear(24, 2)
    
    def forward(self, x):
        # convolution 1
        out = self.conv1(x)
        # sigmoid activation 1
        out = F.sigmoid(out)
        # max pool 1
        out = self.maxpool1(out)
        
        # convolution 2
        out = self.conv2(out)
        # sigmoid activation 2
        out = F.sigmoid(out)
        # max pool 2
        out = self.maxpool2(out)

        # convolution 3
        out = self.conv3(out)
        # sigmoid activation 3
        out = F.sigmoid(out)
        # max pool 3
        out = self.maxpool3(out)

        # fully connected layer
        out = self.fc1(out)

        return out
    
# instantate the convolution model
model = ConvModel()
print(model)

# list of parameters
params = list(model.parameters())
print("length parameters: ", len(params))
print("output_size: ", params[0].size())

