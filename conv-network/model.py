# Conv-Network mit PyTorch
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2

class ConvModel(nn.Module):
    def __init__(self) -> None:
        super(ConvModel, self).__init__()
        self.filter_size = 3 # -> 6 conv layers
        
        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=self.filter_size)

        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=self.filter_size)

        # convolution 3
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=self.filter_size)

        # convolution 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=self.filter_size)

        # convolution 5
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=self.filter_size)

        # convolution 6
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=self.filter_size)
        
        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=32, out_features=16)
        
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=16, out_features=8)

        # fully connected layer 3
        self.fc3 = nn.Linear(in_features=8, out_features=4)

        # fully connected layer 4
        self.fc4 = nn.Linear(in_features=4, out_features=2)
    
    def forward(self, x):
        # test shape: (1, 13, 13)

        # convolution 1
        out = self.conv1(x)
        # print("\nconvolution 1: ", torch._shape_as_tensor(out)) # shape: (4, 11, 11)
        # sigmoid activation 1
        out = F.sigmoid(out)

        # convolution 2
        out = self.conv2(out)
        # print("convolution 2: ", torch._shape_as_tensor(out)) # shape: (6, 9, 9)
        # sigmoid activation 2
        out = F.sigmoid(out)

        # convolution 3
        out = self.conv3(out)
        # print("convolution 3: ", torch._shape_as_tensor(out)) # shape: (8, 7, 7)
        # sigmoid activation 3
        out = F.sigmoid(out)

        # convolution 4
        out = self.conv4(out)
        # print("convolution 4: ", torch._shape_as_tensor(out)) # shape: (12, 5, 5)
        # sigmoid activation 4
        out = F.sigmoid(out)

        # convolution 5
        out = self.conv5(out)
        # print("convolution 5: ", torch._shape_as_tensor(out)) # shape: (24, 3, 3)
        # sigmoid activation 5
        out = F.sigmoid(out)

        # convolution 6
        out = self.conv6(out)
        # print("convolution 6: ", torch._shape_as_tensor(out)) # shape: (32, 1 1)
        # sigmoid activation 6
        out = F.sigmoid(out)

        # Flatten the output from conv layers
        out = torch.flatten(out, 1) # Flatten from [(batch_size,) 32, 1, 1] to [(batch_size,) 32, 1]
        out = torch.flatten(out, 0) # Flatten from [(batch_size,) 32, 1] to [(batch_size,) 32] 

        # fully connected layers with sigmoid activations
        out = self.fc1(out)
        out = F.sigmoid(out)

        out = self.fc2(out)
        out = F.sigmoid(out)

        out = self.fc3(out)
        out = F.sigmoid(out)

        out = self.fc4(out)
        out = F.sigmoid(out)

        return out # (B, 2) --> A und B Farbchannel
    
