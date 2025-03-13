# Conv-Network mit PyTorch
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import cv2

class ConvModel(nn.Module):
    def __init__(self,
                 layer_1_in_channels_param: int,
                 layer_1_out_channels_param: int,
                 layer_2_in_channels_param: int,
                 layer_2_out_channels_param: int,
                 layer_3_in_channels_param: int,
                 layer_3_out_channels_param: int,
                 layer_4_in_channels_param: int,
                 layer_4_out_channels_param: int,
                 layer_5_in_channels_param: int,
                 layer_5_out_channels_param: int,
                 layer_6_in_channels_param: int,
                 layer_6_out_channels_param: int,
                 linear_layer_1_in_features_param: int,
                 linear_layer_1_out_features_param: int,
                 linear_layer_2_in_features_param: int,
                 linear_layer_2_out_features_param: int,
                 linear_layer_3_in_features_param: int,
                 linear_layer_3_out_features_param: int
            ) -> None:

        super(ConvModel, self).__init__()
        self.filter_size = 3 # -> 6 conv layers

        # convolution layer 1 parameters
        self.layer_1_in_channels = layer_1_in_channels_param
        self.layer_1_out_channels = layer_1_out_channels_param

        # convolution layer 2 parameters
        self.layer_2_in_channels = layer_2_in_channels_param
        self.layer_2_out_channels = layer_2_out_channels_param

        # convolution layer 3 parameters
        self.layer_3_in_channels = layer_3_in_channels_param
        self.layer_3_out_channels = layer_3_out_channels_param

        # convolution layer 4 parameters
        self.layer_4_in_channels = layer_4_in_channels_param
        self.layer_4_out_channels = layer_4_out_channels_param

        # convolution layer 5 parameters
        self.layer_5_in_channels = layer_5_in_channels_param
        self.layer_5_out_channels = layer_5_out_channels_param

        # convolution layer 6 parameters
        self.layer_6_in_channels = layer_6_in_channels_param
        self.layer_6_out_channels = layer_6_out_channels_param

        # linear layer 1 parameters
        self.linear_layer_1_in_features = linear_layer_1_in_features_param
        self.linear_layer_1_out_features = linear_layer_1_out_features_param

        # linear layer 2 parameters
        self.linear_layer_2_in_features = linear_layer_2_in_features_param
        self.linear_layer_2_out_features = linear_layer_2_out_features_param

        # linear layer 3 parameters
        self.linear_layer_3_in_features = linear_layer_3_in_features_param
        self.linear_layer_3_out_features = linear_layer_3_out_features_param


        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=self.layer_1_in_channels, out_channels=self.layer_1_out_channels, kernel_size=self.filter_size)

        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=self.layer_2_in_channels, out_channels=self.layer_2_out_channels, kernel_size=self.filter_size)

        # convolution 3
        self.conv3 = nn.Conv2d(in_channels=self.layer_3_in_channels, out_channels=self.layer_3_out_channels, kernel_size=self.filter_size)

        # convolution 4
        self.conv4 = nn.Conv2d(in_channels=self.layer_4_in_channels, out_channels=self.layer_4_out_channels, kernel_size=self.filter_size)

        # convolution 5
        self.conv5 = nn.Conv2d(in_channels=self.layer_5_in_channels, out_channels=self.layer_5_out_channels, kernel_size=self.filter_size)

        # convolution 6
        self.conv6 = nn.Conv2d(in_channels=self.layer_6_in_channels, out_channels=self.layer_6_out_channels, kernel_size=self.filter_size)
        
        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=self.linear_layer_1_in_features, out_features=self.linear_layer_1_out_features)
        
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=self.linear_layer_2_in_features, out_features=self.linear_layer_2_out_features)

        # fully connected layer 3
        self.fc3 = nn.Linear(in_features=self.linear_layer_3_in_features, out_features=self.linear_layer_3_out_features)


    # noinspection PyPep8Naming
    def forward(self, x):
        # test shape: (1, 13, 13)
        LeakyReLU = nn.LeakyReLU()

        # convolution 1
        out = self.conv1(x)
        out = LeakyReLU(out)

        # convolution 2
        out = self.conv2(out)
        out = LeakyReLU(out)

        # convolution 3
        out = self.conv3(out)
        out = LeakyReLU(out)

        # convolution 4
        out = self.conv4(out)
        out = LeakyReLU(out)

        # convolution 5
        out = self.conv5(out)
        out = LeakyReLU(out)

        # convolution 6
        out = self.conv6(out)
        out = LeakyReLU(out)

        # Flatten
        out = torch.flatten(out, 1) # Flatten from [(batch_size,) 32, 1, 1] to [(batch_size,) 32, 1]


        # fc 1
        out = self.fc1(out)
        out = LeakyReLU(out)

        # fc 2
        out = self.fc2(out)
        out = LeakyReLU(out)

        # fc 3
        out = self.fc3(out)
        out = LeakyReLU(out)

        return out # shape: tensor([2])
    
