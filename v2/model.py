import torch
import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self,
            ) -> None:

        super(ConvModel, self).__init__()
        self.filter_size = 3 # -> 6 conv layers

        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.filter_size)
        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=self.filter_size)
        # convolution 3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.filter_size)
        # convolution 4
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.filter_size)
        # convolution 5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.filter_size)
        # convolution 6
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.filter_size)
        
        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=128, out_features=32)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    # noinspection PyPep8Naming
    def forward(self, x):
        # test shape: (1, 13, 13)
        LeakyReLU = nn.LeakyReLU()

        x = self.conv1(x)
        x = LeakyReLU(x)

        x = self.conv2(x)
        x = LeakyReLU(x)

        x = self.conv3(x)
        x = LeakyReLU(x)

        x = self.conv4(x)
        x = LeakyReLU(x)

        x = self.conv5(x)
        x = LeakyReLU(x)

        x = self.conv6(x)
        x = LeakyReLU(x)

        # Flatten
        x = torch.flatten(x, 1) # Flatten from [(batch_size,) 32, 1, 1] to [(batch_size,) 32, 1]

        x = self.fc1(x)
        x = LeakyReLU(x)

        x = self.fc2(x)
        return x # shape: tensor([2])
    
