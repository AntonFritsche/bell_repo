import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
    def __init__(self,
                 section_size: int = 13,
            ) -> None:
        super(ConvModel, self).__init__()
        assert(section_size % 2 == 1)
        self.filter_size = 3 # -> 6 conv layers
        self.num_layers = section_size // 2
        self.plane_expansion = 2

        self.layers = []
        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else 4*2**i
            out_channels = 4*2**(i+1)

            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.filter_size))
            self.layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*self.layers)

        # # convolution 1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.filter_size)
        # # convolution 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=self.filter_size)
        # # convolution 3
        # self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.filter_size)
        # # convolution 4
        # self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.filter_size)
        # # convolution 5
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.filter_size)
        # # convolution 6
        # self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.filter_size)

        in_features = 4*2**self.num_layers

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    # noinspection PyPep8Naming
    def forward(self, x):
        # all conv + LeakyReLU layers
        x = self.layers(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)

        return x

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
    
