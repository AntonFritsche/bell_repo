import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
    def __init__(self,
                 section_size: int,
            ) -> None:
        super(ConvModel, self).__init__()
        assert(section_size % 2 == 1)
        self.filter_size = 3
        self.num_layers = section_size // 2
        if self.num_layers > 7:
            self.num_layers = 7
        self.plane_expansion = 2

        self.layers = []
        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else 4*2**i
            out_channels = 4*2**(i+1)

            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.filter_size))
            self.layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*self.layers)

        in_features = 4*2**self.num_layers

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        # all conv + LeakyReLU layers
        x = self.layers(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)

        return x
    
class ConvModel_v2(nn.Module):
    def __init__(self,#
                 section_size: int,
            ) -> None:
        super(ConvModel_v2, self).__init__()
        assert (section_size % 2 == 1)
        self.filter_size = 3
        self.layers = []

        self.layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=self.filter_size, padding=1))
        self.layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*self.layers)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=256, out_features=32)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        # all conv + LeakyReLU layers
        x = self.layers(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)

        return x