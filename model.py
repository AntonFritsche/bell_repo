import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel_v1(nn.Module):
    def __init__(self,
                 section_size: int,
            ) -> None:
        super(ConvModel_v1, self).__init__()
        assert(section_size % 2 == 1)
        self.filter_size = 3
        self.num_layers = section_size // 2
        self.plane_expansion = 2
        self.layers = []

        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else 4 * 2**i
            out_channels = 4 * 2**(i + 1)

            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.filter_size
                )
            )
            self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)
        self.input_features = 4 * 2**self.num_layers

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=self.input_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.layers(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x

class ConvModel_v2(nn.Module):
    def __init__(self,
                 section_size: int,
            ) -> None:
        super(ConvModel_v2, self).__init__()
        assert (section_size % 2 == 1)
        self.filter_size = 3
        self.num_layers = section_size // 2
        self.plane_expansion = 2
        self.layers = []

        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else 4 * 2**i
            out_channels = 4 * 2**(i + 1)

            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.filter_size
                )
            )
            self.layers.append(nn.ReLU())
            if i % 2 == 1:
                self.layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*self.layers)
        self.input_features = 4 * 2**self.num_layers

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=self.input_features, out_features=64, bias=True)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        # fully connected layer 3
        self.fc3 = nn.Linear(in_features=32, out_features=2, bias=True)

    def forward(self, x):
        # all conv + LeakyReLU layers
        x = self.layers(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)

        return x
    
class ConvModel_v3(nn.Module):
    def __init__(self,
                 section_size: int,
            ) -> None:
        super(ConvModel_v3, self).__init__()
        self.filter_size = 3
        self.num_layers = section_size // 2
        self.plane_expansion = 2

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.filter_size)
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.filter_size)
        self.relu_2 = nn.ReLU()

        self.batch_norm_1 = nn.BatchNorm2d(16)

        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.filter_size)
        self.relu_3 = nn.ReLU()

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.filter_size)
        self.relu_4 = nn.ReLU()

        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.filter_size)
        self.relu_5 = nn.ReLU()

        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.filter_size)
        self.relu_6 = nn.ReLU()

        self.batch_norm_3 = nn.BatchNorm2d(256)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.relu_7 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.relu_8 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)

        x = self.batch_norm_1(x)

        x = self.conv_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.relu_4(x)

        x = self.batch_norm_2(x)

        x = self.conv_5(x)
        x = self.relu_5(x)

        x = self.conv_6(x)
        x = self.relu_6(x)

        x = self.batch_norm_3(x)

        x = self.average_pooling(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu_7(x)
        x = self.fc2(x)
        x = self.relu_8(x)
        x = self.fc3(x)

        return x
