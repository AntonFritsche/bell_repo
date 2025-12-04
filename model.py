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

        self.block_1 = []
        for i in range(self.num_layers // 2):
            in_channels = 1 if i == 0 else 4 * 2**i
            out_channels = 4 * 2**(i+1)

            print(in_channels, out_channels)

            self.block_1.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.filter_size,
                    bias=True
                ))
            self.block_1.append(nn.LeakyReLU(True))
        in_features = 4 * 2 ** (self.num_layers // 2)
        self.block_1.append(nn.BatchNorm2d(in_features))
        self.block_1 = nn.Sequential(*self.block_1)
        
        print(in_features)

        self.block_2 = []
        for i in range(self.num_layers // 2):
            in_channels = in_features if i == 0 else int((2**i) * (in_features/2))
            out_channels = in_channels if i == self.num_layers else int((2**(i + 1)) * (in_features/2))
            
            print(in_channels, out_channels)

            self.block_2.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.filter_size,
                    bias=True
                ))
            self.block_2.append(nn.LeakyReLU(True))
        in_features = int((2**(self.num_layers // 2)) * (in_features/2))
        self.block_2.append(nn.BatchNorm2d(in_features))
        
        print(in_features)
        self.block_2 = nn.Sequential(*self.block_2)

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=64, bias=True)
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        # fully connected layer 3
        self.fc3 = nn.Linear(in_features=32, out_features=2, bias=True)

    def forward(self, x):
        # all conv + LeakyReLU layers
        x = self.block_1(x)
        x = self.block_2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
