import torch
from torch import nn
import numpy as np

class XRDEncoder(nn.Module):
    def __init__(self, num_channels=1):
        super(XRDEncoder, self).__init__()

        self.num_channels = num_channels
        layer_list = [
            nn.Sequential(
                nn.Conv1d(in_channels=self.num_channels, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU()
            )
        ]
        curr_num_channels = 64
        for _ in range(3):
            # downsize to half the previous size
            layer_list.append(nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
            for i in range(2):
                layer_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=curr_num_channels, out_channels=curr_num_channels//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(num_features=curr_num_channels//2),
                    nn.ReLU()
                ))
                curr_num_channels = curr_num_channels // 2
        self.layers = nn.Sequential(*layer_list)
        self.linear = nn.Linear(in_features=64, out_features=256)

    def forward(self, x):
        result = self.layers(x)
        result = self.linear(result.reshape(x.size(0), -1))
        return result

class XRDRegressor(nn.Module):
    def __init__(self, latent_dim=256, xrd_dim=512, out_channels=1):
        super(XRDRegressor, self).__init__()

        self.high_dim_proj = nn.Linear(latent_dim, xrd_dim)

        layer_list = [
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU()
            )
        ]

        curr_num_channels = 64
        for _ in range(3):
            for _ in range(2):
                layer_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=curr_num_channels, out_channels=curr_num_channels//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(num_features=curr_num_channels//2),
                    nn.ReLU()
                ))
                curr_num_channels = curr_num_channels // 2
        # dont end on ReLU
        layer_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        result = self.high_dim_proj(x)
        result = result.unsqueeze(1)
        result = self.layers(result)
        return result