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
    def __init__(self, latent_dim=256, xrd_dim=512, num_blocks=4):
        super(XRDRegressor, self).__init__()

        self.num_blocks = num_blocks

        self.high_dim_proj = nn.Linear(latent_dim, xrd_dim)

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU()
        )

        self.dense_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=8*i, out_channels=8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(num_features=8),
                nn.ReLU()
            ) 
            for i in range(1,self.num_blocks+1)]
        )

        self.final_conv = (nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        batch_size = x.shape[0]

        # project to XRD dimensionality
        x = self.high_dim_proj(x)
        x = x.unsqueeze(1)
        assert x.shape == (batch_size, 1, 512)

        # do first convolution
        x = self.first_conv(x)
        assert x.shape == (batch_size, 8, 512)

        # densely connected conv blocks
        x_history = [x]
        for i, the_block in enumerate(self.dense_blocks):
            assert len(x_history) == i + 1 # make sure we are updating the history list
            curr_input = torch.cat(x_history, dim=1)
            assert curr_input.shape == (batch_size, 8 * (i + 1), 512)
            x = the_block(curr_input)
            x_history.append(x) # add new result to running list
            assert x.shape == (batch_size, 8, 512)
        assert len(x_history) == len(self.dense_blocks) + 1 # make sure we hit all the blocks

        # final conv to get one channel
        x = self.final_conv(x)
        # squeeze to get rid of dummy dim
        x = x.squeeze(1)
        assert x.shape == (batch_size, 512)

        return x