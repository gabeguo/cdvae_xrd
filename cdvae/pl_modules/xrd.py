import torch
from torch import nn
import numpy as np

class XRDDenseRegressor(nn.Module):
    def __init__(self, latent_dim=256, xrd_dim=512, num_blocks=4):
        super(XRDDenseRegressor, self).__init__()

        self.num_blocks = num_blocks

        self.high_dim_proj = nn.Sequential(
            nn.Linear(latent_dim, xrd_dim),
            nn.BatchNorm1d(num_features=xrd_dim),
            nn.ReLU(),
            nn.Linear(xrd_dim, xrd_dim),
            nn.BatchNorm1d(num_features=xrd_dim),
            nn.ReLU()
        )

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

        # normalize
        max_by_xrd = torch.max(x, 1)[0].reshape(batch_size, 1).expand(-1, 512)
        min_by_xrd = torch.min(x, 1)[0].reshape(batch_size, 1).expand(-1, 512)
        assert max_by_xrd.shape == x.shape
        assert min_by_xrd.shape == x.shape
        x = (x - min_by_xrd) / (max_by_xrd - min_by_xrd)
        assert torch.isclose(torch.min(x), torch.tensor(0.0))
        assert torch.isclose(torch.max(x), torch.tensor(1.0))

        return x
    

class XRDConvRegressor(nn.Module):
    def __init__(self, latent_dim=256, xrd_dim=512):
        raise ValueError('conv regressor is deprecated')
        super(XRDConvRegressor, self).__init__()

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
        result = result.squeeze(1)
        return result