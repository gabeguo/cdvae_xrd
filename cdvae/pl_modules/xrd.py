import torch
from torch import nn
import numpy as np

class DiffractionPatternEmbedder(nn.Module):
    def __init__(self, xrd_dims=512, latent_dims=256, num_blocks=3, num_channels=6):
        super(DiffractionPatternEmbedder, self).__init__()

        self.num_blocks = num_blocks
        self.xrd_dims = xrd_dims
        self.latent_dims = latent_dims
        self.num_channels = num_channels

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.num_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([self.num_channels, self.xrd_dims]),
            nn.ReLU()
        )

        self.conv_blocks_1 = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=self.num_channels*(i+1), out_channels=self.num_channels, 
                          kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([self.num_channels, self.xrd_dims]),
                nn.ReLU()
            ) 
            for i in range(self.num_blocks)]
        )

        self.transition = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels*(self.num_blocks+1), out_channels=self.num_channels*2, 
                      kernel_size=1, bias=False),
            nn.LayerNorm([self.num_channels*2, self.xrd_dims]),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # decrease dimensionality to self.xrd_dims // 2
        )

        self.conv_blocks_2 = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=self.num_channels*(i+2), out_channels=self.num_channels, 
                          kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([self.num_channels, self.xrd_dims // 2]),
                nn.ReLU()
            ) 
            for i in range(self.num_blocks)]
        )

        self.lastfc = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels*(self.num_blocks+2), out_channels=1, kernel_size=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(self.xrd_dims // 2, self.latent_dims),
            nn.BatchNorm1d(num_features=self.latent_dims)
        )

        print('classifier free guidance with XRD embedding')

        return

    """
    Input is (n_channels, 512)-dimensional
    Output is 256-dimensional
    """
    def forward(self, batch):
        assert len(batch.num_atoms.shape) == 1
        batch_size = batch.num_atoms.shape[0]
        assert batch.y.shape == (batch_size * self.xrd_dims, 1)
        x = batch.y.reshape(batch_size, 1, self.xrd_dims) # make it channels (N, C, L)

        assert self.xrd_dims == 512
        assert self.latent_dims == 256

        # first convolution: give it some channels
        x = self.first_conv(x)
        assert len(x.shape) == 3
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.num_channels
        assert x.shape[2] == self.xrd_dims

        # first group of densely connected conv blocks - size: (batch_size x num_channels x 1024)
        x_history_1 = [x]
        for i, the_block in enumerate(self.conv_blocks_1):
            assert len(x_history_1) == i + 1 # make sure we are updating the history list
            x = the_block(torch.cat(x_history_1, dim=1))
            x_history_1.append(x) # add new result to running list
            assert len(x.shape) == 3
            assert x.shape[1] == self.num_channels
            assert x.shape[2] == self.xrd_dims
        assert len(x_history_1) == len(self.conv_blocks_1) + 1 # make sure we hit all the blocks
 
        # transition layer: downsize combo of all previous feature maps to (batch_size, (2 * num_channels), 512)
        x = self.transition(torch.cat(x_history_1, dim=1))
        assert len(x.shape) == 3
        assert x.shape[1] == self.num_channels * 2
        assert x.shape[2] == self.xrd_dims // 2

        # second group of densely connected conv blocks
        x_history_2 = [x] # start with downsized
        for i, the_block in enumerate(self.conv_blocks_2):
            assert len(x_history_2) == i + 1 # make sure we are updating the history list
            x = the_block(torch.cat(x_history_2, dim=1))
            x_history_2.append(x) # add new result to running list
            assert len(x.shape) == 3
            assert x.shape[1] == self.num_channels
            assert x.shape[2] == self.xrd_dims // 2
        assert len(x_history_2) == len(self.conv_blocks_2) + 1 # make sure we hit all the blocks

        # get final output
        x_final = self.lastfc(torch.cat(x_history_2, dim=1))

        assert len(x_final.shape) == 2
        assert x_final.shape[0] == batch_size
        assert x_final.shape[1] == self.latent_dims
        return x_final


class XRDDenseRegressor(nn.Module):
    def __init__(self, latent_dim=256, xrd_dim=512, num_blocks=4):
        super(XRDDenseRegressor, self).__init__()
        raise ValueError('do not use this one: we should directly condition on XRD')

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