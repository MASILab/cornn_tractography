# Module Definitions
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
import torch.nn as nn
import numpy as np

# Class Definitions

class TrilinearInterpolator(nn.Module):

    def __init__(self):

        super(TrilinearInterpolator, self).__init__()

    def forward(self, img, trid, trii):

        assert len(img.shape) == 5 and img.shape[0] == 1, 'img must be a 5D tensor (batch, channel, x, y, z) with batch = 1.'
        img = torch.permute(img, dims=(2, 3, 4, 1, 0)).squeeze(-1) # (b=1, c, x, y, z) => (x, y, z, c)
        img = torch.flatten(img, start_dim=0, end_dim=2) # (x, y, z, c) => (xyz, c)

        # Source: https://www.wikiwand.com/en/Trilinear_interpolation

        xd = trid[:, 0].unsqueeze(1)
        yd = trid[:, 1].unsqueeze(1)
        zd = trid[:, 2].unsqueeze(1)
        
        c000 = img[trii[:, 0], :]
        c100 = img[trii[:, 1], :]
        c010 = img[trii[:, 2], :]
        c001 = img[trii[:, 3], :]
        c110 = img[trii[:, 4], :]
        c101 = img[trii[:, 5], :]
        c011 = img[trii[:, 6], :]
        c111 = img[trii[:, 7], :]

        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd

        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd

        c = c0*(1-zd) + c1*zd

        return c

class DetCNNFake(nn.Module):

    def __init__(self):

        super(DetCNNFake, self).__init__()

    def forward(self, img):

        return img    

class DetConvProj(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=0):

        super(DetConvProj, self).__init__()

        if kernel_size == 0:
            self.c = lambda x : x
        else:
            self.c = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, img):

        return self.c(img)

class DetRNN(nn.Module):

    def __init__(self, in_features, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=4):
    
        super(DetRNN, self).__init__()

        self.interp = TrilinearInterpolator()
        self.fc = nn.Sequential(nn.Linear(in_features, fc_width), *[self.block(fc_width, fc_width) for _ in range(fc_depth)])
        self.rnn = nn.GRU(input_size=fc_width, hidden_size=rnn_width, num_layers=rnn_depth)

        self.azi = nn.Sequential(nn.Linear(fc_width + rnn_width, 1), nn.Tanh())        # just rnn_width if fodtest without res
        self.ele = nn.Sequential(nn.Linear(fc_width + rnn_width, 1), nn.Sigmoid())

    def block(self, in_features, out_features):
    
        return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features, track_running_stats=False), nn.LeakyReLU(0.1))

    def forward(self, img, trid, trii, h=None): # Take either a packed sequence (train) or batch x feature tensor (gen) and return a (padded) seq x batch x feature tensor

        # Check input types

        seq = isinstance(trid, nn.utils.rnn.PackedSequence) and isinstance(trii, nn.utils.rnn.PackedSequence)

        # Interpolate imaging features

        if seq:
            trid_data = trid.data
            trii_data = trii.data
        else:
            trid_data = trid
            trii_data = trii
        z = self.interp(img, trid_data, trii_data)

        # Embed through FC

        x = self.fc(z) # FC requires batch x features (elements x features for packed sequence)
        if seq:
            x = nn.utils.rnn.PackedSequence(x, batch_sizes=trid.batch_sizes, sorted_indices=trid.sorted_indices, unsorted_indices=trid.unsorted_indices)
        else:
            x = x.unsqueeze(0)

        # Propagate through RNN 

        if h is not None:
            p, h = self.rnn(x, h) # RNN takes only packed sequences or seq x batch x feature tensors
        else:
            p, h = self.rnn(x)

        if seq: # p is a packed sequence (ele x feat) or seq=1 x batch x feat tensor
            y = torch.cat((x.data, p.data), dim=-1)
        else:
            y = torch.cat((x, p), dim=-1)

        # Format output

        a = np.pi * self.azi(y)
        e = np.pi * self.ele(y)
        
        dx = 1 * torch.sin(e) * torch.cos(a)
        dy = 1 * torch.sin(e) * torch.sin(a)
        dz = 1 * torch.cos(e)
        ds = torch.cat((dx, dy, dz), dim=-1)

        if seq:
            ds = nn.utils.rnn.PackedSequence(ds, batch_sizes=trid.batch_sizes, sorted_indices=trid.sorted_indices, unsorted_indices=trid.unsorted_indices)
            ds, _ = nn.utils.rnn.pad_packed_sequence(ds, batch_first=False)
            x = x.data
        
        return ds, a, e, h, x
