'''
Implementation of a low pass filter as a torch.nn.Module.

It considers the input to be of shape (B, C, H, W) where each
element (:,:,:,i) is a signal. Each such signal is filtered using 
the chosen low pass filter.
This filter is not learnable.
'''

import pywt
import numpy as np
import torch
import torch.nn as nn

class LowPassFilter(nn.Module):
    def __init__(self, name="db3"):
        super().__init__()
        w = pywt.Wavelet(name)
        enc_low, _, _, _ = w.filter_bank

        self.conv = nn.Conv1d(1, 1, len(enc_low), padding=int((len(enc_low)-1)/2))
        self.conv.weight.data = torch.tensor(np.array(enc_low)).view(1,1,len(enc_low)).float()
        self.conv.requires_grad_(False)
    
    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N*C*H, 1, W)
        x = self.conv(x)
        x = x.view(N, C, H, x.size(2))

        return x