'''
Refactor class as a torch.nn.Module.

Reshape the input tensor so that it goes from
(B, C, H, W) to (B, C/7, 7*H, W)
'''

import torch
import torch.nn as nn

class Refactor(nn.Module):
    def __init__(self, factor):
        super().__init__()

        self.factor = factor
    
    def forward(self, x):
        N, C, H, W = x.size()
        new_C = int(C / self.factor)
        new_H = int(H * self.factor)

        return x.view(N, new_C, new_H, W)