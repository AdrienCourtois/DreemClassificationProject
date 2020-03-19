'''
Mish activation function, as a torch.nn.Module.

Implementation of:
Mish: A Self Regularized Non-Monotonic Neural Activation Function
Diganta Misra
https://arxiv.org/abs/1908.08681
'''

import torch
import torch.nn as nn

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.exp(x)))