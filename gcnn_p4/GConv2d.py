import torch
import torch.nn as nn
import torch.nn.functional as F

class GConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = 


