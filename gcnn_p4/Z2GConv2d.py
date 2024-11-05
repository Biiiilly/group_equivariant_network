import torch
import torch.nn as nn
import torch.nn.functional as F

class Z2GConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Z2GConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Say kernel_size = n if the kernel has size nxn
        self.kernel_size = kernel_size 

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.basis = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        outputs = []

    def rotate_filter()



