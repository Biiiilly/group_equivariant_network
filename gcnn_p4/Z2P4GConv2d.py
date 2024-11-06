import torch
import torch.nn as nn
import torch.nn.functional as F

class Z2P4GConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(Z2P4GConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Say kernel_size = n if the kernel has size nxn
        self.kernel_size = kernel_size 

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x):

        outputs = []

        for i in range(4):
            rotated_weight = torch.rot90(self.weight, i, [2, 3])
            output = F.conv2d(x, rotated_weight, bias=self.bias, padding='same')
            outputs.append(output)

        gconv_output = torch.stack(outputs, dim=1)
        return gconv_output
