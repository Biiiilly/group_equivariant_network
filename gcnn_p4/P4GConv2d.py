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
            output = F.conv2d(x, rotated_weight, bias=None, padding='same')
            outputs.append(output)

        return torch.stack(outputs, dim=2)
    

class P4P4GConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(P4P4GConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels * 4, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x):

        xshape = x.size()
        x = x.view(xshape[0], xshape[1]*xshape[2], xshape[3], xshape[4])

        outputs = []

        for i in range(4):
            rotated_weight = torch.rot90(self.weight, i, [2, 3])
            output = F.conv2d(x, rotated_weight, bias=self.bias, padding='same')
            outputs.append(output)

        return torch.stack(outputs, dim=2)


def GConv2d_MaxPooling(x, kernel_size):

    size = x.size()
    x = x.view(size[0], size[1]*size[2], size[3], size[4])
    x = F.max_pool2d(x, kernel_size)
    x = x.view(size[0], size[1], size[2], int(size[3]/kernel_size), int(size[4]/kernel_size))

    return x


class P4GConvNet(nn.Module):

    def __init__(self):

        super(P4GConvNet, self).__init__()
        # self.batck_size = batch_size
        self.gconv1 = Z2P4GConv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.gconv2 = P4P4GConv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc = nn.Linear(16 * 4 * 7 * 7, 10)

    def forward(self, x):

        x = self.gconv1(x)
        x = GConv2d_MaxPooling(x, 2)
        x = self.gconv2(x)
        x = GConv2d_MaxPooling(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
