import torch
import torch.nn as nn
import torch.nn.functional as F

class P4GConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(P4GConv2d, self).__init__()
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

        gconv = torch.stack(outputs, dim=1)
        gconv_output = gconv.view(gconv.size()[0], gconv.size()[1] * gconv.size()[2],
                                    gconv.size()[3], gconv.size()[4])
        return gconv_output
    

class P4GConvNet(nn.Module):

    def __init__(self):

        super(P4GConvNet, self).__init__()
        self.gconv1 = P4GConv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gconv2 = P4GConv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):

        x = self.gconv1(x)
        x = self.pool1(x)
        x = self.gconv2(x)
        x = self.pool2(x)
        x = self.fc(x)

        return x
