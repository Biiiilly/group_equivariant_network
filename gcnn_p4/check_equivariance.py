import torch
import torch.nn as nn
import torch.nn.functional as F
import P4GConv2d as g

'''
We want to check equivariance of Z2P4GConv2d and P4P4Gconv2d in this file.
Approach 1: rotate the image and then put it into the network
Approach 2: put the image into the network and then rotate the reuslt.
So it's equivalent to check whether these two approaches give the same results.
'''

def check_equivariant(network):

    x = torch.randn(1, 3, 28, 28)
    result = network(x)
    
    for i in range(1, 4):
        
        result_rotate = torch.rot90(result, i, (3, 4))
        x_rotate = torch.rot90(x, i, (2, 3))
        rotate_result = network(x_rotate)

        assert torch.allclose(torch.sort(rotate_result, dim=2)[0], torch.sort(rotate_result, dim=2)[0], atol=1e-8)


class test_Z2P4GConvNet(nn.Module):

    def __init__(self):

        super(test_Z2P4GConvNet, self).__init__()
        self.gconv1 = g.Z2P4GConv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.gconv2 = g.P4P4GConv2d(in_channels=16, out_channels=32, kernel_size=3)

    def forward(self, x):

        x = self.gconv1(x)
        x = self.gconv2(x)

        return x
    
model = test_Z2P4GConvNet()
check_equivariant(model)
