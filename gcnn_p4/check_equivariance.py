import torch
import torch.nn as nn
import torch.nn.functional as F
import gcnn_p4.P4GConv2d as g

'''
We want to check equivariance of Z2P4GConv2d and P4P4Gconv2d in this file.
Approach 1: rotate the image and then put it into the network
Approach 2: put the image into the network and then rotate the reuslt.
So it's equivalent to check whether these two approaches give the same results.
'''

def check_equivariant(network, in_channel, out_channel, image_size, batch_size):

    x = torch.randn(batch_size, in_channel, image_size, image_size)
    result = network(x)

    for i in range(1, 4):
        
        result_rotate = torch.rot90(result, i, (3, 4))
        x_rotate = torch.rot90(x, i, (2, 3))
        rotate_result = network(x_rotate)

        for b in range(batch_size):
            for c in range(out_channel):

                result_rotate_fixed = result_rotate[b][c].reshape(4, -1)
                rotate_result_fixed = rotate_result[b][c].reshape(4, -1)

                first_column1 = result_rotate_fixed[:, 0]
                first_column2 = rotate_result_fixed[:, 0]

                sorted_indices1 = torch.argsort(first_column1)
                sorted1 = result_rotate_fixed[sorted_indices1]
                sorted_indices2 = torch.argsort(first_column2)
                sorted2 = rotate_result_fixed[sorted_indices2]

                for ind in range(4):
                    if torch.allclose(sorted1[i], sorted2[i], atol=1e-6) == False:
                        print(f'b:{b}, c:{c}, i:{i}')
                        print(sorted1[i])
                        print(sorted2[i])
                        difference = torch.abs(sorted1[i] - sorted2[i])
                        print("Max absolute difference:", torch.max(difference))
                    assert torch.allclose(sorted1[i], sorted2[i], atol=1e-5)


class test_P4GConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(test_P4GConvNet, self).__init__()
        self.gconv1 = g.Z2P4GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):

        x = self.gconv1(x)
        x = g.GConv2d_MaxPooling(x, 2)

        return x



batch_size = 64
in_channels = 3
out_channels = 64
kernel_size = 3
image_size = 28

model = test_P4GConvNet(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
check_equivariant(model, in_channel=in_channels, out_channel=out_channels, image_size=image_size, batch_size=batch_size)
print('True')
